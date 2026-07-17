"""server.py — Flask HTTP server: routes and entry point."""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from core.analysis import analyze_report
from core.config import LLM_BASE_URL, OPENAI_MODEL, ensure_dirs, RERANK_TOP_K
from core.llm import answer_question, decompose_query, generate_kb_company_report, test_openai_key, fallback_answer
from core.reranker import rerank
from core.store import Store, safe_filename
from core.text_utils import clean_text, query_terms, sanitize_field, unique

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="web/static", template_folder="web")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

_store = Store()

# ---------------------------------------------------------------------------
# Provider presets — returned to the UI so it can populate the picker
# ---------------------------------------------------------------------------

PROVIDERS: list[dict] = [
    {
        "id": "openai",
        "label": "ChatGPT (OpenAI)",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o3-mini"],
        "key_placeholder": "sk-proj-…",
        "color": "#10a37f",
    },
    {
        "id": "anthropic_or",
        "label": "Claude (via OpenRouter)",
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "anthropic/claude-sonnet-4-5",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-haiku",
            "anthropic/claude-3-opus",
        ],
        "key_placeholder": "sk-or-v1-…",
        "color": "#d97706",
    },
    {
        "id": "gemini_or",
        "label": "Gemini (via OpenRouter)",
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "google/gemini-2.0-flash-exp:free",
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash",
            "google/gemini-pro-1.5",
        ],
        "key_placeholder": "sk-or-v1-…",
        "color": "#4285f4",
    },
    {
        "id": "openrouter",
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "deepseek/deepseek-r1:free",
        ],
        "key_placeholder": "sk-or-v1-…",
        "color": "#7c3aed",
    },
    {
        "id": "groq",
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "models": [
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "key_placeholder": "gsk_…",
        "color": "#f97316",
    },
    {
        "id": "ollama",
        "label": "Ollama (local)",
        "base_url": "http://localhost:11434/v1",
        "models": ["llama3.2", "llama3.1", "mistral", "gemma2", "phi3"],
        "key_placeholder": "ollama",
        "color": "#64748b",
    },
    {
        "id": "custom",
        "label": "Custom / Self-hosted",
        "base_url": "",
        "models": [],
        "key_placeholder": "API key…",
        "color": "#6366f1",
    },
]


# ---------------------------------------------------------------------------
# Static / HTML
# ---------------------------------------------------------------------------


@app.route("/")
def index() -> Response:
    return send_from_directory("web", "index.html")


# ---------------------------------------------------------------------------
# GET routes
# ---------------------------------------------------------------------------


@app.route("/api/library")
def library() -> Response:
    try:
        return jsonify(_store.stats())
    except Exception:
        logger.exception("library error")
        resp = jsonify({"error": "Failed to load library."})
        resp.status_code = 500
        return resp


@app.route("/api/books/<int:doc_id>/chunks")
def book_chunks(doc_id: int) -> Response:
    """Return paginated chunks for a document."""
    try:
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)
        return jsonify(_store.get_document_chunks(doc_id, limit, offset))
    except ValueError as exc:
        resp = jsonify({"error": str(exc)})
        resp.status_code = 404
        return resp
    except Exception:
        logger.exception("book-chunks error")
        resp = jsonify({"error": "Failed to load chunks."})
        resp.status_code = 500
        return resp


@app.route("/api/books/<int:doc_id>", methods=["DELETE"])
def delete_book(doc_id: int) -> Response:
    try:
        result = _store.delete_document(doc_id)
        return jsonify(result)
    except ValueError as exc:
        resp = jsonify({"error": str(exc)})
        resp.status_code = 404
        return resp
    except Exception:
        logger.exception("delete-book error")
        resp = jsonify({"error": "Failed to delete document."})
        resp.status_code = 500
        return resp


@app.route("/api/books/<int:doc_id>/reclassify", methods=["POST"])
def reclassify_book(doc_id: int) -> Response:
    """Change a document's source_type and re-index its chunks."""
    data = request.get_json(silent=True) or {}
    new_type = data.get("source_type") or ""
    if new_type not in ("book", "annual_report"):
        resp = jsonify({"error": "source_type must be 'book' or 'annual_report'"})
        resp.status_code = 400
        return resp
    try:
        result = _store.reclassify_document(doc_id, new_type)
        return jsonify(result)
    except ValueError as exc:
        resp = jsonify({"error": str(exc)})
        resp.status_code = 404
        return resp
    except Exception:
        logger.exception("reclassify-book error")
        resp = jsonify({"error": "Failed to reclassify document."})
        resp.status_code = 500
        return resp


@app.route("/api/providers")
def providers() -> Response:
    """Return the list of supported LLM provider presets."""
    return jsonify({"providers": PROVIDERS, "default_base_url": LLM_BASE_URL})


@app.route("/api/reindex", methods=["POST"])
def reindex() -> Response:
    """Re-embed all stored chunks with the current embedding model.

    Call this once after changing EMBED_MODEL so the vector index matches
    the new model's embedding space.
    """
    try:
        count = _store.reindex_all()
        return jsonify({"ok": True, "reindexed_chunks": count})
    except Exception as exc:
        logger.exception("reindex error")
        resp = jsonify({"error": f"Re-index failed: {exc}"})
        resp.status_code = 500
        return resp


# ---------------------------------------------------------------------------
# POST routes
# ---------------------------------------------------------------------------


@app.route("/api/test-key", methods=["POST"])
def test_key() -> Response:
    payload = request.get_json(silent=True) or {}
    api_key = clean_text(payload.get("api_key", ""))
    if not api_key:
        resp = jsonify({"error": "Paste your API key first."})
        resp.status_code = 400
        return resp
    base_url = payload.get("base_url") or LLM_BASE_URL
    model = payload.get("model") or OPENAI_MODEL
    try:
        ok = test_openai_key(api_key, model, base_url)
        return jsonify(
            {
                "ok": ok,
                "message": "API key works."
                if ok
                else "The API key test did not return a response.",
            }
        )
    except Exception as exc:
        logger.exception("test-key error")
        resp = jsonify({"ok": False, "message": str(exc)})
        resp.status_code = 200
        return resp


# ---------------------------------------------------------------------------
# Streaming HTTP endpoint for LLM responses
# ---------------------------------------------------------------------------

from flask import stream_with_context
from core.llm import call_openai_llm, call_openai_llm_stream


@app.route("/api/ask/stream", methods=["POST"])
def ask_stream() -> Response:
    payload = request.get_json(silent=True) or {}
    question = clean_text(payload.get("question", ""))
    if not question:
        resp = jsonify({"error": "Type a question first."})
        resp.status_code = 400
        return resp

    api_key = payload.get("api_key") or ""
    model = payload.get("model") or OPENAI_MODEL
    base_url = payload.get("base_url") or LLM_BASE_URL
    history = payload.get("messages") or []

    def generate():
        t0 = time.time()

        try:
            # Phase 1 — Decompose
            yield _sse({"status": "Analyzing question..."})
            sub_queries = decompose_query(question, api_key, model, base_url, history)
            # Always include the original question for coverage — decomposed
            # sub-queries can drift from the user's actual intent.
            if question not in sub_queries:
                sub_queries.append(question)

            # Phase 2 — Parallel hybrid search across sub-queries
            yield _sse({"status": f"Searching library ({len(sub_queries)} queries)..."})
            all_citations: list[dict] = []
            with ThreadPoolExecutor(max_workers=min(len(sub_queries), 4)) as pool:
                fut_map = {
                    pool.submit(_store.hybrid_search, q, ["book", "annual_report"], 30): q
                    for q in sub_queries
                }
                for fut in as_completed(fut_map):
                    try:
                        all_citations.extend(fut.result())
                    except Exception:
                        pass
            t1 = time.time()
            print(f"[timing] hybrid_search: {t1-t0:.1f}s | sub_queries: {sub_queries}", flush=True)

            citations = unique(all_citations, 50)

            # Phase 2b — Broader fallback if no results found
            if not citations:
                yield _sse({"status": "Broadening search..."})
                broad_terms = query_terms(question)
                if broad_terms:
                    broad_query = " ".join(broad_terms[:15])
                    citations = _store.hybrid_search(
                        broad_query, ["book", "annual_report"], 20
                    )
                    if citations:
                        citations = unique(citations, 20)

            if citations:
                yield _sse({"status": "Ranking results..."})
                citations = rerank(question, citations, top_k=RERANK_TOP_K)
            t2 = time.time()
            print(f"[timing] rerank: {t2-t1:.1f}s | citations: {len(citations)}", flush=True)

            # Phase 3 — Context expansion & multi-strategy query expansion
            if citations:
                yield _sse({"status": "Expanding context..."})
                expanded = _store.expand_context(citations)
                citations = expanded[:RERANK_TOP_K]

                # Strategy A — Keyword-based query expansion from top passages
                yield _sse({"status": "Expanding search (keywords)..."})
                extra_terms: set[str] = set()
                for c in citations[:min(5, len(citations))]:
                    text = c.get("context_text") or c.get("snippet", "")
                    terms = query_terms(text)
                    extra_terms.update(t for t in terms[:8] if t.lower() not in question.lower())
                if extra_terms:
                    expansion = f"{question} {' '.join(list(extra_terms)[:12])}"
                    extra = _store.hybrid_search(expansion, ["book", "annual_report"], 15)
                    if extra:
                        citations = unique(citations + extra, 40)
                        if citations:
                            citations = rerank(question, citations, top_k=RERANK_TOP_K)
                            expanded = _store.expand_context(citations)
                            citations = expanded[:RERANK_TOP_K]

                # Strategy B — LLM-based sub-query expansion (when API key available)
                if api_key and citations:
                    yield _sse({"status": "Expanding search (LLM queries)..."})
                    try:
                        top_texts = []
                        for c in citations[:3]:
                            txt = c.get("context_text") or c.get("snippet", "")
                            if txt:
                                top_texts.append(txt[:300])
                        if top_texts:
                            expansion_prompt = (
                                "Original question: " + question + "\n\n"
                                "Top evidence snippets:\n" +
                                "\n---\n".join(top_texts) + "\n\n"
                                "Generate 2-3 alternative search queries that would find "
                                "additional relevant evidence the current snippets might miss. "
                                "Focus on different phrasings, synonyms, or related aspects. "
                                "Return ONLY the queries, one per line, no numbering."
                            )
                            alt_queries = call_openai_llm(
                                expansion_prompt, [], api_key, model, base_url, history=[]
                            )
                            if alt_queries and alt_queries.strip():
                                alt_lines = [ln.strip() for ln in alt_queries.splitlines() if ln.strip() and len(ln.strip()) > 5]
                                for aq in alt_lines[:3]:
                                    extra = _store.hybrid_search(aq, ["book", "annual_report"], 15)
                                    if extra:
                                        citations = unique(citations + extra, 40)
                                if len(citations) > RERANK_TOP_K:
                                    citations = rerank(question, citations, top_k=RERANK_TOP_K)
                                    expanded = _store.expand_context(citations)
                                    citations = expanded[:RERANK_TOP_K]
                    except Exception:
                        pass

            t3 = time.time()
            print(f"[timing] expansion: {t3-t2:.1f}s | final citations: {len(citations)}", flush=True)

            # Phase 4 — Answer
            if not citations or not api_key:
                answer = fallback_answer(question, citations)
                yield _sse({"token": answer})
                yield _sse({"done": True, "citations": citations, "mode": "rag",
                            "mode_label": "Evidence-based answer", "full_text": answer})
                return

            yield _sse({"status": "Generating answer with LLM..."})
            for chunk in call_openai_llm_stream(question, citations, api_key, model, base_url, history=history):
                if "token" in chunk:
                    yield _sse({"token": chunk["token"]})
                elif "done" in chunk:
                    yield _sse({"done": True, "citations": chunk.get("citations", []),
                                "mode": chunk.get("mode", "llm"),
                                "full_text": chunk.get("full_text", "")})
                    return
                elif "error" in chunk:
                    yield _sse({"error": chunk["error"]})
                    return

        except Exception as exc:
            logger.exception("ask-stream error")
            yield _sse({"error": str(exc)})

    return Response(stream_with_context(generate()), content_type="text/event-stream")


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Event data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.route("/api/ask", methods=["POST"])
def ask() -> Response:
    payload = request.get_json(silent=True) or {}
    question = clean_text(payload.get("question", ""))
    if not question:
        resp = jsonify({"error": "Type a question first."})
        resp.status_code = 400
        return resp
    try:
        result = answer_question(
            _store,
            question,
            payload.get("api_key") or "",
            payload.get("model") or OPENAI_MODEL,
            payload.get("base_url") or LLM_BASE_URL,
            history=payload.get("messages") or [],
            use_iterative=False,  # disable multi-round retrieval for faster responses with slow models
        )
        return jsonify(result)
    except Exception:
        logger.exception("ask error")
        resp = jsonify({"error": "Failed to process question."})
        resp.status_code = 500
        return resp


@app.route("/api/upload-book", methods=["POST"])
def upload_book() -> Response:
    file = request.files.get("book_file")
    if not file or not file.filename:
        resp = jsonify({"error": "Choose a book PDF, EPUB, DOCX, TXT, or MD file."})
        resp.status_code = 400
        return resp

    from core.config import UPLOAD_DIR

    target = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_filename(file.filename)}"
    file.save(str(target))

    title = sanitize_field(request.form.get("book_title") or Path(file.filename).stem)

    # Determine source type: explicit override > filename heuristic
    source_type = request.form.get("source_type") or ""
    if source_type not in ("book", "annual_report"):
        fname_lower = file.filename.lower()
        if any(kw in fname_lower for kw in ("annual report", "báo cáo annual", "báo cáo thường niên", "ar_", "_ar")):
            source_type = "annual_report"
        else:
            source_type = "book"

    try:
        result = _store.add_document(target, title, source_type)
        return jsonify(result)
    except Exception:
        logger.exception("upload-book error")
        resp = jsonify({"error": "Failed to process uploaded book."})
        resp.status_code = 500
        return resp


@app.route("/api/analyze-report", methods=["POST"])
def analyze_report_route() -> Response:
    file = request.files.get("report_file")
    if not file or not file.filename:
        resp = jsonify({"error": "Choose an annual report PDF."})
        resp.status_code = 400
        return resp

    from core.config import UPLOAD_DIR

    target = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_filename(file.filename)}"
    file.save(str(target))

    company = sanitize_field(request.form.get("company") or "Uploaded Company")
    ticker = sanitize_field(request.form.get("ticker") or "UPLOAD", max_len=20)
    api_key = request.form.get("api_key") or ""
    model = request.form.get("model") or OPENAI_MODEL
    base_url = request.form.get("base_url") or LLM_BASE_URL

    try:
        result = analyze_report(_store, target, company, ticker, api_key, model, base_url)
        result["llm_report"] = generate_kb_company_report(
            _store, company, ticker, api_key, model, base_url
        )
        return jsonify(result)
    except Exception:
        logger.exception("analyze-report error")
        resp = jsonify({"error": "Failed to analyze report."})
        resp.status_code = 500
        return resp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Valuation RAG — Flask server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    ensure_dirs()

    # Preload embedding and reranker models at startup so the first
    # query doesn't block for 30-90 seconds loading ML models on CPU.
    print("Loading SentenceTransformer (BAAI/bge-m3)…", flush=True)
    from core.vector_store import load_embedding_model
    load_embedding_model()

    print("Loading CrossEncoder (BAAI/bge-reranker-v2-m3)…", flush=True)
    from core.reranker import load_reranker_model
    load_reranker_model()

    print(f"Valuation RAG running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
