"""server.py — Flask HTTP server: routes and entry point."""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


def _get_real_ip() -> str:
    """Return the client IP, ignoring X-Forwarded-For to prevent spoofing.

    Strips IPv6 zone IDs (e.g. ``fe80::1%eth0`` → ``fe80::1``) so the
    rate limiter sees a canonical address.
    """
    ip = request.remote_addr or "127.0.0.1"
    # Strip IPv6 zone ID (everything after %)
    if "%" in ip:
        ip = ip.split("%", 1)[0]
    return ip

from core.analysis import analyze_report
from core.agent import run_agent
from core.config import LLM_BASE_URL, OPENAI_MODEL, ensure_dirs, RERANK_TOP_K
from core.llm import (
    answer_question,
    call_openai_llm,
    call_openai_llm_stream,
    decompose_query,
    fallback_answer,
    generate_kb_company_report,
    test_openai_key,
)
from core.reranker import rerank
from core.store import Store, safe_filename
from core.text_utils import clean_text, query_terms, sanitize_field, unique

logger = logging.getLogger(__name__)

# Redact API keys from logs (OpenAI, OpenRouter, Groq, Together, etc.)
import re as _re
class _RedactFilter(logging.Filter):
    _API_KEY_RE = _re.compile(r"(sk-[a-zA-Z0-9_\-]{20,}|gsk_[a-zA-Z0-9]{20,}|sk-or-v1-[a-zA-Z0-9_\-]{20,})")
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._API_KEY_RE.sub("[REDACTED]", record.msg)
        if record.args:
            record.args = tuple(
                self._API_KEY_RE.sub("[REDACTED]", str(a)) if isinstance(a, str) else a
                for a in record.args
            )
        return True

logger.addFilter(_RedactFilter())

# ---------------------------------------------------------------------------
# File signature (magic bytes) validation
# ---------------------------------------------------------------------------

_FILE_SIGNATURES: dict[str, list[bytes]] = {
    ".pdf": [b"%PDF"],
    ".epub": [b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"],  # EPUB is ZIP-based
    ".docx": [b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"],  # DOCX is ZIP-based
    ".txt": [],  # No signature, accept any
    ".md": [],   # No signature, accept any
}


def _validate_file_signature(file_stream, expected_ext: str) -> bool:
    """Check that the file's magic bytes match the expected extension."""
    signatures = _FILE_SIGNATURES.get(expected_ext.lower(), [])
    if not signatures:
        return True  # No signature to verify (txt, md)
    header = file_stream.read(8)
    file_stream.seek(0)
    return any(header.startswith(sig) for sig in signatures)


# ---------------------------------------------------------------------------
# ask_stream phase helpers
# ---------------------------------------------------------------------------

def _phase_decompose(question: str, api_key: str, model: str, base_url: str, history: list) -> list[str]:
    """Phase 1: Decompose question into sub-queries."""
    sub_queries = decompose_query(question, api_key, model, base_url, history)
    if question not in sub_queries:
        sub_queries.append(question)
    return sub_queries


def _phase_search(store: Store, sub_queries: list[str]) -> list[dict]:
    """Phase 2: Parallel hybrid search across sub-queries."""
    all_citations: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(sub_queries), 4)) as pool:
        fut_map = {
            pool.submit(store.hybrid_search, q, ["book", "annual_report"], 30): q
            for q in sub_queries
        }
        for fut in as_completed(fut_map):
            try:
                all_citations.extend(fut.result())
            except Exception:
                pass
    return unique(all_citations, 50)


def _phase_fallback_search(store: Store, question: str) -> list[dict]:
    """Phase 2b: Broader fallback search if no results."""
    broad_terms = query_terms(question)
    if not broad_terms:
        return []
    broad_query = " ".join(broad_terms[:15])
    citations = store.hybrid_search(broad_query, ["book", "annual_report"], 20)
    return unique(citations, 20) if citations else []


def _phase_rerank(question: str, citations: list[dict]) -> list[dict]:
    """Phase 2c: Rerank citations by relevance."""
    if citations:
        return rerank(question, citations, top_k=RERANK_TOP_K)
    return citations


def _phase_expand_context(store: Store, citations: list[dict]) -> list[dict]:
    """Phase 3a: Expand context with neighboring chunks."""
    if citations:
        expanded = store.expand_context(citations)
        return expanded[:RERANK_TOP_K]
    return citations


def _phase_keyword_expansion(store: Store, question: str, citations: list[dict]) -> list[dict]:
    """Phase 3b: Keyword-based query expansion from top passages."""
    if not citations:
        return citations
    extra_terms: set[str] = set()
    for c in citations[:min(5, len(citations))]:
        text = c.get("context_text") or c.get("snippet", "")
        terms = query_terms(text)
        extra_terms.update(t for t in terms[:8] if t.lower() not in question.lower())
    if not extra_terms:
        return citations
    expansion = f"{question} {' '.join(list(extra_terms)[:12])}"
    extra = store.hybrid_search(expansion, ["book", "annual_report"], 15)
    if not extra:
        return citations
    citations = unique(citations + extra, 40)
    citations = rerank(question, citations, top_k=RERANK_TOP_K)
    expanded = store.expand_context(citations)
    return expanded[:RERANK_TOP_K]


def _phase_llm_expansion(store: Store, question: str, api_key: str, model: str, base_url: str, citations: list[dict]) -> list[dict]:
    """Phase 3c: LLM-based sub-query expansion."""
    if not (api_key and citations):
        return citations
    try:
        top_texts = []
        for c in citations[:3]:
            txt = c.get("context_text") or c.get("snippet", "")
            if txt:
                top_texts.append(txt[:300])
        if not top_texts:
            return citations
        expansion_prompt = (
            "Original question: " + question + "\n\n"
            "Top evidence snippets:\n" +
            "\n---\n".join(top_texts) + "\n\n"
            "Generate 2-3 alternative search queries that would find "
            "additional relevant evidence the current snippets might miss. "
            "Focus on different phrasings, synonyms, or related aspects. "
            "Return ONLY the queries, one per line, no numbering."
        )
        alt_queries = call_openai_llm(expansion_prompt, [], api_key, model, base_url, history=[])
        if not alt_queries or not alt_queries.strip():
            return citations
        alt_lines = [ln.strip() for ln in alt_queries.splitlines() if ln.strip() and len(ln.strip()) > 5]
        for aq in alt_lines[:3]:
            extra = store.hybrid_search(aq, ["book", "annual_report"], 15)
            if extra:
                citations = unique(citations + extra, 40)
        if len(citations) > RERANK_TOP_K:
            citations = rerank(question, citations, top_k=RERANK_TOP_K)
            expanded = store.expand_context(citations)
            citations = expanded[:RERANK_TOP_K]
    except Exception:
        pass
    return citations


app = Flask(__name__, static_folder="web/static", template_folder="web")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


@app.before_request
def _csrf_protect() -> Response | None:
    """Reject cross-origin form submissions on mutating methods.

    All POST requests from the SPA include Content-Type: application/json
    (fetch default), which a simple <form> cannot produce.  The upload
    endpoints (/api/upload-book, /api/analyze-report) use multipart/form-data,
    so we additionally require X-Requested-With which is set automatically
    by fetch but not by a browser form submit.
    """
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        ct = (request.content_type or "").lower()
        # JSON payloads are browser-protected (no simple form can send them)
        if "application/json" in ct:
            return None
        # Multipart/form-data requires the X-Requested-With header
        if request.headers.get("X-Requested-With") != "XMLHttpRequest":
            resp = jsonify({"error": "CSRF check failed: missing X-Requested-With header."})
            resp.status_code = 403
            return resp
    return None

# Rate limiting
limiter = Limiter(
    _get_real_ip,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

_store = Store()

# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------

@app.after_request
def _add_security_headers(response: Response) -> Response:
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Referrer policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # HSTS (only effective over HTTPS; harmless on HTTP)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    # Basic CSP - adjust if you load external resources
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://fonts.googleapis.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://fonts.gstatic.com; "
        "font-src 'self' https://fonts.gstatic.com https://fonts.googleapis.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    return response

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
    except ValueError:
        resp = jsonify({"error": "Document not found."})
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
    except ValueError:
        resp = jsonify({"error": "Document not found."})
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
    except ValueError:
        resp = jsonify({"error": "Document not found."})
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
@limiter.limit("5 per hour")
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
        resp = jsonify({"error": "Re-index failed."})
        resp.status_code = 500
        return resp


# ---------------------------------------------------------------------------
# POST routes
# ---------------------------------------------------------------------------


@app.route("/api/test-key", methods=["POST"])
@limiter.limit("10 per hour")
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
    except Exception:
        logger.exception("test-key error")
        resp = jsonify({"ok": False, "message": "Test failed."})
        resp.status_code = 200
        return resp


# ---------------------------------------------------------------------------
# Streaming HTTP endpoint for LLM responses
# ---------------------------------------------------------------------------

from flask import stream_with_context


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Event data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.route("/api/ask/stream", methods=["POST"])
@limiter.limit("30 per minute")
def ask_stream() -> Response:
    payload = request.get_json(silent=True) or {}
    question = clean_text(payload.get("question", ""))
    if not question:
        resp = jsonify({"error": "Type a question first."})
        resp.status_code = 400
        return resp
    if len(question) > 2048:
        resp = jsonify({"error": "Question too long (max 2048 characters)."})
        resp.status_code = 400
        return resp

    api_key = payload.get("api_key") or ""
    model = payload.get("model") or OPENAI_MODEL
    base_url = payload.get("base_url") or LLM_BASE_URL
    history = payload.get("messages") or []
    # "agent" (default): LLM-driven retrieval loop. "classic": fixed
    # deterministic fallback pipeline.
    mode = payload.get("mode") or "agent"
    use_agent = mode == "agent" and bool(api_key)

    def generate_agent():
        try:
            for event in run_agent(_store, question, api_key, model, base_url, history):
                yield _sse(event)
        except Exception:
            logger.exception("ask-stream agent error")
            yield _sse({"error": "An internal error occurred."})

    def generate():
        t0 = time.time()

        try:
            # Phase 1 — Decompose
            yield _sse({"status": "Analyzing question..."})
            sub_queries = _phase_decompose(question, api_key, model, base_url, history)

            # Phase 2 — Parallel hybrid search across sub-queries
            yield _sse({"status": f"Searching library ({len(sub_queries)} queries)..."})
            citations = _phase_search(_store, sub_queries)
            t1 = time.time()
            logger.info("timing hybrid_search: %.1fs | sub_queries: %s", t1 - t0, sub_queries)

            # Phase 2b — Broader fallback if no results found
            if not citations:
                yield _sse({"status": "Broadening search..."})
                citations = _phase_fallback_search(_store, question)

            # Phase 2c — Rerank
            citations = _phase_rerank(question, citations)
            t2 = time.time()
            logger.info("timing rerank: %.1fs | citations: %d", t2 - t1, len(citations))

            # Phase 3a — Context expansion
            citations = _phase_expand_context(_store, citations)

            # Phase 3b — Keyword-based query expansion
            yield _sse({"status": "Expanding search (keywords)..."})
            citations = _phase_keyword_expansion(_store, question, citations)

            # Phase 3c — LLM-based sub-query expansion
            if api_key and citations:
                yield _sse({"status": "Expanding search (LLM queries)..."})
                citations = _phase_llm_expansion(_store, question, api_key, model, base_url, citations)

            t3 = time.time()
            logger.info("timing expansion: %.1fs | final citations: %d", t3 - t2, len(citations))

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

        except Exception:
            logger.exception("ask-stream error")
            yield _sse({"error": "An internal error occurred."})

    gen = generate_agent() if use_agent else generate()
    return Response(stream_with_context(gen), content_type="text/event-stream")


@app.route("/api/ask", methods=["POST"])
@limiter.limit("30 per minute")
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
@limiter.limit("10 per hour")
def upload_book() -> Response:
    file = request.files.get("book_file")
    if not file or not file.filename:
        resp = jsonify({"error": "Choose a book PDF, EPUB, DOCX, TXT, or MD file."})
        resp.status_code = 400
        return resp

    # Validate file extension
    allowed_ext = {".pdf", ".epub", ".docx", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_ext:
        resp = jsonify({"error": f"File type '{suffix}' not allowed. Allowed: {', '.join(allowed_ext)}"})
        resp.status_code = 400
        return resp

    # Validate file size (max 50 MB)
    file.seek(0, 2)  # seek to end
    size = file.tell()
    file.seek(0)
    if size > 50 * 1024 * 1024:
        resp = jsonify({"error": "File too large. Maximum size: 50 MB"})
        resp.status_code = 400
        return resp

    # Validate file signature (magic bytes)
    if not _validate_file_signature(file.stream, suffix):
        resp = jsonify({"error": f"File content does not match extension '{suffix}'"})
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
@limiter.limit("5 per hour")
def analyze_report_route() -> Response:
    file = request.files.get("report_file")
    if not file or not file.filename:
        resp = jsonify({"error": "Choose an annual report PDF."})
        resp.status_code = 400
        return resp

    # Validate file extension (PDF only for annual reports)
    suffix = Path(file.filename).suffix.lower()
    if suffix != ".pdf":
        resp = jsonify({"error": "Only PDF files allowed for annual reports."})
        resp.status_code = 400
        return resp

    # Validate file size (max 50 MB)
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > 50 * 1024 * 1024:
        resp = jsonify({"error": "File too large. Maximum size: 50 MB"})
        resp.status_code = 400
        return resp

    # Validate file signature (magic bytes)
    if not _validate_file_signature(file.stream, suffix):
        resp = jsonify({"error": "File content does not match PDF format"})
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
    import os

    parser = argparse.ArgumentParser(description="Valuation RAG — Flask server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--data-dir", default=None, help="Data directory (overrides DATA_DIR env var)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Debug mode only allowed if explicitly enabled via env var (safety for production)
    debug_mode = args.debug and os.environ.get("VALUATION_RAG_DEBUG") == "1"
    if args.debug and not debug_mode:
        print("WARNING: --debug ignored. Set VALUATION_RAG_DEBUG=1 to enable.", flush=True)

    if args.data_dir:
        import core.config as cfg
        cfg.DATA_DIR = Path(args.data_dir).resolve()
        cfg.ROOT = cfg.DATA_DIR
        cfg.UPLOAD_DIR = cfg.ROOT / "uploads"
        cfg.DB_PATH = cfg.ROOT / "rag.sqlite3"
        cfg.CHROMA_DIR = cfg.ROOT / "chroma"

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
    app.run(host=args.host, port=args.port, debug=debug_mode, threaded=True)


if __name__ == "__main__":
    main()
