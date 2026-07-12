"""server.py — Flask HTTP server: routes and entry point."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from core.analysis import analyze_report
from core.config import LLM_BASE_URL, OPENAI_MODEL, ensure_dirs
from core.llm import answer_question, generate_kb_company_report, test_openai_key
from core.store import Store, safe_filename
from core.text_utils import clean_text, sanitize_field

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="web/static", template_folder="web")

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
        return jsonify(Store().stats())
    except Exception:
        logger.exception("library error")
        return jsonify({"error": "Failed to load library."}), 500


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
        count = Store().reindex_all()
        return jsonify({"ok": True, "reindexed_chunks": count})
    except Exception as exc:
        logger.exception("reindex error")
        return jsonify({"error": f"Re-index failed: {exc}"}), 500


# ---------------------------------------------------------------------------
# POST routes
# ---------------------------------------------------------------------------


@app.route("/api/test-key", methods=["POST"])
def test_key() -> Response:
    payload = request.get_json(silent=True) or {}
    api_key = clean_text(payload.get("api_key", ""))
    if not api_key:
        return jsonify({"error": "Paste your API key first."}), 400
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
        return jsonify({"ok": False, "message": str(exc)}), 200


@app.route("/api/ask", methods=["POST"])
def ask() -> Response:
    payload = request.get_json(silent=True) or {}
    question = clean_text(payload.get("question", ""))
    if not question:
        return jsonify({"error": "Type a question first."}), 400
    try:
        result = answer_question(
            Store(),
            question,
            payload.get("api_key") or "",
            payload.get("model") or OPENAI_MODEL,
            payload.get("base_url") or LLM_BASE_URL,
        )
        return jsonify(result)
    except Exception:
        logger.exception("ask error")
        return jsonify({"error": "Failed to process question."}), 500


@app.route("/api/upload-book", methods=["POST"])
def upload_book() -> Response:
    file = request.files.get("book_file")
    if not file or not file.filename:
        return jsonify({"error": "Choose a book PDF, EPUB, DOCX, TXT, or MD file."}), 400

    from core.config import UPLOAD_DIR

    target = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_filename(file.filename)}"
    file.save(str(target))

    title = sanitize_field(request.form.get("book_title") or Path(file.filename).stem)
    try:
        result = Store().add_document(target, title, "book")
        return jsonify(result)
    except Exception:
        logger.exception("upload-book error")
        return jsonify({"error": "Failed to process uploaded book."}), 500


@app.route("/api/analyze-report", methods=["POST"])
def analyze_report_route() -> Response:
    file = request.files.get("report_file")
    if not file or not file.filename:
        return jsonify({"error": "Choose an annual report PDF."}), 400

    from core.config import UPLOAD_DIR

    target = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_filename(file.filename)}"
    file.save(str(target))

    company = sanitize_field(request.form.get("company") or "Uploaded Company")
    ticker = sanitize_field(request.form.get("ticker") or "UPLOAD", max_len=20)
    api_key = request.form.get("api_key") or ""
    model = request.form.get("model") or OPENAI_MODEL
    base_url = request.form.get("base_url") or LLM_BASE_URL

    try:
        store = Store()
        result = analyze_report(store, target, company, ticker)
        result["llm_report"] = generate_kb_company_report(
            store, company, ticker, api_key, model, base_url
        )
        return jsonify(result)
    except Exception:
        logger.exception("analyze-report error")
        return jsonify({"error": "Failed to analyze report."}), 500


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
    print(f"Valuation RAG running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
