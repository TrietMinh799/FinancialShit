"""server.py — Flask HTTP server: routes and entry point."""
from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from core.analysis import analyze_report
from core.config import OPENAI_MODEL, ensure_dirs
from core.llm import answer_question, generate_kb_company_report, test_openai_key
from core.store import Store
from core.text_utils import clean_text

app = Flask(__name__, static_folder="web/static", template_folder="web")


# ---------------------------------------------------------------------------
# Static / HTML
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("web", "index.html")


# ---------------------------------------------------------------------------
# GET routes
# ---------------------------------------------------------------------------

@app.route("/api/library")
def library():
    try:
        return jsonify(Store().stats())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# POST routes
# ---------------------------------------------------------------------------

@app.route("/api/test-key", methods=["POST"])
def test_key():
    payload = request.get_json(silent=True) or {}
    api_key = clean_text(payload.get("api_key", ""))
    if not api_key:
        return jsonify({"error": "Paste your ChatGPT API key first."}), 400
    ok = test_openai_key(api_key, payload.get("model") or OPENAI_MODEL)
    return jsonify({
        "ok": ok,
        "message": "API key works." if ok else "The API key test did not return a response.",
    })


@app.route("/api/ask", methods=["POST"])
def ask():
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
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/upload-book", methods=["POST"])
def upload_book():
    file = request.files.get("book_file")
    if not file or not file.filename:
        return jsonify({"error": "Choose a book PDF, EPUB, DOCX, TXT, or MD file."}), 400

    from core.config import UPLOAD_DIR
    import uuid
    from core.store import safe_filename

    target = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_filename(file.filename)}"
    file.save(str(target))

    title = request.form.get("book_title") or Path(file.filename).stem
    try:
        result = Store().add_document(target, title, "book")
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/analyze-report", methods=["POST"])
def analyze_report_route():
    file = request.files.get("report_file")
    if not file or not file.filename:
        return jsonify({"error": "Choose an annual report PDF."}), 400

    from core.config import UPLOAD_DIR
    import uuid
    from core.store import safe_filename

    target = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_filename(file.filename)}"
    file.save(str(target))

    company = request.form.get("company") or "Uploaded Company"
    ticker = request.form.get("ticker") or "UPLOAD"
    api_key = request.form.get("api_key") or ""
    model = request.form.get("model") or OPENAI_MODEL

    try:
        store = Store()
        result = analyze_report(store, target, company, ticker)
        result["llm_report"] = generate_kb_company_report(store, company, ticker, api_key, model)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


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
