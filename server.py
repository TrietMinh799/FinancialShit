"""server.py — HTTP server: GET/POST route handlers and entry point."""
from __future__ import annotations

import json
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from core.analysis import analyze_report
from core.config import OPENAI_MODEL, ensure_dirs
from core.llm import answer_question, generate_kb_company_report, test_openai_key
from core.store import Store, parse_multipart
from core.text_utils import clean_text

import mimetypes

# Load the single-file frontend once at import time
_HTML_PATH = Path(__file__).parent / "web" / "index.html"
HTML: str = _HTML_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# JSON response helper
# ---------------------------------------------------------------------------

def write_json(handler: BaseHTTPRequestHandler, data: object, status: int = 200) -> None:
    """Serialise *data* as JSON and write it to the HTTP response."""
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    """Single-threaded-safe HTTP handler; server runs it in a thread pool."""

    server_version = "ValuationRAG/0.1"

    # ------------------------------------------------------------------
    # GET routes
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._serve_html()
        elif parsed.path == "/api/library":
            self._handle_library()
        elif parsed.path.startswith("/static/"):
            self._serve_static(parsed.path)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _serve_static(self, path: str) -> None:
        clean_path = Path(path.lstrip("/")).resolve()
        project_dir = Path(__file__).parent.resolve()
        web_static_dir = project_dir / "web" / "static"
        
        target_path = web_static_dir / clean_path.name
        
        if not target_path.exists() or not target_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
            
        mime_type, _ = mimetypes.guess_type(target_path)
        mime_type = mime_type or "application/octet-stream"
        
        body = target_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self) -> None:
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_library(self) -> None:
        try:
            write_json(self, Store().stats())
        except Exception as exc:
            write_json(self, {"error": str(exc)}, 500)

    # ------------------------------------------------------------------
    # POST routes
    # ------------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            store = Store()
            if parsed.path in {"/api/ask", "/api/test-key"}:
                self._handle_json_post(parsed.path, store)
            elif parsed.path == "/api/upload-book":
                self._handle_upload_book(store)
            elif parsed.path == "/api/analyze-report":
                self._handle_analyze_report(store)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            write_json(self, {"error": str(exc)}, 500)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def _handle_json_post(self, path: str, store: Store) -> None:
        payload = self._read_json_body()

        if path == "/api/test-key":
            api_key = clean_text(payload.get("api_key", ""))
            if not api_key:
                write_json(self, {"error": "Paste your ChatGPT API key first."}, 400)
                return
            ok = test_openai_key(api_key, payload.get("model") or OPENAI_MODEL)
            write_json(
                self,
                {
                    "ok": ok,
                    "message": (
                        "API key works." if ok
                        else "The API key test did not return a response."
                    ),
                },
            )
            return

        # /api/ask
        question = clean_text(payload.get("question", ""))
        if not question:
            write_json(self, {"error": "Type a question first."}, 400)
            return
        write_json(
            self,
            answer_question(
                store,
                question,
                payload.get("api_key") or "",
                payload.get("model") or OPENAI_MODEL,
            ),
        )

    def _handle_upload_book(self, store: Store) -> None:
        fields, files = parse_multipart(self)
        path = files.get("book_file")
        if not path:
            write_json(
                self,
                {"error": "Choose a book PDF, EPUB, DOCX, TXT, or MD file."},
                400,
            )
            return
        result = store.add_document(path, fields.get("book_title") or path.stem, "book")
        write_json(self, result)

    def _handle_analyze_report(self, store: Store) -> None:
        fields, files = parse_multipart(self)
        path = files.get("report_file")
        if not path:
            write_json(self, {"error": "Choose an annual report PDF."}, 400)
            return
        company = fields.get("company") or "Uploaded Company"
        ticker = fields.get("ticker") or "UPLOAD"
        result = analyze_report(store, path, company, ticker)
        result["llm_report"] = generate_kb_company_report(
            store,
            company,
            ticker,
            fields.get("api_key") or "",
            fields.get("model") or OPENAI_MODEL,
        )
        write_json(self, result)

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: N802
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {fmt % args}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Valuation RAG HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    ensure_dirs()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Valuation RAG running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
