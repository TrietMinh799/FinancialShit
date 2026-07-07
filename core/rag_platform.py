"""rag_platform.py — Backward-compatibility shim.

All implementation has been split into focused modules:
  config.py      — paths, constants, vocabulary dicts
  text_utils.py  — text cleaning, chunking, snippet helpers
  extractors.py  — PDF / EPUB / DOCX / TXT page extractors
  store.py       — Store (SQLite + FTS), multipart parsing, hashing
  analysis.py    — scoring, analyze_report, SWOT
  llm.py         — call_openai_llm, answer_question, generate_kb_company_report
  server.py      — HTTP Handler, write_json, main()

This file re-exports the full public API so that any code that previously
imported directly from rag_platform continues to work unchanged.
"""
from __future__ import annotations

# Re-export everything from each module so external callers keep working
from core.config import (
    DB_PATH,
    EXECUTION_TERMS,
    GROWTH_TERMS,
    MODEL,
    MOAT_TERMS,
    OPENAI_MODEL,
    OPENROUTER_MODELS_URL,
    RISK_TERMS,
    ROOT,
    RUNS,
    STOPWORDS,
    UPLOAD_DIR,
    ensure_dirs,
)
from core.text_utils import (
    annual_hits,
    chunk_pages,
    clean_text,
    clip,
    matched_labels,
    mention_count,
    query_terms,
    snippet_for,
    strip_markup,
    unique,
)
from core.extractors import (
    extract_docx_pages,
    extract_epub_pages,
    extract_pages,
)
from core.store import (
    Store,
    document_hash,
    parse_content_disposition,
    parse_multipart,
    safe_filename,
)
from core.analysis import analyze_report
from core.llm import (
    answer_question,
    build_context,
    call_openai_llm,
    fallback_answer,
    generate_kb_company_report,
    parse_structured_report,
    test_openai_key,
)
from server import Handler, HTML, main, write_json

__all__ = [
    # config
    "DB_PATH", "EXECUTION_TERMS", "GROWTH_TERMS", "MODEL", "MOAT_TERMS",
    "OPENAI_MODEL", "OPENROUTER_MODELS_URL", "RISK_TERMS", "ROOT", "RUNS",
    "STOPWORDS", "UPLOAD_DIR", "ensure_dirs",
    # text_utils
    "annual_hits", "chunk_pages", "clean_text", "clip", "matched_labels",
    "mention_count", "query_terms", "snippet_for", "strip_markup", "unique",
    # extractors
    "extract_docx_pages", "extract_epub_pages", "extract_pages",
    # store
    "Store", "document_hash", "parse_content_disposition",
    "parse_multipart", "safe_filename",
    # analysis
    "analyze_report",
    # llm
    "answer_question", "build_context", "call_openai_llm", "fallback_answer",
    "generate_kb_company_report", "parse_structured_report", "test_openai_key",
    # server
    "Handler", "HTML", "main", "write_json",
]

if __name__ == "__main__":
    main()
