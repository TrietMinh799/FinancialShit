"""store.py — SQLite document store, FTS search, and multipart form parsing."""

from __future__ import annotations

import contextlib
import hashlib
import re
import sqlite3
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from core.config import DB_PATH, UPLOAD_DIR
from core.extractors import extract_pages
from core.text_utils import chunk_pages, query_terms, sanitize_injection, snippet_for
from core.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Multipart parsing helpers
# ---------------------------------------------------------------------------


def parse_content_disposition(value: str) -> dict[str, str]:
    """Parse a Content-Disposition header value into a key→value dict."""
    result: dict[str, str] = {}
    for part in value.split(";"):
        if "=" in part:
            key, raw = part.strip().split("=", 1)
            result[key.lower()] = raw.strip().strip('"')
    return result


def safe_filename(name: str) -> str:
    """Sanitise *name* to a filesystem-safe string."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return stem or "upload.bin"


def parse_multipart(
    handler: BaseHTTPRequestHandler,
) -> tuple[dict[str, str], dict[str, Path]]:
    """Parse a multipart/form-data request body into (fields, files).

    Files are written to UPLOAD_DIR and returned as Path objects.
    """
    content_type = handler.headers.get("Content-Type", "")
    match = re.search(r"boundary=(.+)", content_type)
    if not match:
        raise ValueError("Missing multipart boundary.")
    boundary = match.group(1).strip().strip('"').encode("utf-8")
    length = int(handler.headers.get("Content-Length", "0"))
    body = handler.rfile.read(length)

    fields: dict[str, str] = {}
    files: dict[str, Path] = {}

    for part in body.split(b"--" + boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--" or b"\r\n\r\n" not in part:
            continue
        raw_headers, data = part.split(b"\r\n\r\n", 1)
        if data.endswith(b"\r\n"):
            data = data[:-2]
        headers: dict[str, str] = {}
        for line in raw_headers.decode("utf-8", errors="ignore").split("\r\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.lower()] = value.strip()
        disposition = parse_content_disposition(headers.get("content-disposition", ""))
        name = disposition.get("name")
        filename = disposition.get("filename")
        if not name:
            continue
        if filename:
            target = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_filename(filename)}"
            target.write_bytes(data)
            files[name] = target
        else:
            fields[name] = data.decode("utf-8", errors="ignore")

    return fields, files


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def document_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of the file at *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Store — SQLite CRUD + FTS
# ---------------------------------------------------------------------------


class Store:
    """Persistent document store backed by SQLite with optional FTS5 search."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = path
        self._vector_store: VectorStore | None = None
        self._init_db()

    def vector_store(self) -> VectorStore:
        """Lazy-initialised vector store for hybrid search."""
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables on first use (idempotent)."""
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS documents ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  title TEXT NOT NULL,"
                "  filename TEXT NOT NULL,"
                "  source_type TEXT NOT NULL,"
                "  content_hash TEXT NOT NULL UNIQUE,"
                "  char_count INTEGER NOT NULL,"
                "  page_count INTEGER NOT NULL,"
                "  created_at TEXT NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS chunks ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,"
                "  chunk_index INTEGER NOT NULL,"
                "  page_start INTEGER,"
                "  page_end INTEGER,"
                "  text TEXT NOT NULL"
                ")"
            )
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts "
                    "USING fts5(text, content='chunks', content_rowid='id')"
                )

    def _has_fts(self, conn: sqlite3.Connection) -> bool:
        return (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_fts'"
            ).fetchone()
            is not None
        )

    def _row_to_search_result(self, row: sqlite3.Row, terms: list[str]) -> dict:
        return {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "title": row["title"],
            "source_type": row["source_type"],
            "page_start": row["page_start"],
            "page_end": row["page_end"],
            "snippet": snippet_for(row["text"], terms),
            "score": row["score"],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(
        self,
        path: Path,
        title: str | None = None,
        source_type: str = "book",
    ) -> dict:
        """Index *path* into the store.  Returns a summary dict.

        If the identical file (by SHA-256) is already indexed, the existing
        record is returned with ``inserted=False``.
        """
        pages = extract_pages(path)
        if not pages:
            raise ValueError("No readable text was found in this file.")
        # Reports are dense; smaller chunks improve retrieval precision.
        if source_type == "annual_report":
            chunks = chunk_pages(pages, chunk_chars=800, overlap=120)
        else:
            chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError("No usable chunks were created from this file.")

        content_hash = document_hash(path)
        char_count = sum(len(text) for _, text in pages)

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, title, source_type, char_count, page_count "
                "FROM documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()

            if existing:
                chunk_count = conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE document_id = ?",
                    (existing["id"],),
                ).fetchone()[0]
                return {
                    "document_id": existing["id"],
                    "title": existing["title"],
                    "source_type": existing["source_type"],
                    "char_count": existing["char_count"],
                    "page_count": existing["page_count"],
                    "chunk_count": chunk_count,
                    "inserted": False,
                }

            cursor = conn.execute(
                "INSERT INTO documents "
                "(title, filename, source_type, content_hash, char_count, page_count, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    title or path.stem,
                    path.name,
                    source_type,
                    content_hash,
                    char_count,
                    len(pages),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            doc_id = cursor.lastrowid
            fts = self._has_fts(conn)

            for index, chunk in enumerate(chunks):
                chunk_text = sanitize_injection(chunk["text"])
                c = conn.execute(
                    "INSERT INTO chunks "
                    "(document_id, chunk_index, page_start, page_end, text) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (doc_id, index, chunk["page_start"], chunk["page_end"], chunk_text),
                )
                if fts:
                    conn.execute(
                        "INSERT INTO chunk_fts(rowid, text) VALUES (?, ?)",
                        (c.lastrowid, chunk_text),
                    )

            # Index into ChromaDB for vector search (with sanitised text)
            sanitised_chunks = [{**ch, "text": sanitize_injection(ch["text"])} for ch in chunks]
            source_type_str = source_type or "book"
            with contextlib.suppress(Exception):
                self.vector_store().index_chunks(
                    doc_id,
                    title or path.stem,
                    source_type_str,
                    sanitised_chunks,
                )

            return {
                "document_id": doc_id,
                "title": title or path.stem,
                "source_type": source_type,
                "char_count": char_count,
                "page_count": len(pages),
                "chunk_count": len(chunks),
                "inserted": True,
            }

    def stats(self) -> dict:
        """Return library-level statistics and the 12 most recent documents."""
        with self._connect() as conn:
            totals = conn.execute(
                "SELECT COUNT(*) AS documents,"
                "       COALESCE(SUM(char_count),0) AS characters,"
                "       COALESCE(SUM(page_count),0) AS pages"
                " FROM documents"
            ).fetchone()
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            recent = conn.execute(
                "SELECT d.id, d.title, d.filename, d.source_type,"
                "       d.char_count, d.page_count, d.created_at,"
                "       COUNT(c.id) AS chunk_count"
                " FROM documents d"
                " LEFT JOIN chunks c ON c.document_id = d.id"
                " GROUP BY d.id"
                " ORDER BY d.created_at DESC"
                " LIMIT 12"
            ).fetchall()
        return {
            "documents": totals["documents"],
            "characters": totals["characters"],
            "pages": totals["pages"],
            "chunks": chunks,
            "recent_documents": [dict(row) for row in recent],
        }

    def search(
        self,
        query: str,
        source_types: list[str] | None = None,
        limit: int = 8,
    ) -> list[dict]:
        """Full-text search with BM25 (FTS5) or LIKE fallback."""
        terms = query_terms(query)
        if not terms:
            return []
        source_types = source_types or ["book"]
        placeholders = ",".join("?" for _ in source_types)

        with self._connect() as conn:
            if self._has_fts(conn):
                try:
                    rows = conn.execute(
                        f"SELECT c.id AS chunk_id, d.id AS document_id, d.title,"
                        f"       d.source_type, c.page_start, c.page_end, c.text,"
                        f"       bm25(chunk_fts) AS score"
                        f" FROM chunk_fts"
                        f" JOIN chunks c ON c.id = chunk_fts.rowid"
                        f" JOIN documents d ON d.id = c.document_id"
                        f" WHERE chunk_fts MATCH ? AND d.source_type IN ({placeholders})"
                        f" ORDER BY score LIMIT ?",
                        [" OR ".join(terms), *source_types, limit],
                    ).fetchall()
                    return [self._row_to_search_result(row, terms) for row in rows]
                except sqlite3.OperationalError:
                    pass  # fall through to LIKE

            like_patterns: list[str] = []
            like_params: list[str] = []
            for term in terms[:6]:
                like_patterns.append("LOWER(c.text) LIKE ?")
                like_params.append(f"%{term}%")
                # Also match ASCII-only terms with wildcard between every character
                # so "cong" matches both "công" and "cong" in the text
                if term.isascii():
                    like_patterns.append("LOWER(c.text) LIKE ?")
                    like_params.append(f"%{'%'.join(term)}%")
            like_clause = " OR ".join(like_patterns)
            rows = conn.execute(
                f"SELECT c.id AS chunk_id, d.id AS document_id, d.title,"
                f"       d.source_type, c.page_start, c.page_end, c.text, 0 AS score"
                f" FROM chunks c"
                f" JOIN documents d ON d.id = c.document_id"
                f" WHERE d.source_type IN ({placeholders}) AND ({like_clause})"
                f" LIMIT ?",
                [*source_types, *like_params, limit],
            ).fetchall()
            return [self._row_to_search_result(row, terms) for row in rows]

    # ------------------------------------------------------------------
    # Hybrid search — Reciprocal Rank Fusion (vector + BM25)
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        source_types: list[str] | None = None,
        limit: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Combine vector (ChromaDB) and BM25 (FTS5) results via RRF.

        Returns the best *limit* results deduplicated and ranked by fused score.
        """
        # 1. Get BM25 results (limit higher for good recall)
        bm25_results = self.search(query, source_types, limit * 3)

        # 2. Get vector results
        vec_results = self.vector_store().search(query, source_types, limit * 3)

        # 3. RRF merge
        seen_ids: set = set()
        fused: list[tuple[float, dict]] = []

        def rank_key(item: dict) -> tuple:
            return (item.get("document_id"), item.get("chunk_id"), item.get("page_start"))

        for rank, item in enumerate(bm25_results):
            key = rank_key(item)
            if key in seen_ids:
                continue
            seen_ids.add(key)
            score = 1.0 / (rrf_k + rank)
            fused.append((score, item))

        for rank, item in enumerate(vec_results):
            key = rank_key(item)
            if key in seen_ids:
                continue
            seen_ids.add(key)
            # BM25 component contribution
            bm25_contrib = 1.0 / (rrf_k + len(bm25_results) + 1)  # rank beyond BM25 list
            score = bm25_contrib + 1.0 / (rrf_k + rank)
            fused.append((score, item))

        # 4. Sort by fused score descending
        fused.sort(key=lambda pair: pair[0], reverse=True)

        return [item for _, item in fused[:limit]]

    # ------------------------------------------------------------------
    # Re-index (e.g. after switching the embedding model)
    # ------------------------------------------------------------------

    def reindex_all(self) -> int:
        """Re-embed every stored chunk into the vector store.

        Reads chunk text back from SQLite (where it is kept verbatim) and
        re-runs the current embedding model. Use this after changing
        ``EMBED_MODEL`` so the vector index matches the new model.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT c.document_id, c.chunk_index, c.page_start, c.page_end, c.text,"
                "       d.title, d.source_type"
                " FROM chunks c"
                " JOIN documents d ON d.id = c.document_id"
                " ORDER BY c.document_id, c.chunk_index"
            ).fetchall()

        by_doc: dict[int, list[dict]] = {}
        meta: dict[int, dict] = {}
        for row in rows:
            doc_id = row["document_id"]
            if doc_id not in by_doc:
                by_doc[doc_id] = []
                meta[doc_id] = {
                    "title": row["title"],
                    "source_type": row["source_type"],
                }
            by_doc[doc_id].append(
                {
                    "text": row["text"],
                    "page_start": row["page_start"],
                    "page_end": row["page_end"],
                }
            )

        total = 0
        for doc_id, chunks in by_doc.items():
            self.vector_store().index_chunks(
                doc_id,
                meta[doc_id]["title"],
                meta[doc_id]["source_type"],
                chunks,
            )
            total += len(chunks)
        return total
