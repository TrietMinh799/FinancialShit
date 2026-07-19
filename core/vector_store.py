"""vector_store.py — ChromaDB-based vector store with multilingual embeddings."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from core.cache import LRUCache
from core.config import CHROMA_DIR, SNIPPET_MAX_CHARS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

_MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
_model: SentenceTransformer | None = None

# Separate HNSW collections so dense report chunks do not dilute reference
# book search results and vice versa.
_BOOKS_COLLECTION = "rag_books_" + _MODEL_NAME.replace("/", "_").replace("-", "_")
_REPORTS_COLLECTION = "rag_reports_" + _MODEL_NAME.replace("/", "_").replace("-", "_")

_COLLECTION_MAP: dict[str, str] = {
    "book": _BOOKS_COLLECTION,
    "annual_report": _REPORTS_COLLECTION,
}


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# Embedding cache: key = hash(text), value = embedding vector
# Using hash instead of full text as key saves memory
_embed_cache = LRUCache(maxsize=2000, default_ttl=None)
_search_cache = LRUCache(maxsize=500, default_ttl=120)


def load_embedding_model() -> None:
    """Preload the embedding model into memory. Call at startup."""
    _get_model()


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with caching (hash-based keys for memory efficiency)."""
    import hashlib

    cached: list[tuple[int, list[float]]] = []
    uncached: list[int] = []
    uncached_texts: list[str] = []

    for i, t in enumerate(texts):
        key = hashlib.sha256(t.encode()).hexdigest()[:32]
        val = _embed_cache.get(key)
        if val is not None:
            cached.append((i, list(val)))  # type: ignore[arg-type]
        else:
            uncached.append(i)
            uncached_texts.append(t)

    if uncached_texts:
        new_vecs: list[list[float]] = _get_model().encode(
            uncached_texts, normalize_embeddings=True
        ).tolist()
        for idx, vec in zip(uncached, new_vecs, strict=False):
            key = hashlib.sha256(texts[idx].encode()).hexdigest()[:32]
            _embed_cache.put(key, vec)
            cached.append((idx, vec))

    cached.sort(key=lambda x: x[0])
    return [v for _, v in cached]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


class VectorStore:
    """ChromaDB-backed vector store for semantic chunk retrieval.

    Maintains separate HNSW collections for ``book`` and ``annual_report``
    chunks so each source type gets its own vector index.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._collections: dict[str, Any] = {}

    def _ensure(self, name: str) -> Any:
        if self._client is None:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    def _resolve(self, source_types: list[str] | None) -> list[str]:
        """Map source type labels to ChromaDB collection names."""
        names: set[str] = set()
        for st in source_types or ["book"]:
            names.add(_COLLECTION_MAP.get(st, _BOOKS_COLLECTION))
        return list(names)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_chunks(
        self,
        document_id: int,
        title: str,
        source_type: str,
        chunks: list[dict],
    ) -> None:
        """Embed and store a batch of chunks into the appropriate collection."""
        if not chunks:
            return
        coll = self._ensure(_COLLECTION_MAP.get(source_type, _BOOKS_COLLECTION))

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for i, chunk in enumerate(chunks):
            ids.append(f"{document_id}_{i}")
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "document_id": str(document_id),
                    "chunk_index": str(i),
                    "title": title,
                    "source_type": source_type,
                    "page_start": str(chunk.get("page_start", "")),
                    "page_end": str(chunk.get("page_end", "")),
                }
            )

        embeddings = _embed(documents)
        coll.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        self.invalidate_search_cache()

    def search(
        self,
        query: str,
        source_types: list[str] | None = None,
        limit: int = 30,
    ) -> list[dict]:
        """Semantic search across one or more collections.

        Results are cached for 120s so repeated/similar queries skip
        ChromaDB lookups and embedding computation.
        """
        cache_key = f"{_MODEL_NAME}||{query}||{sorted(source_types or ['book'])}||{limit}"
        cached = _search_cache.get(cache_key)
        if cached is not None:
            return list(cached)  # type: ignore[arg-type]

        query_emb = _embed([query])
        collection_names = self._resolve(source_types)
        per_coll_limit = limit

        all_results: list[dict] = []

        for coll_name in collection_names:
            coll = self._ensure(coll_name)
            results = coll.query(
                query_embeddings=query_emb,
                n_results=per_coll_limit,
            )

            if not results["ids"] or not results["ids"][0]:
                continue

            ids_list = results["ids"][0]
            distances = results["distances"][0]
            docs_list = results["documents"][0]
            meta_list = results["metadatas"][0]

            for i, id_str in enumerate(ids_list):
                meta = meta_list[i] if meta_list else {}
                text = docs_list[i] or ""
                page_start_val = meta.get("page_start", "")
                page_end_val = meta.get("page_end", "")
                page_start_str = str(page_start_val) if page_start_val else ""
                page_end_str = str(page_end_val) if page_end_val else ""
                all_results.append(
                    {
                        "chunk_id": int(id_str.split("_")[1]) if "_" in id_str else 0,
                        "document_id": int(str(meta.get("document_id", 0))),
                        "title": str(meta.get("title", "")),
                        "source_type": str(meta.get("source_type", "")),
                        "page_start": int(meta["page_start"]) if page_start_str.isdigit() else None,
                        "page_end": int(meta["page_end"]) if page_end_str.isdigit() else None,
                        "snippet": (text[:SNIPPET_MAX_CHARS] + "...") if len(text) > SNIPPET_MAX_CHARS else text,
                        "score": float(distances[i]),
                    }
                )

        all_results.sort(key=lambda x: x["score"])
        result = all_results[:limit]
        _search_cache.put(cache_key, result, ttl=120)
        return result

    def invalidate_search_cache(self) -> None:
        """Clear vector search cache (call after indexing new chunks)."""
        _search_cache.invalidate()

    def clear_all(self) -> None:
        """Delete all collections and re-create them empty.

        Used by ``reindex_all`` to guarantee a clean slate so orphaned
        vectors (from docs deleted only in SQLite) are purged.
        """
        if self._client is None:
            return
        for coll in self._client.list_collections():
            with contextlib.suppress(Exception):
                self._client.delete_collection(coll.name)
        self._collections.clear()
        _search_cache.invalidate()

    def delete_document(self, document_id: int) -> None:
        """Remove every chunk that belongs to *document_id* from all known collections.

        Uses the persistent ChromaDB client to list all collections so chunks
        are cleaned up even if the collection hasn't been accessed yet in this
        session (fixes lazy-init bug where ``_collections`` could be empty).
        """
        if self._client is None:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        for coll in self._client.list_collections():
            try:
                coll.delete(where={"document_id": str(document_id)})
            except Exception as exc:
                logger.warning("ChromaDB delete failed in collection %s: %s", coll.name, exc)
