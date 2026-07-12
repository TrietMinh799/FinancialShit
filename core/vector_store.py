"""vector_store.py — ChromaDB-based vector store with multilingual embeddings."""

from __future__ import annotations

import os
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from core.config import CHROMA_DIR

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


def _embed(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, normalize_embeddings=True).tolist()


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

    def search(
        self,
        query: str,
        source_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Semantic search across one or more collections.

        When *source_types* includes both ``book`` and ``annual_report`` the
        method queries both collections independently and then merges the
        results sorted by cosine distance (ascending).
        """
        query_emb = _embed([query])
        collection_names = self._resolve(source_types)
        per_coll_limit = limit  # fetch *limit* from each collection

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
                        "snippet": (text[:420] + "...") if len(text) > 420 else text,
                        "score": float(distances[i]),
                    }
                )

        all_results.sort(key=lambda x: x["score"])
        return all_results[:limit]

    def delete_document(self, document_id: int) -> None:
        """Remove every chunk that belongs to *document_id* from all collections."""
        for coll in self._collections.values():
            coll.delete(where={"document_id": str(document_id)})
