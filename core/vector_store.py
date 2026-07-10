"""vector_store.py — ChromaDB-based vector store with multilingual embeddings."""
from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from core.config import CHROMA_DIR

# ---------------------------------------------------------------------------
# Embedding model — small multilingual model good for Vietnamese
# ---------------------------------------------------------------------------

_MODEL_NAME = "intfloat/multilingual-e5-small"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _embed(texts: list[str], prefix: str) -> list[list[float]]:
    prefixed = [f"{prefix}{t}" for t in texts]
    return _get_model().encode(prefixed, normalize_embeddings=True).tolist()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


class VectorStore:
    """ChromaDB-backed vector store for semantic chunk retrieval."""

    def __init__(self, collection_name: str = "rag_chunks") -> None:
        self._collection_name = collection_name
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _ensure(self) -> None:
        if self._client is not None:
            return
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

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
        """Embed and store a batch of chunks into ChromaDB."""
        if not chunks:
            return
        self._ensure()

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for i, chunk in enumerate(chunks):
            ids.append(f"{document_id}_{i}")
            documents.append(chunk["text"])
            metadatas.append({
                "document_id": str(document_id),
                "chunk_index": str(i),
                "title": title,
                "source_type": source_type,
                "page_start": str(chunk.get("page_start", "")),
                "page_end": str(chunk.get("page_end", "")),
            })

        embeddings = _embed(documents, prefix="passage: ")
        self._collection.add(
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
        """Semantic search — returns results shaped like ``Store.search``."""
        self._ensure()

        query_emb = _embed([query], prefix="query: ")
        where = None
        if source_types:
            where = {"source_type": {"$in": source_types}}

        results = self._collection.query(
            query_embeddings=query_emb,
            n_results=limit,
            where=where,
        )

        output: list[dict] = []
        if not results["ids"] or not results["ids"][0]:
            return output

        ids_list = results["ids"][0]
        distances = results["distances"][0]
        docs_list = results["documents"][0]
        meta_list = results["metadatas"][0]

        for i, id_str in enumerate(ids_list):
            meta = meta_list[i] if meta_list else {}
            text = docs_list[i] or ""
            output.append({
                "chunk_id": int(id_str.split("_")[1]) if "_" in id_str else 0,
                "document_id": int(meta.get("document_id", 0)),
                "title": meta.get("title", ""),
                "source_type": meta.get("source_type", ""),
                "page_start": int(meta["page_start"]) if meta.get("page_start", "").isdigit() else None,
                "page_end": int(meta["page_end"]) if meta.get("page_end", "").isdigit() else None,
                "snippet": (text[:420] + "...") if len(text) > 420 else text,
                "score": float(distances[i]),
            })

        return output

    def delete_document(self, document_id: int) -> None:
        """Remove every chunk that belongs to *document_id*."""
        self._ensure()
        self._collection.delete(where={"document_id": str(document_id)})
