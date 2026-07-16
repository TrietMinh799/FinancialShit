"""reranker.py — Cross-encoder reranker for improving retrieval precision."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_RERANKER_MODEL)
    return _model


def load_reranker_model() -> None:
    """Preload the reranker model into memory. Call at startup."""
    _get_model()


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Score and re-rank *candidates* by relevance to *query* using a cross-encoder.

    Returns the top *top_k* candidates sorted by relevance (descending).
    Falls back to the original order if the model is unavailable.
    """
    if not candidates:
        return []

    try:
        model = _get_model()
        pairs = [(query, c.get("snippet", "")) for c in candidates]
        scores = model.predict(pairs, show_progress_bar=False)
        scored = list(zip(scores, candidates, strict=False))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]
    except Exception:
        return candidates[:top_k]
