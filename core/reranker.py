"""reranker.py — Cross-encoder reranker for improving retrieval precision."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from core.cache import LRUCache

_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_RERANKER_MODEL)
    return _model


_rerank_cache = LRUCache(maxsize=2000, default_ttl=None)


def load_reranker_model() -> None:
    """Preload the reranker model into memory. Call at startup."""
    _get_model()


def rerank(query: str, candidates: list[dict], top_k: int = 15) -> list[dict]:
    """Score and re-rank *candidates* by relevance to *query* using a cross-encoder.

    Returns the top *top_k* candidates sorted by relevance (descending).
    Falls back to the original order if the model is unavailable.
    Per-(query, snippet) scores are cached to avoid redundant forward passes
    when similar queries hit the same passages.
    """
    if not candidates:
        return []

    try:
        model = _get_model()
        uncached_indices: list[int] = []
        uncached_pairs: list[tuple[str, str]] = []

        for i, c in enumerate(candidates):
            key = f"{query}||{c.get('snippet', '')}"
            val = _rerank_cache.get(key)
            if val is not None:
                c["_rerank_score"] = val
            else:
                uncached_indices.append(i)
                uncached_pairs.append((query, c.get("snippet", "")))

        if uncached_pairs:
            scores = model.predict(uncached_pairs, show_progress_bar=False)
            for idx, score in zip(uncached_indices, scores, strict=False):
                snippet = candidates[idx].get("snippet", "")
                _rerank_cache.put(f"{query}||{snippet}", float(score))
                candidates[idx]["_rerank_score"] = float(score)

        scored = [c for c in candidates if "_rerank_score" in c]
        scored.sort(key=lambda x: x["_rerank_score"], reverse=True)
        result = scored[:top_k]
        for c in result:
            c.pop("_rerank_score", None)
        return result
    except Exception:
        return candidates[:top_k]
