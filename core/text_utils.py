"""text_utils.py — Pure text-processing helpers with no I/O or DB side-effects."""
from __future__ import annotations

import html
import re

from core.config import (
    EXECUTION_TERMS,
    GROWTH_TERMS,
    MOAT_TERMS,
    RISK_TERMS,
    STOPWORDS,
)


# ---------------------------------------------------------------------------
# Basic cleaning
# ---------------------------------------------------------------------------

def clean_text(value: str | None) -> str:
    """Collapse whitespace and strip leading/trailing space."""
    return re.sub(r"\s+", " ", value or "").strip()


def strip_markup(value: str | None) -> str:
    """Remove HTML/XML tags and unescape entities, then clean whitespace."""
    value = re.sub(r"(?is)<(script|style).*?</\1>", " ", value or "")
    value = re.sub(r"(?s)<[^>]+>", " ", value)
    return clean_text(html.unescape(value))


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def query_terms(query: str, limit: int = 14) -> list[str]:
    """Extract meaningful tokens from *query* for keyword search."""
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", query.lower()):
        if len(token) < 3 or token in STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= limit:
            break
    return terms


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_chars: int = 1400,
    overlap: int = 180,
) -> list[dict]:
    """Split page texts into overlapping chunks suitable for FTS indexing."""
    chunks: list[dict] = []
    step = max(400, chunk_chars - overlap)
    for page_number, text in pages:
        start = 0
        while start < len(text):
            chunk = clean_text(text[start : start + chunk_chars])
            if len(chunk) >= 120:
                chunks.append(
                    {"page_start": page_number, "page_end": page_number, "text": chunk}
                )
            start += step
    return chunks


# ---------------------------------------------------------------------------
# Snippet extraction
# ---------------------------------------------------------------------------

def snippet_for(text: str, terms: list[str], max_chars: int = 420) -> str:
    """Return a short excerpt of *text* that is most likely to contain *terms*."""
    lower = text.lower()
    positions = [lower.find(term) for term in terms if lower.find(term) >= 0]
    if not positions:
        return clean_text(text[:max_chars])
    center = min(positions)
    start = max(0, center - max_chars // 3)
    return clean_text(text[start : start + max_chars])


# ---------------------------------------------------------------------------
# Vocabulary matching
# ---------------------------------------------------------------------------

def matched_labels(text: str, vocab: dict[str, str]) -> list[str]:
    """Return sorted list of human-readable labels whose terms appear in *text*."""
    lower = text.lower()
    return sorted({label for term, label in vocab.items() if term in lower})


def mention_count(text: str, vocab: dict[str, str]) -> int:
    """Count total keyword occurrences across all vocabulary terms."""
    lower = text.lower()
    return sum(lower.count(term) for term in vocab)


def clip(value: float, low: float = 0, high: float = 100) -> int:
    """Clamp *value* to [*low*, *high*] and return as int."""
    return int(max(low, min(high, round(value))))


# ---------------------------------------------------------------------------
# Result deduplication
# ---------------------------------------------------------------------------

def unique(items: list[dict], limit: int = 10) -> list[dict]:
    """Deduplicate search results by (document_id, chunk_id, page_start, snippet)."""
    seen: set = set()
    result: list[dict] = []
    for item in items:
        key = (
            item.get("document_id"),
            item.get("chunk_id"),
            item.get("page_start"),
            item.get("snippet"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= limit:
            break
    return result


# ---------------------------------------------------------------------------
# Annual-report keyword search (no DB)
# ---------------------------------------------------------------------------

def annual_hits(
    pages: list[tuple[int, str]], query: str, limit: int = 6
) -> list[dict]:
    """Score raw pages against *query* terms and return top hits."""
    terms = query_terms(query)
    scored: list[dict] = []
    for page_number, text in pages:
        lower = text.lower()
        score = sum(lower.count(term) for term in terms)
        if score:
            scored.append(
                {
                    "title": "Annual report",
                    "page_start": page_number,
                    "page_end": page_number,
                    "snippet": snippet_for(text, terms),
                    "score": score,
                }
            )
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]
