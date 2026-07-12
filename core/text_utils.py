"""text_utils.py — Pure text-processing helpers with no I/O or DB side-effects."""

from __future__ import annotations

import html
import re
import unicodedata

from core.config import (
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
# Prompt-injection sanitisation
# ---------------------------------------------------------------------------

# Phrases commonly used to hijack an LLM via injected document content.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(previous|above|prior|earlier|preceding)\s+(instructions?|prompts?|rules?|context)",
        r"disregard\s+(all\s+)?(previous|above|prior|earlier|preceding)\s+(instructions?|prompts?|rules?|context)",
        r"forget\s+(all\s+)?(previous|above|prior|earlier|preceding)\s+(instructions?|prompts?|rules?|context)",
        r"override\s+(all\s+)?(previous|above|prior|earlier|preceding)\s+(instructions?|prompts?|rules?|context)",
        r"do\s+not\s+follow\s+(the\s+)?(system|previous|above|prior)\s+(prompt|instructions?|rules?)",
        r"you\s+are\s+now\s+(a|an)\s+",
        r"new\s+(instructions?|role|persona|identity)\s*[:.]",
        r"act\s+as\s+(a|an|if)\s+",
        r"switch\s+(to|into)\s+(a\s+)?(new\s+)?(role|mode|persona)",
        r"system\s*:\s*",
        r"<\s*/?\s*system\s*>",
        r"<<\s*SYS\s*>>",
        r"\[INST\]",
        r"\[/INST\]",
        r"BEGIN\s+(INSTRUCTION|SYSTEM|PROMPT)",
        r"END\s+(INSTRUCTION|SYSTEM|PROMPT)",
        r"###\s*(instruction|system|prompt)",
    )
]

_INJECTION_REPLACEMENT = "[content removed]"


def sanitize_injection(text: str) -> str:
    """Neutralise common prompt-injection phrases in document text.

    Replaces each matched pattern with a harmless placeholder so the
    surrounding financial content is preserved.
    """
    for pattern in _INJECTION_PATTERNS:
        text = pattern.sub(_INJECTION_REPLACEMENT, text)
    return text


# Max length for short user-supplied metadata fields (title, company, ticker).
_FIELD_MAX_LEN = 200


def sanitize_field(value: str, max_len: int = _FIELD_MAX_LEN) -> str:
    """Sanitise a short user-supplied metadata field.

    * Collapses whitespace
    * Strips control characters and common injection markers
    * Truncates to *max_len*
    """
    value = clean_text(value)
    # Remove control chars (except normal whitespace)
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    # Strip role/tag markers that could shift the LLM's role perception
    for pattern in _INJECTION_PATTERNS:
        value = pattern.sub("", value)
    return value[:max_len].strip()


# ---------------------------------------------------------------------------
# Vietnamese diacritic removal
# ---------------------------------------------------------------------------


def remove_diacritics(text: str) -> str:
    """Strip diacritics from Vietnamese/Latin text (e.g., ``công`` → ``cong``)."""
    normalized = unicodedata.normalize("NFD", text)
    ascii_chars: list[str] = []
    for ch in normalized:
        if unicodedata.combining(ch):
            continue
        # đ (U+0111) → d, Đ (U+0110) → D
        if ch == "\u0111":
            ascii_chars.append("d")
        elif ch == "\u0110":
            ascii_chars.append("D")
        else:
            ascii_chars.append(ch)
    return "".join(ascii_chars)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def query_terms(query: str, limit: int = 14) -> list[str]:
    """Extract meaningful tokens from *query* for keyword search.

    Also extracts diacritic-free alternatives so that typing
    ``"cong ty"`` still matches ``"công ty"`` via LIKE fallback.
    """
    terms: list[str] = []
    for raw in (query, remove_diacritics(query)):
        for token in re.findall(r"[^\W\d_][^\W_]+", raw.lower()):
            if len(token) < 3 or token in STOPWORDS:
                continue
            if token not in terms:
                terms.append(token)
                if len(terms) >= limit:
                    return terms
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
                chunks.append({"page_start": page_number, "page_end": page_number, "text": chunk})
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


def annual_hits(pages: list[tuple[int, str]], query: str, limit: int = 6) -> list[dict]:
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
