"""text_utils.py — Pure text-processing helpers with no I/O or DB side-effects."""

from __future__ import annotations

import html
import re
import unicodedata

from core.config import (
    CHUNK_CHARS,
    CHUNK_MIN_LEN,
    CHUNK_MIN_STEP,
    CHUNK_OVERLAP,
    QUERY_TERM_LIMIT,
    SNIPPET_MAX_CHARS,
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


def query_terms(query: str, limit: int = QUERY_TERM_LIMIT) -> list[str]:
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

# Common abbreviation list to avoid false sentence splits
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "e.g.", "i.e.",
    "inc.", "ltd.", "llc.", "corp.", "co.", "u.s.", "u.k.", "e.u.", "p.m.", "a.m.",
    "no.", "vol.", "ch.", "fig.", "tab.", "sec.", "pp.", "ed.", "trans.", "rev.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "oct.", "nov.", "dec.",
    "mon.", "tue.", "wed.", "thu.", "fri.", "sat.", "sun.",
}


def _split_headers(text: str) -> list[str]:
    """Split text on markdown/ATX headers, numbered sections, or ALL-CAPS lines.

    Returns a list of sections. Headers are kept attached to the following content.
    """
    # Patterns: # Header, ## Header, 1. Title, Chapter 1, ALL CAPS LINE (3+ words)
    header_pattern = re.compile(
        r"(?m)^("
        r"(?:#{1,6}\s+.+)"              # Markdown headers
        r"|(?:\d+\.\s+[A-Z].+)"         # Numbered sections: "1. Introduction"
        r"|(?:[A-Z][A-Z\s]{2,}:)"       # ALL CAPS with colon: "INTRODUCTION:"
        r"|(?:Chapter\s+\d+[:\.\s])"    # Chapter N
        r"|(?:Section\s+\d+[:\.\s])"    # Section N
        r"|(?:Appendix\s+[A-Z][:\.\s])" # Appendix A
        r")",
        re.MULTILINE,
    )
    parts = header_pattern.split(text)
    if not parts or len(parts) == 1:
        return [text]

    # Reconstruct: header + following content
    sections = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        content = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append(f"{header}\n{content}".strip())
    return sections


def _split_paragraphs(text: str) -> list[str]:
    """Split text on blank lines (double newline), preserving single newlines.
    
    Also merges header-like lines (markdown headers, numbered sections, ALL CAPS)
    with the following paragraph so they don't become standalone fragments.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned = [clean_text(p) for p in paragraphs if p.strip()]
    
    # Merge header-like lines with following paragraph
    merged: list[str] = []
    header_pattern = re.compile(
        r"^("
        r"(?:#{1,6}\s+.+)"
        r"|(?:\d+\.\s+[A-Z].+)"
        r"|(?:[A-Z][A-Z\s]{2,}:)"
        r"|(?:Chapter\s+\d+[:\.\s])"
        r"|(?:Section\s+\d+[:\.\s])"
        r"|(?:Appendix\s+[A-Z][:\.\s])"
        r")$"
    )
    i = 0
    while i < len(cleaned):
        p = cleaned[i]
        if (i + 1 < len(cleaned) and 
            not p.rstrip().endswith(('.', '!', '?', '。', '！', '？')) and
            header_pattern.match(p.strip())):
            # Header line - merge with next paragraph
            merged.append(p + " " + cleaned[i + 1])
            i += 2
        else:
            merged.append(p)
            i += 1
    return merged


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting common abbreviations."""
    # Split on sentence-ending punctuation followed by whitespace + capital letter
    # or end of string. Avoid splitting on abbreviations.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[。！？])\s*", text)
    # Post-process to fix abbreviation false splits
    merged: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if merged and any(merged[-1].lower().endswith(abbr) for abbr in _ABBREVIATIONS):
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


def _merge_to_target(
    units: list[str],
    target_chars: int,
    overlap_chars: int,
    min_len: int,
    min_step: int = 0,
) -> list[str]:
    """Greedily merge text units until target_chars is reached, then emit chunk.

    Carries overlap_chars from end of previous chunk to start of next.
    When *min_step* > 0, ensures each new chunk starts at least *min_step*
    characters after the previous chunk's start (in cumulative character offsets),
    skipping overlapping units that would violate the step constraint.

    Tries to split at natural boundaries (transition words, paragraph breaks)
    when nearing target_chars instead of splitting mid-sentence.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    cum_offset = 0  # cumulative character offset for min_step enforcement

    # Transition words that indicate a natural boundary
    _transitions = re.compile(
        r"^(However|Moreover|Furthermore|Nevertheless|In addition|"
        r"Additionally|Therefore|Thus|Consequently|As a result|"
        r"On the other hand|In contrast|Specifically|In particular|"
        r"First|Second|Third|Finally|In summary|To illustrate|"
        r"For example|For instance|Notably|Importantly)\b",
        re.IGNORECASE,
    )

    for unit in units:
        unit = unit.strip()
        if not unit:
            cum_offset += 0  # empty units have 0 characters
            continue
        ulen = len(unit)

        # If a single unit exceeds target, split it further
        if ulen > target_chars:
            if current:
                chunk_text = clean_text(" ".join(current))
                chunks.append(chunk_text)
                current = []
                current_len = 0
                cum_offset += len(chunk_text)
            # Fallback: character split for this oversized unit
            for i in range(0, ulen, target_chars):
                part = unit[i : i + target_chars]
                if len(part) >= min_len:
                    chunks.append(part)
            cum_offset += ulen
            continue

        # Check whether adding this unit would exceed target
        if current_len + ulen > target_chars and current:
            # Try to split at a natural boundary (transition word) in current
            split_at = None
            for i, cu in enumerate(current):
                if _transitions.match(cu) and i > max(1, len(current) // 3):
                    split_at = i
                    break
            if split_at is not None:
                pre = current[:split_at]
                post = current[split_at:]
                chunk_text = clean_text(" ".join(pre))
                if len(chunk_text) >= min_len:
                    chunks.append(chunk_text)
                cum_offset += len(chunk_text)
                current = post
                current_len = sum(len(u) for u in post)
                # Still may need to emit if still over target
                if current_len + ulen > target_chars and current:
                    chunk_text = clean_text(" ".join(current))
                    if len(chunk_text) >= min_len:
                        chunks.append(chunk_text)
                    cum_offset += len(chunk_text)
                    current = []
                    current_len = 0
            else:
                chunk_text = clean_text(" ".join(current))
                if len(chunk_text) >= min_len:
                    chunks.append(chunk_text)
                cum_offset += len(chunk_text)

            # Start new chunk with overlap from previous, respecting min_step
            overlap_text = chunks[-1][-overlap_chars:] if chunks else ""
            # Enforce min_step: skip overlap if it would place us too close
            if min_step > 0 and chunks:
                last_start = cum_offset - len(chunks[-1])
                potential_start = cum_offset - len(overlap_text)
                if potential_start - last_start < min_step:
                    overlap_text = ""
            current = [overlap_text, unit] if overlap_text else [unit]
            current_len = len(overlap_text) + ulen
        else:
            current.append(unit)
            current_len += ulen

    # Emit final chunk
    if current:
        merged = clean_text(" ".join(current))
        if len(merged) >= min_len:
            chunks.append(merged)

    return chunks


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_chars: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split page texts into semantically-coherent overlapping chunks.

    Recursively splits on headers -> paragraphs -> sentences -> characters,
    then merges units until target size is reached.
    """
    chunks: list[dict] = []
    for page_number, text in pages:
        text = clean_text(text)
        if not text:
            continue

        # Recursive splitting: headers -> paragraphs -> sentences
        sections = _split_headers(text)
        all_sentences: list[str] = []
        for sec in sections:
            for para in _split_paragraphs(sec):
                all_sentences.extend(_split_sentences(para))

        # Merge sentences into target-sized chunks
        page_chunks = _merge_to_target(
            all_sentences, chunk_chars, overlap, CHUNK_MIN_LEN, CHUNK_MIN_STEP
        )

        for ch in page_chunks:
            chunks.append({"page_start": page_number, "page_end": page_number, "text": ch})

    return chunks


# ---------------------------------------------------------------------------
# Snippet extraction
# ---------------------------------------------------------------------------


def snippet_for(text: str, terms: list[str], max_chars: int = SNIPPET_MAX_CHARS) -> str:
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


def unique(items: list[dict], limit: int = 50) -> list[dict]:
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


def annual_hits(pages: list[tuple[int, str]], query: str, limit: int = 10) -> list[dict]:
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
