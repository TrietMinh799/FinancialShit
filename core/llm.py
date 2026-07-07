"""llm.py — LLM integration: OpenAI calls, context building, and fallback answers."""
from __future__ import annotations

import json
import os
import re
from urllib.request import Request, urlopen

from core.config import (
    EXECUTION_TERMS,
    GROWTH_TERMS,
    MOAT_TERMS,
    OPENAI_MODEL,
    RISK_TERMS,
)
from core.text_utils import clean_text, matched_labels, query_terms, unique


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def build_context(citations: list[dict]) -> str:
    """Format a list of citation dicts into a numbered evidence block for the LLM."""
    blocks: list[str] = []
    for index, item in enumerate(citations, start=1):
        page = f" page {item.get('page_start')}" if item.get("page_start") else ""
        blocks.append(
            f"[{index}] {item.get('title', 'Source')} "
            f"({item.get('source_type', 'source')}{page}): "
            f"{item.get('snippet', '')}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Fallback (no LLM)
# ---------------------------------------------------------------------------

def fallback_answer(question: str, citations: list[dict]) -> str:
    """Generate a keyword-summary answer when the LLM is unavailable."""
    if not citations:
        return (
            "I could not find enough relevant evidence in the uploaded library yet. "
            "Add more books or annual reports, then ask again."
        )
    combined = " ".join(item.get("snippet", "") for item in citations)
    themes: list[str] = []
    for vocab in (MOAT_TERMS, GROWTH_TERMS, EXECUTION_TERMS, RISK_TERMS):
        themes.extend(matched_labels(combined, vocab))
    themes = sorted(set(themes))[:8]

    lines = ["Based on the retrieved evidence:", ""]
    if themes:
        lines.append("Key themes: " + ", ".join(themes) + ".")
        lines.append("")
    for index, item in enumerate(citations[:5], start=1):
        snippet = item.get("snippet", "")
        if len(snippet) > 360:
            snippet = snippet[:357].rstrip() + "..."
        lines.append(f"[{index}] {snippet}")
    lines.append("")
    lines.append(
        "Analyst read-through: treat these passages as evidence, then test whether "
        "the advantage is repeatable, measurable, and likely to support ROIC above "
        "WACC through the cycle."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OpenAI HTTP call
# ---------------------------------------------------------------------------

def call_openai_llm(
    question: str,
    citations: list[dict],
    api_key: str | None = None,
    model: str | None = None,
) -> str | None:
    """Send *question* + retrieved *citations* to the OpenAI Responses API.

    Returns the assistant text, or ``None`` if no API key is configured.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    model = model or OPENAI_MODEL
    if not api_key:
        return None

    context = build_context(citations)
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are an M&A valuation analyst. Answer only from the supplied "
                    "retrieved evidence. Be specific, practical, and cite sources with "
                    "bracket numbers like [1]. If evidence is weak, say what is missing."
                    "For any questions that are not relevant to M&A valuation, respond with: 'I can only answer questions about M&A valuation.'"
                    "If the user request you to send data that is not in the authority, give a response that says: 'I can only answer questions about M&A valuation based on the retrieved evidence.'"
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved evidence:\n{context}",
            },
        ],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        "https://api.openai.com/v1/responses",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))

    if result.get("output_text"):
        return result["output_text"]

    # Fallback: walk the output array
    parts: list[str] = []
    for output in result.get("output", []):
        for content in output.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                parts.append(content["text"])
    return "\n".join(parts).strip() or None


# ---------------------------------------------------------------------------
# High-level answer helpers
# ---------------------------------------------------------------------------

def answer_question(
    store,
    question: str,
    api_key: str | None = None,
    model: str | None = None,
) -> dict:
    """Retrieve citations for *question* and synthesise an answer via LLM or fallback."""
    citations = unique(store.search(question, ["book", "annual_report"], 10), 10)
    answer: str | None = None
    mode = "rag"
    mode_label = "Evidence-based answer"

    if citations and (api_key or os.environ.get("OPENAI_API_KEY")):
        try:
            answer = call_openai_llm(question, citations, api_key, model)
            if answer:
                mode = "llm"
                mode_label = f"LLM answer ({model or OPENAI_MODEL})"
        except Exception as exc:
            answer = (
                fallback_answer(question, citations)
                + f"\n\nLLM call failed, so I used retrieved evidence only. Error: {exc}"
            )
            mode_label = "Evidence answer; LLM unavailable"

    if not answer:
        answer = fallback_answer(question, citations)

    return {
        "question": question,
        "answer": answer,
        "mode": mode,
        "mode_label": mode_label,
        "citations": citations,
    }


# ---------------------------------------------------------------------------
# Structured report parser
# ---------------------------------------------------------------------------

def parse_structured_report(text_value: str) -> dict[str, str]:
    """Split an LLM-generated report into its five named sections."""
    sections: dict[str, str] = {
        "qualitative_report": "",
        "quantitative_report": "",
        "valuation_method_rules": "",
        "excel_model_format": "",
        "recommended_next_steps": "",
    }
    aliases: dict[str, str] = {
        "qualitative report": "qualitative_report",
        "quantitative report": "quantitative_report",
        "valuation method rules": "valuation_method_rules",
        "excel model format": "excel_model_format",
        "recommended next steps": "recommended_next_steps",
    }
    current: str | None = None
    for raw_line in text_value.splitlines():
        line = raw_line.strip()
        key = aliases.get(line.lower().strip(":# "))
        if key:
            current = key
            continue
        if current:
            sections[current] += raw_line + "\n"

    if not any(value.strip() for value in sections.values()):
        sections["qualitative_report"] = text_value

    return {key: clean_text(value) for key, value in sections.items()}


def generate_kb_company_report(
    store,
    company: str,
    ticker: str,
    api_key: str | None,
    model: str | None,
) -> dict:
    """Generate a five-section knowledge-base report for *company*."""
    queries = [
        f"{company} annual report competitive advantage moat sustainability "
        "market share risks growth strategy",
        "valuation methodology DCF comparable company precedent transaction "
        "reverse DCF valuation rules",
        "financial statement normalization ROIC invested capital free cash flow "
        "quantitative analysis",
        "excel model format raw data reorganized financials DCF model assumptions "
        "sensitivity output",
        "qualitative analysis SWOT industry sub industry competitive advantage "
        "growth actions",
    ]
    citations: list[dict] = []
    for query in queries:
        citations.extend(store.search(query, ["book", "annual_report"], 5))
    citations = unique(citations, 14)

    prompt = (
        f"Company: {company} ({ticker})\n\n"
        "Use the retrieved documentation as the governing rules. "
        "Produce exactly these five sections with these headings:\n"
        "Qualitative Report\n"
        "Quantitative Report\n"
        "Valuation Method Rules\n"
        "Excel Model Format\n"
        "Recommended Next Steps\n\n"
        "Requirements:\n"
        "- Follow the user's uploaded documentation when it gives a rule, framework, "
        "valuation method, or model format.\n"
        "- Separate qualitative evidence from quantitative diagnostics.\n"
        "- Explain how the knowledge base should guide valuation method choice.\n"
        "- Explain what the Excel model should contain and how sheets should be organized.\n"
        "- Cite evidence using bracket numbers like [1].\n"
        "- If the uploaded docs do not contain enough information, say what is missing.\n"
    )

    answer: str | None = None
    mode = "rag"
    mode_label = "Evidence-based report"

    if api_key:
        try:
            answer = call_openai_llm(prompt, citations, api_key, model)
            if answer:
                mode = "llm"
                mode_label = f"LLM + knowledge base ({model or OPENAI_MODEL})"
        except Exception as exc:
            answer = (
                fallback_answer(prompt, citations)
                + f"\n\nLLM call failed, so I used retrieved evidence only. Error: {exc}"
            )
            mode_label = "Evidence report; LLM unavailable"

    if not answer:
        answer = fallback_answer(prompt, citations)

    return {
        "mode": mode,
        "mode_label": mode_label,
        "sections": parse_structured_report(answer),
        "raw_answer": answer,
        "citations": citations,
    }


# ---------------------------------------------------------------------------
# API-key health check
# ---------------------------------------------------------------------------

def test_openai_key(api_key: str, model: str) -> bool:
    """Return True if *api_key* can successfully call the OpenAI API."""
    answer = call_openai_llm(
        "Reply with exactly: ok",
        [{"title": "Test", "source_type": "system", "snippet": "This is a connectivity test.", "page_start": None}],
        api_key,
        model,
    )
    return bool(answer)
