"""analysis.py — Rule-based scoring engine and annual-report analyser."""

from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime
from pathlib import Path

from core.config import (
    EXECUTION_TERMS,
    GROWTH_TERMS,
    MAX_RUNS,
    MOAT_TERMS,
    RERANK_TOP_K,
    RISK_TERMS,
    RUNS,
)
from core.extractors import extract_pages
from core.llm import call_openai_llm
from core.reranker import rerank
from core.store import Store
from core.text_utils import (
    annual_hits,
    clip,
    matched_labels,
    mention_count,
    unique,
)

# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def _compute_scores(
    report_text: str,
    moat: list[str],
    growth: list[str],
    execution: list[str],
    risks: list[str],
) -> dict:
    """Compute five sub-scores and an overall score from term-match counts."""
    moat_mentions = mention_count(report_text, MOAT_TERMS)
    growth_mentions = mention_count(report_text, GROWTH_TERMS)
    execution_mentions = mention_count(report_text, EXECUTION_TERMS)
    risk_mentions = mention_count(report_text, RISK_TERMS)

    log_scale = max(1.0, math.log10(max(1000, len(report_text))))

    moat_score = clip(
        42
        + len(moat) * 5
        + min(24, moat_mentions / log_scale)
        - min(18, risk_mentions / (log_scale * 1.8))
    )
    growth_score = clip(
        40
        + len(growth) * 5
        + min(26, growth_mentions / log_scale)
        - min(14, risk_mentions / (log_scale * 2.2))
    )
    execution_score = clip(42 + len(execution) * 6 + min(22, execution_mentions / log_scale))
    financial_score = clip(
        58 - min(24, risk_mentions / log_scale) + (8 if "Risk management" in execution else 0)
    )
    risk_score = clip(35 + len(risks) * 6 + min(28, risk_mentions / log_scale))

    overall = clip(
        moat_score * 0.30
        + growth_score * 0.30
        + execution_score * 0.18
        + financial_score * 0.14
        + (100 - risk_score) * 0.08
    )

    return {
        "moat_score": moat_score,
        "growth_score": growth_score,
        "execution_score": execution_score,
        "financial_score": financial_score,
        "risk_score": risk_score,
        "overall": overall,
    }


def _moat_rating(moat_score: int) -> tuple[str, str]:
    """Map a numeric moat score to a (rating, assessment) pair."""
    if moat_score >= 75:
        return (
            "Strong",
            "Competitive advantage appears durable, supported by several "
            "repeatable moat signals in the report.",
        )
    if moat_score >= 60:
        return (
            "Moderate",
            "Competitive advantage is visible, but its durability depends on "
            "execution and industry pressure.",
        )
    return (
        "Developing",
        "The report shows some advantage signals, but sustainability needs more evidence.",
    )


# ---------------------------------------------------------------------------
# Strategic actions
# ---------------------------------------------------------------------------


def _build_growth_actions(moat: list[str], growth: list[str], risks: list[str]) -> list[str]:
    """Derive prioritised management actions from signal lists."""
    actions: list[str] = []

    if "Capacity expansion" in growth or "Investment program" in growth:
        actions.append(
            "Prioritize expansion projects where incremental ROIC can stay "
            "above WACC through the cycle."
        )
    if "Cost advantage" in moat or "Vertical integration" in moat:
        actions.append(
            "Protect the cost position by locking in input efficiency, "
            "logistics advantage, and operating utilization."
        )
    if "Export growth" in growth or "Market expansion" in growth:
        actions.append(
            "Diversify demand channels and avoid depending on one geography, "
            "customer group, or trade regime."
        )
    if "Brand strength" in moat or "Customer relationships" in moat:
        actions.append(
            "Convert customer relationships into recurring volume, pricing "
            "power, and product mix improvement."
        )
    if "Input-cost exposure" in risks or "Commodity exposure" in risks:
        actions.append(
            "Build a mid-cycle margin plan with procurement discipline, "
            "hedging policy, and inventory risk controls."
        )

    # Always-on actions
    actions += [
        "Define the core moat in measurable terms: share, margin spread, "
        "retention, cost curve, or utilization.",
        "Separate growth capex from maintenance capex and require hurdle-rate "
        "discipline for new projects.",
        "Track leading indicators in the pitch book: market share, volume, "
        "price spread, utilization, leverage, and ROIC.",
    ]
    return actions


# ---------------------------------------------------------------------------
# LLM-based reasoning layer
# ---------------------------------------------------------------------------


def _build_reasoning_prompt(
    company: str,
    ticker: str,
    moat_signals: list[str],
    growth_signals: list[str],
    execution_signals: list[str],
    risk_signals: list[str],
    scores: dict,
    book_passages: list[dict],
) -> tuple[str, list[dict]]:
    """Build the prompt + citation list for the reasoned-analysis LLM call."""
    moat_str = ", ".join(moat_signals) if moat_signals else "(none detected)"
    growth_str = ", ".join(growth_signals) if growth_signals else "(none detected)"
    exec_str = ", ".join(execution_signals) if execution_signals else "(none detected)"
    risk_str = ", ".join(risk_signals) if risk_signals else "(none detected)"

    kb_lines: list[str] = []
    for i, p in enumerate(book_passages, 1):
        src = p.get("title", "Source")
        snippet = p.get("snippet", "")
        kb_lines.append(f"[{i}] {src}:\n{snippet}")
    kb_text = "\n\n".join(kb_lines) if kb_lines else "(No book passages retrieved)"

    prompt = (
        f"Company: {company} ({ticker})\n\n"
        "A keyword-based scan of the annual report detected the following signals "
        "and produced preliminary scores (0-100, higher is better). "
        "Relevant passages from the uploaded valuation knowledge base are provided below.\n\n"
        "--- PRELIMINARY SIGNALS ---\n"
        f"Moat signals: {moat_str}\n"
        f"Growth signals: {growth_str}\n"
        f"Execution signals: {exec_str}\n"
        f"Risk signals: {risk_str}\n\n"
        "--- PRELIMINARY SCORES ---\n"
        f"Moat Sustainability: {scores['moat_score']}/100\n"
        f"Growth Capacity: {scores['growth_score']}/100\n"
        f"Execution Quality: {scores['execution_score']}/100\n"
        f"Financial Resilience: {scores['financial_score']}/100\n"
        f"Risk Pressure: {scores['risk_score']}/100\n"
        f"Overall: {scores['overall']}/100\n\n"
        "--- KNOWLEDGE BASE PASSAGES ---\n"
        f"{kb_text}\n\n"
        "Based on the preliminary findings and the knowledge base above, provide "
        "a reasoned analysis as JSON with these fields:\n"
        "{\n"
        '  "score_commentary": {\n'
        '    "moat": "2-3 sentence analysis of competitive advantage, referencing '
        "both report signals and knowledge base principles...\",\n"
        '    "growth": "2-3 sentence analysis of growth prospects...",\n'
        '    "execution": "2-3 sentence analysis of execution quality...",\n'
        '    "financial": "2-3 sentence analysis of financial resilience...",\n'
        '    "risk": "2-3 sentence analysis of key risks..."\n'
        "  },\n"
        '  "reasoned_swot": {\n'
        '    "strengths": ["Strength 1 with brief evidence...", "Strength 2..."],\n'
        '    "weaknesses": ["Weakness 1...", "Weakness 2..."],\n'
        '    "opportunities": ["Opportunity 1...", "Opportunity 2..."],\n'
        '    "threats": ["Threat 1...", "Threat 2..."]\n'
        "  },\n"
        '  "key_considerations": "2-3 paragraph summary of what an analyst should '
        "focus on, referencing valuation principles from the knowledge base...\",\n"
        '  "score_adjustments": {\n'
        '    "moat": 0,\n'
        '    "growth": 0,\n'
        '    "execution": 0,\n'
        '    "financial": 0,\n'
        '    "risk": 0\n'
        "  }\n"
        "}\n\n"
        "RULES:\n"
        "- Each SWOT item must be 10-25 words and reference specific evidence.\n"
        "- Score adjustments must be integers from -10 to +10.\n"
        "- If knowledge base passages are insufficient, rely on your own valuation "
        "training (ROIC, WACC, moat durability, capex discipline, etc.).\n"
        "- Output ONLY valid JSON, no surrounding text or markdown."
    )
    return prompt, book_passages


def _parse_reasoning_response(text: str) -> dict | None:
    """Extract JSON from the LLM reasoning response."""
    text = text.strip()
    # Try to find a JSON block delimited by ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        # Assume the entire response is JSON
        brace = text.find("{")
        if brace >= 0:
            text = text[brace:]
            end = text.rfind("}")
            if end >= 0:
                text = text[: end + 1]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def reason_analysis(
    store: Store,
    company: str,
    ticker: str,
    report_text: str,
    moat_signals: list[str],
    growth_signals: list[str],
    execution_signals: list[str],
    risk_signals: list[str],
    scores: dict,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    tone: str = "basic",
) -> dict:
    """Use LLM + book knowledge to produce reasoned commentary on scores and SWOT.

    Returns an enriched analysis dict with *score_commentary*, *reasoned_swot*,
    *key_considerations*, and *score_adjustments*. Falls back to an empty dict
    (no reasoning) when the LLM is unavailable.
    """
    if not api_key:
        return {}

    # Retrieve relevant book passages for reasoning context
    kb_queries = [
        f"{company} moat competitive advantage sustainable strategy",
        f"{company} growth capital allocation ROIC investment",
        f"{company} financial risk balance sheet leverage cash flow",
        "valuation methodology DCF ROIC WACC framework analyst approach",
    ]
    book_cites: list[dict] = []
    for q in kb_queries:
        book_cites.extend(store.hybrid_search(q, ["book"], 5))
    book_cites = unique(book_cites, 8)

    if book_cites:
        rerank_query = (
            f"{company} moat growth risk valuation ROIC WACC competitive advantage"
        )
        book_cites = rerank(rerank_query, book_cites, top_k=RERANK_TOP_K)

    prompt, citations = _build_reasoning_prompt(
        company, ticker,
        moat_signals, growth_signals, execution_signals, risk_signals,
        scores, book_cites,
    )
    
    if tone == "professional":
        prompt = "[TONE: Professional - Write your commentary using formal financial terminology and precise language.]\n\n" + prompt
    elif tone == "expert":
        prompt = "[TONE: Expert - Write your commentary as a seasoned Managing Director. Enrich the analysis with your own financial reasoning. Mark insights not found in the evidence with [Analysis] or [Inference].]\n\n" + prompt

    # We use a strict system prompt here to enforce JSON, overriding the default chat system prompt.
    system_prompt = "You are an M&A valuation analyst. Output ONLY valid JSON."

    try:
        raw = call_openai_llm(prompt, citations, api_key, model, base_url, system_prompt=system_prompt)
        if not raw:
            return {}
        parsed = _parse_reasoning_response(raw)
        if not parsed:
            return {}
        return {
            "score_commentary": parsed.get("score_commentary", {}),
            "reasoned_swot": parsed.get("reasoned_swot", {}),
            "key_considerations": parsed.get("key_considerations", ""),
            "score_adjustments": parsed.get("score_adjustments", {}),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze_report(
    store: Store,
    report_path: Path,
    company: str,
    ticker: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    tone: str = "basic",
) -> dict:
    """Parse *report_path*, score it, and persist the result in RUNS.

    Also searches *store* for supporting knowledge-base citations.
    When an API key is provided, uses the LLM to produce a reasoned commentary
    on the scores and SWOT, grounded in the uploaded knowledge base.
    """
    pages = list(extract_pages(report_path))
    report_text = " ".join(text for _, text in pages)

    # Persist the report in the document store
    doc_info = store.add_document(report_path, f"{company} annual report", "annual_report")
    if not doc_info.get("inserted", True):
        report_path.unlink(missing_ok=True)

    # Signal detection
    moat = matched_labels(report_text, MOAT_TERMS)
    growth = matched_labels(report_text, GROWTH_TERMS)
    execution = matched_labels(report_text, EXECUTION_TERMS)
    risks = matched_labels(report_text, RISK_TERMS)

    scores = _compute_scores(report_text, moat, growth, execution, risks)
    moat_rating, moat_assessment = _moat_rating(scores["moat_score"])
    growth_rating = (
        "High" if scores["overall"] >= 75 else ("Medium" if scores["overall"] >= 60 else "Watch")
    )

    # SWOT
    strengths = moat[:5] or ["Operational advantage requires more supporting evidence"]
    weaknesses = risks[:5] or ["No dominant risk theme detected in the report text"]
    opportunities = growth[:5] or ["Growth option set requires deeper market evidence"]
    threats = risks[:5] or ["Competitive and macro risk should still be tested"]

    # Citation queries
    queries = [
        f"{company} sustainable competitive advantage moat scale switching costs "
        "cost advantage barriers",
        f"{company} company situation competitive pressure risks market share capital allocation",
        f"{company} growth capacity expansion market penetration innovation "
        "returns on invested capital",
        "actions to grow sustainably competitive advantage reinvestment "
        "capital allocation strategy",
    ]
    knowledge: list[dict] = []
    report_cites: list[dict] = []
    for q in queries:
        knowledge.extend(store.search(q, ["book"], 4))
        report_cites.extend(annual_hits(pages, q, 4))

    result = {
        "run_id": uuid.uuid4().hex,
        "project": {
            "company": company,
            "ticker": ticker,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "report": {
            "filename": report_path.name,
            "pages": len(pages),
            "characters": len(report_text),
        },
        "library": store.stats(),
        "scores": {
            "overall_growth_score": scores["overall"],
            "growth_rating": growth_rating,
            "moat_sustainability": scores["moat_score"],
            "growth_capacity": scores["growth_score"],
            "execution_quality": scores["execution_score"],
            "financial_resilience": scores["financial_score"],
            "risk_pressure": scores["risk_score"],
        },
        "competitive_advantage": {
            "rating": moat_rating,
            "assessment": moat_assessment,
            "signals": moat,
            "concerns": risks,
        },
        "company_situation": {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "opportunities": opportunities,
            "threats": threats,
        },
        "growth_actions": _build_growth_actions(moat, growth, risks)[:7],
        "citations": {
            "knowledge": unique(knowledge, 10),
            "annual_report": unique(report_cites, 10),
        },
    }

    # LLM-reasoned analysis layer (enriches scores/SWOT with book-grounded narrative)
    reasoned = reason_analysis(
        store, company, ticker, report_text,
        moat, growth, execution, risks, scores,
        api_key, model, base_url, tone,
    )
    if reasoned:
        result["reasoned_analysis"] = reasoned

    RUNS[result["run_id"]] = result
    RUNS.move_to_end(result["run_id"])
    while len(RUNS) > MAX_RUNS:
        RUNS.popitem(last=False)
    return result
