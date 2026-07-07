"""analysis.py — Rule-based scoring engine and annual-report analyser."""
from __future__ import annotations

import math
import uuid
from datetime import datetime

from core.config import (
    EXECUTION_TERMS,
    GROWTH_TERMS,
    MOAT_TERMS,
    RISK_TERMS,
    RUNS,
)
from core.extractors import extract_pages
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
    execution_score = clip(
        42 + len(execution) * 6 + min(22, execution_mentions / log_scale)
    )
    financial_score = clip(
        58
        - min(24, risk_mentions / log_scale)
        + (8 if "Risk management" in execution else 0)
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
        "The report shows some advantage signals, but sustainability needs "
        "more evidence.",
    )


# ---------------------------------------------------------------------------
# Strategic actions
# ---------------------------------------------------------------------------

def _build_growth_actions(
    moat: list[str], growth: list[str], risks: list[str]
) -> list[str]:
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
# Main entry point
# ---------------------------------------------------------------------------

def analyze_report(store, report_path, company: str, ticker: str) -> dict:
    """Parse *report_path*, score it, and persist the result in RUNS.

    Also searches *store* for supporting knowledge-base citations.
    """
    pages = extract_pages(report_path)
    report_text = " ".join(text for _, text in pages)

    # Persist the report in the document store
    store.add_document(report_path, f"{company} annual report", "annual_report")

    # Signal detection
    moat = matched_labels(report_text, MOAT_TERMS)
    growth = matched_labels(report_text, GROWTH_TERMS)
    execution = matched_labels(report_text, EXECUTION_TERMS)
    risks = matched_labels(report_text, RISK_TERMS)

    scores = _compute_scores(report_text, moat, growth, execution, risks)
    moat_rating, moat_assessment = _moat_rating(scores["moat_score"])
    growth_rating = (
        "High" if scores["overall"] >= 75
        else ("Medium" if scores["overall"] >= 60 else "Watch")
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
        f"{company} company situation competitive pressure risks market share "
        "capital allocation",
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

    RUNS[result["run_id"]] = result
    return result
