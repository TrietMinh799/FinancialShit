"""agent.py — Agentic RAG loop: an LLM drives retrieval actions.

The agent runs a Think -> Act -> Observe loop. At each step the LLM sees the
question, a compact summary of evidence gathered so far, and the list of
available tools. It replies with a single JSON action. The loop executes the
tool, appends a compact observation, and repeats until the LLM calls
``final_answer`` or ``max_iterations`` is reached.

The initial harvest is deliberately lighter than the classic pipeline (decompose
+ search + rerank + expand only — no keyword or LLM expansion).  This saves
time and tokens for simple questions and gives the agent room to drive targeted
searches for complex or tricky ones.

Tool selection uses plain JSON output (not the native function-calling API) so
it works with every OpenAI-compatible provider, including Ollama and free
OpenRouter models. Malformed replies fall back gracefully.

``run_agent`` is a generator yielding SSE-compatible event dicts:
    {"status": "..."}          progress updates for the UI
    {"agent_step": {...}}      structured log of each tool call
    {"token": "..."}           final-answer tokens (streamed)
    {"done": True, ...}        terminal event with citations
    {"error": "..."}           terminal error event
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import OPENAI_MODEL, RERANK_TOP_K
from core.llm import (
    _normalize_base_url,
    _post_chat_completion,
    call_openai_llm,
    call_openai_llm_stream,
    decompose_query,
    fallback_answer,
)
from core.reranker import rerank
from core.text_utils import query_terms, unique

logger = logging.getLogger(__name__)

# Hard ceiling on agent iterations (each costs one LLM round-trip)
MAX_ITERATIONS = 4
# Per-decision LLM timeout (seconds). Decisions are small completions.
DECISION_TIMEOUT = 45
# How many evidence summaries the agent sees per step (keeps context bounded)
EVIDENCE_SUMMARY_LIMIT = 20
SNIPPET_PREVIEW_CHARS = 180

_TOOL_DESCRIPTIONS = """\
Available tools (respond with EXACTLY ONE JSON object, nothing else):

1. search — retrieve passages from the library (vector + keyword hybrid).
   {"tool": "search", "args": {"query": "<English search query>"}}
   Use when: a specific aspect of the question has no evidence yet.

2. expand — fetch full text + neighboring chunks for current top evidence.
   {"tool": "expand", "args": {}}
   Use when: top passages are truncated and need full context before answering.

3. expand_keywords — keyword-based query expansion from current top passages.
   {"tool": "expand_keywords", "args": {}}
   Use when: current evidence covers the topic but you want to cast a wider net
   to catch related passages.

4. expand_llm — LLM-generated alternative search queries.
   {"tool": "expand_llm", "args": {}}
   Use when: a different phrasing or angle might turn up better evidence.

5. final_answer — stop gathering and produce the answer from current evidence.
   {"tool": "final_answer", "args": {}}
   Use when: every key aspect of the question has supporting evidence, or
   further searching is unlikely to help.

Rules:
- Output ONLY the JSON object. No prose, no markdown fences, no explanations.
- Search queries must be in English (the library is English).
- Do not repeat a search query you already tried.
- Call final_answer by iteration {max_iterations} at the latest."""

_AGENT_SYSTEM_PROMPT = (
    "You are a retrieval agent for a financial knowledge base (M&A, valuation, "
    "company reports). Your job is to gather evidence that directly answers "
    "the user's question.\n\n"
    "For each step:\n"
    "1. Break the question into key facts needed to answer it.\n"
    "2. Check current evidence — which key facts are covered and which are "
    "missing?\n"
    "3. Pick the best tool to fill the biggest gap.\n\n"
    + _TOOL_DESCRIPTIONS
)


_DECIDE_SIMPLE_PROMPT = (
    "You are a retrieval agent. You have gathered {count} passages of evidence.\n"
    "If the evidence covers all key aspects of the question, call final_answer.\n"
    "If important aspects are still missing, call search with a targeted query.\n"
    "Return ONLY a JSON object: "
    '{{"tool": "final_answer"}} or {{"tool": "search", "args": {{"query": "..."}}}}.'
)


def _summarize_evidence(citations: list[dict]) -> str:
    """Compact evidence listing the agent sees each step (bounded size)."""
    if not citations:
        return "(no evidence gathered yet)"
    lines = []
    for i, c in enumerate(citations[:EVIDENCE_SUMMARY_LIMIT], start=1):
        snippet = (c.get("snippet") or c.get("context_text") or "")[:SNIPPET_PREVIEW_CHARS]
        page = f" p.{c.get('page_start')}" if c.get("page_start") else ""
        lines.append(f"[{i}] {c.get('title', '?')} ({c.get('source_type', '?')}{page}): {snippet}")
    if len(citations) > EVIDENCE_SUMMARY_LIMIT:
        lines.append(f"... and {len(citations) - EVIDENCE_SUMMARY_LIMIT} more")
    return "\n".join(lines)


def _parse_action(text: str) -> dict | None:
    """Extract the first JSON object from the LLM reply. None if unparseable."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        action = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(action, dict) or "tool" not in action:
        return None
    return action


def _decide(
    question: str,
    citations: list[dict],
    log: list[str],
    iteration: int,
    api_key: str,
    model: str,
    base_url: str,
) -> dict | None:
    """One LLM round-trip: return the parsed action dict, or None on failure."""
    user_content = (
        f"Question: {question}\n\n"
        f"Iteration: {iteration + 1} of {MAX_ITERATIONS}\n\n"
        f"Actions taken so far:\n"
        + ("\n".join(log) if log else "(none)")
        + f"\n\nEvidence gathered ({len(citations)} passages):\n"
        + _summarize_evidence(citations)
        + "\n\nWhat is your next action? Reply with one JSON object."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _AGENT_SYSTEM_PROMPT.replace(
                "{max_iterations}", str(MAX_ITERATIONS))},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
    }
    try:
        result = _post_chat_completion(payload, api_key, base_url, DECISION_TIMEOUT)
        text = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return _parse_action(text)
    except Exception as exc:
        logger.warning("Agent decision call failed: %s", exc)
        return None


def _decide_simple(
    question: str,
    citations: list[dict],
    api_key: str,
    model: str,
    base_url: str,
) -> dict | None:
    """Simpler fallback decision call — used when the full _decide fails."""
    prompt = _DECIDE_SIMPLE_PROMPT.format(count=len(citations))
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{_summarize_evidence(citations)}"},
    ]
    payload = {"model": model, "messages": messages, "temperature": 0.1}
    try:
        result = _post_chat_completion(payload, api_key, base_url, 30)
        text = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return _parse_action(text)
    except Exception as exc:
        logger.warning("Agent simple decision call failed: %s", exc)
        return None


def _initial_harvest(
    store,
    question: str,
    api_key: str,
    model: str,
    base_url: str,
    history: list | None,
) -> list[dict]:
    """Light initial harvest — returns up to RERANK_TOP_K enriched citations.

    Does decompose + parallel hybrid search + fallback + rerank + context
    expansion.  Keyword-based and LLM-based expansion are deliberately left
    for the agent loop so the LLM can drive them when gaps are detected.
    """
    # Phase 1 — Decompose
    sub_queries = decompose_query(question, api_key, model, base_url, history)
    if question not in sub_queries:
        sub_queries.append(question)

    # Phase 2 — Parallel hybrid search
    all_citations: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(sub_queries), 4)) as pool:
        fut_map = {
            pool.submit(store.hybrid_search, q, ["book", "annual_report"], 30): q
            for q in sub_queries
        }
        for fut in as_completed(fut_map):
            try:
                all_citations.extend(fut.result())
            except Exception:
                pass
    citations = unique(all_citations, 50)

    # Phase 2b — Broader fallback if no results
    if not citations:
        broad_terms = query_terms(question)
        if broad_terms:
            broad_query = " ".join(broad_terms[:15])
            citations = store.hybrid_search(broad_query, ["book", "annual_report"], 20)
            citations = unique(citations, 20)

    # Phase 2c — Rerank
    if citations:
        citations = rerank(question, citations, top_k=RERANK_TOP_K)

    # Phase 3a — Context expansion
    if citations:
        citations = store.expand_context(citations)[:RERANK_TOP_K]

    return citations


def run_agent(
    store,
    question: str,
    api_key: str,
    model: str | None = None,
    base_url: str | None = None,
    history: list | None = None,
    max_iterations: int = MAX_ITERATIONS,
):
    """Agentic RAG loop. Yields SSE-compatible event dicts.

    Requires *api_key* — callers must route no-key requests to the classic
    pipeline instead.
    """
    model = model or OPENAI_MODEL
    base_url = _normalize_base_url(base_url)

    log: list[str] = []
    tried_queries: set[str] = set()

    # ------------------------------------------------------------------
    # Initial harvest — light seed evidence (no keyword/LLM expansion yet)
    # ------------------------------------------------------------------
    yield {"status": "Agent: initial search..."}
    citations = _initial_harvest(store, question, api_key, model, base_url, history)
    log.append(f"initial harvest: {len(citations)} passages")
    yield {"status": f"Agent: {len(citations)} passages gathered, refining..."}

    # ------------------------------------------------------------------
    # Agent loop — refine evidence with targeted actions
    # ------------------------------------------------------------------
    for iteration in range(max_iterations):
        action = _decide(question, citations, log, iteration, api_key, model, base_url)

        # Retry once with a simpler prompt before giving up
        if action is None:
            log.append("decision failed, retrying simple...")
            yield {"status": "Agent re-evaluating..."}
            action = _decide_simple(question, citations, api_key, model, base_url)

        if action is None:
            log.append("decision failed again -> final_answer")
            break

        tool = str(action.get("tool", "")).strip().lower()
        args = action.get("args") or {}

        if tool == "final_answer":
            log.append("final_answer")
            yield {"agent_step": {"iteration": iteration + 1, "tool": tool}}
            break

        elif tool == "search":
            query = str(args.get("query", "")).strip() or question
            if query.lower() in tried_queries:
                log.append(f"search (skipped duplicate): {query}")
                continue
            tried_queries.add(query.lower())
            yield {"status": f"Agent searching: {query[:80]}..."}
            yield {"agent_step": {"iteration": iteration + 1, "tool": tool, "query": query}}
            results = store.hybrid_search(query, ["book", "annual_report"], 30)
            citations = unique(citations + results, 50)
            log.append(f"search: \"{query}\" -> {len(results)} results, {len(citations)} total")

        elif tool == "expand":
            if citations:
                yield {"status": "Agent expanding context..."}
                yield {"agent_step": {"iteration": iteration + 1, "tool": tool}}
                citations = store.expand_context(citations)[:RERANK_TOP_K]
                log.append(f"expand -> {len(citations)} enriched passages")
            else:
                log.append("expand (skipped: no evidence)")

        elif tool == "expand_keywords":
            if citations:
                yield {"status": "Agent expanding via keywords..."}
                yield {"agent_step": {"iteration": iteration + 1, "tool": tool}}
                extra_terms: set[str] = set()
                for c in citations[:min(5, len(citations))]:
                    text = c.get("context_text") or c.get("snippet", "")
                    terms = query_terms(text)
                    extra_terms.update(t for t in terms[:8] if t.lower() not in question.lower())
                if extra_terms:
                    expansion = f"{question} {' '.join(list(extra_terms)[:12])}"
                    extra = store.hybrid_search(expansion, ["book", "annual_report"], 15)
                    if extra:
                        citations = unique(citations + extra, 40)
                        citations = rerank(question, citations, top_k=RERANK_TOP_K)
                        citations = store.expand_context(citations)[:RERANK_TOP_K]
                        log.append(f"expand_keywords -> +{len(extra)} new, {len(citations)} total")
                    else:
                        log.append("expand_keywords: no new results")
                else:
                    log.append("expand_keywords: no extra terms extracted")
            else:
                log.append("expand_keywords (skipped: no evidence)")

        elif tool == "expand_llm":
            if citations and api_key:
                yield {"status": "Agent expanding via LLM queries..."}
                yield {"agent_step": {"iteration": iteration + 1, "tool": tool}}
                try:
                    top_texts = []
                    for c in citations[:3]:
                        txt = c.get("context_text") or c.get("snippet", "")
                        if txt:
                            top_texts.append(txt[:300])
                    if top_texts:
                        expansion_prompt = (
                            "Original question: " + question + "\n\n"
                            "Top evidence snippets:\n" +
                            "\n---\n".join(top_texts) + "\n\n"
                            "Generate 2-3 alternative search queries that would find "
                            "additional relevant evidence the current snippets might miss. "
                            "Focus on different phrasings, synonyms, or related aspects. "
                            "Return ONLY the queries, one per line, no numbering."
                        )
                        alt_queries = call_openai_llm(expansion_prompt, [], api_key, model, base_url, history=[])
                        if alt_queries and alt_queries.strip():
                            alt_lines = [ln.strip() for ln in alt_queries.splitlines() if ln.strip() and len(ln.strip()) > 5]
                            new_total = 0
                            for aq in alt_lines[:3]:
                                extra = store.hybrid_search(aq, ["book", "annual_report"], 15)
                                if extra:
                                    citations = unique(citations + extra, 40)
                                    new_total += len(extra)
                            if new_total:
                                citations = rerank(question, citations, top_k=RERANK_TOP_K)
                                citations = store.expand_context(citations)[:RERANK_TOP_K]
                                log.append(f"expand_llm -> +{new_total} new, {len(citations)} total")
                            else:
                                log.append("expand_llm: no new results")
                        else:
                            log.append("expand_llm: LLM returned no queries")
                except Exception:
                    log.append("expand_llm: call failed")
            else:
                log.append("expand_llm (skipped: no api_key or evidence)")

        else:
            log.append(f"unknown tool \"{tool}\" (ignored)")

    # ------------------------------------------------------------------
    # Finalize: rerank + expand, then stream the answer
    # ------------------------------------------------------------------
    if not citations:
        yield {"status": "Agent found no evidence; broad search..."}
        citations = unique(
            store.hybrid_search(question, ["book", "annual_report"], 20), 20
        )

    if citations:
        if len(citations) > RERANK_TOP_K:
            citations = rerank(question, citations, top_k=RERANK_TOP_K)
        if not any(c.get("context_text") for c in citations):
            citations = store.expand_context(citations)[:RERANK_TOP_K]

    # Quality check — if evidence is still thin, do a final broad search
    if citations and len(citations) < 5 and api_key:
        yield {"status": "Agent gathering more evidence..."}
        extra = store.hybrid_search(question, ["book", "annual_report"], 15)
        if extra:
            citations = unique(citations + extra, 30)
            citations = rerank(question, citations, top_k=RERANK_TOP_K)
            citations = store.expand_context(citations)[:RERANK_TOP_K]

    if not citations:
        answer = fallback_answer(question, citations)
        yield {"token": answer}
        yield {"done": True, "citations": [], "mode": "rag",
               "mode_label": "Evidence-based answer", "full_text": answer,
               "agent_log": log}
        return

    yield {"status": "Generating answer with LLM..."}
    for chunk in call_openai_llm_stream(
        question, citations, api_key, model, base_url, history=history
    ):
        if "token" in chunk:
            yield {"token": chunk["token"]}
        elif "done" in chunk:
            yield {"done": True, "citations": chunk.get("citations", citations),
                   "mode": "agent", "mode_label": f"Agentic answer ({model})",
                   "full_text": chunk.get("full_text", ""), "agent_log": log}
            return
        elif "error" in chunk:
            yield {"error": chunk["error"]}
            return

    yield {"done": True, "citations": citations, "mode": "agent",
           "mode_label": f"Agentic answer ({model})", "full_text": "",
           "agent_log": log}
