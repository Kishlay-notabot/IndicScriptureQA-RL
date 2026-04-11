"""
Reward computation for IndicScriptureQA — LLM-as-a-Judge.
Uses an LLM (via OpenAI client) to evaluate both factual accuracy and
semantic structure quality. Falls back to lightweight token heuristics
if the LLM call fails.
Environment variables (shared with inference.py):
  API_BASE_URL   LLM endpoint
  MODEL_NAME     Model identifier
  HF_TOKEN       API key
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Tuple

from openai import OpenAI

from models import ActionType, EnvState, StructuralMeta


# ═══════════════════════════════════════════════════════════════════════════════
# LLM CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ["API_BASE_URL"],   
            api_key=os.environ["API_KEY"],         
        )
        try:
                _client.chat.completions.create(
                    model=_get_model(),
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
        except Exception as e:
                print(f"[WARN] Warmup failed: {e}", flush=True)

    return _client


def _get_model() -> str:
    return os.environ.get("MODEL_NAME", "gpt-4o-mini")


def _llm_judge(system: str, user_prompt: str) -> Optional[dict]:
    """Call the LLM and parse a JSON response. Returns None on any failure."""
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except Exception as exc:
        print(f"[JUDGE] LLM call failed, using fallback: {exc}", flush=True)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# JUDGE PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM = (
    "You are an expert judge evaluating answers about Indic scriptures "
    "(Vedas, Upanishads, Ramayana, Mahabharata, Bhagavad Gita, Puranas). "
    "You evaluate both factual accuracy and semantic structure quality.\n\n"
    "Respond with ONLY a valid JSON object. No markdown fences, no "
    "explanation, no text outside the JSON braces."
)


def _terminal_accept_prompt(state: EnvState) -> str:
    return json.dumps({
        "task": "Score the candidate answer against the reference on all axes.",
        "question": state.question,
        "candidate_answer": state.current_answer,
        "reference_answer": state.ground_truth_answer,
        "candidate_citations": state.current_citations,
        "expected_citations": state.ground_truth_citations,
        "structural_requirements": {
            "required_terms": state.structural_meta.required_terms,
            "required_sections": state.structural_meta.required_sections,
            "expected_order": state.structural_meta.expected_order,
            "banned_terms": state.structural_meta.banned_terms,
        },
        "output_format": {
            "factual_score": "0.0-1.0: semantic accuracy of candidate vs reference",
            "citation_score": "0.0-1.0: fraction of expected citations covered",
            "terminology_score": "-0.5 to 1.0: correct Sanskrit/domain terms present; NEGATIVE if banned terms found",
            "completeness_score": "0.0-1.0: all required conceptual sections covered",
            "ordering_score": "0.0-1.0: concepts appear in expected logical sequence",
            "coherence_score": "0.0-1.0: smooth transitions, balanced structure, readable flow",
            "feedback": "one-sentence summary of quality",
        },
    }, indent=2)


def _terminal_reject_prompt(state: EnvState) -> str:
    return json.dumps({
        "task": "Judge whether this answer deserves rejection.",
        "question": state.question,
        "candidate_answer": state.current_answer,
        "reference_answer": state.ground_truth_answer,
        "structural_requirements": {
            "required_terms": state.structural_meta.required_terms,
            "banned_terms": state.structural_meta.banned_terms,
        },
        "output_format": {
            "answer_is_flawed": "boolean: true if the answer has significant factual or structural problems",
            "feedback": "one-sentence explanation",
        },
    }, indent=2)


def _step_delta_prompt(
    state: EnvState,
    action_type: ActionType,
    old_answer: str,
    new_answer: str,
) -> str:
    if action_type == ActionType.EDIT:
        focus = "Focus on FACTUAL improvement (60%) and STRUCTURAL improvement (40%)."
    else:
        focus = (
            "Focus primarily on STRUCTURAL improvement (ordering, terminology, "
            "coherence). Penalise heavily if factual content was lost."
        )
    return json.dumps({
        "task": f"Evaluate whether this {action_type.value} improved the answer.",
        "focus": focus,
        "question": state.question,
        "old_answer": old_answer,
        "new_answer": new_answer,
        "reference_answer": state.ground_truth_answer,
        "structural_requirements": {
            "required_terms": state.structural_meta.required_terms,
            "required_sections": state.structural_meta.required_sections,
            "expected_order": state.structural_meta.expected_order,
            "banned_terms": state.structural_meta.banned_terms,
        },
        "output_format": {
            "factual_delta": "-1.0 to 1.0 (positive = factual improvement)",
            "structural_delta": "-1.0 to 1.0 (positive = structural improvement)",
            "feedback": "one-sentence explanation of what changed",
        },
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK HEURISTICS (used when LLM is unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9\u0900-\u097F]+", text.lower()) if t]


def _token_f1(candidate: str, reference: str) -> float:
    cand = set(_tokenize(candidate))
    ref = set(_tokenize(reference))
    if not cand or not ref:
        return 0.0
    common = cand & ref
    if not common:
        return 0.0
    p, r = len(common) / len(cand), len(common) / len(ref)
    return 2 * p * r / (p + r)


def _citation_recall_heuristic(predicted: List[str], ground_truth: List[str]) -> float:
    if not ground_truth:
        return 1.0
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    gt = [norm(g) for g in ground_truth]
    pr = [norm(p) for p in predicted]
    matched = sum(1 for g in gt if any(g in p or p in g for p in pr))
    return matched / len(gt)


# ═══════════════════════════════════════════════════════════════════════════════
# PER-STEP REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def step_reward(
    state: EnvState, action_type: ActionType, payload: str | None,
) -> Tuple[float, str]:
    """Compute per-step reward and feedback. Uses LLM judge for EDIT/RESTRUCTURE."""

    # ── RETRIEVE ──────────────────────────────────────────────────────────
    if action_type == ActionType.RETRIEVE:
        if state.retrieval_count >= 3:
            return -0.15, "Redundant retrieval — already retrieved 3 times."
        elif state.available_passages:
            return 0.05, "Passages retrieved."
        else:
            return -0.05, "No passages available for retrieval."

    # ── CITE ──────────────────────────────────────────────────────────────
    if action_type == ActionType.CITE:
        if not payload:
            return -0.05, "Empty citation."
        cr = _citation_recall_heuristic([payload], state.ground_truth_citations)
        if cr > 0:
            return 0.15, "Correct citation added."
        return -0.05, "Citation does not match expected sources."

    # ── ACCEPT / REJECT ──────────────────────────────────────────────────
    if action_type in (ActionType.ACCEPT, ActionType.REJECT):
        return 0.0, ""

    # ── EDIT / RESTRUCTURE — LLM judge ───────────────────────────────────
    if not payload:
        return -0.10, f"Empty {action_type.value.lower()} — no content provided."

    old_answer = state.current_answer
    result = _llm_judge(
        JUDGE_SYSTEM,
        _step_delta_prompt(state, action_type, old_answer, payload),
    )

    if result is not None:
        fd = max(-1.0, min(1.0, float(result.get("factual_delta", 0.0))))
        sd = max(-1.0, min(1.0, float(result.get("structural_delta", 0.0))))
        fb = result.get("feedback", "")

        if action_type == ActionType.EDIT:
            combined = 0.6 * fd + 0.4 * sd
            if combined > 0.03:
                return 0.20 + combined, f"Edit improved (fact Δ{fd:+.2f}, struct Δ{sd:+.2f}). {fb}"
            elif combined < -0.03:
                return -0.20, f"Edit degraded (fact Δ{fd:+.2f}, struct Δ{sd:+.2f}). {fb}"
            else:
                return -0.05, f"Edit had negligible effect. {fb}"

        else:  # RESTRUCTURE
            if fd < -0.10:
                return -0.25, f"Restructure lost factual content (fact Δ{fd:+.2f}). {fb}"
            elif sd > 0.05:
                return 0.25 + sd, f"Restructure improved structure (Δ{sd:+.2f}). {fb}"
            elif sd < -0.03:
                return -0.15, f"Restructure degraded structure (Δ{sd:+.2f}). {fb}"
            else:
                return -0.05, f"Restructure had negligible effect. {fb}"

    # ── Fallback: token-F1 delta ──────────────────────────────────────────
    old_sim = _token_f1(old_answer, state.ground_truth_answer)
    new_sim = _token_f1(payload, state.ground_truth_answer)
    delta = new_sim - old_sim
    label = action_type.value

    if delta > 0.03:
        return 0.20 + delta, f"{label} improved (Δ{delta:+.2f}, fallback scoring)."
    elif delta < -0.03:
        return -0.20, f"{label} degraded (Δ{delta:+.2f}, fallback scoring)."
    return -0.05, f"{label} negligible effect (fallback scoring)."


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def terminal_reward(
    state: EnvState, action_type: ActionType,
) -> Tuple[float, str]:
    """Terminal reward using LLM-as-a-judge, with heuristic fallback."""

    # ── REJECT ────────────────────────────────────────────────────────────
    if action_type == ActionType.REJECT:
        result = _llm_judge(JUDGE_SYSTEM, _terminal_reject_prompt(state))
        if result is not None:
            is_flawed = result.get("answer_is_flawed", True)
            fb = result.get("feedback", "")
            if is_flawed:
                return 0.30, f"Correctly rejected a flawed answer. {fb}"
            else:
                return -0.50, f"Incorrectly rejected a valid answer. {fb}"
        if not state.answer_is_correct:
            return 0.30, "Correctly rejected a flawed answer (fallback)."
        return -0.50, "Incorrectly rejected a valid answer (fallback)."

    # ── ACCEPT — LLM judge ────────────────────────────────────────────────
    result = _llm_judge(JUDGE_SYSTEM, _terminal_accept_prompt(state))

    if result is not None:
        fs  = max(0.0, min(1.0, float(result.get("factual_score", 0.0))))
        cs  = max(0.0, min(1.0, float(result.get("citation_score", 0.0))))
        ts  = max(-0.5, min(1.0, float(result.get("terminology_score", 0.0))))
        comp = max(0.0, min(1.0, float(result.get("completeness_score", 0.0))))
        os_ = max(0.0, min(1.0, float(result.get("ordering_score", 0.0))))
        coh = max(0.0, min(1.0, float(result.get("coherence_score", 0.0))))
        fb  = result.get("feedback", "")

        # structural composite
        struct_score = 0.30 * max(ts, 0.0) + 0.25 * comp + 0.25 * os_ + 0.20 * coh
        if ts < 0:
            struct_score += 0.15 * ts
        struct_score = max(0.0, min(1.0, struct_score))

        efficiency = 0.20 * (state.steps_remaining / state.max_steps)

        terminal = 0.90 * fs + 0.30 * cs + 0.70 * struct_score + efficiency

        if fs < 0.3 and struct_score < 0.3:
            terminal -= 0.50
            quality = "poor"
        elif fs < 0.5:
            quality = "mediocre"
        else:
            quality = "good"

        feedback = (
            f"Accepted a {quality} answer "
            f"(fact={fs:.2f}, cite={cs:.2f}, struct={struct_score:.2f} "
            f"[term={ts:.2f} comp={comp:.2f} ord={os_:.2f} coh={coh:.2f}]). {fb}"
        )
        return terminal, feedback

    # ── Fallback: heuristic scoring ───────────────────────────────────────
    fs = _token_f1(state.current_answer, state.ground_truth_answer)
    cs = _citation_recall_heuristic(
        state.current_citations, state.ground_truth_citations,
    )
    efficiency = 0.20 * (state.steps_remaining / state.max_steps)
    terminal = 0.90 * fs + 0.30 * cs + efficiency

    if fs < 0.3:
        terminal -= 0.50
        quality = "poor"
    elif fs < 0.5:
        quality = "mediocre"
    else:
        quality = "good"

    return terminal, f"Accepted a {quality} answer (fact={fs:.2f}, cite={cs:.2f}, fallback)."


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════

MAX_REASONABLE_REWARD = 2.80


def normalize_score(cumulative_reward: float) -> float:
    """Clamp cumulative reward into [0, 1]."""
    return max(0.0, min(1.0, cumulative_reward / MAX_REASONABLE_REWARD))
