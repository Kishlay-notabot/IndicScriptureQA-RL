from __future__ import annotations

import re
from typing import List, Tuple

from models import ActionType, EnvState


# Text utilities

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
    p = len(common) / len(cand)
    r = len(common) / len(ref)
    return 2 * p * r / (p + r)


def _citation_recall(predicted: List[str], ground_truth: List[str]) -> float:
    if not ground_truth:
        return 1.0
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    gt = [norm(g) for g in ground_truth]
    pr = [norm(p) for p in predicted]
    matched = sum(1 for g in gt if any(g in p or p in g for p in pr))
    return matched / len(gt)


# Structural working

def _terminology_score(answer: str, required: List[str], banned: List[str]) -> float:
    lower = answer.lower()
    if required:
        found = sum(1 for t in required if t.lower() in lower)
        score = found / len(required)
    else:
        score = 1.0

    penalty = sum(0.25 for t in banned if t.lower() in lower)
    return max(-0.5, score - penalty)


def _completeness_score(answer: str, required_sections: List[str]) -> float:
    if not required_sections:
        return 1.0
    lower = answer.lower()
    covered = 0
    for section in required_sections:
        keywords = _tokenize(section)
        if not keywords:
            covered += 1
            continue
        hits = sum(1 for k in keywords if k in lower)
        if hits / len(keywords) >= 0.5:
            covered += 1
    return covered / len(required_sections)


def _ordering_score(answer: str, expected_order: List[str]) -> float:
    if len(expected_order) < 2:
        return 1.0

    lower = answer.lower()
    positions: List[float] = []
    for concept in expected_order:
        keywords = _tokenize(concept)
        if not keywords:
            positions.append(-1)
            continue
        found_pos = []
        for kw in keywords:
            idx = lower.find(kw)
            if idx >= 0:
                found_pos.append(idx)
        if found_pos:
            positions.append(sum(found_pos) / len(found_pos))
        else:
            positions.append(-1)

    valid = [(i, p) for i, p in enumerate(positions) if p >= 0]
    if len(valid) < 2:
        return 0.0

    in_order = 0
    total = len(valid) - 1
    for a in range(total):
        if valid[a][1] <= valid[a + 1][1]:
            in_order += 1
    return in_order / total


def _coherence_score(answer: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?।]+", answer) if s.strip()]
    if len(sentences) <= 1:
        return 0.3

    lengths = [len(s.split()) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    if mean_len == 0:
        return 0.2
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    cv = (variance ** 0.5) / mean_len
    uniformity = max(0.0, 1.0 - cv)

    transitions = [
        "therefore", "however", "moreover", "furthermore", "thus",
        "in addition", "specifically", "for example", "in contrast",
        "as a result", "this means", "which", "because", "while",
        "not only", "also", "finally", "first", "second",
    ]
    lower = answer.lower()
    marker_count = sum(1 for t in transitions if t in lower)
    marker_score = min(1.0, marker_count / 3)

    return 0.5 * uniformity + 0.5 * marker_score


def _structural_composite(answer: str, state: EnvState) -> float:
    meta = state.structural_meta
    ts = _terminology_score(answer, meta.required_terms, meta.banned_terms)
    comp = _completeness_score(answer, meta.required_sections)
    ords = _ordering_score(answer, meta.expected_order)
    coh = _coherence_score(answer)

    score = 0.30 * max(ts, 0.0) + 0.25 * comp + 0.25 * ords + 0.20 * coh
    if ts < 0:
        score += 0.15 * ts
    return max(0.0, min(1.0, score))


# Per-step reward

def step_reward(
    state: EnvState, action_type: ActionType, payload: str | None,
) -> Tuple[float, str]:

    # RETRIEVE
    if action_type == ActionType.RETRIEVE:
        if state.retrieval_count >= 3:
            return -0.15, "Redundant retrieval — already retrieved 3 times."
        elif state.available_passages:
            return 0.05, "Passages retrieved."
        else:
            return -0.05, "No passages available for retrieval."

    # CITE
    if action_type == ActionType.CITE:
        if not payload:
            return -0.05, "Empty citation."
        cr = _citation_recall([payload], state.ground_truth_citations)
        if cr > 0:
            return 0.15, "Correct citation added."
        return -0.05, "Citation does not match expected sources."

    # ACCEPT / REJECT — handled by terminal_reward
    if action_type in (ActionType.ACCEPT, ActionType.REJECT):
        return 0.0, ""

    # EDIT / RESTRUCTURE
    if not payload:
        return -0.10, f"Empty {action_type.value.lower()} — no content provided."

    old_answer = state.current_answer

    old_f1 = _token_f1(old_answer, state.ground_truth_answer)
    new_f1 = _token_f1(payload, state.ground_truth_answer)
    fact_delta = new_f1 - old_f1

    old_struct = _structural_composite(old_answer, state)
    new_struct = _structural_composite(payload, state)
    struct_delta = new_struct - old_struct

    label = action_type.value

    if action_type == ActionType.EDIT:
        combined = 0.6 * fact_delta + 0.4 * struct_delta
        if combined > 0.03:
            return 0.20 + combined, (
                f"{label} improved (fact Δ{fact_delta:+.2f}, struct Δ{struct_delta:+.2f})."
            )
        elif combined < -0.03:
            return -0.20, (
                f"{label} degraded (fact Δ{fact_delta:+.2f}, struct Δ{struct_delta:+.2f})."
            )
        else:
            return -0.05, f"{label} had negligible effect."

    else:  # RESTRUCTURE
        if fact_delta < -0.10:
            return -0.25, f"Restructure lost factual content (fact Δ{fact_delta:+.2f})."
        elif struct_delta > 0.05:
            return 0.25 + struct_delta, f"Restructure improved structure (Δ{struct_delta:+.2f})."
        elif struct_delta < -0.03:
            return -0.15, f"Restructure degraded structure (Δ{struct_delta:+.2f})."
        else:
            return -0.05, f"Restructure had negligible effect."


# Terminal rewards

def terminal_reward(
    state: EnvState, action_type: ActionType,
) -> Tuple[float, str]:

    # REJECT
    if action_type == ActionType.REJECT:
        if not state.answer_is_correct:
            return 0.30, "Correctly rejected a flawed answer."
        return -0.50, "Incorrectly rejected a valid answer."

    # ACCEPT
    fs = _token_f1(state.current_answer, state.ground_truth_answer)
    cs = _citation_recall(state.current_citations, state.ground_truth_citations)
    struct = _structural_composite(state.current_answer, state)
    efficiency = 0.20 * (state.steps_remaining / state.max_steps)

    terminal = 0.90 * fs + 0.30 * cs + 0.70 * struct + efficiency

    if fs < 0.3 and struct < 0.3:
        terminal -= 0.50
        quality = "poor"
    elif fs < 0.5:
        quality = "mediocre"
    else:
        quality = "good"

    # detailed breakdown for feedback
    meta = state.structural_meta
    ts = _terminology_score(state.current_answer, meta.required_terms, meta.banned_terms)
    comp = _completeness_score(state.current_answer, meta.required_sections)
    ords = _ordering_score(state.current_answer, meta.expected_order)
    coh = _coherence_score(state.current_answer)

    feedback = (
        f"Accepted a {quality} answer "
        f"(fact={fs:.2f}, cite={cs:.2f}, struct={struct:.2f} "
        f"[term={ts:.2f} comp={comp:.2f} ord={ords:.2f} coh={coh:.2f}])."
    )
    return terminal, feedback


# Score Normalization

MAX_REASONABLE_REWARD = 2.80


def normalize_score(cumulative_reward: float) -> float:
    return max(0.0, min(1.0, cumulative_reward / MAX_REASONABLE_REWARD))

