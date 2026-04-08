"""
Reward computation for IndicScriptureQA.

Two evaluation axes, weighted into a single scalar:
  A. Factual quality  — token-F1 similarity to ground truth, citation recall
  B. Structural quality — coherence, completeness, terminology, ordering

All scoring is zero-dependency (no ML models) so the env runs on 2 vCPU / 8 GB.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from models import ActionType, EnvState, StructuralMeta


# ═══════════════════════════════════════════════════════════════════════════════
# A. FACTUAL SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    """Lowercase split on non-alphanumeric (keeps Devanagari chars)."""
    return [t for t in re.split(r"[^a-zA-Z0-9\u0900-\u097F]+", text.lower()) if t]


def token_f1(candidate: str, reference: str) -> float:
    """Token-level F1 between candidate and reference. Returns 0–1."""
    cand_toks = _tokenize(candidate)
    ref_toks = _tokenize(reference)
    if not cand_toks or not ref_toks:
        return 0.0
    cand_set = set(cand_toks)
    ref_set = set(ref_toks)
    common = cand_set & ref_set
    if not common:
        return 0.0
    precision = len(common) / len(cand_set)
    recall = len(common) / len(ref_set)
    return 2 * precision * recall / (precision + recall)


def _normalize_citation(c: str) -> str:
    return re.sub(r"\s+", " ", c.strip().lower())


def citation_recall(predicted: List[str], ground_truth: List[str]) -> float:
    """Fraction of ground-truth citations matched (fuzzy substring)."""
    if not ground_truth:
        return 1.0
    gt_norms = [_normalize_citation(g) for g in ground_truth]
    pred_norms = [_normalize_citation(p) for p in predicted]
    matched = 0
    for gt in gt_norms:
        for pred in pred_norms:
            if gt in pred or pred in gt:
                matched += 1
                break
    return matched / len(gt_norms)


# ═══════════════════════════════════════════════════════════════════════════════
# B. STRUCTURAL SCORING
# ═══════════════════════════════════════════════════════════════════════════════

# ── B1. Terminology precision ────────────────────────────────────────────────

def terminology_score(answer: str, meta: StructuralMeta) -> float:
    """
    Checks:
      +  required_terms present   → recall over required_terms
      -  banned_terms present     → hard penalty per banned term found
    Returns float in [-1.0, 1.0].
    """
    answer_lower = answer.lower()

    # required term recall
    if meta.required_terms:
        hits = sum(1 for t in meta.required_terms if t.lower() in answer_lower)
        term_recall = hits / len(meta.required_terms)
    else:
        term_recall = 1.0

    # banned term penalty
    ban_penalty = 0.0
    if meta.banned_terms:
        for bt in meta.banned_terms:
            if bt.lower() in answer_lower:
                ban_penalty += 0.25
        ban_penalty = min(ban_penalty, 1.0)

    return term_recall - ban_penalty


# ── B2. Completeness (section coverage) ──────────────────────────────────────

def completeness_score(answer: str, meta: StructuralMeta) -> float:
    """
    Heuristic: for each required_section, check whether characteristic
    keywords from that section label appear in the answer.
    Returns 0–1 (fraction of sections covered).
    """
    if not meta.required_sections:
        return 1.0
    answer_lower = answer.lower()
    covered = 0
    for section in meta.required_sections:
        # use the keywords from the section label itself
        section_keywords = _tokenize(section)
        # count a section as covered if ≥ half its keywords appear
        if section_keywords:
            hits = sum(1 for kw in section_keywords if kw in answer_lower)
            if hits / len(section_keywords) >= 0.5:
                covered += 1
    return covered / len(meta.required_sections)


# ── B3. Logical ordering (sequence adherence) ────────────────────────────────

def ordering_score(answer: str, meta: StructuralMeta) -> float:
    """
    Checks whether concepts in expected_order appear in the correct sequence
    in the answer. Uses first-occurrence position of each concept's keywords.
    Returns 0–1.
    """
    if len(meta.expected_order) < 2:
        return 1.0

    answer_lower = answer.lower()
    positions: List[int] = []

    for concept in meta.expected_order:
        keywords = _tokenize(concept)
        # find earliest position of any keyword
        earliest = len(answer_lower) + 1
        for kw in keywords:
            idx = answer_lower.find(kw)
            if idx != -1 and idx < earliest:
                earliest = idx
        positions.append(earliest)

    # count correctly ordered adjacent pairs
    correct_pairs = sum(
        1 for i in range(len(positions) - 1) if positions[i] <= positions[i + 1]
    )
    return correct_pairs / (len(positions) - 1)


# ── B4. Coherence (transition quality + sentence structure) ──────────────────

_TRANSITION_MARKERS = {
    "therefore", "however", "moreover", "furthermore", "thus", "consequently",
    "specifically", "in contrast", "for example", "similarly", "additionally",
    "because", "since", "although", "while", "first", "second", "third",
    "finally", "in particular", "notably", "according to", "this means",
    "as a result", "in other words",
}


def coherence_score(answer: str) -> float:
    """
    Lightweight coherence proxy:
      - Sentence count (more than 1 sentence expected)
      - Transition markers (discourse connectives)
      - Sentence-length variance (very uneven → lower coherence)
    Returns 0–1.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
    if len(sentences) <= 1:
        return 0.3  # single sentence is structurally weak for these tasks

    # transition marker density
    answer_lower = answer.lower()
    marker_count = sum(1 for m in _TRANSITION_MARKERS if m in answer_lower)
    marker_density = min(marker_count / max(len(sentences) - 1, 1), 1.0)

    # sentence length variance (normalised). Very uneven → incoherent.
    lengths = [len(s.split()) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    if mean_len == 0:
        return 0.2
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    cv = (variance ** 0.5) / mean_len  # coefficient of variation
    uniformity = max(0.0, 1.0 - cv)  # lower CV → more uniform → higher score

    # blend: 50 % markers, 30 % uniformity, 20 % baseline for multi-sentence
    return 0.5 * marker_density + 0.3 * uniformity + 0.2


# ── Composite structural score ───────────────────────────────────────────────

def structural_quality(answer: str, meta: StructuralMeta) -> Tuple[float, dict]:
    """
    Weighted composite of all structural axes.
    Returns (score_0_to_1, breakdown_dict).
    """
    ts = terminology_score(answer, meta)
    cs = completeness_score(answer, meta)
    os_ = ordering_score(answer, meta)
    coh = coherence_score(answer)

    # weights
    composite = (
        0.30 * max(ts, 0.0)   # terminology  (clamp negatives to 0 for composite)
      + 0.25 * cs              # completeness
      + 0.25 * os_             # ordering
      + 0.20 * coh             # coherence
    )

    # apply banned-term penalty on top
    if ts < 0:
        composite += 0.15 * ts  # propagate penalty

    composite = max(0.0, min(1.0, composite))

    breakdown = {
        "terminology": round(ts, 3),
        "completeness": round(cs, 3),
        "ordering": round(os_, 3),
        "coherence": round(coh, 3),
        "composite": round(composite, 3),
    }
    return composite, breakdown


# ═══════════════════════════════════════════════════════════════════════════════
# PER-STEP REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def step_reward(state: EnvState, action_type: ActionType, payload: str | None) -> Tuple[float, str]:
    """
    Compute per-step reward and feedback message.
    Now accounts for structural improvement on EDIT and RESTRUCTURE.
    """
    reward = 0.0
    feedback = ""

    if action_type == ActionType.RETRIEVE:
        if state.retrieval_count >= 3:
            reward = -0.15
            feedback = "Redundant retrieval — you've already retrieved 3 times."
        elif state.available_passages:
            reward = 0.05
            feedback = "Passages retrieved."
        else:
            reward = -0.05
            feedback = "No passages available for retrieval."

    elif action_type == ActionType.EDIT:
        if not payload:
            reward = -0.10
            feedback = "Empty edit — no content provided."
        else:
            # factual delta
            old_sim = token_f1(state.current_answer, state.ground_truth_answer)
            new_sim = token_f1(payload, state.ground_truth_answer)
            fact_delta = new_sim - old_sim

            # structural delta
            old_struct, _ = structural_quality(state.current_answer, state.structural_meta)
            new_struct, bk = structural_quality(payload, state.structural_meta)
            struct_delta = new_struct - old_struct

            combined_delta = 0.6 * fact_delta + 0.4 * struct_delta

            if combined_delta > 0.03:
                reward = 0.20 + combined_delta
                feedback = f"Edit improved answer (fact Δ{fact_delta:+.2f}, struct Δ{struct_delta:+.2f})."
            elif combined_delta < -0.03:
                reward = -0.20
                feedback = f"Edit degraded answer (fact Δ{fact_delta:+.2f}, struct Δ{struct_delta:+.2f})."
            else:
                reward = -0.05
                feedback = "Edit had negligible effect."

    elif action_type == ActionType.RESTRUCTURE:
        if not payload:
            reward = -0.10
            feedback = "Empty restructure — no content provided."
        else:
            # restructure should preserve facts but improve structure
            old_sim = token_f1(state.current_answer, state.ground_truth_answer)
            new_sim = token_f1(payload, state.ground_truth_answer)
            fact_delta = new_sim - old_sim

            old_struct, _ = structural_quality(state.current_answer, state.structural_meta)
            new_struct, bk = structural_quality(payload, state.structural_meta)
            struct_delta = new_struct - old_struct

            if fact_delta < -0.10:
                # restructure destroyed factual content
                reward = -0.25
                feedback = f"Restructure lost factual content (fact Δ{fact_delta:+.2f}). Use EDIT if changing facts."
            elif struct_delta > 0.05:
                reward = 0.25 + struct_delta
                feedback = (
                    f"Restructure improved structure (Δ{struct_delta:+.2f}). "
                    f"Breakdown: term={bk['terminology']:.2f} comp={bk['completeness']:.2f} "
                    f"order={bk['ordering']:.2f} coh={bk['coherence']:.2f}"
                )
            elif struct_delta < -0.03:
                reward = -0.15
                feedback = f"Restructure degraded structure (Δ{struct_delta:+.2f})."
            else:
                reward = -0.05
                feedback = "Restructure had negligible structural effect."

    elif action_type == ActionType.CITE:
        if not payload:
            reward = -0.05
            feedback = "Empty citation."
        else:
            cr = citation_recall([payload], state.ground_truth_citations)
            if cr > 0:
                reward = 0.15
                feedback = "Correct citation added."
            else:
                reward = -0.05
                feedback = "Citation does not match expected sources."

    elif action_type in (ActionType.ACCEPT, ActionType.REJECT):
        pass  # terminal rewards handled separately

    return reward, feedback


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def terminal_reward(state: EnvState, action_type: ActionType) -> Tuple[float, str]:
    """
    Terminal reward blends factual quality AND structural quality.
    """
    if action_type == ActionType.REJECT:
        if not state.answer_is_correct:
            return 0.30, "Correctly rejected a flawed answer."
        else:
            return -0.50, "Incorrectly rejected a valid answer."

    # ── ACCEPT ────────────────────────────────────────────────────────────
    # factual component
    answer_sim = token_f1(state.current_answer, state.ground_truth_answer)
    cit_score = citation_recall(state.current_citations, state.ground_truth_citations)

    # structural component
    struct_score, struct_breakdown = structural_quality(
        state.current_answer, state.structural_meta
    )

    # efficiency bonus (0–0.2)
    efficiency = 0.20 * (state.steps_remaining / state.max_steps)

    # weighted terminal reward
    terminal = (
        0.90 * answer_sim        # factual similarity    (max 0.90)
      + 0.30 * cit_score         # citation recall       (max 0.30)
      + 0.70 * struct_score      # structural quality    (max 0.70)
      + efficiency               # efficiency bonus      (max 0.20)
    )
    # theoretical max ≈ 2.10

    # penalty for accepting a still-bad answer
    if answer_sim < 0.3 and struct_score < 0.3:
        terminal -= 0.50
        quality_label = "poor"
    elif answer_sim < 0.5:
        quality_label = "mediocre"
    else:
        quality_label = "good"

    feedback = (
        f"Accepted a {quality_label} answer "
        f"(fact={answer_sim:.2f}, cite={cit_score:.2f}, struct={struct_score:.2f} "
        f"[term={struct_breakdown['terminology']:.2f} "
        f"comp={struct_breakdown['completeness']:.2f} "
        f"ord={struct_breakdown['ordering']:.2f} "
        f"coh={struct_breakdown['coherence']:.2f}])"
    )

    return terminal, feedback


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════

# theoretical max: terminal ~2.10 + step bonuses ~0.5 ≈ 2.6
MAX_REASONABLE_REWARD = 2.80


def normalize_score(cumulative_reward: float) -> float:
    """Clamp cumulative reward into [0, 1]."""
    score = cumulative_reward / MAX_REASONABLE_REWARD
    return max(0.0, min(1.0, score))
