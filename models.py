"""Typed Pydantic models for the IndicScriptureQA environment."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Action Space ──────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    RETRIEVE     = "RETRIEVE"      # Retrieve source passages (payload = optional query)
    EDIT         = "EDIT"          # Replace current answer   (payload = new answer text)
    RESTRUCTURE  = "RESTRUCTURE"   # Reorganise answer flow   (payload = restructured text)
    CITE         = "CITE"          # Attach a citation        (payload = e.g. "Bhagavad Gita 2.47")
    ACCEPT       = "ACCEPT"        # Accept answer as final    (terminal)
    REJECT       = "REJECT"        # Reject answer entirely    (terminal)


class Action(BaseModel):
    action_type: ActionType
    payload: Optional[str] = None


# ── Structural metadata (hidden from agent, used by grader) ──────────────────

class StructuralMeta(BaseModel):
    """Describes the *expected* semantic structure of a correct answer."""
    required_terms: List[str] = Field(
        default_factory=list,
        description="Sanskrit / domain terms the answer must contain.",
    )
    required_sections: List[str] = Field(
        default_factory=list,
        description="Conceptual aspects the answer should cover (order-independent).",
    )
    expected_order: List[str] = Field(
        default_factory=list,
        description="Concepts that should appear in this logical sequence.",
    )
    banned_terms: List[str] = Field(
        default_factory=list,
        description="Terms that indicate a common misconception if present.",
    )


# ── Observation Space ─────────────────────────────────────────────────────────

class Observation(BaseModel):
    question: str
    current_answer: str
    retrieved_passages: List[str] = Field(default_factory=list)
    current_citations: List[str] = Field(default_factory=list)
    steps_remaining: int
    task_name: str
    feedback: Optional[str] = None
    # structural hints exposed to the agent (non-spoiler)
    structural_hints: List[str] = Field(
        default_factory=list,
        description="High-level hints about expected answer structure.",
    )


# ── Step Result ───────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: dict = Field(default_factory=dict)


# ── Internal State (superset of observation + grading internals) ──────────────

class EnvState(BaseModel):
    # observable
    question: str
    current_answer: str
    retrieved_passages: List[str] = Field(default_factory=list)
    current_citations: List[str] = Field(default_factory=list)
    steps_remaining: int = 0
    task_name: str = ""
    feedback: Optional[str] = None
    structural_hints: List[str] = Field(default_factory=list)

    # hidden / grading — factual
    original_answer: str = ""
    ground_truth_answer: str = ""
    ground_truth_citations: List[str] = Field(default_factory=list)
    available_passages: List[str] = Field(default_factory=list)
    answer_is_correct: bool = False        # overall: facts AND structure both OK
    factual_is_correct: bool = False       # facts alone are OK (structure may be bad)

    # hidden / grading — structural
    structural_meta: StructuralMeta = Field(default_factory=StructuralMeta)

    # episode bookkeeping
    step_count: int = 0
    max_steps: int = 8
    done: bool = False
    cumulative_reward: float = 0.0
    rewards: List[float] = Field(default_factory=list)
    retrieval_count: int = 0
    edit_count: int = 0
    restructure_count: int = 0

    def to_observation(self) -> Observation:
        return Observation(
            question=self.question,
            current_answer=self.current_answer,
            retrieved_passages=list(self.retrieved_passages),
            current_citations=list(self.current_citations),
            steps_remaining=self.steps_remaining,
            task_name=self.task_name,
            feedback=self.feedback,
            structural_hints=list(self.structural_hints),
        )
