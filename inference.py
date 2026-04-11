import json
import os
import textwrap
from typing import Dict, List, Optional

import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.environ.get("PING_URL", "http://localhost:8000")
BENCHMARK = "indic_scripture_qa"
TEMPERATURE = 0.4
MAX_TOKENS = 600

TASKS = [
    {"name": "verify-factual",    "max_steps": 5},
    {"name": "correct-and-cite",  "max_steps": 8},
    {"name": "fix-hallucination", "max_steps": 12},
]

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert agent that both CORRECTS hallucinations and IMPROVES the
semantic structure of answers about Indic scriptures (Vedas, Upanishads,
Ramayana, Mahabharata, Bhagavad Gita, Puranas).
Each turn you receive an observation with:
  - question, current_answer, retrieved_passages, current_citations,
    steps_remaining, feedback, structural_hints
You must reply with EXACTLY ONE JSON object (no markdown, no explanation):
{
  "action_type": "RETRIEVE" | "EDIT" | "RESTRUCTURE" | "CITE" | "ACCEPT" | "REJECT",
  "payload": "<string or null>"
}
Actions:
  RETRIEVE    — fetch source passages to verify facts
  EDIT        — rewrite the answer to fix factual errors AND improve content
  RESTRUCTURE — reorganise the answer's flow, ordering, and coherence WITHOUT
                changing facts (use when facts are right but structure is poor)
  CITE        — add a scripture citation (e.g. "Bhagavad Gita 2.47")
  ACCEPT      — finalise when answer is both accurate and well-structured
  REJECT      — only if the answer is fundamentally unsalvageable
Strategy:
  1. RETRIEVE first (1–2 times) to get authoritative source passages.
  2. Check facts against retrieved passages. EDIT to fix any errors.
  3. Read structural_hints. If the answer's flow, terminology, or completeness
     is poor, use RESTRUCTURE to reorganise it.
  4. CITE relevant scripture references.
  5. ACCEPT when the answer is factually accurate, well-structured, uses
     correct Sanskrit terminology, and covers all required aspects.
  6. Be efficient — fewer steps score higher.
Evaluation axes (the grader checks ALL of these):
  - Factual similarity to ground truth
  - Citation accuracy
  - Terminology precision (correct Sanskrit/domain terms, no misconception markers)
  - Completeness (all required conceptual aspects covered)
  - Logical ordering (concepts in proper sequence)
  - Coherence (smooth transitions, balanced sentence structure)
""")


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Env HTTP helpers ──────────────────────────────────────────────────────────

def env_reset(task_name: str, scenario_index: int = 0) -> Dict:
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_name": task_name, "scenario_index": scenario_index},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action_type: str, payload: Optional[str] = None) -> Dict:
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"action_type": action_type, "payload": payload},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── Agent ─────────────────────────────────────────────────────────────────────

def build_user_prompt(obs: Dict, step: int) -> str:
    return json.dumps({
        "step": step,
        "question": obs["question"],
        "current_answer": obs["current_answer"],
        "retrieved_passages": obs["retrieved_passages"],
        "current_citations": obs["current_citations"],
        "steps_remaining": obs["steps_remaining"],
        "feedback": obs.get("feedback"),
        "structural_hints": obs.get("structural_hints", []),
    }, indent=2)


def get_agent_action(client: OpenAI, obs: Dict, step: int) -> Dict:
    """Ask the LLM for the next action."""
    user_prompt = build_user_prompt(obs, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except Exception as exc:
        print(f"[DEBUG] LLM parse error: {exc}", flush=True)
        if step <= 2:
            return {"action_type": "RETRIEVE", "payload": None}
        return {"action_type": "ACCEPT", "payload": None}


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_name: str, max_steps: int, scenario_index: int = 0) -> float:
    """Run one episode. Returns score in [0, 1]."""
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = env_reset(task_name, scenario_index)
        obs = result["observation"]

        for step in range(1, max_steps + 1):
            if result.get("done", False):
                break

            agent_action = get_agent_action(client, obs, step)
            action_type = agent_action.get("action_type", "ACCEPT")
            payload = agent_action.get("payload")

            result = env_step(action_type, payload)
            obs = result["observation"]
            reward = result.get("reward", 0.0)
            done = result.get("done", False)

            rewards.append(reward)
            steps_taken = step

            action_str = f"{action_type}({payload!r})" if payload else action_type
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                score = result.get("info", {}).get("score", 0.0)
                break

        success = score >= 0.10

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores: Dict[str, float] = {}

    for task in TASKS:
        task_name = task["name"]
        max_steps = task["max_steps"]
        score = run_task(client, task_name, max_steps, scenario_index=0)
        all_scores[task_name] = score
        print(flush=True)

    print("=" * 60, flush=True)
    print("BASELINE RESULTS", flush=True)
    for name, sc in all_scores.items():
        print(f"  {name:25s}  score={sc:.3f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    print(f"  {'AVERAGE':25s}  score={avg:.3f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
