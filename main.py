"""
we've made the following endpoints: 
Endpoints:
  POST /reset   — start a new episode
  POST /step    — take an action
  GET  /state   — get current environment state
  GET  /health  — liveness check
  GET  /tasks   — list available tasks
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import IndicScriptureQAEnv
from models import Action, ActionType
from tasks import TASKS

app = FastAPI(
    title="IndicScriptureQA",
    description=(
        "OpenEnv environment for evaluating agents on Indic scripture "
        "hallucination correction AND semantic structure quality."
    ),
    version="1.1.0",
)

_env = IndicScriptureQAEnv()



# ── Request / Response schemas 

class ResetRequest(BaseModel):
    task_name: str = "verify-factual"
    scenario_index: Optional[int] = None


class StepRequest(BaseModel):
    action_type: str     # RETRIEVE | EDIT | RESTRUCTURE | CITE | ACCEPT | REJECT
    payload: Optional[str] = None


class TaskInfo(BaseModel):
    name: str
    description: str
    max_steps: int
    num_scenarios: int



# ── Endpoints

@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    try:
        result = _env.reset(
            task_name=body.task_name,
            scenario_index=body.scenario_index,
        )
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(body: StepRequest):
    try:
        action_type = ActionType(body.action_type.upper())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_type {body.action_type!r}. Must be one of: {[a.value for a in ActionType]}",
        )
    try:
        action = Action(action_type=action_type, payload=body.payload)
        result = _env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    try:
        s = _env.state()
        return s.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    return [
        TaskInfo(
            name=cfg.name,
            description=cfg.description,
            max_steps=cfg.max_steps,
            num_scenarios=len(cfg.scenarios),
        ).model_dump()
        for cfg in TASKS.values()
    ]


# root - this get reflected in the HF Space (since we don't have a gradio interface)

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "IndicScriptureQA OpenEnv",
        "endpoints": ["/reset", "/step", "/state", "/health", "/tasks"]
    }
