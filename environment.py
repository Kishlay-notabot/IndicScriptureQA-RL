from __future__ import annotations

import random
from typing import Optional

from models import Action, ActionType, EnvState, Observation, StepResult, StructuralMeta
from rewards import normalize_score, step_reward, terminal_reward
from tasks import TASKS, Scenario, TaskConfig


class IndicScriptureQAEnv:

    def __init__(self) -> None:
        self._state: Optional[EnvState] = None

    def reset(
        self,
        task_name: str = "verify-factual",
        scenario_index: Optional[int] = None,
    ) -> StepResult:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task {task_name!r}. Choose from {list(TASKS)}")

        cfg: TaskConfig = TASKS[task_name]
        if scenario_index is not None:
            idx = scenario_index % len(cfg.scenarios)
        else:
            idx = random.randint(0, len(cfg.scenarios) - 1)

        sc: Scenario = cfg.scenarios[idx]

        self._state = EnvState(
            question=sc.question,
            current_answer=sc.given_answer,
            original_answer=sc.given_answer,
            ground_truth_answer=sc.ground_truth_answer,
            ground_truth_citations=list(sc.ground_truth_citations),
            available_passages=list(sc.available_passages),
            answer_is_correct=sc.answer_is_correct,
            factual_is_correct=sc.factual_is_correct,
            structural_meta=sc.structural_meta,
            structural_hints=list(sc.structural_hints),
            task_name=task_name,
            max_steps=cfg.max_steps,
            steps_remaining=cfg.max_steps,
            step_count=0,
            done=False,
            cumulative_reward=0.0,
            rewards=[],
            retrieval_count=0,
            edit_count=0,
            restructure_count=0,
            feedback="Episode started. Examine the answer for factual accuracy AND semantic structure.",
        )
        return StepResult(observation=self._state.to_observation(), reward=0.0, done=False)


    def step(self, action: Action) -> StepResult:
        s = self._state
        if s is None:
            raise RuntimeError("Call reset() before step().")
        if s.done:
            raise RuntimeError("Episode already finished. Call reset().")

        s.step_count += 1
        s.steps_remaining -= 1
        act = action.action_type
        payload = (action.payload or "").strip()

        reward = 0.0
        feedback = ""
        done = False

        # action dispatch
        if act == ActionType.RETRIEVE:
            s.retrieval_count += 1
            if s.available_passages:
                idx = (s.retrieval_count - 1) % len(s.available_passages)
                passage = s.available_passages[idx]
                if passage not in s.retrieved_passages:
                    s.retrieved_passages.append(passage)
            reward, feedback = step_reward(s, act, payload)

        elif act == ActionType.EDIT:
            s.edit_count += 1
            reward, feedback = step_reward(s, act, payload)
            if payload:
                s.current_answer = payload

        elif act == ActionType.RESTRUCTURE:
            s.restructure_count += 1
            reward, feedback = step_reward(s, act, payload)
            if payload:
                s.current_answer = payload

        elif act == ActionType.CITE:
            if payload and payload not in s.current_citations:
                s.current_citations.append(payload)
            reward, feedback = step_reward(s, act, payload)

        elif act == ActionType.ACCEPT:
            t_reward, feedback = terminal_reward(s, act)
            reward = t_reward
            done = True

        elif act == ActionType.REJECT:
            t_reward, feedback = terminal_reward(s, act)
            reward = t_reward
            done = True

        else:
            reward = -0.10
            feedback = f"Unknown action type: {act}"

        # check steps
        if not done and s.steps_remaining <= 0:
            t_reward, t_fb = terminal_reward(s, ActionType.ACCEPT)
            reward += t_reward - 0.20
            feedback += f" | Forced termination (step limit). {t_fb}"
            done = True

        # book-keep the logs 
        s.rewards.append(reward)
        s.cumulative_reward += reward
        s.done = done
        s.feedback = feedback

        info = {}
        if done:
            info["score"] = normalize_score(s.cumulative_reward)
            info["cumulative_reward"] = s.cumulative_reward

        return StepResult(
            observation=s.to_observation(),
            reward=reward,
            done=done,
            info=info,
        )

    # update state
    def state(self) -> EnvState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state.model_copy(deep=True)
