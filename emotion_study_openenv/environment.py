from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

from .models import Action, EpisodeMetrics, StudentState, TaskSpec, TransitionResult
from .tasks import TASKS


class EmotionStudyEnv:
    def __init__(self) -> None:
        self.tasks: List[TaskSpec] = TASKS
        self._task_index = 0
        self.current_task: Optional[TaskSpec] = None
        self.student_state: Optional[StudentState] = None
        self.student_message: str = ""
        self.steps_remaining: int = 0
        self.metrics = EpisodeMetrics()
        self.history: List[Dict[str, object]] = []

    def reset(self, task_id: Optional[str] = None) -> Dict[str, object]:
        self.current_task = self._resolve_task(task_id)
        self.student_state = deepcopy(self.current_task.initial_state)
        self.student_message = self.current_task.initial_message
        self.steps_remaining = self.current_task.max_steps
        self.metrics = EpisodeMetrics()
        self.history = []
        return self._observation()

    def step(self, action: Dict[str, object]) -> Dict[str, object]:
        if self.current_task is None or self.student_state is None:
            raise RuntimeError("Environment must be reset before calling step().")

        parsed_action = Action.model_validate(action)
        previous_state = self.student_state.model_copy(deep=True)
        self._apply_action(parsed_action.action_type)
        self.steps_remaining -= 1
        self.metrics.steps_taken += 1
        self.metrics.last_action = parsed_action.action_type

        reward = self._compute_reward(previous_state, self.student_state, parsed_action.action_type)
        self.metrics.total_reward += reward
        done = self.steps_remaining <= 0 or self._is_success()
        self.metrics.success = self._is_success()
        self.metrics.score = min(1.0, round(self.metrics.total_reward / max(self.metrics.steps_taken, 1), 4))

        result = TransitionResult(
            observation=self._observation(),
            reward=round(reward, 4),
            done=done,
            info={
                "success": self.metrics.success,
                "steps_remaining": self.steps_remaining,
            },
        )
        return result.model_dump()

    def state(self) -> Dict[str, object]:
        if self.current_task is None or self.student_state is None:
            raise RuntimeError("Environment has no active state. Call reset() first.")

        return {
            "task": self.current_task.model_dump(),
            "student_state": self.student_state.model_dump(),
            "student_message": self.student_message,
            "steps_remaining": self.steps_remaining,
            "history": self.history,
            "episode_metrics": self.metrics.model_dump(),
        }

    def _resolve_task(self, task_id: Optional[str]) -> TaskSpec:
        if task_id:
            for task in self.tasks:
                if task.id == task_id:
                    return task
            raise ValueError(f"Unknown task_id: {task_id}")

        task = self.tasks[self._task_index % len(self.tasks)]
        self._task_index += 1
        return task

    def _observation(self) -> Dict[str, object]:
        if self.current_task is None or self.student_state is None:
            raise RuntimeError("Environment has no active task.")

        return {
            "task": {
                "id": self.current_task.id,
                "title": self.current_task.title,
                "difficulty": self.current_task.difficulty,
                "lesson_topic": self.current_task.lesson_topic,
                "student_persona": self.current_task.student_persona,
            },
            "student_message": self.student_message,
            "student_state": self.student_state.model_dump(),
            "steps_remaining": self.steps_remaining,
            "available_actions": Action.model_json_schema()["properties"]["action_type"]["enum"],
            "history_tail": self.history[-3:],
        }

    def _apply_action(self, action_type: str) -> None:
        assert self.current_task is not None
        assert self.student_state is not None

        state = self.student_state
        delta = {
            "focus": 0,
            "stress": 0,
            "comprehension": 0,
            "trust": 0,
            "energy": 0,
            "confidence": 0,
        }

        if action_type == "acknowledge_emotion":
            delta.update({"stress": -2, "trust": 2, "focus": 1})
            self.student_message = "That helped. I feel a little less pressured now."
        elif action_type == "adjust_difficulty":
            if state.confidence >= 8 and state.comprehension <= 5:
                delta.update({"comprehension": 2, "trust": 1, "confidence": -2})
                self.student_message = "Okay, that example showed me I was missing something important."
            else:
                delta.update({"comprehension": 2, "focus": 1})
                self.student_message = "This version makes more sense. I can follow it now."
        elif action_type == "play_calming_music":
            delta.update({"stress": -1, "focus": 1, "energy": 1})
            self.student_message = "That made the study session feel calmer."
        elif action_type == "suggest_break":
            delta.update({"stress": -1, "energy": 2, "focus": 1})
            self.student_message = "A short break helped me reset a bit."
        elif action_type == "give_micro_quiz":
            if state.comprehension >= 5:
                delta.update({"comprehension": 1, "focus": 1, "confidence": -1})
                self.student_message = "That quick check helped me test what I really know."
            else:
                delta.update({"stress": 1, "trust": -1, "focus": -1})
                self.student_message = "That quiz felt too soon and made me more unsure."
        elif action_type == "set_study_goal":
            delta.update({"focus": 2, "trust": 1})
            self.student_message = "A smaller goal sounds manageable. I can try that."
        elif action_type == "offer_encouragement":
            delta.update({"trust": 1, "focus": 1, "stress": -1})
            self.student_message = "Thanks, that makes me feel more willing to keep going."

        if action_type in self.current_task.preferred_actions:
            delta["trust"] += 1
        if action_type in self.current_task.discouraged_actions:
            delta["trust"] -= 1
            delta["stress"] += 1

        self._apply_delta(delta)
        self.history.append(
            {
                "action_type": action_type,
                "student_message": self.student_message,
                "student_state": self.student_state.model_dump(),
            }
        )

    def _apply_delta(self, delta: Dict[str, int]) -> None:
        assert self.student_state is not None
        for field_name, change in delta.items():
            current_value = getattr(self.student_state, field_name)
            updated_value = max(0, min(10, current_value + change))
            setattr(self.student_state, field_name, updated_value)

    def _compute_reward(self, previous: StudentState, current: StudentState, action_type: str) -> float:
        assert self.current_task is not None

        focus_gain = max(0, current.focus - previous.focus) * 0.12
        stress_reduction = max(0, previous.stress - current.stress) * 0.12
        comprehension_gain = max(0, current.comprehension - previous.comprehension) * 0.16
        trust_gain = max(0, current.trust - previous.trust) * 0.08
        energy_gain = max(0, current.energy - previous.energy) * 0.05
        preferred_bonus = 0.08 if action_type in self.current_task.preferred_actions else 0.0
        discouraged_penalty = 0.1 if action_type in self.current_task.discouraged_actions else 0.0
        success_bonus = 0.15 if self._is_success() else 0.0

        reward = (
            focus_gain
            + stress_reduction
            + comprehension_gain
            + trust_gain
            + energy_gain
            + preferred_bonus
            + success_bonus
            - discouraged_penalty
        )
        return max(0.0, min(1.0, reward))

    def _is_success(self) -> bool:
        assert self.current_task is not None
        assert self.student_state is not None

        criteria = self.current_task.success_criteria
        state = self.student_state
        return (
            state.focus >= criteria.min_focus
            and state.stress <= criteria.max_stress
            and state.comprehension >= criteria.min_comprehension
            and state.trust >= criteria.min_trust
        )
