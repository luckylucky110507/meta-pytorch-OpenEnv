from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal[
    "acknowledge_emotion",
    "adjust_difficulty",
    "play_calming_music",
    "suggest_break",
    "give_micro_quiz",
    "set_study_goal",
    "offer_encouragement",
]


class StudentState(BaseModel):
    focus: int = Field(ge=0, le=10)
    stress: int = Field(ge=0, le=10)
    comprehension: int = Field(ge=0, le=10)
    trust: int = Field(ge=0, le=10)
    energy: int = Field(ge=0, le=10)
    confidence: int = Field(ge=0, le=10)


class SuccessCriteria(BaseModel):
    min_focus: int = Field(ge=0, le=10)
    max_stress: int = Field(ge=0, le=10)
    min_comprehension: int = Field(ge=0, le=10)
    min_trust: int = Field(ge=0, le=10)


class TaskSpec(BaseModel):
    id: str
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    lesson_topic: str
    student_persona: str
    initial_message: str
    initial_state: StudentState
    success_criteria: SuccessCriteria
    max_steps: int = Field(ge=3, le=12)
    preferred_actions: List[ActionType]
    discouraged_actions: List[ActionType]


class Action(BaseModel):
    action_type: ActionType


class TransitionResult(BaseModel):
    observation: Dict[str, object]
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, object]


class EpisodeMetrics(BaseModel):
    total_reward: float = 0.0
    score: float = 0.0
    success: bool = False
    steps_taken: int = 0
    last_action: Optional[ActionType] = None
