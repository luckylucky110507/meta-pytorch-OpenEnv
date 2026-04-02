from __future__ import annotations

import uuid
from typing import Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from emotion_study_openenv import EmotionStudyEnv


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    action_type: str


app = FastAPI(title="Emotion-Aware Study Companion OpenEnv")
_episode_id: Optional[str] = None
env = EmotionStudyEnv()


@app.get("/")
def healthcheck() -> Dict[str, object]:
    return {
        "name": "emotion-aware-study-companion",
        "status": "ok",
        "supports": ["reset", "step", "state"],
    }


@app.post("/reset")
def reset(payload: ResetRequest) -> Dict[str, object]:
    global _episode_id
    _episode_id = str(uuid.uuid4())
    observation = env.reset(task_id=payload.task_id)
    return {
        "episode_id": _episode_id,
        "observation": observation,
    }


@app.post("/step")
def step(payload: StepRequest) -> Dict[str, object]:
    result = env.step({"action_type": payload.action_type})
    result["episode_id"] = _episode_id
    return result


@app.get("/state")
def state() -> Dict[str, object]:
    data = env.state()
    data["episode_id"] = _episode_id
    return data


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
