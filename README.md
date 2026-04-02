# Emotion-Aware Study Companion OpenEnv

This project is a submission-ready OpenEnv-style environment for the hackathon brief shown in your screenshots. It models a realistic study-support workflow where an AI agent must adapt to a student's emotional and cognitive state while helping them make progress on a lesson.

## Why This Fits The Brief

- Real-world task: study support and academic coaching
- Standard environment API: `reset()`, `step()`, and `state()`
- Typed task and state models
- Three graded tasks with increasing difficulty
- Partial reward shaping for meaningful progress
- Baseline `inference.py` using the `OpenAI` client for LLM decisions
- `openenv.yaml` included
- Dockerfile included for Hugging Face Spaces deployment

## Project Structure

```text
emotion-study-openenv/
|-- Dockerfile
|-- README.md
|-- inference.py
|-- openenv.yaml
|-- requirements.txt
|-- validate_submission.py
|-- server/
|   `-- app.py
`-- emotion_study_openenv/
    |-- __init__.py
    |-- environment.py
    |-- models.py
    `-- tasks.py
```

This keeps only the important submission pieces:

- `emotion_study_openenv/` for the actual environment logic
- `server/app.py` for deployment
- `inference.py` for the baseline agent
- `openenv.yaml`, `Dockerfile`, and `requirements.txt` for packaging
- `validate_submission.py` for a quick local smoke test

## Environment Overview

The agent receives an observation describing:

- The active lesson topic
- The student's latest message
- Current focus, stress, comprehension, trust, and energy levels
- The intervention actions available on that turn

The agent chooses one action per step:

- `acknowledge_emotion`
- `adjust_difficulty`
- `play_calming_music`
- `suggest_break`
- `give_micro_quiz`
- `set_study_goal`
- `offer_encouragement`

The student responds deterministically based on their current state and the chosen intervention. This makes scoring reproducible for evaluation.

## Tasks

1. `calm_and_recover`
   Difficulty: easy
   Goal: stabilize a stressed student and restore focus.

2. `rebuild_focus`
   Difficulty: medium
   Goal: reduce distraction and increase engagement before checking understanding.

3. `challenge_with_confidence_control`
   Difficulty: hard
   Goal: handle an overconfident student by calibrating difficulty and improving real comprehension.

## Reward Design

Each step returns a shaped reward in `[0.0, 1.0]` based on:

- Focus improvement
- Stress reduction
- Comprehension improvement
- Trust improvement
- Energy preservation

Episodes also include a final success bonus when the student reaches the task-specific target zone. This creates meaningful partial progress signals instead of only terminal pass/fail outcomes.

## Local Setup

```bash
pip install -r requirements.txt
```

Set these environment variables for the baseline LLM policy:

```bash
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=your_key_here
HF_TOKEN=optional_for_space_deploy
```

Notes:

- `API_BASE_URL` maps to the hackathon's requested LLM endpoint variable.
- `OPENAI_API_KEY` is used by the OpenAI client.
- If `OPENAI_API_KEY` is not set, `HF_TOKEN` is used as a fallback API key.
- `HF_TOKEN` is not required for local execution, but is kept for deployment alignment.

## Run The Baseline

```bash
python inference.py
```

The script prints structured logs in `[START]`, `[STEP]`, and `[END]` format and runs all three tasks with reproducible scores.

## Quick Validation

```bash
python validate_submission.py
```

This checks for the required submission files and runs a deterministic smoke test against all three tasks without needing an LLM call.

## Python API Example

```python
from emotion_study_openenv import EmotionStudyEnv

env = EmotionStudyEnv()
obs = env.reset(task_id="calm_and_recover")
done = False

while not done:
    action = {"action_type": "acknowledge_emotion"}
    result = env.step(action)
    done = result["done"]

print(env.state())
```

## Hugging Face Spaces

The included Dockerfile now starts the FastAPI environment server expected by a Space deployment:

```bash
docker build -t emotion-study-openenv .
docker run --rm -p 8000:8000 emotion-study-openenv
```

The baseline script remains local:

```bash
python inference.py
```

### Create The Space

1. Go to `https://huggingface.co/spaces`
2. Click `Create new Space`
3. Set:
   - Owner: your Hugging Face username
   - Space name: `emotion-study-openenv`
   - SDK: `Docker`
   - Visibility: Public or Private

After creation, your URLs will be:

- Repo URL: `https://huggingface.co/spaces/<your-username>/emotion-study-openenv`
- Space URL: `https://<your-username>-emotion-study-openenv.hf.space`

Example:

- Repo URL: `https://huggingface.co/spaces/lucky/emotion-study-openenv`
- Space URL: `https://lucky-emotion-study-openenv.hf.space`

### Upload The Project

Upload the full contents of this folder:

- `Dockerfile`
- `openenv.yaml`
- `requirements.txt`
- `server/`
- `emotion_study_openenv/`
- `inference.py`
- `README.md`
- `validate_submission.py`

You can upload through the Hugging Face web UI or by pushing the folder as a git repository.

### Environment Variables

In your Space settings, add:

- `API_BASE_URL`
- `MODEL_NAME`
- `OPENAI_API_KEY`
- `HF_TOKEN` if your deployment flow needs it

Suggested values:

```text
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=your_key_here
```

### Expected Live Endpoints

Once the Space is running, these should be available:

- `GET /`
- `POST /reset`
- `POST /step`
- `GET /state`

For example:

- `https://<your-username>-emotion-study-openenv.hf.space/`
- `https://<your-username>-emotion-study-openenv.hf.space/state`

## Design Assumptions

- Webcam emotion detection from the screenshot idea is represented as a realistic simulated student-state signal instead of requiring actual camera input.
- The environment is intentionally deterministic so judges can reproduce scores.
- The LLM is used for action selection, while the world dynamics and grading remain transparent and testable.
- The exact official validator payload shape was not available in the screenshots, so the project is aligned to the public OpenEnv scaffold and FastAPI Space manifest format.
