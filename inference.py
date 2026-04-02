import json
import os
from typing import Any, Dict

from openai import OpenAI

from emotion_study_openenv import EmotionStudyEnv


ACTION_TYPES = [
    "acknowledge_emotion",
    "adjust_difficulty",
    "play_calming_music",
    "suggest_break",
    "give_micro_quiz",
    "set_study_goal",
    "offer_encouragement",
]


def log_event(tag: str, payload: Dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, ensure_ascii=True)}")


def build_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    api_base = os.getenv("API_BASE_URL")
    if not api_key or not api_base:
        return None
    return OpenAI(api_key=api_key, base_url=api_base)


def heuristic_action(observation: Dict[str, Any]) -> str:
    state = observation["student_state"]
    if state["stress"] >= 7:
        return "acknowledge_emotion"
    if state["energy"] <= 3:
        return "suggest_break"
    if state["focus"] <= 4:
        return "set_study_goal"
    if state["comprehension"] <= 5:
        return "adjust_difficulty"
    return "give_micro_quiz"


def llm_action(client: OpenAI, model_name: str, observation: Dict[str, Any]) -> str:
    prompt = {
        "role": "system",
        "content": (
            "You are selecting one study-coaching intervention for a deterministic "
            "student support environment. Return only a JSON object with a single "
            'key named "action_type". Choose one valid action that best improves '
            "long-term focus, comprehension, and emotional stability."
        ),
    }
    user_message = {
        "role": "user",
        "content": json.dumps(
            {
                "observation": observation,
                "valid_action_types": ACTION_TYPES,
            },
            ensure_ascii=True,
        ),
    }
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[prompt, user_message],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    action_type = parsed.get("action_type", "")
    if action_type not in ACTION_TYPES:
        return heuristic_action(observation)
    return action_type


def choose_action(client: OpenAI | None, model_name: str, observation: Dict[str, Any]) -> str:
    if client is None:
        return heuristic_action(observation)
    try:
        return llm_action(client, model_name, observation)
    except Exception:
        return heuristic_action(observation)


def run_task(env: EmotionStudyEnv, client: OpenAI | None, model_name: str, task_id: str) -> Dict[str, Any]:
    observation = env.reset(task_id=task_id)
    log_event(
        "START",
        {
            "task_id": task_id,
            "difficulty": observation["task"]["difficulty"],
            "lesson": observation["task"]["lesson_topic"],
        },
    )

    done = False
    step_index = 0
    total_reward = 0.0

    while not done:
        action_type = choose_action(client, model_name, observation)
        result = env.step({"action_type": action_type})
        observation = result["observation"]
        total_reward += result["reward"]
        step_index += 1
        log_event(
            "STEP",
            {
                "task_id": task_id,
                "step": step_index,
                "action_type": action_type,
                "reward": round(result["reward"], 4),
                "done": result["done"],
                "student_message": observation["student_message"],
                "student_state": observation["student_state"],
            },
        )
        done = result["done"]

    final_state = env.state()
    score = round(final_state["episode_metrics"]["score"], 4)
    log_event(
        "END",
        {
            "task_id": task_id,
            "steps": step_index,
            "total_reward": round(total_reward, 4),
            "score": score,
            "success": final_state["episode_metrics"]["success"],
        },
    )
    return {
        "task_id": task_id,
        "score": score,
        "steps": step_index,
        "success": final_state["episode_metrics"]["success"],
    }


def main() -> None:
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    env = EmotionStudyEnv()
    client = build_client()
    task_ids = [task.id for task in env.tasks]
    summary = [run_task(env, client, model_name, task_id) for task_id in task_ids]
    average_score = round(sum(item["score"] for item in summary) / len(summary), 4)
    log_event("END", {"summary": summary, "average_score": average_score})


if __name__ == "__main__":
    main()
