from pathlib import Path

from emotion_study_openenv import EmotionStudyEnv


REQUIRED_FILES = [
    "README.md",
    "Dockerfile",
    "requirements.txt",
    "openenv.yaml",
    "inference.py",
]


def validate_files(project_root: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (project_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def smoke_test_environment() -> None:
    env = EmotionStudyEnv()
    action_plan = {
        "calm_and_recover": [
            "acknowledge_emotion",
            "play_calming_music",
            "adjust_difficulty",
        ],
        "rebuild_focus": [
            "set_study_goal",
            "suggest_break",
            "give_micro_quiz",
            "offer_encouragement",
        ],
        "challenge_with_confidence_control": [
            "adjust_difficulty",
            "give_micro_quiz",
            "offer_encouragement",
        ],
    }

    for task in env.tasks:
        observation = env.reset(task_id=task.id)
        done = False
        step_index = 0
        while not done and step_index < len(action_plan[task.id]):
            action_type = action_plan[task.id][step_index]
            result = env.step({"action_type": action_type})
            observation = result["observation"]
            done = result["done"]
            step_index += 1
        state = env.state()
        if state["episode_metrics"]["steps_taken"] <= 0:
            raise AssertionError(f"Task {task.id} did not progress during smoke test.")
        if "student_state" not in observation:
            raise AssertionError(f"Task {task.id} returned an invalid observation.")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    validate_files(project_root)
    smoke_test_environment()
    print("Validation passed.")


if __name__ == "__main__":
    main()
