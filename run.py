"""Unified entry point for TER experiments."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TER experiments")
    parser.add_argument(
        "--task",
        choices=["test_time_training", "self_supervised", "ttt", "ssl"],
        required=True,
        help="Experiment track",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    normalized_task = args.task
    if args.task == "ttt":
        normalized_task = "test_time_training"
    elif args.task == "ssl":
        normalized_task = "self_supervised"

    print(f"[TASK] {normalized_task}")
    print(f"[CONFIG] {config_path}")
    if normalized_task == "test_time_training":
        print("[TODO] Wire this command to src/test_time_training.py")
    else:
        print("[TODO] Wire this command to src/self_supervised.py")


if __name__ == "__main__":
    main()
