#!/usr/bin/env python3
"""Rollout finetuning CLI.

A convenience wrapper around train.py that simplifies rollout finetuning setup.
It automatically resolves the checkpoint path from an existing experiment and
sets up the output directory.

Usage:
    python rollout_finetune.py <experiment_name> [additional_hydra_overrides]

Example:
    python rollout_finetune.py 2026-01-18_13-50-fierce-eagle
    python rollout_finetune.py 2026-01-18_13-50-fierce-eagle constants.training.epochs=2
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Rollout finetuning CLI wrapper for train.py",
        epilog="Additional arguments are passed through to train.py as Hydra overrides.",
    )
    parser.add_argument(
        "experiment_name",
        type=str,
        help="Name of the source experiment to finetune from (e.g., 2026-01-18_13-50-fierce-eagle)",
    )
    parser.add_argument(
        "--root_results",
        type=str,
        default="_results",
        help="Root directory for experiment results (default: _results)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (skips git checks)",
    )

    args, extra_args = parser.parse_known_args()

    # Resolve paths
    source_experiment_dir = Path(args.root_results) / args.experiment_name
    checkpoint_path = source_experiment_dir / "checkpoints" / "latest.pt"
    rft_experiment_name = f"rft_{args.experiment_name}"
    rft_experiment_path = Path(args.root_results) / rft_experiment_name

    # Validate source experiment exists
    assert source_experiment_dir.exists(), (
        f"Source experiment not found: {source_experiment_dir}"
    )

    if not checkpoint_path.exists():
        checkpoints_dir = source_experiment_dir / "checkpoints"
        available = list(checkpoints_dir.iterdir()) if checkpoints_dir.exists() else []
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Available: {[f.name for f in available]}"
        )

    # Build train.py command
    train_script = Path(__file__).parent / "train.py"

    cmd = [
        sys.executable,
        str(train_script),
        "--config_name", "rollout_finetuning",
        "--root_results", args.root_results,
        f"constants.load_weights_from_experiment={checkpoint_path}",
        f"info.experiment_name={rft_experiment_name}",
        f"info.experiment_path={rft_experiment_path}",
    ]

    if args.debug:
        cmd.append("--debug")

    # Pass through any additional Hydra overrides
    cmd.extend(extra_args)

    logging.info("Starting rollout finetuning:")
    logging.info(f"  Source: {source_experiment_dir}")
    logging.info(f"  Checkpoint: {checkpoint_path}")
    logging.info(f"  Output: {rft_experiment_path}")

    # Execute train.py
    result = subprocess.run(cmd, cwd=os.getcwd())
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
