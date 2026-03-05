import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
)

import torch
from torch.amp.grad_scaler import GradScaler

from otter.models.architectures.schedulers import RolloutHorizonScheduler


@dataclass
class TrainState:
    epoch: int
    step: int
    best_val_loss: float = float("inf")


@dataclass
class RunState:
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    scaler_state: Dict[str, Any]
    rollout_horizon_scheduler_state: Dict[str, Any]
    epoch: int
    step: int
    best_val_loss: float = float("inf")

    @staticmethod
    def from_objects(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: GradScaler,
        rollout_horizon_scheduler: RolloutHorizonScheduler,
        epoch: int,
        step: int,
        best_val_loss: float = float("inf"),
    ) -> "RunState":
        """Create RunState from torch objects."""
        return RunState(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=lr_scheduler.state_dict(),  # type: ignore[no-untyped-call]
            scaler_state=scaler.state_dict(),
            rollout_horizon_scheduler_state=rollout_horizon_scheduler.state_dict(),
            epoch=epoch,
            step=step,
            best_val_loss=best_val_loss,
        )

    @staticmethod
    def from_dict(state_dict: Dict[str, Any]) -> "RunState":
        """Create RunState from a dictionary loaded from disk.
        """
        return RunState(
            model_state=state_dict["model_state"],
            optimizer_state=state_dict["optimizer_state"],
            scheduler_state=state_dict["scheduler_state"],
            scaler_state=state_dict["scaler_state"],
            # Old checkpoints may not have this key
            rollout_horizon_scheduler_state=state_dict.get(
                "rollout_horizon_scheduler_state", {}
            ),
            epoch=state_dict["epoch"],
            step=state_dict["step"],
            best_val_loss=state_dict["best_val_loss"],
        )

    @staticmethod
    def load(path: Path) -> "RunState":
        """Load RunState from disk."""
        try:
            state_dict = torch.load(path, weights_only=False, map_location="cpu")
            return RunState.from_dict(state_dict)
        except Exception as e:
            logging.error(f"Failed to load checkpoint from {path}: {e}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert RunState to a dictionary that can be saved to disk."""
        return {
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "scaler_state": self.scaler_state,
            "rollout_horizon_scheduler_state": self.rollout_horizon_scheduler_state,
            "epoch": self.epoch,
            "step": self.step,
            "best_val_loss": self.best_val_loss,
        }

    def save(self, path: Path) -> None:
        """Save RunState to disk using atomic write to prevent corruption."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first for crash safety
        tmp_path = path.parent / f"{path.name}.tmp"
        try:
            torch.save(self.to_dict(), tmp_path)
            # Atomically replace the target file
            # On POSIX systems, Path.replace() is atomic when on same filesystem
            tmp_path.replace(path)
            logging.debug(f"Successfully saved checkpoint to {path}")
        except Exception as e:
            # Clean up temp file if write failed
            if tmp_path.exists():
                tmp_path.unlink()
            logging.error(f"Failed to save checkpoint to {path}: {e}")
            raise

    def restore(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: GradScaler,
        rollout_horizon_scheduler: RolloutHorizonScheduler,
    ) -> None:
        """Restore state to provided objects."""
        model.load_state_dict(self.model_state)
        optimizer.load_state_dict(self.optimizer_state)
        lr_scheduler.load_state_dict(self.scheduler_state)
        scaler.load_state_dict(self.scaler_state)
        rollout_horizon_scheduler.load_state_dict(
            self.rollout_horizon_scheduler_state
        )


def load_run_state_from_latest_checkpoint(
    experiment: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    rollout_horizon_scheduler: RolloutHorizonScheduler,
) -> TrainState:
    """
    Load the initial state of the run from the given experiment_name.

    Args:
        experiment: The experiment object, loaded from the yaml config.
        optimizer: The torch optimizer.
        lr_scheduler: The torch learning rate scheduler.
        scaler: The torch amp gradient scaler object.
    Returns:
        The TrainState object with the initial epoch and step number.
    """
    checkpoint_dir = get_checkpoint_dir(experiment)
    logging.info(f"Looking for checkpoints in {checkpoint_dir}")

    try:
        path = get_latest_checkpoint(checkpoint_dir)
        logging.info(f"Found checkpoint: {path}")

        # Load complete run state if available
        run_state = RunState.load(path)
        run_state.restore(
            model=experiment.model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            rollout_horizon_scheduler=rollout_horizon_scheduler,
        )
        step = run_state.step
        epoch = run_state.epoch
        best_val_loss = run_state.best_val_loss
        logging.info(f"Successfully restored state from {path} at step {step}")

        return TrainState(epoch=epoch, step=step, best_val_loss=best_val_loss)

    except Exception as e:
        logging.error(f"Failed to load checkpoint from {checkpoint_dir}: {e}")
        raise


def get_checkpoint_dir(experiment: Any) -> Path:
    return Path(experiment.info.experiment_path) / "checkpoints"


def get_latest_checkpoint(path: Path) -> Path:
    """Get the latest checkpoint file in the given directory."""
    if (path / "latest.pt").exists():
        return path / "latest.pt"
    checkpoints = list(path.glob("*.pt"))
    # Exclude "best" checkpoints for the purpose of getting the latest.
    checkpoints = [c for c in checkpoints if "best" not in c.name]
    if not checkpoints:
        error = FileNotFoundError(f"No checkpoint files found in {path}")
        raise error
    return max(checkpoints, key=lambda x: str(x))


def save_run_state(
    experiment: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    rollout_horizon_scheduler: RolloutHorizonScheduler,
    epoch: int,
    step: int,
    checkpoint_name: str,
    best_val_loss: float = float("inf"),
) -> None:
    """Save the complete run state to disk."""
    logging.info(f"Saving checkpoint {checkpoint_name} at step {step}")
    run_state = RunState.from_objects(
        model=experiment.model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        rollout_horizon_scheduler=rollout_horizon_scheduler,
        epoch=epoch,
        step=step,
        best_val_loss=best_val_loss,
    )
    logging.info("Checkpoint saved.")

    path = get_checkpoint_dir(experiment) / checkpoint_name
    run_state.save(path)
