from datetime import datetime
from typing import Any, Callable, Dict, Literal, Tuple

import torch

import wandb
from otter.models.losses.loss_fn import LossInfo


class WandbLogger:
    """
    A thin wrapper around wandb active for one training step.

    This avoids passing step to every wandb.log call, reducing visual clutter.
    """

    def __init__(self, step: int):
        self.step = step

    def timed(
        self,
        log_name: str,
        f: Callable[..., Any],
        *f_args: Any,
        **f_kwargs: Any,
    ) -> Any:
        """Run the function f and log the time it took to run to wandb."""
        return run_and_log_time_to_wandb(
            log_name, self.step, f, *f_args, **f_kwargs
        )

    def log(self, log_dict: Dict[str, Any]) -> None:
        wandb.log(log_dict, step=self.step)


def run_with_timing(
    f: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Tuple[float, Any]:
    start = datetime.now()
    result = f(*args, **kwargs)
    end = datetime.now()
    return (end - start).total_seconds(), result


def run_and_log_time_to_wandb(
    log_name: str,
    step: int,
    f: Callable[..., Any],
    *f_args: Any,
    **f_kwargs: Any,
) -> Any:
    time, result = run_with_timing(f, *f_args, **f_kwargs)
    wandb.log({f"time/{log_name}": time}, step=step)
    return result


def log_loss_info_to_wandb(
    loss_info: LossInfo,
    prefix: str,
) -> None:
    for k, v in loss_info.items():
        wandb.log({f"{prefix}/{k}": v}, commit=False)


def get_norm_and_abs_max(
    model: torch.nn.Module, measurable: Literal["gradients", "parameters"]
) -> tuple[float, float]:
    max_quantity = 0.0
    norm: float = 0.0
    for param in model.parameters():
        quantity = param.grad if measurable == "gradients" else param
        if quantity is None:
            continue
        norm += float(quantity.norm() ** 2)
        abs_param_quantity = quantity.abs().max().item()
        max_quantity = max(max_quantity, abs_param_quantity)
    return norm**0.5, max_quantity
