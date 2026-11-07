"""One-step ensembles obtained from repeated complete model forwards."""

from contextlib import contextmanager
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch import nn

from otter.data.datasets import ForecastingSample
from otter.models.architectures.moe import LoadBalancingLosses
from otter.models.architectures.utils import aggregate_load_balancing_losses
from otter.models.distributions import Delta
from otter.models.forecasting_model import ForecastingModel


_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


@contextmanager
def _dropout_mode(model: nn.Module, enabled: bool) -> Iterator[None]:
    """Enable explicit dropout without enabling other evaluation-time noise.

    Training keeps its usual module modes (including auxiliary objectives).
    Evaluation first puts all children in evaluation mode, then enables only
    dropout modules. In particular, transformer drop-path and MoE gating stay
    in evaluation mode. Restore every flag, including heterogeneous modes,
    even when a forward raises.
    """
    modes = [(module, module.training) for module in model.modules()]
    try:
        if not model.training:
            model.eval()
        for module, _ in modes:
            if isinstance(module, _DROPOUT_TYPES):
                module.training = enabled
        yield
    finally:
        for module, training in modes:
            module.training = training


def predict_members(
    model: ForecastingModel,
    batch: ForecastingSample,
    num_members: int,
    grad_ckpt_start_timestep: int = 1,
    *,
    enable_dropout: bool = True,
    load_balancing_losses: Optional[List[LoadBalancingLosses]] = None,
) -> torch.Tensor:
    """Predict independent dropout members of the same one-step context.

    Returns a tensor with axes ``(batch, member, lat, lon, lead, variable)``
    in physical units, with the one-element lead axis retained. Each member
    runs the whole forecasting model; sampling a single Delta repeatedly
    would instead give identical members. No tensors are detached, and RNG
    state is neither reset nor otherwise managed here. Seed once before the
    whole call to reproduce an ensemble.

    Call ``model.eval()`` and use ``torch.no_grad()`` for inference. Explicit
    dropout modules (the configured attention projection and SwiGLU dropout)
    remain stochastic. Functional attention dropout is left in evaluation
    mode; the dropout-CRPS configuration sets its probability to zero.

    The optional collector receives one auxiliary aggregate per member, if
    any member has auxiliary losses. An absent member contribution counts as
    zero. Otter's existing mean aggregation therefore averages over members,
    keeping the scale independent of ensemble size. The model, output head,
    and checkpoint parameter names are unchanged.
    """
    if not isinstance(num_members, int) or isinstance(num_members, bool):
        raise ValueError("num_members must be a positive integer")
    if num_members < 1:
        raise ValueError("num_members must be a positive integer")
    lead_times = np.asarray(batch.trg_timedelta_hours)
    if (
        lead_times.ndim not in (1, 2)
        or lead_times.size == 0
        or lead_times.shape[-1] != 1
        or batch.trg.ndim != 5
        or batch.trg.shape[-2] != 1
    ):
        raise ValueError("predict_members requires exactly one target lead")
    if np.any(lead_times != model.temporal_resolution_hours):
        raise ValueError(
            "predict_members supports one temporal-resolution step only"
        )
    # Checkpoint recomputation happens after this helper restores module modes.
    # Ordinary training preserves the same modes and supports checkpointing;
    # temporarily changing modes requires the uncheckpointed one-step forward.
    if torch.is_grad_enabled() and grad_ckpt_start_timestep < 1:
        for module in model.modules():
            if isinstance(module, _DROPOUT_TYPES):
                intended_mode = enable_dropout
            else:
                intended_mode = module.training if model.training else False
            if module.training != intended_mode:
                raise ValueError(
                    "Changing module modes with gradients requires "
                    "grad_ckpt_start_timestep >= 1 for one-step ensembles"
                )

    predictions = []
    member_auxiliary = []
    with _dropout_mode(model, enabled=enable_dropout):
        for _ in range(num_members):
            distribution, auxiliary = model(
                batch,
                grad_ckpt_start_timestep=grad_ckpt_start_timestep,
            )
            if not isinstance(distribution, Delta):
                raise TypeError("Dropout ensembles require a Delta output head")
            if distribution.mean.shape != batch.trg.shape:
                raise ValueError("Forecast and one-step target shapes differ")
            predictions.append(distribution.mean)
            if load_balancing_losses is not None:
                member_auxiliary.append(
                    aggregate_load_balancing_losses(auxiliary, mean=True)
                )

    if load_balancing_losses is not None and any(
        auxiliary is not None for auxiliary in member_auxiliary
    ):
        zero = predictions[0].new_zeros(())
        load_balancing_losses.extend(
            auxiliary
            if auxiliary is not None
            else LoadBalancingLosses(zero, zero)
            for auxiliary in member_auxiliary
        )

    return torch.stack(predictions, dim=1)
