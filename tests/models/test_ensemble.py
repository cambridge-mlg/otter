"""MC-dropout behaviour using small, checkpoint-compatible model fixtures."""

import numpy as np
import pytest
import torch
from torch import nn

from otter.data.datasets import ForecastingSample
from otter.models.architectures.moe import LoadBalancingLosses
from otter.models.distributions import Delta, Gaussian
from otter.models.ensemble import predict_members
from otter.models.losses.loss_fn import gather_load_balancing_losses


class DropoutModel(nn.Module):
    temporal_resolution_hours = 6

    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalisation = nn.BatchNorm1d(1)
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.seen_batches = []
        self.outputs = []
        self.seen_modes = []
        self.fail = False
        self.gaussian = False
        self.sparse_auxiliary = False

    def forward(self, batch, grad_ckpt_start_timestep):
        self.seen_batches.append(batch)
        self.seen_modes.append(
            (self.training, self.dropout.training, self.normalisation.training)
        )
        assert grad_ckpt_start_timestep == 1
        if self.fail:
            raise RuntimeError("test forward failure")
        prediction = self.dropout(batch.ctx) * self.weight
        if prediction.requires_grad:
            prediction.retain_grad()
        self.outputs.append(prediction)
        auxiliary = LoadBalancingLosses(self.weight * 2, self.weight * 3)
        if self.gaussian:
            return Gaussian(prediction, torch.zeros_like(prediction), 0.1), []
        if self.sparse_auxiliary and len(self.outputs) % 2 == 0:
            return Delta(prediction), []
        return Delta(prediction), [auxiliary]


@pytest.fixture
def batch():
    shape = (2, 3, 16, 1, 1)
    return ForecastingSample(
        ctx=torch.ones(shape),
        zero_time=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[h]"),
        trg_timedelta_hours=np.array([[6], [6]], dtype=np.int32),
        trg=torch.zeros(shape),
    )


def modes(model):
    return [module.training for module in model.modules()]


def test_same_batch_independent_draws_and_whole_ensemble_seed(batch):
    model = DropoutModel().eval()
    before = modes(model)
    torch.manual_seed(123)
    first = predict_members(model, batch, num_members=3)
    assert first.shape == (2, 3, 3, 16, 1, 1)
    assert not torch.equal(first[:, 0], first[:, 1])
    assert not torch.equal(first[:, 1], first[:, 2])
    assert all(seen is batch for seen in model.seen_batches)
    assert modes(model) == before
    assert model.seen_modes == [(False, True, False)] * 3
    torch.manual_seed(123)
    second = predict_members(model, batch, num_members=3)
    assert torch.equal(first, second)


@pytest.mark.parametrize("probability, enabled", [(0.0, True), (0.5, False)])
def test_disabled_dropout_agrees(batch, probability, enabled):
    model = DropoutModel(probability).eval()
    members = predict_members(model, batch, 3, enable_dropout=enabled)
    assert torch.equal(members[:, 0], members[:, 1])
    assert torch.equal(members[:, 1], members[:, 2])


def test_every_member_preserves_gradients_and_auxiliary_scale(batch):
    model = DropoutModel().train()
    auxiliary = []
    members = predict_members(model, batch, 4, load_balancing_losses=auxiliary)
    loss, _ = gather_load_balancing_losses(auxiliary)
    assert loss.item() == 10.0
    (members.mean() + loss).backward()
    assert torch.isfinite(model.weight.grad)
    assert model.weight.grad.abs() > 0
    for output in model.outputs:
        assert output.grad is not None
        assert output.grad.abs().sum() > 0
    assert model.seen_modes == [(True, True, True)] * 4


def test_missing_auxiliary_contributions_are_averaged_as_zero(batch):
    model = DropoutModel()
    model.sparse_auxiliary = True
    auxiliary = []
    predict_members(model, batch, 4, load_balancing_losses=auxiliary)
    loss, _ = gather_load_balancing_losses(auxiliary)
    assert len(auxiliary) == 4
    assert loss.item() == 5.0


@pytest.mark.parametrize("fail", [False, True])
def test_restores_heterogeneous_modes_even_on_failure(batch, fail):
    model = DropoutModel().eval()
    model.normalisation.train()
    model.dropout.train()
    model.fail = fail
    before = modes(model)
    if fail:
        with pytest.raises(RuntimeError, match="test forward failure"):
            predict_members(model, batch, 2)
    else:
        with torch.no_grad():
            members = predict_members(model, batch, 2)
        assert not members.requires_grad
        assert model.seen_modes == [(False, True, False)] * 2
    assert modes(model) == before


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_rejects_invalid_member_count(batch, count):
    with pytest.raises(ValueError, match="positive integer"):
        predict_members(DropoutModel(), batch, count)


@pytest.mark.parametrize("leads", [[[6, 12], [6, 12]], [[12], [12]], []])
def test_rejects_rollout_or_empty_leads(batch, leads):
    batch.trg_timedelta_hours = np.asarray(leads)
    with pytest.raises(ValueError, match="one target lead|one temporal"):
        predict_members(DropoutModel(), batch, 2)


def test_rejects_non_delta_head(batch):
    model = DropoutModel()
    model.gaussian = True
    with pytest.raises(TypeError, match="Delta output"):
        predict_members(model, batch, 2)


@pytest.mark.parametrize("training,enabled", [(False, True), (True, False)])
def test_rejects_checkpoint_recomputation_with_changed_modes(
    batch, training, enabled
):
    model = DropoutModel().train(training)
    before = modes(model)
    with pytest.raises(ValueError, match="Changing module modes"):
        predict_members(
            model,
            batch,
            2,
            grad_ckpt_start_timestep=0,
            enable_dropout=enabled,
        )
    assert modes(model) == before
