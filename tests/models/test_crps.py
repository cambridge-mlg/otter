import numpy as np
import pytest
import torch
import xarray as xr

from otter.data.datasets import ForecastingSample
from otter.models.losses import crps, weighting


def sample(target):
    return ForecastingSample(
        ctx=torch.zeros_like(target),
        zero_time=np.array(["2020-01-01"], dtype="datetime64[h]"),
        trg_timedelta_hours=np.array([[6]], dtype=np.int32),
        trg=target,
    )


@pytest.mark.parametrize("target,expected", [(5.0, 2.0), (2.0, 0.0)])
def test_two_member_fair_crps_and_permutation(target, expected):
    members = torch.tensor([1.0, 3.0]).reshape(1, 2, 1, 1, 1, 1)
    batch = sample(torch.full((1, 1, 1, 1, 1), target))
    for forecasts in (members, members.flip(1)):
        pointwise = crps.fair_crps(forecasts, batch.trg)
        assert pointwise.shape == batch.trg.shape
        assert pointwise.item() == expected
        loss, info = crps.apply_loss(
            members=forecasts,
            batch=batch,
            trg_variables_and_levels=["temperature"],
            apply_latitude_weighting=False,
        )
        assert loss.item() == expected
        assert info == {
            "temperature/time_00": expected,
            "time_00": expected,
            "fair_crps": expected,
        }


def test_float32_differences_and_reductions_under_autocast():
    # Subtraction in half precision would overflow before the reductions.
    members = torch.tensor(
        [-60000.0, 60000.0], dtype=torch.float16, requires_grad=True
    ).reshape(1, 2, 1, 1, 1, 1)
    target = torch.full((1, 1, 1, 1, 1), 60000.0, dtype=torch.float16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, _ = crps.apply_loss(
            members=members,
            batch=sample(target),
            trg_variables_and_levels=["temperature"],
            apply_latitude_weighting=False,
        )
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    gradient = torch.autograd.grad(loss, members)[0]
    assert torch.isfinite(gradient).all()


def test_gradients_reach_every_member():
    # Each member takes each rank at one longitude, avoiding an identically
    # zero gradient for any entire member in the fair two-member estimator.
    members = torch.tensor(
        [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]],
        requires_grad=True,
    ).reshape(1, 3, 1, 3, 1, 1)
    loss, _ = crps.apply_loss(
        members=members,
        batch=sample(torch.full((1, 1, 3, 1, 1), 5.0)),
        trg_variables_and_levels=["temperature"],
        apply_latitude_weighting=False,
    )
    gradients = torch.autograd.grad(loss, members)[0]
    assert torch.isfinite(gradients).all()
    assert torch.all(gradients.abs().sum(dim=(0, 2, 3, 4, 5)) > 0)


def test_latitude_pressure_surface_and_std_weights_applied_once(monkeypatch):
    variables = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "geopotential/level_100",
        "geopotential/level_1000",
    ]
    calls = []

    def load_std(diff, stats_name):
        calls.append((diff, stats_name))
        return xr.DataArray(
            [2.0, 4.0, 3.0, 8.0],
            dims=["variable_and_level"],
            coords={"variable_and_level": variables},
        )

    monkeypatch.setattr(weighting, "load_stacked_std_data_array", load_std)
    # The poles have much larger errors. Cosine weighting leaves the equator.
    target = torch.tensor([10.0, 2.0, 10.0]).reshape(1, 3, 1, 1, 1)
    target = target * torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss, info = crps.apply_loss(
        members=torch.zeros(1, 2, 3, 1, 1, 4),
        batch=sample(target),
        trg_variables_and_levels=variables,
        weighting_fns=[
            weighting.apply_pressure_weighting,
            weighting.apply_surface_weighting,
            weighting.apply_division_by_variable_std,
        ],
        stats_name="fixture_stats",
    )
    expected = [1.0, 0.1, 2.0 / 11.0, 10.0 / 11.0]
    assert loss.item() == pytest.approx(sum(expected))
    for variable, value in zip(variables, expected):
        assert info[f"{variable}/time_00"] == pytest.approx(value)
    assert calls == [(False, "fixture_stats")]


def test_mean_rmse_and_unbiased_spread():
    members = torch.tensor([1.0, 3.0]).reshape(1, 2, 1, 1, 1, 1)
    batch = sample(torch.full((1, 1, 1, 1, 1), 5.0))
    for metric, expected in [
        ("ensemble_mean_rmse", 3.0),
        ("ensemble_spread", 2.0**0.5),
    ]:
        loss, info = crps.apply_loss(
            members=members,
            batch=batch,
            trg_variables_and_levels=["temperature"],
            apply_latitude_weighting=False,
            metric=metric,
        )
        assert loss.item() == pytest.approx(expected)
        assert info[metric] == pytest.approx(expected)


def test_batch_time_mean_and_variable_sum():
    members = torch.zeros(2, 2, 1, 1, 2, 2)
    target = torch.tensor([1.0, 3.0, 5.0, 7.0]).reshape(2, 1, 1, 2, 1)
    target = target.expand(2, 1, 1, 2, 2)
    loss, info = crps.apply_loss(
        members=members,
        batch=sample(target),
        trg_variables_and_levels=["a", "b"],
        apply_latitude_weighting=False,
    )
    assert loss.item() == 8.0
    assert info["a/time_00"] == 3.0
    assert info["a/time_01"] == 5.0


@pytest.mark.parametrize(
    "members,target,error",
    [
        (torch.zeros(1, 2, 1), torch.zeros(1), "Expected members"),
        (
            torch.zeros(1, 2, 1, 1, 1, 1),
            torch.zeros(2, 1, 1, 1, 1),
            "shapes do not match",
        ),
        (
            torch.zeros(1, 1, 1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1, 1),
            "at least two",
        ),
    ],
)
def test_invalid_ensemble_shapes(members, target, error):
    with pytest.raises(ValueError, match=error):
        crps.fair_crps(members, target)


def test_factory_selects_metric_and_preserves_batch_contract(monkeypatch):
    monkeypatch.setattr(
        crps, "get_trg_vars_and_levels", lambda excluded: ["temperature"]
    )
    loss_fn = crps.make_loss_fn(
        trg_variables_to_remove=[], apply_latitude_weighting=False
    )
    loss, _ = loss_fn(
        members=torch.tensor([1.0, 3.0]).reshape(1, 2, 1, 1, 1, 1),
        batch=sample(torch.full((1, 1, 1, 1, 1), 5.0)),
    )
    assert loss.item() == 2.0
    with pytest.raises(ValueError, match="Unknown ensemble metric"):
        crps.make_loss_fn(trg_variables_to_remove=[], metric="missing")
