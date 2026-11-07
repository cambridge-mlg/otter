"""Conversion checks using random tiny Swin checkpoints, never real weights.

These tests run the normal trainer and checkpoint machinery on CPU. They use
bundled normalization statistics, synthetic weather, eager execution and the
existing PyTorch attention fallback; no data or checkpoints are downloaded.
"""

from functools import partial

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.amp import GradScaler

from otter.data.datasets import ForecastingSample
from otter.experiment import train
from otter.experiment.run_state import RunState
from otter.experiment.utils import (
    _get_config,
    load_model_weights_from_experiment,
)
from otter.models.architectures.schedulers import RolloutHorizonScheduler


class Logger:
    records = None

    def __init__(self, step=0):
        self.step = step

    def timed(self, name, function, *args, **kwargs):
        return function(*args, **kwargs)

    def log(self, values):
        if self.records is not None:
            self.records.append(values)


class RecordingAdam(torch.optim.AdamW):
    def __init__(self, model):
        self.named_parameters = dict(model.named_parameters())
        self.gradients = {}
        self.step_calls = 0
        super().__init__(model.parameters(), lr=1e-4)

    def step(self, closure=None):
        self.gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in self.named_parameters.items()
            if parameter.grad is not None
        }
        self.step_calls += 1
        return super().step(closure)


class Batches:
    """Keep ForecastingSample objects intact inside allow_objects configs."""

    def __init__(self, *batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


class EnsembleObjective:
    def __init__(self, objective):
        self.objective = objective
        self.calls = []

    def __call__(self, *, members, batch):
        assert isinstance(batch, ForecastingSample)
        assert batch.ctx.ndim == 5
        self.calls.append(
            (members.shape[0], members.shape[1], torch.is_grad_enabled())
        )
        loss, info = self.objective(members=members, batch=batch)
        assert loss.ndim == 0 and torch.isfinite(loss)
        return loss, info


@pytest.fixture(autouse=True)
def eager_without_wandb(monkeypatch):
    # Swin blocks are compiled by default; these are small eager CPU fixtures.
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    monkeypatch.setattr(train, "WandbLogger", Logger)
    monkeypatch.setattr(train.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.wandb, "finish", lambda: None)


def tiny_config(name="dropout_crps", dropout=0.35):
    return _get_config(
        "otter.experiment.config",
        name,
        [
            f"constants.training.dropout_rate={dropout}",
            "constants.training.drop_path_rate=0",
            "constants.training.num_training_steps=10",
            "constants.training.num_lr_decay_steps=10",
            "model.data_grid_num_lat=9",
            "model.data_grid_num_lon=16",
            "model.backbone_network.height=9",
            "model.backbone_network.width=16",
            "model.backbone_network.token_dimensions=[8,8]",
            "model.backbone_network.num_blocks_per_stage=[1]",
            "model.backbone_network.num_bottom_blocks=1",
            "model.backbone_network.num_heads=2",
            "model.backbone_network.window_size=2",
            "model.backbone_network.use_rope=false",
            "+model.backbone_network.use_efficient_attention=false",
            "feedforward_network.hidden_ratio=0.5",
            "model.time_embedding.num_scales=2",
            "model.backbone_network.pos_embedding.num_scales=4",
        ],
    )


def make_model(config):
    return instantiate(config.model, device="cpu")


def make_batch(model):
    generator = torch.Generator().manual_seed(17)
    shape = (1, 9, 16, 4, len(model.ctx_variables_and_levels))
    context = (
        torch.randn(shape, generator=generator) * model.ctx_std
        + model.ctx_mean
    )
    target = (
        context[..., :1, model.idx_residual].clone() + 0.1 * model.trg_std
    )
    return ForecastingSample(
        ctx=context,
        trg=target,
        zero_time=np.array(["2020-01-01T00"], dtype="datetime64[h]"),
        trg_timedelta_hours=np.array([[6]], dtype=np.int32),
    )


def horizon():
    return RolloutHorizonScheduler.pretraining_scheduler(10, 6)


def snapshot(model):
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def assert_finite_backbone_update(model, optimizer, before):
    gradients = [
        value
        for name, value in optimizer.gradients.items()
        if name.startswith("backbone_network.")
    ]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)
    assert any(value.abs().sum() > 0 for value in gradients)
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
        if name.startswith("backbone_network.")
    )


def update(model, objective, *, accumulation=1, members=None):
    optimizer = RecordingAdam(model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
    rollout = horizon()
    train.train_step(
        model=model,
        loss_fn=objective,
        train_loader=iter([make_batch(model) for _ in range(accumulation)]),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        scaler=GradScaler("cpu", enabled=False),
        rollout_horizon_scheduler=rollout,
        wandb_logger=Logger(),
        clip_grad_norm=10.0,
        mixed_precision_dtype=torch.bfloat16,
        grad_ckpt_start_timestep=1,
        accumulate_batches=accumulation,
        device="cpu",
        ensemble_members=members,
    )
    assert optimizer.step_calls == 1
    assert scheduler.last_epoch == 1
    assert rollout.current_step == 1
    return optimizer


def test_strict_fixture_conversion_and_accumulated_update(tmp_path):
    deterministic = make_model(tiny_config("base", dropout=0))
    source = tmp_path / "deterministic.pt"
    torch.save({"model_state": deterministic.state_dict()}, source)
    config = tiny_config()
    model = make_model(config)
    assert model.state_dict().keys() == deterministic.state_dict().keys()
    load_model_weights_from_experiment(source, model)
    for name, value in deterministic.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])

    objective = EnsembleObjective(instantiate(config.loss_fn))
    before = snapshot(model)
    optimizer = update(model, objective, accumulation=2, members=2)
    assert objective.calls == [(1, 2, True), (1, 2, True)]
    assert_finite_backbone_update(model, optimizer, before)


def test_validation_is_stochastic_read_only_and_restores_modes():
    config = tiny_config()
    model = make_model(config).train()
    dropout = next(
        m for m in model.modules() if isinstance(m, torch.nn.Dropout)
    )
    dropout.eval()  # Deliberately heterogeneous prior modes.
    modes = [module.training for module in model.modules()]
    before = snapshot(model)
    parameter = next(model.parameters())
    parameter.grad = torch.full_like(parameter, 0.25)
    prior_gradient = parameter.grad.clone()
    predictions = []

    def observe(module, inputs, output):
        assert not module.training
        assert not torch.is_grad_enabled()
        assert all(
            layer.training
            for layer in module.modules()
            if isinstance(layer, torch.nn.Dropout)
        )
        predictions.append(output[0].mean.detach().clone())

    handle = model.register_forward_hook(observe)
    try:
        losses, _, _ = train.run_validation_epoch(
            model=model,
            device="cpu",
            dataloader=[make_batch(model)],
            loss_fns=instantiate(config.val_loss_fns),
            mixed_precision_dtype=torch.bfloat16,
            ensemble_members=3,
        )
    finally:
        handle.remove()
    assert len(predictions) == 3
    assert not torch.equal(predictions[0], predictions[1])
    assert all(np.isfinite(value).all() for value in losses.values())
    assert losses["ensemble_spread"][0] > 0
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(parameter.grad, prior_gradient)
    assert all(
        torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )

    def fail(**kwargs):
        raise RuntimeError("validation objective failed")

    with pytest.raises(RuntimeError, match="validation objective failed"):
        train.run_validation_epoch(
            model,
            "cpu",
            [make_batch(model)],
            {"train_loss": fail},
            torch.bfloat16,
            ensemble_members=2,
        )
    assert [module.training for module in model.modules()] == modes


def test_deterministic_training_and_validation_contract():
    config = tiny_config("base", dropout=0)
    model = make_model(config)
    before = snapshot(model)
    optimizer = update(model, instantiate(config.loss_fn))
    assert_finite_backbone_update(model, optimizer, before)
    losses, _, _ = train.run_validation_epoch(
        model,
        "cpu",
        [make_batch(model)],
        instantiate(config.val_loss_fns),
        torch.bfloat16,
    )
    assert set(losses) == {"train_loss", "raw_rmse", "raw_lwrmse"}
    assert all(np.isfinite(value).all() for value in losses.values())


def test_checkpoint_selection_uses_forecast_crps(monkeypatch):
    monkeypatch.setattr(
        train, "run_validation_epoch",
        lambda **kwargs: (
            {"train_loss": [104.0], "fair_crps": [4.0]},
            {
                "train_loss": [{"total_load_balancing_loss": 100.0}],
                "fair_crps": [{"fair_crps": 4.0}],
            },
            {},
        ),
    )
    selected = train.validation_epoch(
        None, "cpu", [], {}, Logger(), torch.bfloat16, ensemble_members=2
    )
    assert selected == 4.0


def make_experiment(config, path, source, *, resume, max_steps):
    model = make_model(config)
    objective = EnsembleObjective(instantiate(config.loss_fn))
    validation = dict(instantiate(config.val_loss_fns))
    validation["train_loss"] = objective
    return OmegaConf.create(
        {
            "model": model,
            "loss_fn": objective,
            "val_loss_fns": validation,
            "optimizer": RecordingAdam,
            "lr_scheduler": partial(
                torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5
            ),
            "rollout_horizon_scheduler": horizon(),
            "train_dataloader": Batches(make_batch(model)),
            "valid_dataloader": Batches(make_batch(model)),
            "info": {
                "experiment_name": "dropout-fixture",
                "experiment_path": str(path),
                "start_from_checkpoint": resume,
            },
            "constants": {
                "load_weights_from_experiment": str(source),
                "save_best_checkpoint": True,
                "save_checkpoint_every": 1,
                "save_persistent_checkpoint_every": 100,
                "ensemble": {
                    "enabled": True,
                    "train_members": 2,
                    "validation_members": 3,
                },
                "data": {"accumulate_batches": 1},
                "training": {
                    "mixed_precision_dtype": torch.bfloat16,
                    "clip_grad_norm": 10.0,
                    "grad_ckpt_start_timestep": 1,
                    "max_steps": max_steps,
                    "validate_every": 1,
                    "epochs": 1,
                },
            },
        },
        flags={"allow_objects": True},
    )


def test_main_converts_saves_and_resumes_without_original_source(
    tmp_path,
    monkeypatch,
):
    source_model = make_model(tiny_config("base", dropout=0))
    source_state = snapshot(source_model)
    source = tmp_path / "source.pt"
    # Invalid source optimizer data proves conversion loads model weights only.
    torch.save(
        {
            "model_state": source_state,
            "optimizer_state": {"source_only": True},
        },
        source,
    )
    config = tiny_config()
    first = make_experiment(
        config, tmp_path / "run", source, resume=False, max_steps=1
    )
    monkeypatch.setattr(train, "assert_single_device", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train, "initialize_wandb", lambda config: None)
    monkeypatch.setattr(
        train, "GradScaler", lambda device: GradScaler("cpu", enabled=False)
    )
    monkeypatch.setattr(
        train, "initialize_experiment", lambda device: (first, config)
    )
    loaded = []

    def load_weights(path, model):
        result = load_model_weights_from_experiment(path, model)
        loaded.append(snapshot(result))
        return result

    monkeypatch.setattr(
        train, "load_model_weights_from_experiment", load_weights
    )
    train.main()
    assert len(loaded) == 1
    assert all(
        torch.equal(loaded[0][name], value)
        for name, value in source_state.items()
    )
    assert first.loss_fn.calls == [(1, 2, True)]
    checkpoint = tmp_path / "run" / "checkpoints" / "latest.pt"
    saved = RunState.load(checkpoint)
    assert saved.step == 1
    assert saved.scheduler_state["last_epoch"] == 1
    assert saved.rollout_horizon_scheduler_state["current_step"] == 1
    assert all(
        state["step"].item() == 1
        for state in saved.optimizer_state["state"].values()
    )

    source.unlink()
    resumed = make_experiment(
        config, tmp_path / "run", source, resume=True, max_steps=2
    )
    monkeypatch.setattr(
        train, "initialize_experiment", lambda device: (resumed, config)
    )
    restore = train.load_run_state_from_latest_checkpoint
    restored = []

    def restore_state(**kwargs):
        result = restore(**kwargs)
        restored.append(snapshot(kwargs["experiment"].model))
        assert result.step == 1
        assert kwargs["lr_scheduler"].last_epoch == 1
        assert kwargs["rollout_horizon_scheduler"].current_step == 1
        assert all(
            state["step"].item() == 1
            for state in kwargs["optimizer"].state.values()
        )
        return result

    monkeypatch.setattr(
        train, "load_run_state_from_latest_checkpoint", restore_state
    )
    train.main()
    assert len(loaded) == 1  # No attempt to reload the now-missing source.
    assert all(
        torch.equal(restored[0][name], value)
        for name, value in saved.model_state.items()
    )
    assert resumed.loss_fn.calls == [(1, 3, False), (1, 2, True)]
    final = RunState.load(checkpoint)
    assert final.step == 2
    assert final.scheduler_state["last_epoch"] == 2
    assert final.rollout_horizon_scheduler_state["current_step"] == 2
    assert all(
        state["step"].item() == 2
        for state in final.optimizer_state["state"].values()
    )
    assert np.isfinite(final.best_val_loss)
    assert (checkpoint.parent / "best.pt").exists()
