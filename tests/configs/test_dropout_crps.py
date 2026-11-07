"""Compose the conversion recipe without loading weather data or a checkpoint."""

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from otter.experiment.utils import _get_config, _parse_args
from otter.models.distributions import Delta


def _config(overrides=()) -> DictConfig:
    return _get_config(
        "otter.experiment.config",
        "dropout_crps",
        ["constants.training.num_training_steps=12", *overrides],
    )


def test_dropout_config_keeps_deterministic_parameter_architecture() -> None:
    config = _config()
    source = _get_config("otter.experiment.config", "base", [])
    source_model = OmegaConf.to_container(source.model, resolve=True)
    ensemble_model = OmegaConf.to_container(config.model, resolve=True)
    for model in (source_model, ensemble_model):
        assert isinstance(model, dict)
        backbone = model["backbone_network"]
        assert isinstance(backbone, dict)
        del backbone["dropout_rate"]
        del backbone["drop_path_rate"]
        del backbone["feedforward_network"]["dropout_rate"]
    assert source_model == ensemble_model
    assert config.constants.num_stats == Delta.num_stats == 1
    assert config.model.distribution_type is Delta
    assert config.constants.training.drop_path_rate == 0
    assert config.model.backbone_network.attn_dropout_rate == 0


def test_dropout_config_all_horizons_are_one_six_hour_step() -> None:
    config = _config()
    assert config.constants.data.temporal_resolution_hours == 6
    for loader in (
        config.train_dataloader,
        config.valid_dataloader,
        config.test_dataloader,
    ):
        assert loader.dataset.num_target_frames == 1
        assert loader.dataset.temporal_resolution_hours == 6
    scheduler = instantiate(config.rollout_horizon_scheduler)
    for _ in range(24):
        assert scheduler.get_rollout_steps() == 1
        assert scheduler.get_rollout_time_hours().tolist() == [6]
        scheduler.step()


def test_dropout_config_instantiates_only_ensemble_objectives() -> None:
    config = _config()
    # Composition must remove the base distribution-loss kwargs and metrics.
    assert "loss_fn" not in config.loss_fn
    assert set(config.val_loss_fns) == {
        "train_loss",
        "fair_crps",
        "ensemble_mean_rmse",
        "ensemble_spread",
    }
    assert callable(instantiate(config.loss_fn))
    for metric in config.val_loss_fns.values():
        assert callable(instantiate(metric))


def test_documented_conversion_command_and_independent_batch_controls() -> None:
    args, overrides = _parse_args(
        [
            "--config_name",
            "dropout_crps",
            "--debug",
            "constants.load_weights_from_experiment="
            "_results/deterministic-experiment/checkpoints/latest.pt",
            "constants.ensemble.train_members=2",
            "constants.ensemble.validation_members=8",
            "constants.data.memory_max_batch_size=1",
            "constants.data.accumulate_batches=4",
        ]
    )
    config = _config(overrides)
    assert args.config_name == "dropout_crps"
    assert args.debug
    assert not args.resume_experiment
    assert config.constants.load_weights_from_experiment == (
        "_results/deterministic-experiment/checkpoints/latest.pt"
    )
    assert config.constants.ensemble.train_members == 2
    assert config.constants.ensemble.validation_members == 8
    assert config.train_dataloader.batch_size == 1
    assert config.constants.data.accumulate_batches == 4
    assert config.constants.data.batch_size == 4

    changed = _config(
        [
            "constants.ensemble.train_members=3",
            "constants.ensemble.validation_members=5",
            "constants.data.memory_max_batch_size=2",
            "constants.data.accumulate_batches=3",
        ]
    )
    assert changed.constants.ensemble.train_members == 3
    assert changed.constants.ensemble.validation_members == 5
    assert changed.train_dataloader.batch_size == 2
    assert changed.constants.data.accumulate_batches == 3
    assert changed.constants.data.batch_size == 6


def test_documented_resume_command() -> None:
    args, overrides = _parse_args(
        ["--resume_experiment", "dropout-finetuning-experiment", "--debug"]
    )
    assert args.resume_experiment == "dropout-finetuning-experiment"
    assert args.debug
    assert overrides == []
