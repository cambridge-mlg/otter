import argparse
import importlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Sized

import git
import hydra
import torch
import xarray as xr
import yaml
from hydra import compose
from hydra.utils import instantiate
from omegaconf import (
    DictConfig,
    OmegaConf,
)

import wandb
from otter.experiment.namegen import _generate_experiment_name

Experiment = DictConfig


def load_experiment_config(
    experiment_name: str,
    override_args: list[str],
    results_dir: Path | str = "_results",
) -> DictConfig:
    cfg_dir = Path(results_dir) / experiment_name / "config.yaml"

    _register_resolvers()
    with open(cfg_dir, "r") as f:
        cfg = OmegaConf.load(f)

    overrides = OmegaConf.from_cli(override_args)
    cfg = OmegaConf.merge(cfg, overrides)
    OmegaConf.set_struct(cfg, False)

    assert isinstance(cfg, DictConfig)
    return cfg


def get_model_parameters(
    model: torch.nn.Module,
) -> Iterator[torch.nn.Parameter]:
    return model.parameters()


def get_num_training_steps(
    num_epochs: int,
    dataloader: Sized | torch.utils.data.DataLoader[Any],
    accumulate_batches: int = 1,
) -> int:
    return (num_epochs * len(dataloader)) // accumulate_batches


def get_num_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def create_module_summary(experiment: Experiment) -> str:
    module_summary = get_module_summary(experiment.model)
    num_params = get_num_parameters(experiment.model)
    return module_summary + f"\nTotal # parameters: {num_params / 1e6:.1f}M."


def get_module_attr(string: str) -> Any:
    module_name, class_name = string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_model_weights_from_experiment(
    path: Path | str,
    model: torch.nn.Module,
) -> torch.nn.Module:
    state_dict = torch.load(path, weights_only=False)
    model.load_state_dict(state_dict["model_state"])
    return model


def initialize_experiment(
    device: str,
    args_list: Optional[Sequence[str]] = None,
) -> tuple[Experiment, DictConfig]:
    """Initialize experiment by parsing the config file, checking that the
    repo is clean, creating a path for the experiment, and creating a
    writer for tensorboard.

    Arguments:
        device: device to run the experiment on.
        args_list: list of arguments to parse.

    Returns:
        experiment: experiment config object.
        config: DictConfig object containing the config.
    """

    args, extra_args = _parse_args(args_list)
    # If we are resuming, set the config_dir to the experiment path.
    if bool(args.resume_experiment):
        config_dir = os.path.join(
            os.path.abspath(args.root_results), args.resume_experiment
        )
        if os.path.exists(config_dir):
            # The config name is just the "config.yaml" file in that directory.
            args.config_name = "config"
            logging.info(f"Resuming experiment from {config_dir}")
        else:
            logging.warning(
                f"Experiment path {config_dir} does not exist, starting new experiment."
            )
            config_dir = None
    else:
        config_dir = None

    config = _get_config(
        args.config_module, args.config_name, extra_args, config_dir=config_dir
    )
    # Set the start_from_checkpoint flag if we found a config when resuming.
    config.info.start_from_checkpoint = config_dir is not None
    config.info.root_results = args.root_results

    logging.info("Config:\n" + OmegaConf.to_yaml(config))

    try:
        repo = git.Repo(search_parent_directories=True)
    except git.exc.InvalidGitRepositoryError:
        repo = None
        logging.warning("No git repository found, skipping repo checks.")
    # Check that the repo is clean, and up to date with remote branch,
    # if not in debug mode.
    if not args.debug and repo is not None:
        _assert_repo_is_clean(repo)
        # If config contains commit hash, assert that it matches the current commit hash.
        if "commit" in config:
            _assert_correct_commit_hash(config, repo)

    # If the config does not contain an experiment name, generate one.
    if config.info.experiment_name is None:
        experiment_name = (
            args.resume_experiment
            if args.resume_experiment
            else _generate_experiment_name()
        )

        experiment_path = _create_experiment_path(config, experiment_name)

        config.info.experiment_name = experiment_name
        config.info.experiment_path = experiment_path

        _write_config(config, experiment_path, repo)
    elif config.info.experiment_path is not None:
        # experiment_name was set via override - check if we need to create the path and write config
        experiment_path = config.info.experiment_path
        config_path = os.path.join(experiment_path, "config.yaml")
        if not os.path.exists(config_path):
            _create_experiment_path(config, config.info.experiment_name)
            _write_config(config, experiment_path, repo)

    logging.info("Instantiating dependencies.")

    config.model.device = device
    experiment: DictConfig = instantiate(config)
    return experiment, config


def _parse_args(
    args_list: Optional[Sequence[str]],
) -> tuple[argparse.Namespace, list[str]]:
    """
    Parse CLI arguments.

    Arguments:
        args_list: Optional list of arguments to parse.

    Returns:
        args: Known parsed arguments.
        extra_args: Additional arguments to be logged/stored.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--resume_experiment",
        type=str,
        required=False,
        help="Resume the experiment with the provided given name",
    )
    parser.add_argument(
        "--config_module", type=str, default="otter.experiment.config"
    )
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--root_results", type=str, default="_results")

    return parser.parse_known_args(args=args_list)


def initialize_wandb(config: DictConfig) -> None:
    config_dict: dict = OmegaConf.to_container(config)  # type: ignore
    experiment_name = config.info.experiment_name
    log_code = config.constants.log_code
    wandb.init(
        project="otter-deterministic",
        entity="otter-weather",
        name=experiment_name,
        id=experiment_name,
        config=config_dict,
    )
    if log_code:
        log_code_to_wandb()


def _assert_repo_is_clean(repo: git.Repo) -> None:
    if repo.is_dirty():
        raise AssertionError("Repo is dirty, please commit changes.")

    if _has_commits_ahead(repo):
        raise AssertionError("Repo has commits ahead, please push changes.")


def _has_commits_ahead(repo: git.Repo) -> bool:
    """Check if there are commits ahead in the local current branch.

    Arguments:
        repo: git repo object.

    Returns:
        has_commits_ahead: True if there are commits ahead, False otherwise.
    """
    if repo.head.is_detached:
        assert not repo.is_dirty(), "Repo is dirty, please commit changes."
        return False

    else:
        current_branch = repo.active_branch.name
        commits = list(
            repo.iter_commits(f"origin/{current_branch}..{current_branch}")
        )
        return len(commits) > 0


def _assert_correct_commit_hash(config: DictConfig, repo: git.Repo) -> None:
    """
    Asserts that the current commit hash matches the one stored in the config.
    """
    if "commit" not in config:
        raise ValueError("Config does not have a 'commit' field.")

    config_commit_hash = config.commit
    current_commit_hash = _get_current_commit_hash(repo)

    if config_commit_hash != current_commit_hash:
        raise ValueError(
            f"Config commit hash {config_commit_hash} does not match current commit hash {current_commit_hash}."
        )


def _get_config(
    config_module: str,
    config_name: str,
    extra_args: list[str],
    config_dir: Optional[str] = None,
) -> DictConfig:
    """
    Reads the config file, evaluates expressions, and merges it with overrides.

    When loading from a config_dir (e.g. when resuming an experiment), logs
    diagnostic information about the directory contents to aid debugging.
    """
    _register_resolvers()

    # Initialize experiment.
    # Here, we can either load from a config module or from a config directory.
    if config_dir is None:
        with hydra.initialize_config_module(
            config_module=config_module, version_base="1.3"
        ):
            config = compose(config_name=config_name, overrides=extra_args)
    else:
        # This is a folder-level override, and it won't exist in the reified config.yaml file.
        overrides = [a for a in extra_args if not a.startswith("host=")]
        try:
            with hydra.initialize_config_dir(
                config_dir=config_dir, version_base="1.3"
            ):
                config = compose(config_name=config_name, overrides=overrides)
        except Exception as e:
            logging.error(f"Failed to load config from {config_dir}: {e}")
            raise

    OmegaConf.set_struct(config, False)

    return config


def _register_resolvers() -> None:
    # Register custom resolvers if not already registered.
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    if not OmegaConf.has_resolver("get_module_attr"):
        OmegaConf.register_new_resolver("get_module_attr", get_module_attr)


def _get_current_commit_hash(repo: git.Repo) -> str:
    """Get the current commit hash of the local repo.

    Arguments:
        repo: git repo object.

    Returns:
        commit_hash: current commit hash.
    """
    if repo.head.is_detached:
        return repo.commit(repo.head.object).hexsha

    else:
        return repo.head.commit.hexsha


def _create_experiment_path(config: DictConfig, experiment_name: str) -> str:
    """Creates a path for the experiment, and create it if it doesn't exist.

    Arguments:
        config: config object.
        experiment_name: name of the experiment.
    Returns:
        experiment_path: path to the experiment.
    """

    path = os.path.join(
        config.info.root_results,
        experiment_name,
    )

    if not os.path.exists(path):
        logging.info(f"Making path for experiment results: {path}.")
        os.makedirs(path)

    return path


def _write_config(
    config: DictConfig, path: str, repo: git.Repo | None
) -> None:
    # Write config to file together with commit hash
    path = f"{path}/config.yaml"
    logging.info(f"Writing reified config to {path}.")
    with open(path, "w") as file:
        hash = (
            _get_current_commit_hash(repo) if repo is not None else "unknown"
        )
        config_to_save: DictConfig = OmegaConf.to_container(config)  # type: ignore
        config_to_save.update({"commit": hash})
        yaml.dump(config_to_save, file, indent=4, sort_keys=False)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def get_module_summary(module: torch.nn.Module, indent: int = 0) -> str:
    summary = ""
    total_params = count_parameters(module)
    summary += (
        f"{' ' * indent}Module: {module.__class__.__name__} "
        f"| Parameters: {total_params / 1e6:.1f}M\n"
    )
    for name, sub_module in module.named_children():
        summary += f"{' ' * (indent + 2)}Submodule: {name}\n"
        summary += get_module_summary(sub_module, indent + 2)
    return summary


def wandb_log_code_exclude_fn(path: str) -> bool:
    # Create regex patterns that:
    patterns = [
        # matches all paths starting with ".".
        re.compile(r"\..*"),
        # matches all paths starting with "_".
        re.compile(r"_.*"),
        # matches all paths containing "/wandb"
        re.compile(r"wandb.*"),
        # matches all paths containing "otter.egg-info"
        re.compile(r"otter.egg-info.*"),
        # matches the "tests" directory
        re.compile(r"tests.*"),
        # matches all paths containing "__pycache__"
        re.compile(r"__pycache__.*"),
    ]

    return any(bool(pattern.match(path)) for pattern in patterns)


def log_code_to_wandb() -> None:
    # Create code artifact to log to wandb.
    code_artifact = wandb.Artifact(name="source", type="code")
    # Walk through all files and directories from the current directory.
    for root, dirs, files in os.walk("."):
        # Filter out unwanted directories and files.
        dirs[:] = [d for d in dirs if not wandb_log_code_exclude_fn(d)]

        for file in files:
            file_path = os.path.join(root, file)
            # Get relative path to preserve directory structure.
            rel_path = os.path.relpath(file_path, start=".")

            # Only add non-excluded files.
            if not wandb_log_code_exclude_fn(rel_path):
                code_artifact.add_file(file_path, name=rel_path)

    # Log code artifact to wandb.
    wandb.log_artifact(code_artifact)


def load_std_dataset() -> xr.DataArray:
    from otter.data.normalisation.utils import load_statistic
    from otter.data.utils import (
        split_into_different_variables_along_dim,
        stack_dataset_variable_and_levels,
    )

    std_dataset = load_statistic("std")
    mean_dataset_split = split_into_different_variables_along_dim(
        std_dataset, "level"
    )
    return stack_dataset_variable_and_levels(mean_dataset_split)


def get_adamw_optimizer(
    module: torch.nn.Module, optimizer_kwargs: Dict[str, Any]
) -> torch.optim.AdamW:
    return torch.optim.AdamW(params=module.parameters(), **optimizer_kwargs)


class CombinedOptimizer(torch.optim.Optimizer):
    """A simple wrapper to combine multiple optimizers into one."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers
        self.param_groups = [
            pg for opt in optimizers for pg in opt.param_groups
        ]

    def step(self) -> None:  # type: ignore
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Resets the gradients of all optimized :class:`torch.Tensor` s."""
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer as a :class:`dict`."""
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the optimizer state."""
        optimizer_states = state_dict["optimizers"]
        if len(optimizer_states) != len(self.optimizers):
            raise ValueError(
                "Loaded state dict contains a different number of optimizers."
            )

        for opt, sd in zip(self.optimizers, optimizer_states):
            opt.load_state_dict(sd)

        # The child optimizers have replaced their param_groups objects
        # during load_state_dict, so we must update our references.
        self.param_groups = [
            pg for opt in self.optimizers for pg in opt.param_groups
        ]


def get_muon_optimizer(
    model: torch.nn.Module,
    muon_momentum: float,
    adamw_lr: float,
    adamw_betas: tuple[float, float],
    adamw_weight_decay: float,
) -> CombinedOptimizer:
    """Get a Muon optimizer for the given module."""
    muon_params = []
    adamw_params = []

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    num_muon_params = sum(p.numel() for p in muon_params)
    num_adamw_params = sum(p.numel() for p in adamw_params)
    logging.info(
        f"Using Muon optimizer for {num_muon_params / 1e6:.1f}M parameters"
    )
    logging.info(
        f"Using AdamW optimizer for {num_adamw_params / 1e6:.1f}M parameters"
    )

    opt_muon = torch.optim.Muon(
        muon_params,
        lr=adamw_lr,
        momentum=muon_momentum,
        weight_decay=adamw_weight_decay,
        # Should allow us to use AdamW LR and weight decay.
        adjust_lr_fn="match_rms_adamw",
    )

    opt_adamw = torch.optim.AdamW(
        adamw_params,
        lr=adamw_lr,
        betas=adamw_betas,
        weight_decay=adamw_weight_decay,
    )

    return CombinedOptimizer([opt_muon, opt_adamw])


def get_cosine_lr_scheduler(
    optimizer: torch.optim.Optimizer, lr_scheduler_kwargs: Dict[str, Any]
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, **lr_scheduler_kwargs
    )


def get_linear_lr_scheduler(
    optimizer: torch.optim.Optimizer, num_steps: int, end_factor: float
) -> torch.optim.lr_scheduler.LinearLR:
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=end_factor,
        total_iters=num_steps,
    )


def get_constant_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    factor: float,
    total_iters: int,
) -> torch.optim.lr_scheduler.ConstantLR:
    return torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=factor, total_iters=total_iters
    )


def get_sequential_warmup_cosine_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    eta_min: float,
) -> torch.optim.lr_scheduler.SequentialLR:
    """
    Returns a scheduler that warms up linearly and then decays using cosine annealing.
    Uses native torch.optim.lr_scheduler.SequentialLR.
    """
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
    )

    decay_steps = max(1, total_steps - warmup_steps - 1)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=decay_steps, eta_min=eta_min
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
