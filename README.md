# Otter

## Getting started
The project uses uv to manage dependencies.
If you haven't installed uv you can do so by
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then clone the repo, go to its root, create a virtual environment and sync dependencies.
```bash
git clone git@github.com:cambridge-mlg/otter.git
cd otter
uv venv
source .venv/bin/activate
uv sync
```
Last, if you want to commit code you probably want to install pre-commit hooks, to run pre-commit checks before each commit.
```bash
pre-commit install
```
To download and pre-process the ERA5 data:
```bash
# This will download the 64x32 and 240x121 grid ERA5 data from WeatherBench2.
# The larger of the two datasets, the 240x121 grid, requires about 660GB of storage.
# By default, the data will be stored under ./_data in the repo root.
python otter/data/source/download_weatherbench_data.py

# This will post-process convert it to pure-numpy format for faster loading.
python otter/data/source/convert_saved_zarr_to_memmap.py

# Normalisation factors for the ERA5 data are already computed and version controlled
# in the package, but if you want to re-generate the factors you can use the command
# python otter/data/normalisation/compute_stats.py
```

## Training

To train a deterministic model, from the repo root, run:
```bash
# This will train a model with logging to wandb. It will also check your local branch
# is clean, i.e. that it has no changes relative to the remote branch.
wandb online; python otter/experiment/train.py --config_name base

# You can disable this check by adding the flag --debug at the end of the command, as in
wandb online; python otter/experiment/train.py --config_name base --debug
```


### RFT
You can rollout finetune a pretrained model using:
```bash
python otter/experiment/rollout_finetune.py <experiment_name> <other overrides>
```

### One-step MC-dropout fine-tuning

Convert a deterministic `base` checkpoint to a six-hour dropout ensemble through
the normal training entry point. Run from the repo root with the prepared ERA5
data and exactly one visible CUDA device. Replace `deterministic-experiment`
with the source experiment's name:

```bash
python -m otter.experiment.train --config_name dropout_crps --debug \
  constants.load_weights_from_experiment=_results/deterministic-experiment/checkpoints/latest.pt \
  constants.ensemble.train_members=2 \
  constants.ensemble.validation_members=8 \
  constants.data.memory_max_batch_size=1 \
  constants.data.accumulate_batches=4
```

The checkpoint override names a **file** whose `model_state` loads strictly into
the unchanged Delta head and backbone, starting a new experiment with a fresh
optimizer. Match the source architecture, variables, context, grid and
normalization settings if it differs from `base`. `--debug` permits uncommitted
changes. The command syntax was reviewed against the CLI; its parsing and
overrides have tests in `tests/configs/test_dropout_crps.py`.

Members, examples per minibatch (`memory_max_batch_size`) and accumulated
minibatches (`accumulate_batches`) are separate controls. The example draws two
members per example and accumulates four one-example minibatches per update.
Each member runs the complete shared model on the same context. Configure dropout
with `constants.training.dropout_rate`: it executes in SwiGLU's gated hidden
activations and attention output projections. Stochastic depth and
attention-probability dropout are zero.

Training and validation each request one target: six hours. These are controlled
by `constants.data.temporal_resolution_hours`, `train_num_target_frames` and
`valid_num_target_frames`; `rollout_horizon_scheduler` also fixes both its minimum
and maximum to one step. `constants.num_evaluation_rollout_hours` is six as well.
The model returns physical units, so fair CRPS applies latitude weighting once,
then the existing pressure, surface-variable and inverse-standard-deviation
weights once, with coefficient power one. Validation reports weighted
`val/fair_crps`, ensemble-mean RMSE and ensemble spread (the spatial RMS of the
unbiased member standard deviation). RMSE and spread also have per-variable
diagnostics in physical units. `best.pt` is selected using forecast fair CRPS.

Resume the resulting fine-tuning experiment by its **new** experiment name:

```bash
python -m otter.experiment.train \
  --resume_experiment dropout-finetuning-experiment --debug
```

Resuming restores that run's model, optimizer, schedulers, gradient scaler and
training state, skipping the original source checkpoint. Ensemble losses receive
individual members and the full `ForecastingSample`, including its context, for
future objectives. Rollout fine-tuning, MMD-FUSE and latent-space objectives are
outside this recipe.

For ensemble inference with a loaded fine-tuned model and a batch requesting
only lead time six hours:

```python
import torch
from otter.models.ensemble import predict_members

model.eval()
with torch.no_grad():
    members = predict_members(model, batch, num_members=8)
# [batch, member, latitude, longitude, 1, variable_and_level]
```

The helper temporarily enables the dropout layers and restores their previous
modes afterward. Reset the seed before the whole call to reproduce an ensemble
draw in a supported test environment.

Verification in this change is limited to static review: no training or tests
were run on this laptop, and no real pretrained checkpoint was available.
The new tests define fixture-based conversion, dropout, fair-CRPS and resume
checks. On a suitable training machine, run them and the conversion command above
with a real matching checkpoint to verify acceptance:

```bash
TORCH_COMPILE_DISABLE=1 python -m pytest \
  tests/models/test_ensemble.py tests/models/test_crps.py \
  tests/configs/test_dropout_crps.py tests/experiment/test_dropout_training.py \
  tests/experiment/test_run_state.py
```

The optional existing deterministic configuration checks require weather test
data and network access:

```bash
OTTER_RUN_ALL_TESTS=true TORCH_COMPILE_DISABLE=1 python -m pytest \
  tests/configs/test_config_deterministic.py
```

### 0.25-degree training
The 0.25-degree (1440x721) configs read the WeatherBench2 ERA5 zarr from `../otter-data/era5_1440x721.zarr`; override `data_source.path` to read it from elsewhere. Train from scratch, then rollout finetune from the final checkpoint:
```bash
python otter/experiment/train.py --config_name quarter_res_scratch

python otter/experiment/train.py --config_name quarter_res_rollout_finetune \
  constants.load_weights_from_experiment=_results/<experiment_name>/checkpoints/latest.pt
```

## Weatherbench Eval

After a model has been trained, it can be used to generate cached forecasts for later evaluation. To do so for a given `experiment_name`, run:
```bash
python otter/evaluate/save_forecasts.py <experiment_name>
```
The experiment name is the same as in wandb or in the _results directory, e.g. `2025-11-06_19-08-bold-wolf`.
For a run trained on another machine, point it at a local copy of the data with `--data_path`.

For weatherbench model evaluation (to save evaluation metrics), run:
```bash
python otter/evaluate/weatherbench.py --experiment_name=<experiment_name>
```

Or in one command:
```bash
EXP_NAME=<experiment_name> && CUDA_VISIBLE_DEVICES=<> python otter/evaluate/save_forecasts.py $EXP_NAME && python otter/evaluate/weatherbench.py --experiment_name=$EXP_NAME --lead_time_stop 30
```


Finally, these results can be plotted and saved to the _plots folder using
```
python otter/evaluate/plot_weatherbench_output.py <experiment_name>
```

## Pretrained models on Hugging Face

Trained models can be published on the Hugging Face Hub. Each Hub repo holds one experiment's weights, its config, a model card, and a reference forecast made with the original training checkpoint.

[`examples/forecast.ipynb`](examples/forecast.ipynb) goes from nothing to a scored, plotted forecast.

To forecast with a published model, set up the repo as in [Getting started](#getting-started) and run, on a machine with a CUDA GPU:
```bash
python -m otter.hub forecast <org>/<name> --init_time 2020-07-01T00 --output forecast.zarr
```
This reads the initial conditions from the public WeatherBench2 ERA5 and writes a 10-day forecast, following WeatherBench2 conventions. Add `--end_init_time` to forecast from a range of inits, `--lead_time_hours`, `--headline_only` and `--lead_frequency_hours` to save less, and `--revision <commit hash>` to pin the model version. The config is instantiated with Hydra, which can run arbitrary code, so only load repos you trust.

To publish an experiment, log in with `hf auth login` (or set `HF_TOKEN`), then, on a machine with a GPU, run:
```bash
python -m otter.hub export <experiment_name> hub_export/<name> \
  --reference_init_time 2020-07-01T00 --push_to_hub <org>/<name>
```
The repo is created private; pass `--public` to create it public. Fill in the generated model card (`README.md` in the repo) with the model's scores and its training data. Then check that the published model reproduces its reference forecast:
```bash
OTTER_HUB_TEST_REPOS=<org>/<name> pytest tests/hub/test_hub_e2e.py -s
```

## Citation

```bibtex
@article{diaconu2026otter,
  title={Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting},
  author={Diaconu, Cristiana and Scholz, Jonas and Shysheya, Aliaksandra and Markou, Stratis and Mukhopadhyay, Payel and Cranmer, Miles and Turner, Richard E},
  journal={arXiv preprint arXiv:2606.26421},
  year={2026}
}
```

## License

MIT; see [LICENSE](LICENSE). The published model weights are under the same licence.
