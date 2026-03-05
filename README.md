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

## Weatherbench Eval

After a model has been trained, it can be used to generate cached forecasts for later evaluation. To do so for a given `experiment_name`, run:
```bash
python otter/evaluate/save_forecasts.py <experiment_name>
```
The experiment name is the same as in wandb or in the _results directory, e.g. `2025-11-06_19-08-bold-wolf`.

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
