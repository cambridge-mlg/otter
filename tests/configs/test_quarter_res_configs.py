"""Checks that the 0.25-degree configs compose and build a model."""

import pytest
from hydra.utils import instantiate

from otter.data.utils import ZarrDatasource
from otter.experiment.utils import _get_config


@pytest.mark.parametrize(
    "config_name",
    ["quarter_res_scratch", "quarter_res_rollout_finetune"],
)
def test_quarter_res_config(config_name: str) -> None:
    config = _get_config(
        "otter.experiment.config",
        config_name,
        [
            # Its data path must not leak into the quarter configs.
            "host=default",
            # Reduce model size to keep test fast and prevent OOMs.
            "model.backbone_network.token_dimensions=[16,16,16]",
            "model.backbone_network.num_heads=2",
            "model.backbone_network.num_blocks_per_stage=[1,1]",
            "model.backbone_network.num_bottom_blocks=1",
        ],
    )

    assert config.data_source._target_ == (
        f"{ZarrDatasource.__module__}.ZarrDatasource.from_path"
    )
    assert config.data_source.path.endswith("era5_1440x721.zarr")
    assert config.model.data_grid_num_lat == 721
    assert config.model.data_grid_num_lon == 1440
    # 721 x 1440 maps onto the 1.5-degree model's 120 x 240 token grid.
    assert config.model.backbone_network.patch_size == 6
    assert list(config.model.backbone_network.patchify_kernel_size) == [7, 6]
    assert config.constants.data.num_context_frames == 4

    config.model.device = "cpu"
    instantiate(config.model)
