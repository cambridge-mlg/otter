from functools import partial

import pytest
import torch

from otter.models.architectures import attention
from otter.models.architectures.attention import EfficientMultiHeadAttention
from otter.models.architectures.embeddings import SpatialEmbedding
from otter.models.architectures.mlp import MLP
from otter.models.architectures.transformer import SwinTransformer


@pytest.mark.parametrize(
    "training, expected_p",
    [(True, 0.25), (False, 0.0)],
)
def test_attention_dropout_only_in_training(
    monkeypatch: pytest.MonkeyPatch, training: bool, expected_p: float
) -> None:
    dropout_probabilities = []

    def fake_attention(q, k, v, attn_bias=None, p=0.0):
        dropout_probabilities.append(p)
        return torch.zeros_like(q)

    monkeypatch.setattr(
        attention, "memory_efficient_attention", fake_attention
    )

    layer = EfficientMultiHeadAttention(
        num_heads=2, token_dim=8, attn_drop=0.25
    )
    layer.train(training)
    x = torch.randn(3, 5, 8)
    layer(x, x, x)

    assert dropout_probabilities == [expected_p]


def test_swin_transformer_attention_dropout_rate() -> None:
    model = SwinTransformer(
        in_channels=3,
        height=8,
        width=8,
        patch_size=1,
        patchify_kernel_size=(1, 1),
        token_dimensions=8,
        output_dimension=3,
        window_size=4,
        num_heads=2,
        num_blocks_per_stage=[1],
        num_bottom_blocks=1,
        pos_embedding=SpatialEmbedding(8, 0.1, 720, 8),
        feedforward_network=partial(
            MLP, hidden_features=16, num_hidden_layers=1
        ),
        dropout_rate=0.1,
        drop_path_rate=0.0,
        use_rope=False,
        attn_dropout_rate=0.25,
    )
    layers = [
        module
        for module in model.modules()
        if isinstance(module, EfficientMultiHeadAttention)
    ]

    assert layers
    assert all(layer.attn_drop == 0.25 for layer in layers)
    assert all(layer.proj_drop.p == 0.1 for layer in layers)
