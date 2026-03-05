import pytest
import torch

from otter.models.architectures.moe import (
    MixtureOfExperts,
    _softmax_then_top_k,
    _top_k_then_softmax,
)


@pytest.mark.parametrize(
    "token_dim, num_experts, num_hidden_layers, "
    "hidden_features, use_gating_bias, apply_softmax_before_top_k,"
    " w_importance, w_load, k, token_batch_dim",
    [
        (64, 16, 2, 32, False, False, 1.0, 1.0, 8, 16),
        (32, 4, 2, 32, True, False, 0.5, 0.0, 3, 8),
        (32, 4, 2, 32, True, False, 0.5, 0.5, 1, 8),
    ],
)
def test_moe_forward(
    token_dim: int,
    num_experts: int,
    num_hidden_layers: int,
    hidden_features: int,
    use_gating_bias: bool,
    apply_softmax_before_top_k: bool,
    w_importance: float,
    w_load: float,
    k: int,
    token_batch_dim: int,
) -> None:
    moe = MixtureOfExperts(
        in_features=token_dim,
        output_features=token_dim,
        num_experts=num_experts,
        num_hidden_layers=num_hidden_layers,
        hidden_features=hidden_features,
        use_gating_bias=use_gating_bias,
        apply_softmax_before_top_k=apply_softmax_before_top_k,
        k=k,
        w_importance=w_importance,
        w_load=w_load,
    )
    x = torch.rand((token_batch_dim, token_dim))
    output, _ = moe(x)
    expected_shape = (token_batch_dim, token_dim)
    assert output.shape == expected_shape


@pytest.mark.parametrize(
    "token_dim, num_experts, num_hidden_layers, "
    "hidden_features, use_gating_bias, apply_softmax_before_top_k,"
    " w_importance, w_load, k, token_batch_dim",
    [
        (64, 16, 2, 32, False, False, 1.0, 1.0, 8, 4),
        (32, 16, 2, 32, True, False, 3.0, 0.0, 10, 8),
        (32, 4, 2, 32, False, False, 0.0, 0.5, 2, 8),
    ],
)
def test_output_computation(
    token_dim: int,
    num_experts: int,
    num_hidden_layers: int,
    hidden_features: int,
    use_gating_bias: bool,
    apply_softmax_before_top_k: bool,
    w_importance: float,
    w_load: float,
    k: int,
    token_batch_dim: int,
) -> None:
    moe = MixtureOfExperts(
        in_features=token_dim,
        output_features=token_dim,
        num_experts=num_experts,
        num_hidden_layers=num_hidden_layers,
        hidden_features=hidden_features,
        use_gating_bias=use_gating_bias,
        apply_softmax_before_top_k=apply_softmax_before_top_k,
        k=k,
        w_importance=w_importance,
        w_load=w_load,
    ).eval()
    x = torch.rand((token_batch_dim, token_dim))
    output_moe, _ = moe(x)

    logits, _, _ = moe.noisy_gate_logits(x)

    gate_probs, gate_indices = (
        _softmax_then_top_k(logits, k=k, dim=1)
        if apply_softmax_before_top_k
        else _top_k_then_softmax(logits, k=k, dim=1)
    )  # shape (token_batch, k), (token_batch, k)

    output = torch.zeros_like(x)
    for i in range(token_batch_dim):
        for prob, idx in zip(gate_probs[i], gate_indices[i]):
            expert_output, _ = moe.experts[idx](x[i].unsqueeze(0))
            output[i] += prob * expert_output.squeeze(0)

    assert torch.allclose(output, output_moe, atol=1e-6)
