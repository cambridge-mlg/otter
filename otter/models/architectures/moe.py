from dataclasses import dataclass
from typing import (
    Optional,
    Tuple,
)

import torch
import torch.distributions as dist

from otter.models.architectures.mlp import MLP


@dataclass
class LoadBalancingLosses:
    importance_loss: torch.Tensor
    load_loss: torch.Tensor


def _softmax_then_top_k(
    logits: torch.Tensor, k: int, dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.topk(torch.softmax(logits, dim=dim), k=k, dim=dim)


def _top_k_then_softmax(
    logits: torch.Tensor, k: int, dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    top_k_logits, top_k_indices = torch.topk(logits, k=k, dim=dim)
    return torch.softmax(top_k_logits, dim=dim), top_k_indices


class MixtureOfExperts(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        output_features: int,
        num_experts: int,
        hidden_features: int,
        num_hidden_layers: int,
        k: int,
        use_gating_bias: bool = False,
        apply_softmax_before_top_k: bool = False,
        w_importance: float = 1.0,
        w_load: float = 1.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        # NOTE: current implementation does not support different input and
        # output dimensions.
        assert in_features == output_features
        self.in_features = in_features

        # Expert MLPs.
        self.experts = torch.nn.ModuleList(
            [
                MLP(
                    in_features=in_features,
                    num_hidden_layers=num_hidden_layers,
                    hidden_features=hidden_features,
                    output_features=output_features,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_experts)
            ]
        )
        self.num_experts = num_experts

        # Layer for gating the experts.
        self.linear_layer = torch.nn.Linear(
            in_features, num_experts, bias=use_gating_bias
        )

        # Layer for adding noise to the gating logits.
        self.linear_gate_noise_layer = torch.nn.Linear(
            in_features, num_experts, bias=use_gating_bias
        )
        self.gaussian_noise_std = num_experts**-0.5

        # Softplus activation for the noise layer.
        self.softplus = torch.nn.Softplus()

        # Whether to apply softmax before or after the top-k operation.
        self.apply_softmax_before_top_k = apply_softmax_before_top_k

        self.w_importance = w_importance
        self.w_load = w_load

        self.k = k

    def noisy_gate_logits(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Computes noise-perturbed gating logits for each expert.

        The gating logits are computed according to the formula:

            logit = (Wg * x + bg) + (epsilon * softplus(Wn * x + bn))

        where Wg and Wn are learnable weight matrices, bg and bn are learnable
        bias vectors, which are used only if use_gating_bias is True, and
        epsilon is standard Gaussian noise.

        Args:
            x (torch.Tensor): Input tensor of shape (token_batch, token_dim).

        Returns:
            torch.Tensor: Gating logits of shape (token_batch, num_experts).
        """

        # Apply gating layer.
        logits: torch.Tensor = self.linear_layer(x)

        if self.training:
            # Draw standard Gaussian noise and scale by input-dependent factor.
            noise: torch.Tensor = (
                torch.randn(*logits.shape, device=x.device)
                * self.gaussian_noise_std
            )
            noise_std: torch.Tensor = self.softplus(
                self.linear_gate_noise_layer(x)
            )
            noise = noise * noise_std

            return logits + noise, noise_std, logits

        return logits, None, None

    def importance_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """Computes importance loss weighted by w_importance.

        Loss is computed as follows:

            Imp = sum_i gate_probs[i] # summation over batch dimension
            Loss_importance = w_importance * var(Imp)/mean(Imp)**2

        Args:
            gate_probs (torch.Tensor): matrix of expert weights of shape
            (token_batch, num_experts).

        Returns:
            torch.Tensor: Resulting importance loss.
        """
        assert gate_probs.dim() == 2
        assert gate_probs.shape[1] == self.num_experts

        eps = 1e-10
        importance = gate_probs.sum(dim=0)  # shape (num_experts,)
        loss_importance: torch.Tensor = (
            self.w_importance
            * importance.var()
            / (importance.mean() ** 2 + eps)
        )
        return loss_importance

    def load_loss(
        self,
        gate_logits: torch.Tensor,
        noise_std: torch.Tensor | None,
        gate_logits_without_noise: torch.Tensor | None,
        gates_khot: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Computes load loss weighted by w_load.

        Loss is computed as follows:

            P(x) = CDF(((Wg * x + bg) - kth_excluding(H(x), k, i))/
            Softplus(Wn * x + bn))

            Load = sum_x P(x) # summation over batch dimension
            Loss_load = w_load * (std(Load)/mean(Load))^2

        Args:
            gate_logits_dict (Dict[str, torch.Tensor]): dict containing the
            output of self.noisy_gate_logits.
            gates_khot (torch.Tensor): matrix of khot representation of the
            active gates (token_batch, num_experts).
            k (int): Number of experts to route each token to.

        Returns:
            torch.Tensor: Resulting load loss.
        """

        assert gates_khot.dim() == 2
        assert gates_khot.shape[1] == self.num_experts
        assert k < self.num_experts

        if self.training:
            assert gate_logits_without_noise is not None
            assert noise_std is not None

            # NOTE: this topk could probably be avoided, but it requires
            # splitting _softmax_then_top_k function
            top_k_plus_one_logits, _ = torch.topk(
                gate_logits, k=k + 1, dim=1
            )  # shape (token_batch, k+1)
            # kth_excluding is a tensor of shape (token_batch, num_experts)
            # whose element at indices [i, j] contains the kth largest entry
            # of gate_logits excluding the entry at index j itself.
            # Therefore, we have two cases:
            #
            # 1. if the gate is inactive, gates_khot[i, j] is 0, and the kth
            #    largest entry is in top_k_plus_one_logits[:, -2]
            #
            # 2. if the gate is active, gates_khot[i, j] is 1, and the kth
            #    largest entry is in top_k_plus_one_logits[:, -1]
            #
            #  The following expression evaluates this in parallel.
            in_logits = top_k_plus_one_logits[:, -1:]  # shape (token_batch, 1)
            out_logits = top_k_plus_one_logits[
                :, -2:-1
            ]  # shape (token_batch, 1)
            kth_excluding = (
                in_logits * gates_khot + (1 - gates_khot) * out_logits
            )  # shape (token_batch, num_experts)
            cdf_input = (gate_logits_without_noise - kth_excluding) / noise_std
            probs = dist.Normal(0.0, 1.0).cdf(cdf_input)
            load = probs.sum(dim=0)  # shape (num_experts,)
        else:
            # If the model is not in the training mode, the gate probabilities
            # are deterministic and the load is computed using gates_khot
            # tensor, as we do not need it to be differentiable.
            load = gates_khot.sum(dim=0).type(
                gate_logits.dtype
            )  # shape (num_experts,)

        eps = 1e-10
        loss_load: torch.Tensor = (
            self.w_load * load.var() / (load.mean() ** 2 + eps)
        )
        return loss_load

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[LoadBalancingLosses]]:
        """Performs forward pass through mixture of experts layer.

        NOTE: this layer includes a resuidual connection by default.

        Args:
            x (torch.Tensor): Input tensor of shape (token_batch, token_dim).

        Returns:
            torch.Tensor: Output tensor of shape (token_batch, token_dim).
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input tensor has shape {x.shape}, "
                f"expected {self.in_features}"
            )

        k = self.k
        pred = x.reshape(
            -1, self.in_features
        )  # shape (token_batch, token_dim)

        # Compute noisy gate logits.
        gate_logits, noise_std, logits_wo_noise = self.noisy_gate_logits(pred)

        # Apply keep_top_k operation and softmax, or the other way around,
        # to obtain sparse gate probabilities. By sparse, we mean that only
        # the top-k probabilities are returned.
        gate_probs, gate_indices = (
            _softmax_then_top_k(gate_logits, k=k, dim=1)
            if self.apply_softmax_before_top_k
            else _top_k_then_softmax(gate_logits, k=k, dim=1)
        )  # shape (token_batch, k), (token_batch, k)

        # Scatter gate probabilities, converting them to a tensor of shape
        # (token_batch, num_experts), that contains all zeroes except for
        # the top-k probabilities for each token.
        gate_probs_expanded = torch.zeros_like(
            gate_logits, device=pred.device, requires_grad=True
        )  # shape (token_batch, num_experts)
        gate_probs_expanded = torch.scatter_add(
            gate_probs_expanded, dim=1, index=gate_indices, src=gate_probs
        )  # shape (token_batch, num_experts)

        # Scatter the gate indices, converting them to a tensor of shape
        # (token_batch, num_experts), that contains a k-hot representation
        # of the active gates. This is used for indexing the tokens later.
        gates_khot = torch.zeros(
            pred.shape[0],
            self.num_experts,
            dtype=torch.int32,
            device=pred.device,
        )  # shape (token_batch, num_experts)

        ones = torch.ones(
            *gate_indices.shape, dtype=torch.int32, device=pred.device
        )  # shape (token_batch, k)
        gates_khot.scatter_(
            dim=1, index=gate_indices, src=ones
        )  # shape (token_batch, num_experts)

        output = torch.zeros_like(pred, device=pred.device, requires_grad=True)

        for i, expert in enumerate(self.experts):
            # Get booleans encoding whether each token goes to the i-th expert.
            token_in_expert = gates_khot[:, i].bool()  # shape (token_batch,)

            # Apply the i-th expert to its tokens.
            expert_output, _ = expert(
                pred[token_in_expert]
            )  # shape (expert_tokens, token_dim)

            # Compute the product of the result of the expert
            # and the gate probabilities corresponding to the tokens.
            probs = gate_probs_expanded[:, i]  # shape (token_batch,)
            probs = probs[token_in_expert]  # shape (expert_tokens,)

            product = expert_output * probs[:, None]

            # Repeat index to match shape of product along second dimension.
            index = torch.nonzero(token_in_expert)
            index = index.repeat(1, product.shape[1])

            # Scatter the product back into the output tensor.
            output = torch.scatter_add(output, dim=0, index=index, src=product)

        output = output.reshape(x.shape)

        # Compute importance and load loss
        importance_loss = self.importance_loss(gate_probs_expanded)
        load_loss = self.load_loss(
            gate_logits=gate_logits,
            noise_std=noise_std,
            gate_logits_without_noise=logits_wo_noise,
            gates_khot=gates_khot,
            k=k,
        )

        load_balancing_losses = LoadBalancingLosses(
            importance_loss=importance_loss,
            load_loss=load_loss,
        )
        return output, load_balancing_losses
