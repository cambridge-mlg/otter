from typing import (
    Callable,
    List,
    Optional,
    Tuple,
)

import torch
from torch import nn

from otter.models.architectures.attention import EfficientMultiHeadAttention
from otter.models.architectures.moe import LoadBalancingLosses


def extract_subimages(x: torch.Tensor, subimage_size: int) -> torch.Tensor:
    """Extract non-overlapping windows from input tensor `x`.

    Arguments:
        x: input tensor of shape
        subimage_size: size of the extracted subimages

    Returns:
        output tensor of shape (B, H//S, S, W//S, S, D)
    """
    B, H, W, D = x.shape
    S = subimage_size

    assert H % S == 0, (
        f"Height {H} must be divisible by window size {S} ({x.shape=})"
    )
    assert W % S == 0, (
        f"Width {W} must be divisible by window size {S} ({x.shape=})"
    )

    return torch.reshape(x, [B, H // S, S, W // S, S, D])


def extract_subimages_and_fold_into_batch_dim(
    x: torch.Tensor,
    subimage_size: int,
) -> torch.Tensor:
    """Extract non-overlapping windows from input tensor `x` and fold the
    spatial dimensions into the batch dimension.

    Arguments:
        x: input tensor of shape (B, H, W, D)
        subimage_size: size of the extracted subimages

    Returns:
        output tensor of shape (B*(H//S)*(W//S), S**2, D)
    """
    B, H, W, D = x.shape
    S = subimage_size
    x = extract_subimages(x, subimage_size)  # (B, H//S, S, W//S, S, D)
    x = torch.permute(x, [0, 1, 3, 2, 4, 5])  # (B, H//S, W//S, S, S, D)
    return torch.reshape(x, [B * (H // S) * (W // S), S**2, D])


def extract_subimages_and_fold_into_token_dim(
    x: torch.Tensor,
    subimage_size: int,
) -> torch.Tensor:
    """Extract non-overlapping windows from input tensor `x` and fold the
    spatial dimensions into the token dimension.

    Arguments:
        x: input tensor of shape (B, H, W, D)
        subimage_size: size of the extracted subimages

    Returns:
        output tensor of shape (B, H, W, D)
    """
    B, H, W, D = x.shape
    S = subimage_size
    x = extract_subimages(x, subimage_size)  # (B, H//S, S, W//S, S, D)
    x = torch.permute(x, [0, 1, 3, 2, 4, 5])  # (B, H//S, W//S, S, S, D)
    return torch.reshape(x, [B, H // S, W // S, S**2 * D])


def combine_subimages(
    x: torch.Tensor,
    original_shape: torch.Size | Tuple[int, int, int, int],
    subimage_size: int,
) -> torch.Tensor:
    """Combine subimages extracted from input tensor `x`.

    Arguments:
        x: input tensor of shape (B, H//S, S, W//S, S, D)
        original_shape: shape of original tensor

    Returns:
        output tensor of shape (B, H, W, D)
    """
    B, H, W, D = original_shape
    S = subimage_size

    assert H % S == 0, f"Height {H} must be divisible by window size {S}"
    assert W % S == 0, f"Width {W} must be divisible by window size {S}"

    return torch.reshape(x, [B, H, W, D])


def unfold_from_batch_dim_and_combine_subimages(
    x: torch.Tensor,
    original_shape: torch.Size | Tuple[int, int, int, int],
    subimage_size: int,
) -> torch.Tensor:
    """Unfold the batch dimension of input tensor `x` and combine
    the subimages.

    Arguments:
        x: input tensor of shape (B*(H//S)*(W//S), S**2, D)
        original_shape: shape of original tensor
        subimage_size: size of the extracted subimages

    Returns:
        output tensor of shape (B, H, W, D)
    """

    B, H, W, D = original_shape
    S = subimage_size

    x = torch.reshape(x, [B, H // S, W // S, S, S, D])
    x = torch.permute(x, [0, 1, 3, 2, 4, 5])  # (B, H//S, S, W//S, S, D)
    return combine_subimages(x, original_shape, S)


def unfold_from_token_dim_and_combine_subimages(
    x: torch.Tensor,
    original_shape: torch.Size | Tuple[int, int, int, int],
    subimage_size: int,
) -> torch.Tensor:
    """Unfold the token dimension of input tensor `x` and combine
    the subimages.

    Arguments:
        x: input tensor of shape (B, H // S, W // S, S**2 * D)
        original_shape: shape of original tensor
        window_size: size of the extracted subimages

    Returns:
        output tensor of shape (B, H, W, D)
    """

    B, H, W, D = original_shape
    S = subimage_size

    x = torch.reshape(x, [B, H // S, W // S, S, S, D])
    x = torch.permute(x, [0, 1, 3, 2, 4, 5])  # (B, H//S, S, W//S, S, D)
    return combine_subimages(x, original_shape, S)


def shift_horizontally_and_vertically(
    x: torch.Tensor, shift: int
) -> torch.Tensor:
    """Shift windows in the input tensor `x` by shift along its width and
    height. For example, using shift == 1 (and fixing an index for the B and D
    dimensions), the corresponding image would change as follows:

                      Original                Shifted
                 -----------------       -----------------
                |  x   x   x   o  |     |  *   +   +   +  |
                |  x   x   x   o  |     |  o   x   x   x  |
                |  x   x   x   o  |     |  o   x   x   x  |
                |  +   +   +   *  |     |  o   x   x   x  |
                 -----------------       -----------------

    Arguments:
        x: input tensor of shape (B, H, W, D)
        shift: amount of shift to apply

    Returns:
        output tensor of shape (B, H, W, D)
    """
    if len(x.shape) != 4:
        raise ValueError("Input tensor must have shape (B, H, W, D)")
    return torch.roll(x, (shift, shift), dims=(1, 2))


def shifted_window_pole_bias(
    height: int,
    width: int,
    window_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Attention bias for SwinTransformerBlock's shifted windows that blocks
    attention across the poles, but not across the longitude wrap.

    Returns:
        (H//S * W//S, S**2, S**2) bias: -inf where attention is blocked, 0
        elsewhere.
    """
    shift = window_size // 2
    # After the shift, the first window row holds the last `shift` rows of
    # the grid above its first `window_size - shift` rows.
    row_labels = torch.zeros(height, device=device)
    row_labels[:shift] = 1
    row_labels[shift:window_size] = 2
    labels = row_labels[None, :, None, None].expand(1, height, width, 1)
    windows = extract_subimages_and_fold_into_batch_dim(labels, window_size)
    blocked = windows != windows.transpose(1, 2)  # (num_windows, S**2, S**2)
    return torch.zeros(blocked.shape, dtype=dtype, device=device).masked_fill(
        blocked, float("-inf")
    )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        feedforward_network: Callable[[int, int], nn.Module],
        num_heads: int,
        dropout_rate: float,
        drop_path_rate: float,
        attn_dropout_rate: float = 0.0,
        use_efficient_attention: bool = True,
    ):
        super().__init__()

        self.mhsa: Callable[
            [torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
            torch.Tensor,
        ]

        if use_efficient_attention:
            self.efficient_mha_layer = EfficientMultiHeadAttention(
                num_heads=num_heads,
                token_dim=token_dim,
                attn_drop=attn_dropout_rate,
                proj_drop=dropout_rate,
            )
            self.mhsa = lambda x, freq_cis, attn_bias: (
                self.efficient_mha_layer(
                    x, x, x, freq_cis=freq_cis, attn_bias=attn_bias
                )
            )

        else:
            self.mha_layer = nn.MultiheadAttention(
                embed_dim=token_dim,
                num_heads=num_heads,
                dropout=attn_dropout_rate,
            )
            self.mhsa = lambda x, _, __: (self.mha_layer(x, x, x)[0])

        self.drop_path_rate = drop_path_rate
        self.ffn = feedforward_network(token_dim, token_dim)

        self.ln1 = nn.LayerNorm(token_dim)
        self.ln2 = nn.LayerNorm(token_dim)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[LoadBalancingLosses]]:
        """Apply the transformer block to input tokens `x`.

        Args:
            x: input tensor of shape (B, N, D)

        Returns:
            output tensor of shape (B, N, D)
        """
        # Do drop-path at the block level to skip computation during training.
        if self.training and self.drop_path_rate > 0:
            if torch.rand(1) < self.drop_path_rate:
                return x, None

            scale_factor = 1.0 / (1.0 - self.drop_path_rate)
        else:
            scale_factor = 1.0

        x = x + self.mhsa(self.ln1(x), freq_cis, attn_bias) * scale_factor

        x_ffn, load_balancing_losses = self.ffn(self.ln2(x))
        x = x + x_ffn * scale_factor

        return x, load_balancing_losses


@torch.compile()
class SwinTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        token_dim: int,
        feedforward_network: Callable[[int, int], nn.Module],
        num_heads: int,
        window_size: int,
        dropout_rate: float,
        drop_path_rate: float,
        attn_dropout_rate: float = 0.0,
        use_attention_mask: bool = False,
        use_efficient_attention: bool = True,
    ):
        super().__init__()

        if use_attention_mask and not use_efficient_attention:
            raise ValueError(
                "use_attention_mask requires use_efficient_attention."
            )
        # Whether the shifted windows block attention across the poles.
        self.use_attention_mask = use_attention_mask

        # Transformer block applied before the shift operation.
        self.first_block = TransformerBlock(
            token_dim=token_dim,
            feedforward_network=feedforward_network,
            num_heads=num_heads,
            use_efficient_attention=use_efficient_attention,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            attn_dropout_rate=attn_dropout_rate,
        )

        # Transformer block applied after the shift operation.
        self.second_block = TransformerBlock(
            token_dim=token_dim,
            feedforward_network=feedforward_network,
            num_heads=num_heads,
            use_efficient_attention=use_efficient_attention,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            attn_dropout_rate=attn_dropout_rate,
        )

        # Window size for extracting subimages.
        self.window_size = window_size

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, List[LoadBalancingLosses]]:
        """Apply the Swin Transformer block to input tokens `x`.

        Arguments:
            x: input tensor of shape (B, H, W, D)

        Returns:
            output tensor of shape (B, H, W, D)
        """
        original_shape = x.shape
        S = self.window_size

        load_balancing_losses_list: List[LoadBalancingLosses] = list()
        # Apply first transformer block, extracting windows, applying
        # the transformer block to them, and re-combining them to the
        # original image.
        x = extract_subimages_and_fold_into_batch_dim(
            x, S
        )  # (B*H//S*W//S, S**2, D)
        x, load_balancing_losses = self.first_block(
            x, freq_cis=freq_cis
        )  # (B*H//S*W//S, S**2, D)

        if load_balancing_losses is not None:
            load_balancing_losses_list.append(load_balancing_losses)

        x = unfold_from_batch_dim_and_combine_subimages(
            x, original_shape, S
        )  # (B, H, W, D)

        # Apply second transformer block same as the first block, but
        # shifting the windows before and after the block.
        x = shift_horizontally_and_vertically(x, S // 2)  # (B, H, W, D)
        x = extract_subimages_and_fold_into_batch_dim(
            x, S
        )  # (B*H//S*W//S, S**2, D)

        B, H, W, _ = original_shape
        attn_bias = None
        # With a single row of windows, every window spans both poles.
        if self.use_attention_mask and H > S:
            attn_bias = shifted_window_pole_bias(
                H, W, S, x.dtype, x.device
            ).repeat(B, 1, 1)

        x, load_balancing_losses = self.second_block(
            x, freq_cis=freq_cis, attn_bias=attn_bias
        )  # (B*H//S*W//S, S**2, D)

        if load_balancing_losses is not None:
            load_balancing_losses_list.append(load_balancing_losses)

        x = unfold_from_batch_dim_and_combine_subimages(
            x, original_shape, S
        )  # (B, H, W, D)
        x = shift_horizontally_and_vertically(x, -(S // 2))  # (B, H, W, D)

        return x, load_balancing_losses_list


def make_swin_stage(
    token_dim: int,
    feedforward_network: Callable[[int, int], nn.Module],
    num_heads: int,
    num_swin_blocks: int,
    window_size: int,
    dropout_rate: float,
    drop_path_rate: float,
    attn_dropout_rate: float = 0.0,
    use_attention_mask: bool = False,
    use_efficient_attention: bool = True,
) -> torch.nn.ModuleList:
    return torch.nn.ModuleList(
        [
            SwinTransformerBlock(
                token_dim=token_dim,
                feedforward_network=feedforward_network,
                num_heads=num_heads,
                window_size=window_size,
                use_attention_mask=use_attention_mask,
                use_efficient_attention=use_efficient_attention,
                dropout_rate=dropout_rate,
                drop_path_rate=drop_path_rate,
                attn_dropout_rate=attn_dropout_rate,
            )
            for _ in range(num_swin_blocks)
        ]
    )
