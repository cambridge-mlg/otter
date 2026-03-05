from collections.abc import Callable
from typing import (
    List,
    Optional,
    Sequence,
)

import torch
from torch import nn

from otter.models.architectures.embeddings import (
    PositionEmbedding,
    SpatialEmbedding,
)
from otter.models.architectures.moe import LoadBalancingLosses
from otter.models.architectures.rope import precompute_freqs_cis_2d
from otter.models.architectures.tokeniser import ImageTokeniser
from otter.models.architectures.transformer_blocks import make_swin_stage
from otter.models.architectures.utils import (
    aggregate_load_balancing_losses,
    conv_channel_last,
)


class SwinTransformer(nn.Module):
    def __init__(
        self,
        token_dimensions: Sequence[int] | int,
        output_dimension: int,
        window_size: int,
        num_heads: int,
        in_channels: int,
        height: int,
        width: int,
        patch_size: int,
        patchify_kernel_size: tuple[int, int],
        num_blocks_per_stage: Sequence[int],
        num_bottom_blocks: int,
        dropout_rate: float,
        drop_path_rate: float,
        feedforward_network: Callable[[int, int], nn.Module],
        pos_embedding: PositionEmbedding | SpatialEmbedding,
        use_rope: bool,
        use_efficient_attention: bool = True,
        use_simple_skip: bool = False,
    ):
        super().__init__()

        assert len(num_blocks_per_stage) > 0

        if isinstance(token_dimensions, int):
            first_token_dim = token_dimensions
        else:
            first_token_dim = token_dimensions[0]

        if isinstance(token_dimensions, int):
            num_stages = len(num_blocks_per_stage)
            token_dimensions = [token_dimensions] * (num_stages + 1)

        head_dim = first_token_dim // num_heads

        if use_rope:
            freq_cis = precompute_freqs_cis_2d(head_dim, window_size)
            # Windows are flattened to 1D, so we need to reshape the frequency
            # tensor to match the flattened window shape.
            # head_dim // 2 is used because these are complex numbers.
            freq_cis = freq_cis.reshape(window_size**2, head_dim // 2)

            self.register_buffer("freq_cis", freq_cis)
        else:
            self.freq_cis = None

        token_in_out_dims = list(
            zip(token_dimensions[:-1], token_dimensions[1:])
        )

        down_swin_stages = []
        up_swin_stages = []
        patch_downsampling_layers = []
        patch_upsampling_layers = []
        fusion_conv_layers = []

        for num_blocks, (t1, t2) in zip(
            num_blocks_per_stage,
            token_in_out_dims,
        ):
            down_swin_stages.append(
                make_swin_stage(
                    token_dim=t1,
                    feedforward_network=feedforward_network,
                    num_heads=num_heads,
                    num_swin_blocks=num_blocks,
                    window_size=window_size,
                    use_efficient_attention=use_efficient_attention,
                    dropout_rate=dropout_rate,
                    drop_path_rate=drop_path_rate,
                )
            )
            patch_downsampling_layers.append(
                nn.Conv2d(
                    in_channels=t1,
                    out_channels=t2,
                    kernel_size=2,
                    stride=2,
                )
            )

        self.bottom_swin_stage = make_swin_stage(
            token_dim=token_dimensions[-1],
            feedforward_network=feedforward_network,
            num_heads=num_heads,
            num_swin_blocks=num_bottom_blocks,
            window_size=window_size,
            use_efficient_attention=use_efficient_attention,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
        )

        for num_blocks, (t2, t1) in zip(
            num_blocks_per_stage[::-1],
            token_in_out_dims[::-1],
        ):
            patch_upsampling_layers.append(
                nn.ConvTranspose2d(
                    in_channels=t1,
                    out_channels=t2,
                    kernel_size=2,
                    stride=2,
                )
            )
            if not use_simple_skip:
                fusion_conv_layers.append(
                    nn.Conv2d(
                        in_channels=t2 + t2,
                        out_channels=t2,
                        kernel_size=1,  # 1x1 conv is efficient
                        padding=0,
                    )
                )

            up_swin_stages.append(
                make_swin_stage(
                    token_dim=t2,
                    feedforward_network=feedforward_network,
                    num_heads=num_heads,
                    num_swin_blocks=num_blocks,
                    window_size=window_size,
                    use_efficient_attention=use_efficient_attention,
                    dropout_rate=dropout_rate,
                    drop_path_rate=drop_path_rate,
                )
            )

        self.down_swin_stages = torch.nn.ModuleList(down_swin_stages)
        self.up_swin_stages = torch.nn.ModuleList(up_swin_stages)
        self.patch_downsampling_layers = torch.nn.ModuleList(
            patch_downsampling_layers
        )
        self.patch_upsampling_layers = torch.nn.ModuleList(
            patch_upsampling_layers
        )
        if not use_simple_skip:
            self.fusion_conv_layers = torch.nn.ModuleList(fusion_conv_layers)
        self.use_simple_skip = use_simple_skip

        self.tokeniser = ImageTokeniser(
            in_channels=in_channels,
            token_dimension=first_token_dim,
            patch_size=patch_size,
            kernel_size=patchify_kernel_size,
        )

        self.embedding = pos_embedding

        self.final_conv = nn.ConvTranspose2d(
            in_channels=first_token_dim,
            out_channels=output_dimension,
            stride=(patch_size, patch_size),
            kernel_size=patchify_kernel_size,
        )

        self.width = width
        self.height = height

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[LoadBalancingLosses]]:
        """Apply vision transformer to batch of images.

        Arguments:
            x: input image tensor of shape (B, H, W, C)

        Returns:
            output logits tensor of shape (B, H, W, C)
        """
        x = self.tokeniser(x)
        x = self.embedding(x)

        skips = []
        load_balancing_losses_list: List[LoadBalancingLosses] = list()

        for downsampling_conv, down_swin_stage in zip(
            self.patch_downsampling_layers,
            self.down_swin_stages,
        ):
            for swin_transformer_block in down_swin_stage:
                x, load_balancing_losses_per_block = swin_transformer_block(
                    x, freq_cis=self.freq_cis
                )
                load_balancing_losses_list += load_balancing_losses_per_block
            skips.append(x)
            x = downsampling_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        for swin_transformer_block in self.bottom_swin_stage:
            x, load_balancing_losses_per_block = swin_transformer_block(
                x, freq_cis=self.freq_cis
            )
            load_balancing_losses_list += load_balancing_losses_per_block

        if self.use_simple_skip:
            for upsampling_conv, up_swin_stage, skip in zip(
                self.patch_upsampling_layers,
                self.up_swin_stages,
                skips[::-1],
            ):
                x = upsampling_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x = x + skip

                for swin_transformer_block in up_swin_stage:
                    x, load_balancing_losses_per_block = (
                        swin_transformer_block(x, freq_cis=self.freq_cis)
                    )
                    load_balancing_losses_list += (
                        load_balancing_losses_per_block
                    )
        else:
            for upsampling_conv, up_swin_stage, skip, fusion_conv in zip(
                self.patch_upsampling_layers,
                self.up_swin_stages,
                skips[::-1],
                self.fusion_conv_layers,
            ):
                x = upsampling_conv(x.permute(0, 3, 1, 2))
                x = torch.cat((x, skip.permute(0, 3, 1, 2)), dim=1)
                x = fusion_conv(x)
                x = x.permute(0, 2, 3, 1)

                for swin_transformer_block in up_swin_stage:
                    x, load_balancing_losses_per_block = (
                        swin_transformer_block(x, freq_cis=self.freq_cis)
                    )
                    load_balancing_losses_list += (
                        load_balancing_losses_per_block
                    )

        x = conv_channel_last(self.final_conv, x)

        return x, aggregate_load_balancing_losses(load_balancing_losses_list)
