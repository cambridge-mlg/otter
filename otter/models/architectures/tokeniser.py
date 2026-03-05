import torch
from torch import nn

from otter.models.architectures.utils import conv_channel_last


class ImageTokeniser(nn.Module):
    def __init__(
        self,
        in_channels: int,
        token_dimension: int,
        patch_size: int,
        kernel_size: tuple[int, int],
    ):
        super().__init__()

        self.patch_size = patch_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=token_dimension,
            kernel_size=kernel_size,
            stride=(patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenise the image `x`, applying a strided convolution.
        This is equivalent to splitting the image into patches,
        and then linearly projecting each one of these using a
        shared linear projection.

        Arguments:
            x: image input tensor of shape (B, W, H, C)

        Returns:
            output tensor of shape (B, N, D)
        """
        return conv_channel_last(self.conv, x)
