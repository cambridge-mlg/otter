from typing import Optional, Tuple

import einops
import numpy as np
import numpy.typing as npt
import torch
from torch import nn

import otter.models.architectures.spherical_harmonics.spherical_harmonics_ylm as sh_gen

TEMPORAL_EMBEDDING_DEFAULT_MIN_HOUR_SCALE = 3.0
TEMPORAL_EMBEDDING_DEFAULT_MAX_HOUR_SCALE = 8760.0
REFERENCE_DATETIME = np.datetime64("1979-01-01T00:00:00")
SPATIAL_EMBEDDING_DEFAULT_MIN_SCALE = 0.1
SPATIAL_EMBEDDING_DEFAULT_MAX_SCALE = 720


class PositionEmbedding(nn.Module):
    def __init__(
        self,
        token_dimension: int,
        height: int,
        width: int,
        init_scale: float = 1e-1,
    ):
        super().__init__()

        self.embeddings = nn.Parameter(
            init_scale * torch.randn(height, width, token_dimension),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position embeddings to input tensor.

        Arguments:
            x: input tensor of shape (B, H, W, D)

        Returns:
            output tensor of shape (B, H, W, D)
        """
        return x + self.embeddings[None, :, :, :]


class FourierExpansionEmbedding(nn.Module):
    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        num_scales: int,
    ) -> None:
        super().__init__()

        # Set up scales in log space.
        log_scales = torch.linspace(
            torch.log(torch.tensor(np.array(min_scale))),
            torch.log(torch.tensor(np.array(max_scale))),
            num_scales,
        )
        self.scales = torch.exp(log_scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create Fourier embeddings from input tensor x.

        Arguments:
            x: input tensor of shape (...)

        Returns:
            output tensor of shape (..., 2 * num_scales)
        """

        # TODO: Check if we want to use float64 here like in Aurora
        x = x.double()

        scales = self.scales.to(x.device)

        # Compute the Fourier embeddings.
        prod = torch.einsum("...d, s->...ds", x, scales**-1.0)

        sin = torch.sin(2 * torch.pi * prod)  # (..., num_scales)
        cos = torch.cos(2 * torch.pi * prod)  # (..., num_scales)

        return torch.cat([sin, cos], dim=-1).float()  # (..., 2 * num_scales)


def _get_hours_from_reference_time(
    x: npt.NDArray[np.datetime64],
) -> npt.NDArray[np.int32]:
    return (x - REFERENCE_DATETIME).astype("timedelta64[h]").astype(int)


class TimeEmbedding(FourierExpansionEmbedding):
    """Fourier embedding class that works with"""

    def __init__(
        self,
        num_scales: int,
        min_scale: float = TEMPORAL_EMBEDDING_DEFAULT_MIN_HOUR_SCALE,
        max_scale: float = TEMPORAL_EMBEDDING_DEFAULT_MAX_HOUR_SCALE,
    ) -> None:
        super().__init__(min_scale, max_scale, num_scales)

    def forward(
        self,
        x: npt.NDArray[np.datetime64],
    ) -> torch.Tensor:  # type: ignore
        hours = _get_hours_from_reference_time(x)
        return super().forward(torch.tensor(hours).float())


class SpatialEmbedding(FourierExpansionEmbedding):
    """Spatial position embedding based on Fourier Embeddings"""

    def __init__(
        self,
        num_scales: int,
        min_scale: float = SPATIAL_EMBEDDING_DEFAULT_MIN_SCALE,
        max_scale: float = SPATIAL_EMBEDDING_DEFAULT_MAX_SCALE,
        token_dimension: Optional[int] = None,
    ) -> None:
        super().__init__(min_scale, max_scale, num_scales // 2)

        if token_dimension is not None:
            self.linear = nn.Linear(2 * num_scales, token_dimension)

        self.num_scales = num_scales

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add Fourier-based positional embeddings to the input tensor.

        Arguments:
            x (torch.Tensor): Input tensor of shape (B, H, W, D)

        Returns:
            torch.Tensor: Output tensor of shape (B, H, W, D_final)
        """

        # Get lat-lon dimensions from tensor
        lat_dim, lon_dim = x.shape[1], x.shape[2]

        # Set up lat-lon coordinates
        lat = torch.linspace(-90, 90, lat_dim, device=x.device)
        lon = torch.linspace(
            0, 360 - (360 / lon_dim), lon_dim, device=x.device
        )

        # Get lat and lon embeddings.

        lat = super().forward(lat)  # (B, lat_dim, num_scales//2)
        lon = super().forward(lon)  # (B, lon_dim, num_scales//2)

        # Broadcast
        lat = einops.repeat(lat, "l d -> l L d", L=lon_dim)
        lon = einops.repeat(lon, "L d -> l L d", l=lat_dim)
        spatial_embedding = torch.cat([lat, lon], dim=-1)
        spatial_embedding = einops.repeat(
            spatial_embedding, "h w d -> b h w d", b=x.shape[0]
        )

        spatial_embedding = self.linear(spatial_embedding)
        return x + spatial_embedding


class SphericalHarmonicsEmbedding(nn.Module):
    """
    Spatial embedding using pre-generated analytical Spherical Harmonics.

    Adapts geographic coordinates (Lat/Lon) to the physics convention (Theta/Phi)
    expected by the SymPy-generated formulas.
    """

    def __init__(
        self,
        token_dimension: int,
        max_degree: int = 20,  # L_max. Total features = (L+1)^2
    ) -> None:
        super().__init__()

        self.max_degree = max_degree
        self.num_features = (max_degree + 1) ** 2

        # Linear projection from SH basis to transformer token dimension
        self.linear = nn.Linear(self.num_features, token_dimension)

        # Cache buffer to store the computed map (computed once per resolution)
        self.register_buffer("cached_embedding", None, persistent=False)
        self.cached_embedding: Optional[torch.Tensor] = None
        self.cached_shape: Optional[Tuple[int, int]] = None

    def _get_physics_grid(
        self, H: int, W: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the grid in Physics coordinates (Colatitude, Longitude).
        """
        # 1. LATITUDE -> COLATITUDE (Theta)
        # Geography: -90 (South) to +90 (North)
        # Physics:    pi (South) to  0 (North)
        # Transformation: Colat = 90 - Lat

        lat_deg = torch.linspace(-90, 90, H, device=device)
        colat_deg = 90.0 - lat_deg  # Now 180 (South) to 0 (North)

        # 2. LONGITUDE -> AZIMUTH (Phi)
        # Geography: -180 to 180
        # Physics:    Accepts 0-2pi or -pi to pi.
        # We assume standard W covers the full globe.
        lon_deg = torch.linspace(0, 360, W + 1, device=device)[:-1]

        # Meshgrid
        colat_grid, lon_grid = torch.meshgrid(
            colat_deg, lon_deg, indexing="ij"
        )

        # Convert to Radians for the SH functions
        theta = torch.deg2rad(colat_grid)  # Colatitude
        phi = torch.deg2rad(lon_grid)  # Longitude

        return theta, phi

    def _compute_basis_from_generated_file(
        self, theta: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        """
        Iterates through the generated functions to build the basis tensor.
        This is slow in Python loop, but only runs ONCE when grid size changes.
        """
        features_list = []

        # The generated script usually provides a helper SH(m, l, phi, theta)
        # OR we can iterate the naming convention Yl{l}_m{m}

        for degree in range(self.max_degree + 1):
            for order in range(-degree, degree + 1):
                # We call the specific JIT-compiled function from the module
                # Note: The generated script defined SH(m, l, phi, theta)
                # We use that dispatcher if available, otherwise we construct the name.
                try:
                    # Try using the dispatcher defined in the generated file footer
                    y_lm = sh_gen.SH(order, degree, phi, theta)  # type: ignore[no-untyped-call]
                    if isinstance(y_lm, (float, int)):
                        # Create a tensor of shape (H, W) filled with this constant
                        # Uses theta's device and dtype automatically
                        y_lm = torch.full_like(theta, float(y_lm))
                    elif isinstance(y_lm, torch.Tensor) and y_lm.ndim == 0:
                        # Expand scalar tensor to (H, W)
                        y_lm = y_lm.expand_as(theta)
                except AttributeError:
                    # Fallback: construct function name if SH() helper isn't exported
                    # Name format from script: f"Yl{l}_m{m}".replace("-", "_minus_")
                    order_str = str(order).replace("-", "_minus_")
                    fname = f"Yl{degree}_m{order_str}"
                    func = getattr(sh_gen, fname)
                    y_lm = func(theta, phi)

                features_list.append(y_lm)

        # Stack along the last dimension
        # Shape: (H, W, (L+1)^2)
        return torch.stack(features_list, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, D)
        """
        B, H, W, D = x.shape

        # 1. Check Cache
        if self.cached_embedding is None or self.cached_shape != (H, W):
            # Compute Grid in PHYSICS coordinates
            theta, phi = self._get_physics_grid(H, W, x.device)

            # Compute Basis (Slow loop, runs once)
            sh_features = self._compute_basis_from_generated_file(theta, phi)

            # Update Cache
            self.cached_embedding = sh_features
            self.cached_shape = (H, W)

        # 2. Projection
        emb = self.cached_embedding  # (H, W, num_features)
        assert emb is not None, "Embedding cache failed to initialize"

        emb = emb.unsqueeze(0)  # (1, H, W, num_features)

        # Linear projection to token dimension
        emb = self.linear(emb)  # (1, H, W, token_dim)

        return x + emb
