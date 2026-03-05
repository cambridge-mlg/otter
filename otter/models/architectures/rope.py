import torch


# From https://github.com/meta-llama/llama/blob/main/llama/model.py
def precompute_freqs_cis_1d(
    dim: int, end: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def precompute_freqs_cis_2d(
    head_dim: int, window_size: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    Precompute the 2D frequency tensor for complex exponentials (cis).

    Reasoning:
    Instead of rotating the keys and queries (each of dimensionality
    head_dim) by one frequency vector, associated with the position
    of the token in the 1d sequence, we rotate the top half of the
    vector by the first frequency vector, and the bottom half of the
    vector by the second frequency vector, each associated with one
    spatial dimension.

    :param head_dim: The dimension of tokens in each attention head.
    :param window_size: Number of patches in each SWIN window.
    :param theta: The scaling factor for frequency computation.
    """

    # (window_size, head_dim // 2)
    freqs_lon = precompute_freqs_cis_1d(head_dim // 2, window_size, theta)
    freqs_lat = precompute_freqs_cis_1d(head_dim // 2, window_size, theta)

    # (window_size, 1, head_dim // 2)
    freqs_lon = freqs_lon[:, None, :]
    # (1, window_size, head_dim // 2)
    freqs_lat = freqs_lat[None, :, :]

    # (window_size, window_size, head_dim // 2)
    freqs_lon = freqs_lon.expand(-1, window_size, -1)
    freqs_lat = freqs_lat.expand(window_size, -1, -1)

    # (window_size, window_size, head_dim)
    freq_cis = torch.cat([freqs_lon, freqs_lat], dim=-1)

    return freq_cis


# From https://github.com/meta-llama/llama/blob/main/llama/model.py
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# From https://github.com/meta-llama/llama/blob/main/llama/model.py
def reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [
        d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
    ]
    return freqs_cis.view(*shape)
