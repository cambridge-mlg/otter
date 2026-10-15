from typing import Optional

import torch
from torch import nn
from xformers.ops import memory_efficient_attention

from otter.models.architectures.rope import apply_rotary_emb


class EfficientMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        token_dim: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads

        self.Uq = nn.Linear(token_dim, token_dim, bias=qkv_bias)
        self.Uk = nn.Linear(token_dim, token_dim, bias=qkv_bias)
        self.Uv = nn.Linear(token_dim, token_dim, bias=qkv_bias)

        self.attn_drop = attn_drop
        self.proj = nn.Linear(token_dim, token_dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        freq_cis: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Arguments:
            attn_bias: optional (B, N, N) bias added to the attention logits,
                e.g. -inf to block attention between two tokens.
        """
        assert q.shape == k.shape == v.shape
        B, N, D = q.shape

        q = self.Uq(q).reshape(B, N, self.num_heads, D // self.num_heads)
        k = self.Uk(k).reshape(B, N, self.num_heads, D // self.num_heads)
        v = self.Uv(v).reshape(B, N, self.num_heads, D // self.num_heads)

        if freq_cis is not None:
            q, k = apply_rotary_emb(q, k, freq_cis)

        if attn_bias is not None:
            # xformers takes the bias per head, with rows aligned to 8.
            padded = q.new_zeros(B, self.num_heads, N, -(-N // 8) * 8)
            padded[..., :N] = attn_bias[:, None]
            attn_bias = padded[..., :N]

        x: torch.Tensor = memory_efficient_attention(
            q,
            k,
            v,
            attn_bias=attn_bias,
            p=self.attn_drop if self.training else 0.0,
        )
        x = x.reshape(B, N, D)
        x = self.proj_drop(self.proj(x))

        return x
