import math
from typing import Optional

import torch
import torch.nn as nn

from mc.gaussian_rope import GaussianRoPE
from nanochat.gpt import GPTConfig, MLP, norm


class CrossAttentionBlock(nn.Module):
    """Shared QKV projections + Gaussian RoPE for fixed-span cross-attention."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int,
        sigma_level: float,
    ) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head.")
        if n_head % n_kv_head != 0:
            raise ValueError("n_head must be a multiple of n_kv_head.")

        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = d_model // n_head
        if self.head_dim % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for Gaussian RoPE.")

        self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.gaussian_rope = GaussianRoPE(d_head=self.head_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.sigma_level = float(sigma_level)
        self._last_attn: Optional[torch.Tensor] = None

    @property
    def last_attention_weights(self) -> Optional[torch.Tensor]:
        return self._last_attn

    def forward(self, query: torch.Tensor, children: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, d_model]
            children: [B, G, d_model] or [G, d_model]
        Returns:
            Tensor of shape [B, d_model]
        """
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if query.ndim != 2:
            raise ValueError("query must be [B, d_model].")

        if children.ndim == 2:
            children = children.unsqueeze(0)
        if children.ndim != 3:
            raise ValueError("children must be [B, G, d_model] or [G, d_model].")
        if children.shape[1] == 0:
            raise ValueError("GistNet requires at least one child latent.")
        if query.shape[0] != children.shape[0]:
            raise ValueError("query and children batch sizes must match.")

        B, G = query.shape[0], children.shape[1]
        query_norm = norm(query)  # [B, d_model]
        children_norm = norm(children)  # [B, G, d_model]

        query_seq = query_norm.unsqueeze(1)  # [B, 1, d_model]
        child_seq = children_norm  # [B, G, d_model]

        q = self.q_proj(query_seq).view(B, 1, self.n_head, self.head_dim)
        k = self.k_proj(child_seq).view(B, G, self.n_kv_head, self.head_dim)
        v = self.v_proj(child_seq).view(B, G, self.n_kv_head, self.head_dim)

        query_mu = torch.full(
            (B, 1), float(max(G - 1, 0)), dtype=torch.float32, device=query.device
        )
        query_sigma = torch.full_like(query_mu, self.sigma_level)
        child_mu = torch.arange(G, dtype=torch.float32, device=children.device).view(
            1, G
        ).expand(B, G)
        child_sigma = torch.full_like(child_mu, self.sigma_level)

        q = self._rotate_queries(q, query_mu, query_sigma)  # [B,1,H,D]
        k = self._rotate_keys(k, child_mu, child_sigma)  # [B,G,Hkv,D]

        q = norm(q)  # [B,1,H,D]
        k = norm(k)  # [B,G,Hkv,D]

        q = q.transpose(1, 2)  # [B,H,1,D]
        k = k.permute(0, 2, 1, 3)  # [B,Hkv,G,D]
        v = v.permute(0, 2, 1, 3)  # [B,Hkv,G,D]

        if self.n_head != self.n_kv_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)  # [B,H,G,D]
            v = v.repeat_interleave(repeat, dim=1)  # [B,H,G,D]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,1,G]
        attn = scores.softmax(dim=-1)
        self._last_attn = attn.detach()
        context = torch.matmul(attn, v)  # [B,H,1,D]

        context = context.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        out = self.out_proj(context)  # [B,1,d_model]
        return out.squeeze(1)  # [B, d_model]

    def _rotate_queries(
        self, q: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Gaussian RoPE to query tensor.

        Args:
            q: `[B, 1, n_head, d_head]`
            mu: `[B, 1]`
            sigma: `[B, 1]`
        """
        q_rot, _ = self.gaussian_rope(q, None, mu, sigma)
        if q_rot is None:
            raise RuntimeError("GaussianRoPE returned None for query rotation.")
        return q_rot

    def _rotate_keys(
        self, k: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Gaussian RoPE to key tensor.

        Args:
            k: `[B, G, n_kv_head, d_head]`
            mu: `[B, G]`
            sigma: `[B, G]`
        """
        _, k_rot = self.gaussian_rope(None, k, mu, sigma)
        if k_rot is None:
            raise RuntimeError("GaussianRoPE returned None for key rotation.")
        return k_rot


class GistNet(nn.Module):
    """Local G→1 summarization network backed by Gaussian-RoPE cross-attention."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int,
        sigma_level: float,
        block_size: int,
    ) -> None:
        super().__init__()
        if sigma_level <= 0:
            raise ValueError("sigma_level must be positive.")
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        self.d_model = d_model
        self.sigma_level = float(sigma_level)
        self.block_size = int(block_size)
        self.query_token = nn.Parameter(torch.randn(d_model))
        self.attn = CrossAttentionBlock(
            d_model=d_model,
            n_head=n_head,
            n_kv_head=n_kv_head,
            sigma_level=self.sigma_level,
        )
        self.mlp = MLP(GPTConfig(n_embd=d_model))

    def forward(self, x_children: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_children: [G, d_model] or [B, G, d_model] latents for fixed-size spans.
        Returns:
            Tensor of shape [d_model] (if single block) or [B, d_model].
        """
        orig_ndim = x_children.ndim
        if x_children.ndim == 2:
            x_children = x_children.unsqueeze(0)
        if x_children.ndim != 3:
            raise ValueError("x_children must be [G, d_model] or [B, G, d_model].")
        if x_children.shape[1] != self.block_size:
            raise ValueError(
                f"GistNet expected {self.block_size} children, "
                f"but received {x_children.shape[1]}."
            )

        B = x_children.shape[0]
        query = self.query_token.unsqueeze(0).expand(B, -1).to(
            device=x_children.device, dtype=x_children.dtype
        )  # [B, d_model]
        attn_out = self.attn(query, x_children)  # [B, d_model]
        x = query + attn_out  # [B, d_model]
        x = x + self.mlp(norm(x))  # [B, d_model]
        if orig_ndim == 2:
            return x.squeeze(0)
        return x

    @property
    def last_attention_weights(self) -> Optional[torch.Tensor]:
        """Expose last attention matrix for tests/debugging."""
        return self.attn.last_attention_weights


__all__ = ["CrossAttentionBlock", "GistNet"]
