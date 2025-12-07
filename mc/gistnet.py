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
            children: [G, d_model]
        Returns:
            Tensor of shape [B, d_model]
        """
        if children.ndim != 2:
            raise ValueError("children must be a 2D tensor [G, d_model].")
        if query.ndim != 2:
            raise ValueError("query must be a 2D tensor [B, d_model].")
        if children.shape[0] == 0:
            raise ValueError("GistNet requires at least one child latent.")

        B = query.shape[0]
        G = children.shape[0]
        query_norm = norm(query)  # [B, d_model]
        children_norm = norm(children)  # [G, d_model]

        query_seq = query_norm.unsqueeze(1)  # [B, 1, d_model]
        child_seq = children_norm.unsqueeze(0)  # [1, G, d_model]

        q = self.q_proj(query_seq).view(B, 1, self.n_head, self.head_dim)  # [B,1,H,D]
        k = self.k_proj(child_seq).view(1, G, self.n_kv_head, self.head_dim)  # [1,G,Hkv,D]
        v = self.v_proj(child_seq).view(1, G, self.n_kv_head, self.head_dim)  # [1,G,Hkv,D]

        query_mu = torch.full(
            (B, 1), float(max(G - 1, 0)), dtype=torch.float32, device=query.device
        )
        query_sigma = torch.full_like(query_mu, self.sigma_level)
        child_mu = torch.arange(G, dtype=torch.float32, device=children.device).view(
            1, G
        )
        child_sigma = torch.full_like(child_mu, self.sigma_level)

        q = self._rotate_queries(q, query_mu, query_sigma)  # [B,1,H,D]
        k = self._rotate_keys(k, child_mu, child_sigma)  # [1,G,Hkv,D]

        q = norm(q)  # [B,1,H,D]
        k = norm(k)  # [1,G,Hkv,D]

        q = q.transpose(1, 2)  # [B,H,1,D]
        k = k.transpose(1, 2)  # [1,Hkv,G,D]
        v = v.transpose(1, 2)  # [1,Hkv,G,D]

        if self.n_head != self.n_kv_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)  # [1,H,G,D]
            v = v.repeat_interleave(repeat, dim=1)  # [1,H,G,D]

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
        q_rot, _ = self.gaussian_rope(q, None, mu, sigma)
        if q_rot is None:
            raise RuntimeError("GaussianRoPE returned None for query rotation.")
        return q_rot

    def _rotate_keys(
        self, k: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
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
    ) -> None:
        super().__init__()
        if sigma_level <= 0:
            raise ValueError("sigma_level must be positive.")
        self.d_model = d_model
        self.sigma_level = float(sigma_level)
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
            x_children: [G, d_model] latents for a fixed-size span.
        Returns:
            Tensor of shape [d_model] representing the gist latent.
        """
        if x_children.ndim != 2:
            raise ValueError("x_children must have shape [G, d_model].")
        if x_children.shape[0] == 0:
            raise ValueError("x_children must contain at least one child latent.")

        query = self.query_token.unsqueeze(0).to(
            device=x_children.device, dtype=x_children.dtype
        )  # [1, d_model]
        attn_out = self.attn(query, x_children)  # [1, d_model]
        x = query + attn_out  # [1, d_model]
        x = x + self.mlp(norm(x))  # [1, d_model]
        return x.squeeze(0)

    @property
    def last_attention_weights(self) -> Optional[torch.Tensor]:
        """Expose last attention matrix for tests/debugging."""
        return self.attn.last_attention_weights


__all__ = ["CrossAttentionBlock", "GistNet"]
