import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _build_inv_freq(num_components: int, base: float) -> torch.Tensor:
    """Matches nanochat's RoPE inv-freq construction."""
    idx = torch.arange(0, num_components, dtype=torch.float32)
    return 1.0 / (base ** (idx / max(num_components, 1)))


class GaussianRoPE(nn.Module):
    """Applies time+scale rotary embeddings to Q/K tensors."""

    def __init__(
        self,
        d_head: int,
        *,
        sigma_base: float = 1.0,
        time_base: float = 10000.0,
        scale_base: float = 2.0,
        max_scale_components: Optional[int] = 16,
    ) -> None:
        super().__init__()
        if d_head % 4 != 0:
            raise ValueError(f"d_head={d_head} must be divisible by 4 for GaussianRoPE.")

        self.d_head = d_head
        self.chunk_dim = d_head // 2  # first half=time, second half=scale
        self.pair_dim = self.chunk_dim // 2
        self.sigma_base = sigma_base

        time_inv_freq = _build_inv_freq(self.pair_dim, time_base)
        scale_pair_dim = self.pair_dim
        if max_scale_components is not None:
            scale_pair_dim = min(scale_pair_dim, max_scale_components)
        scale_inv_freq = _build_inv_freq(scale_pair_dim, scale_base)

        self.register_buffer("time_inv_freq", time_inv_freq, persistent=False)
        self.register_buffer("scale_inv_freq", scale_inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: [B, T, n_heads, d_head]
            k: [B, T, n_kv_heads, d_head]
            mu: [B, T] span centers (global or local)
            sigma: [B, T] span sizes
        """
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError("q and k must be [B, T, H, d_head] tensors.")
        if q.shape[:2] != mu.shape or q.shape[:2] != sigma.shape:
            raise ValueError("mu/sigma must match the [B, T] shape of q/k.")
        if q.shape[-1] != self.d_head or k.shape[-1] != self.d_head:
            raise ValueError(
                f"q/k last dim must equal configured d_head={self.d_head}."
            )

        B, T, n_heads, _ = q.shape
        _, _, n_kv_heads, _ = k.shape
        q_flat = q.reshape(B * T, n_heads, self.d_head)  # [N, H, d_head]
        k_flat = k.reshape(B * T, n_kv_heads, self.d_head)  # [N, H_kv, d_head]
        mu_flat = mu.to(q.dtype).reshape(B * T)  # [N]
        sigma_flat = sigma.to(q.dtype).reshape(B * T)  # [N]
        sigma_flat = torch.clamp(sigma_flat, min=1e-8)
        log_sigma = torch.log2(sigma_flat / self.sigma_base)

        time_angles = self._match_frequencies(
            mu_flat[:, None] * self.time_inv_freq.to(q_flat.device), self.pair_dim
        )  # [N, pair_dim]
        scale_angles = self._match_frequencies(
            log_sigma[:, None] * self.scale_inv_freq.to(q_flat.device), self.pair_dim
        )  # [N, pair_dim]

        q_time, q_scale = torch.split(q_flat, self.chunk_dim, dim=-1)
        k_time, k_scale = torch.split(k_flat, self.chunk_dim, dim=-1)
        q_time = self._apply_chunk(q_time, time_angles)
        k_time = self._apply_chunk(k_time, time_angles)
        q_scale = self._apply_chunk(q_scale, scale_angles)
        k_scale = self._apply_chunk(k_scale, scale_angles)
        q_flat = torch.cat([q_time, q_scale], dim=-1)
        k_flat = torch.cat([k_time, k_scale], dim=-1)
        q_rot = q_flat.view(B, T, n_heads, self.d_head)
        k_rot = k_flat.view(B, T, n_kv_heads, self.d_head)
        return q_rot, k_rot

    def _match_frequencies(self, angles: torch.Tensor, target_pairs: int) -> torch.Tensor:
        """Tile or trim frequency rows so they match the needed pair count."""
        current = angles.shape[-1]
        if current == target_pairs:
            return angles
        if current < target_pairs:
            repeat = math.ceil(target_pairs / current)
            angles = angles.repeat(1, repeat)
        return angles[:, :target_pairs]

    def _apply_chunk(self, chunk: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Apply standard RoPE rotation to a chunk [N, H, chunk_dim]."""
        N, H, chunk_dim = chunk.shape
        chunk = chunk.view(N, H, chunk_dim // 2, 2)  # [N, H, pairs, 2]
        cos = angles.cos().to(chunk.dtype).unsqueeze(1)  # [N,1,pairs]
        sin = angles.sin().to(chunk.dtype).unsqueeze(1)  # [N,1,pairs]
        x1 = chunk[..., 0]  # [N,H,pairs]
        x2 = chunk[..., 1]  # [N,H,pairs]
        out0 = x1 * cos - x2 * sin
        out1 = x1 * sin + x2 * cos
        return torch.stack((out0, out1), dim=-1).view(N, H, chunk_dim)


__all__ = ["GaussianRoPE"]
