import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _build_inv_freq(num_components: int, base: float) -> torch.Tensor:
    """Matches nanochat's RoPE inv-freq construction (returns `[num_components]`)."""
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
        q: Optional[torch.Tensor],
        k: Optional[torch.Tensor],
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            q: optional [B, T, n_heads, d_head]
            k: optional [B, T, n_kv_heads, d_head]
            mu: [B, T] coordinates for provided tensor(s)
            sigma: [B, T] span sizes for provided tensor(s)
        """
        if q is None and k is None:
            raise ValueError("GaussianRoPE requires at least one of q or k.")
        if mu is None or sigma is None:
            raise ValueError("mu and sigma must be provided.")

        q_rot: Optional[torch.Tensor] = None
        k_rot: Optional[torch.Tensor] = None
        if q is not None:
            q_rot = self._rotate_tensor(q, mu, sigma, name="q")
        if k is not None:
            k_rot = self._rotate_tensor(k, mu, sigma, name="k")
        return q_rot, k_rot

    def _rotate_tensor(
        self, tensor: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, *, name: str
    ) -> torch.Tensor:
        """
        Rotate a Q/K tensor.

        Args:
            tensor: `[B, T, H, d_head]`
            mu: `[B, T]`
            sigma: `[B, T]`
        """
        if tensor.ndim != 4:
            raise ValueError(f"{name} must be a [B, T, H, d_head] tensor.")
        B, T, _, d_head = tensor.shape
        if d_head != self.d_head:
            raise ValueError(
                f"{name} last dim must equal configured d_head={self.d_head}."
            )
        if mu.shape != (B, T) or sigma.shape != (B, T):
            raise ValueError(
                f"mu/sigma must match the first two dims of {name} (got {mu.shape}, {sigma.shape})."
            )

        tensor_flat = tensor.reshape(B * T, -1, self.d_head)  # [N, H, d_head]
        mu_flat = mu.to(tensor.dtype).reshape(B * T)  # [N]
        sigma_flat = torch.clamp(sigma.to(tensor.dtype), min=1e-8).reshape(B * T)
        log_sigma = torch.log2(sigma_flat / self.sigma_base)

        device = tensor_flat.device
        time_angles = self._match_frequencies(
            mu_flat[:, None] * self.time_inv_freq.to(device), self.pair_dim
        )
        scale_angles = self._match_frequencies(
            log_sigma[:, None] * self.scale_inv_freq.to(device), self.pair_dim
        )

        time_chunk, scale_chunk = torch.split(tensor_flat, self.chunk_dim, dim=-1)
        time_chunk = self._apply_chunk(time_chunk, time_angles)
        scale_chunk = self._apply_chunk(scale_chunk, scale_angles)
        rotated = torch.cat([time_chunk, scale_chunk], dim=-1)
        return rotated.view(B, T, -1, self.d_head)

    def _match_frequencies(self, angles: torch.Tensor, target_pairs: int) -> torch.Tensor:
        """Tile or trim frequency rows so they match the needed pair count `[*, target_pairs]`."""
        current = angles.shape[-1]
        if current == target_pairs:
            return angles
        if current < target_pairs:
            repeat = math.ceil(target_pairs / current)
            angles = angles.repeat(1, repeat)
        return angles[:, :target_pairs]

    def _apply_chunk(self, chunk: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Apply standard RoPE rotation to a chunk `[N, H, chunk_dim]`."""
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
