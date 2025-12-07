from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .gistnet import GistNet


class MegaContextTree:
    """
    Maintains a multi-LOD gist tree backed entirely by dense tensors.

    The tree is grown from LOD0 embeddings (already projected token latents).
    Each higher LOD groups `block_size` children from the previous LOD and
    summarizes them via GistNet. Appends propagate upward incrementally once
    a new block becomes available, so partial tails simply wait for more data.
    """

    def __init__(
        self,
        gistnet: GistNet,
        *,
        max_lods: Optional[int] = None,
        initial_lod0: Optional[torch.Tensor] = None,
    ) -> None:
        if max_lods is not None and max_lods < 1:
            raise ValueError("max_lods must be >= 1 when provided.")

        if not hasattr(gistnet, "block_size"):
            raise ValueError("GistNet must expose a block_size attribute.")
        if gistnet.block_size <= 0:
            raise ValueError("GistNet.block_size must be positive.")

        self.gistnet = gistnet
        self.block_size = int(gistnet.block_size)
        self.max_lods = max_lods

        param = next(gistnet.parameters(), gistnet.query_token)
        self.device = param.device
        self.dtype = param.dtype
        self.d_model = gistnet.d_model

        self.batch_size: Optional[int] = None
        self._levels: List[torch.Tensor] = []  # level tensors, each [B, L_l, d_model]

        if initial_lod0 is not None:
            self.append_lod0(initial_lod0)

    def append_lod0(self, embeddings: torch.Tensor) -> None:
        """
        Append new LOD0 embeddings for each sequence in the batch.

        Args:
            embeddings: tensor of shape [B, N, d_model] (or [N, d_model] if B == 1).
            All sequences in the batch must provide the same number of new LOD0 nodes.
        """
        chunk = self._normalize_batch_input(embeddings)
        if not self._levels:
            self._initialize_levels()
        self._levels[0] = torch.cat([self._levels[0], chunk], dim=1)  # [B, L0+N, d]
        self._propagate_updates()

    def level_latents(self, level: int) -> torch.Tensor:
        """Return the `[B, L_level, d_model]` tensor for the requested LOD."""
        self._require_initialized()
        self._ensure_level(level)
        return self._levels[level]

    def level_counts(self, level: int) -> torch.Tensor:
        """Return `[B]` tensor with the node count per sequence for the LOD."""
        batch = self._require_initialized()
        self._ensure_level(level)
        count = self._levels[level].shape[1]
        return torch.full(
            (batch,),
            count,
            dtype=torch.long,
            device=self.device,
        )

    def padded_level(self, level: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            padded_latents: `[B, L_level, d_model]` dense tensor.
            counts: `[B]` tensor describing the true length per sequence.
        """
        self._require_initialized()
        self._ensure_level(level)
        return self._levels[level], self.level_counts(level)

    def num_levels(self) -> int:
        """Current number of populated LODs (includes empty tails)."""
        return len(self._levels)

    def _normalize_batch_input(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize incoming LOD0 data to `[B, N, d_model]` and capture batch metadata.

        Args:
            value: `[B, N, d_model]` or `[N, d_model]` tensor.
        Returns:
            Tensor shaped `[B, N, d_model]` on the tree's device/dtype.
        """
        tensor = value
        if tensor.ndim == 2:
            if self.batch_size not in (None, 1):
                raise ValueError("Batch inputs must have shape [B, N, d_model].")
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError("Embeddings must have shape [B, N, d_model].")
        if tensor.shape[2] != self.d_model:
            raise ValueError("Tensor d_model dimension mismatch.")
        if tensor.shape[1] == 0:
            raise ValueError("Embeddings must append at least one LOD0 node.")

        if self.batch_size is None:
            self.batch_size = tensor.shape[0]
            self._levels = []
        elif tensor.shape[0] != self.batch_size:
            raise ValueError("Input batch dimension mismatch.")

        return tensor.to(device=self.device, dtype=self.dtype)

    def _propagate_updates(self) -> None:
        if not self._levels:
            return
        batch = self._require_initialized()
        level = 1
        while True:
            if self.max_lods is not None and level >= self.max_lods:
                break
            prev_nodes = self._levels[level - 1]  # [B, L_prev, d_model]
            target = prev_nodes.shape[1] // self.block_size
            if target == 0:
                break
            while len(self._levels) <= level:
                self._levels.append(
                    torch.empty(
                        batch,
                        0,
                        self.d_model,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
            cur_nodes = self._levels[level]  # [B, L_cur, d_model]
            cur_len = cur_nodes.shape[1]
            new_parents = target - cur_len
            if new_parents <= 0:
                level += 1
                continue
            blocks = prev_nodes[:, : target * self.block_size, :].contiguous()  # [B, target*block_size, d]
            blocks = blocks.view(
                batch, target, self.block_size, self.d_model
            )  # [B, target, block_size, d]
            new_blocks = blocks[:, cur_len:target, :, :]  # [B, new_parents, block_size, d]
            flat_blocks = new_blocks.reshape(
                batch * new_parents, self.block_size, self.d_model
            )
            gists = self.gistnet(flat_blocks)  # [batch*new_parents, d_model] or [d_model]
            if flat_blocks.shape[0] == 1 and gists.ndim == 1:
                gists = gists.unsqueeze(0)
            gists = gists.view(batch, new_parents, self.d_model)
            self._levels[level] = torch.cat([cur_nodes, gists], dim=1)
            level += 1

    def _ensure_level(self, level: int) -> None:
        batch = self._require_initialized()
        while len(self._levels) <= level:
            self._levels.append(
                torch.empty(
                    batch,
                    0,
                    self.d_model,
                    dtype=self.dtype,
                    device=self.device,
                )
            )

    def _initialize_levels(self) -> None:
        batch = self._require_initialized()
        if self._levels:
            return
        self._levels.append(
            torch.empty(
                batch,
                0,
                self.d_model,
                dtype=self.dtype,
                device=self.device,
            )
        )

    def _require_initialized(self) -> int:
        if self.batch_size is None:
            raise RuntimeError("MegaContextTree has not received any LOD0 data yet.")
        return self.batch_size


__all__ = ["MegaContextTree"]
