from __future__ import annotations

from enum import IntFlag
from typing import Optional, Tuple

import torch


class WCTFlags(IntFlag):
    NONE = 0
    VIRTUAL = 1 << 0


class WorkingContextTree:
    """
    Maintains a flattened, sparsified slice of the MegaContextTree for transformer inputs.

    Latents live in `[B, max_nodes, d_model]` tensors. Metadata is tracked in tensors:
      - `node_ids`: `[B, max_nodes, 2]` integers storing `(level, node_index)`
      - `flags`: `[B, max_nodes]` bitmasks describing per-node status (virtual, pinned, etc.)
    Partial sequences are denoted via `lengths` per batch entry; positional info is derived
    from `node_ids` when downstream modules need `(mu, sigma)`.
    """

    def __init__(
        self,
        *,
        max_nodes: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive.")

        self.max_nodes = max_nodes
        self._device = device
        self._dtype = dtype

        self.batch_size: Optional[int] = None
        self.d_model: Optional[int] = None
        self.latents: Optional[torch.Tensor] = None
        self.node_ids: Optional[torch.Tensor] = None
        self.flags: Optional[torch.Tensor] = None
        self.lengths: Optional[torch.Tensor] = None

    # --------------------------------------------------------------------- APIs
    def append_nodes(
        self,
        *,
        latents: torch.Tensor,  # [B, K, d_model]
        node_levels: torch.Tensor,  # [B, K]
        node_indices: torch.Tensor,  # [B, K]
        flags: Optional[torch.Tensor] = None,  # [B, K]
    ) -> None:
        """
        Append nodes for all batch entries. Appending any nodes flips
        `tree_causal_ordering` to False (new nodes come after parents).

        Args:
            latents: [B, K, d_model]
            node_levels / node_indices: [B, K] ints describing MCT coordinates
            flags: optional [B, K] WCTFlags masks (defaults to NONE)
        """
        B, K, d_model = self._validate_append_shapes(
            latents, node_levels, node_indices
        )
        if K == 0:
            return
        if flags is not None and flags.shape != (B, K):
            raise ValueError("flags must have shape [B, K].")

        self._ensure_storage_initialized(batch_size=B, d_model=d_model, template=latents)
        assert self.lengths is not None
        assert self.latents is not None
        assert self.node_ids is not None
        assert self.flags is not None

        latents = latents.to(device=self.latents.device, dtype=self.latents.dtype)
        node_levels = node_levels.to(
            device=self.node_ids.device, dtype=torch.long
        )
        node_indices = node_indices.to(
            device=self.node_ids.device, dtype=torch.long
        )
        if flags is None:
            flag_values = torch.full(
                (B, K),
                fill_value=int(WCTFlags.NONE),
                dtype=torch.long,
                device=self.flags.device,
            )
        else:
            flag_values = flags.to(
                device=self.flags.device, dtype=torch.long
            )

        next_lengths = self.lengths + K  # [B]
        if torch.any(next_lengths > self.max_nodes):
            raise ValueError("Appending nodes exceeds max_nodes capacity.")

        for b in range(B):
            start = int(self.lengths[b].item())
            end = start + K
            self.latents[b, start:end] = latents[b]  # [K, d_model]
            self.node_ids[b, start:end, 0] = node_levels[b]  # [K]
            self.node_ids[b, start:end, 1] = node_indices[b]  # [K]
            self.flags[b, start:end] = flag_values[b]  # [K]
            self.lengths[b] = end

    def load_sequence(
        self,
        *,
        latents: torch.Tensor,  # [B, N, d_model]
        node_levels: torch.Tensor,  # [B, N]
        node_indices: torch.Tensor,  # [B, N]
        flags: Optional[torch.Tensor] = None,  # [B, N]
        lengths: Optional[torch.Tensor] = None,  # [B]
    ) -> None:
        """
        Overwrite the WorkingContextTree with a fully formed sequence. This path is
        used by the TrainingContextManager to mirror the entire MegaContextTree in a
        single fused write, avoiding Python-side loops that break torch.compile.

        Args:
            latents: `[B, N, d_model]` tensor containing the flattened nodes.
            node_levels / node_indices: `[B, N]` tensors describing metadata.
            flags: optional `[B, N]` tensor of WCTFlags.
            lengths: optional `[B]` tensor of valid sequence lengths. When omitted,
                every batch entry uses `N`.
        """
        if latents.ndim != 3:
            raise ValueError("latents must have shape [B, N, d_model].")
        if node_levels.shape != latents.shape[:2]:
            raise ValueError("node_levels must have shape [B, N].")
        if node_indices.shape != latents.shape[:2]:
            raise ValueError("node_indices must have shape [B, N].")
        if flags is not None and flags.shape != latents.shape[:2]:
            raise ValueError("flags must have shape [B, N].")

        B, N, d_model = latents.shape
        if lengths is None:
            lengths = torch.full(
                (B,),
                N,
                dtype=torch.long,
                device=latents.device,
            )
        else:
            if lengths.shape != (B,):
                raise ValueError("lengths must have shape [B].")
            lengths = lengths.to(device=latents.device, dtype=torch.long)
        if torch.any(lengths > N):
            raise ValueError("lengths entries cannot exceed the provided sequence.")
        if N > self.max_nodes:
            raise ValueError("Provided sequence exceeds WorkingContextTree capacity.")

        self._ensure_storage_initialized(B, d_model, template=latents)
        assert self.latents is not None
        assert self.node_ids is not None
        assert self.flags is not None
        assert self.lengths is not None

        max_len = N
        self.latents[:, :max_len] = latents.to(
            device=self.latents.device, dtype=self.latents.dtype
        )
        self.node_ids[:, :max_len, 0] = node_levels.to(
            device=self.node_ids.device, dtype=torch.long
        )
        self.node_ids[:, :max_len, 1] = node_indices.to(
            device=self.node_ids.device, dtype=torch.long
        )
        if flags is None:
            self.flags[:, :max_len] = int(WCTFlags.NONE)
        else:
            self.flags[:, :max_len] = flags.to(
                device=self.flags.device, dtype=torch.long
            )
        self.lengths = lengths.to(device=self.lengths.device)

    def mark_virtual(self, mask: torch.Tensor) -> None:
        """
        Mark nodes as virtually removed.

        Args:
            mask: `[B, max_nodes]` boolean tensor; True entries toggle the VIRTUAL bit.
        """
        mask = self._normalize_mask(mask)
        self._require_storage_ready()
        flag_val = int(WCTFlags.VIRTUAL)
        for b in range(self.batch_size):
            length = int(self.lengths[b].item())
            if length == 0:
                continue
            slice_mask = mask[b, :length]
            self.flags[b, :length][slice_mask] |= flag_val

    def reinstate_virtual(self, mask: torch.Tensor) -> None:
        """
        Clear the virtual bit for nodes.

        Args:
            mask: `[B, max_nodes]` boolean tensor; True entries clear the bit.
        """
        mask = self._normalize_mask(mask)
        self._require_storage_ready()
        flag_val = int(WCTFlags.VIRTUAL)
        for b in range(self.batch_size):
            length = int(self.lengths[b].item())
            if length == 0:
                continue
            slice_mask = mask[b, :length]
            self.flags[b, :length][slice_mask] &= ~flag_val

    def omit_nodes(self, mask: torch.Tensor) -> None:  # pragma: no cover
        raise NotImplementedError("Physical omission is disabled in the fully-virtual mode.")

    def compact_virtual(self) -> None:  # pragma: no cover
        raise NotImplementedError("Physical compaction is disabled in the fully-virtual mode.")

    def rebuild_causal_order(self, permutation: torch.Tensor) -> None:  # pragma: no cover
        raise NotImplementedError("Reordering is unnecessary in the fully-virtual mode.")

    # ----------------------------------------------------------------- helpers
    def tensors(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns latents, node_ids, flags, lengths sliced to the maximum active length.

        Returns:
            Tuple `(latents, node_ids, flags, lengths)` with shapes:
              - latents: `[B, max_len, d_model]`
              - node_ids: `[B, max_len, 2]`
              - flags: `[B, max_len]`
              - lengths: `[B]`
        """
        self._require_storage_ready()
        max_len = int(self.lengths.max().item()) if self.lengths is not None else 0
        return (
            self.latents[:, :max_len],  # [B, max_len, d_model]
            self.node_ids[:, :max_len],  # [B, max_len, 2]
            self.flags[:, :max_len],  # [B, max_len]
            self.lengths.clone() if self.lengths is not None else torch.zeros(0),  # [B]
        )

    def _validate_append_shapes(
        self,
        latents: torch.Tensor,
        node_levels: torch.Tensor,
        node_indices: torch.Tensor,
    ) -> Tuple[int, int, int]:
        """
        Validate incoming append buffers.

        Args:
            latents: `[B, K, d_model]`
            node_levels: `[B, K]`
            node_indices: `[B, K]`
        Returns:
            Tuple `(B, K, d_model)` describing inferred batch/block dims.
        """
        if latents.ndim != 3:
            raise ValueError("latents must have shape [B, K, d_model].")
        B, K, d_model = latents.shape
        for tensor, name in [
            (node_levels, "node_levels"),
            (node_indices, "node_indices"),
        ]:
            if tensor.shape != (B, K):
                raise ValueError(f"{name} must have shape [B, K].")
        return B, K, d_model

    def _normalize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Ensure mask is boolean `[B, max_nodes]`."""
        if self.batch_size is None:
            raise RuntimeError("WorkingContextTree has not received any nodes yet.")
        if mask.dtype != torch.bool:
            raise ValueError("mask must be boolean.")
        if mask.shape != (self.batch_size, self.max_nodes):
            raise ValueError("mask must have shape [B, max_nodes].")
        return mask

    def _ensure_storage_initialized(
        self, batch_size: int, d_model: int, template: torch.Tensor
    ) -> None:
        """
        Lazily allocate storage buffers based on the first append.

        Args:
            batch_size: inferred B dimension from `[B, K, d_model]` append.
            d_model: latent width.
            template: tensor sharing device/dtype preferences with the append data.
        """
        if self.batch_size is None:
            device = self._device or template.device
            dtype = self._dtype or template.dtype
            self.batch_size = batch_size
            self.d_model = d_model
            self.latents = torch.empty(
                batch_size,
                self.max_nodes,
                d_model,
                device=device,
                dtype=dtype,
            )
            self.node_ids = torch.full(
                (batch_size, self.max_nodes, 2),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )
            self.flags = torch.full(
                (batch_size, self.max_nodes),
                fill_value=int(WCTFlags.NONE),
                dtype=torch.long,
                device=device,
            )
            self.lengths = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
        else:
            if batch_size != self.batch_size:
                raise ValueError("Input batch dimension mismatch.")
            if d_model != self.d_model:
                raise ValueError("Input d_model dimension mismatch.")

    def _require_storage_ready(self) -> None:
        """Guard that append_nodes has run at least once so buffers exist."""
        if (
            self.latents is None
            or self.node_ids is None
            or self.flags is None
            or self.lengths is None
        ):
            raise RuntimeError("WorkingContextTree has not received any nodes yet.")


__all__ = ["WorkingContextTree", "WCTFlags"]
