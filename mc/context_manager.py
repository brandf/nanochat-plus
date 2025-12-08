from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .gistnet import GistNet
from .mega_context_tree import MegaContextTree
from .working_context_tree import WCTFlags, WorkingContextTree


@dataclass
class ContextOperation:
    """Structured log entry describing context updates."""

    op: str
    batch: int
    indices: Sequence[int]
    reason: Optional[str] = None


@dataclass
class BaseContextManagerConfig:
    max_nodes: int
    tail_token_keep: int = 0
    max_lods: Optional[int] = None
    focus_seed: int = 0
    gistnet_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingContextManagerConfig(BaseContextManagerConfig):
    virtual_dropout_prob: float = 0.0
    holdout_tokens: int = 0
    virtual_dropout_seed: Optional[int] = None


@dataclass
class InferenceContextManagerConfig(BaseContextManagerConfig):
    target_node_count: int = 0
    target_node_growth: int = 0
    target_node_shrink: int = 0


class BaseContextManager:
    """
    Shared scaffolding that owns the MegaContextTree + WorkingContextTree duo.

    Concrete subclasses are responsible for deciding how the trees are reset and
    how masks/focus scores are consumed.
    """

    def __init__(
        self,
        config: BaseContextManagerConfig,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if config.max_nodes <= 0:
            raise ValueError("max_nodes must be positive.")

        self.config = config
        gistnet, inferred_device, inferred_dtype = self._build_gistnet(
            config.gistnet_kwargs
        )
        self.gistnet = gistnet
        self.device = device or inferred_device
        self.dtype = dtype or inferred_dtype
        self.gistnet.to(device=self.device, dtype=self.dtype)

        self.wct = WorkingContextTree(
            max_nodes=config.max_nodes,
            device=self.device,
            dtype=self.dtype,
        )
        self.mct: Optional[MegaContextTree] = None

    # ------------------------------------------------------------------ helpers
    def _new_mct(self) -> MegaContextTree:
        return MegaContextTree(
            self.gistnet,
            max_lods=self.config.max_lods,
        )

    def _new_wct(self) -> WorkingContextTree:
        return WorkingContextTree(
            max_nodes=self.config.max_nodes,
            device=self.device,
            dtype=self.dtype,
        )

    def _ensure_mct(self) -> MegaContextTree:
        if self.mct is None:
            self.mct = self._new_mct()
        return self.mct

    def _placeholder_focus_scores(
        self,
        *,
        node_ids: torch.Tensor,
        lengths: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        """
        Deterministic pseudo-random focus scores derived from `(level, node_idx)`.
        """
        B, max_len, _ = node_ids.shape
        if B == 0 or max_len == 0:
            return torch.zeros(B, max_len, dtype=torch.float32, device=node_ids.device)

        active = (
            torch.arange(max_len, device=node_ids.device)
            .unsqueeze(0)
            .expand(B, -1)
        ) < lengths.unsqueeze(1)
        levels = node_ids[..., 0].to(torch.float32)
        idxs = node_ids[..., 1].to(torch.float32)
        seed_val = float(seed)
        hash_val = levels * 12.9898 + idxs * 78.233 + seed_val
        noise = torch.sin(hash_val) * 43758.5453
        frac = noise - torch.floor(noise)
        scores = torch.where(active, frac * 2.0 - 1.0, torch.zeros_like(frac))
        return scores

    def _must_keep_mask(
        self,
        node_ids: torch.Tensor,
        flags: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a `[B, max_len]` bool mask marking nodes that cannot be removed.

        - Always keep the last token nodes defined by `tail_token_keep`.
        - Always keep the last node per level (root summaries).
        - Nodes already marked virtual are free to drop/compact.
        """
        B, max_len = node_ids.shape[:2]
        device = node_ids.device
        active = (
            torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        ) < lengths.unsqueeze(1)
        levels = node_ids[..., 0]

        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

        # Tail token keep
        tail_keep = self.config.tail_token_keep
        if tail_keep > 0:
            tokens = (levels == 0) & active
            rev_cumsum = torch.cumsum(tokens.flip(-1), dim=-1).flip(-1)
            mask |= tokens & (rev_cumsum <= tail_keep)

        # Last node per level (leverages contiguous per-level ordering)
        same_next = torch.zeros_like(levels, dtype=torch.bool, device=device)
        same_next[:, :-1] = (
            (levels[:, :-1] == levels[:, 1:]) & active[:, :-1] & active[:, 1:]
        )
        last_per_level = active & (~same_next)
        mask |= last_per_level

        virtual_bit = int(WCTFlags.VIRTUAL)
        virtual = (flags & virtual_bit) != 0
        mask &= ~virtual
        return mask

    def _allocate_mask(self, batch: int) -> torch.Tensor:
        return torch.zeros(
            batch,
            self.wct.max_nodes,
            dtype=torch.bool,
            device=self.device,
        )

    def _tree_total_nodes(self) -> int:
        if self.mct is None:
            return 0
        total = 0
        for level in range(self.mct.num_levels()):
            total += self.mct.level_latents(level).shape[1]
        return total

    def _flatten_all_levels(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Flatten the entire MegaContextTree into `[B, N, d]` tensors ordered by LOD.

        Returns:
            latents: `[B, N, d_model]`
            node_levels: `[B, N]`
            node_indices: `[B, N]`
            lengths: `[B]`
        """
        if self.mct is None or self.mct.batch_size is None:
            raise RuntimeError("MegaContextTree is not initialized.")

        latents_per_level = []
        node_levels_per_level = []
        node_indices_per_level = []
        batch = self.mct.batch_size
        device = self.device

        for level in range(self.mct.num_levels()):
            level_latents = self.mct.level_latents(level)  # [B, L, d]
            if level_latents.shape[1] == 0:
                continue
            latents_per_level.append(level_latents)
            level_count = level_latents.shape[1]
            node_levels_per_level.append(
                torch.full(
                    (batch, level_count),
                    level,
                    dtype=torch.long,
                    device=device,
                )
            )
            node_indices_per_level.append(
                torch.arange(
                    level_count,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0).expand(batch, -1)
            )

        if not latents_per_level:
            raise RuntimeError("MegaContextTree did not contain any nodes.")

        latents = torch.cat(latents_per_level, dim=1)
        node_levels = torch.cat(node_levels_per_level, dim=1)
        node_indices = torch.cat(node_indices_per_level, dim=1)
        lengths = torch.full(
            (batch,),
            latents.shape[1],
            dtype=torch.long,
            device=device,
        )
        return latents, node_levels, node_indices, lengths

    def _build_gistnet(
        self,
        kwargs: Dict[str, Any],
    ) -> Tuple[GistNet, torch.device, torch.dtype]:
        if not kwargs:
            raise ValueError("gistnet_kwargs must provide constructor arguments.")
        gistnet = GistNet(**kwargs)
        param = next(gistnet.parameters(), getattr(gistnet, "query_token", None))
        if param is None:
            raise RuntimeError("GistNet has no parameters to infer device/dtype from.")
        return gistnet, param.device, param.dtype


class TrainingContextManager(BaseContextManager):
    """
    Training façade: rebuilds MCT/WCT per batch and applies deterministic masks.
    """

    def __init__(
        self,
        config: TrainingContextManagerConfig,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(config, device=device, dtype=dtype)
        self.config = config
        seed = (
            config.virtual_dropout_seed
            if config.virtual_dropout_seed is not None
            else config.focus_seed
        )
        self._dropout_generator = torch.Generator(device="cpu")
        self._dropout_generator.manual_seed(int(seed))

    def build_batch(self, lod0_latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Materialize a fresh MegaContextTree + WorkingContextTree for a batch.

        Args:
            lod0_latents: `[B, T, d_model]` tensor of token-level latents.
        Returns:
            Dict with WCT tensors plus deterministic masks.
        """
        if lod0_latents.ndim != 3:
            raise ValueError("lod0_latents must be [B, T, d_model].")
        self.mct = self._new_mct()
        self.wct = self._new_wct()
        self.mct.append_lod0(lod0_latents)

        latents, node_levels, node_indices, lengths = self._flatten_all_levels()
        total_nodes = latents.shape[1]
        if total_nodes > self.wct.max_nodes:
            raise ValueError(
                f"WorkingContextTree capacity ({self.wct.max_nodes}) is smaller than "
                f"the tree size ({total_nodes}). Increase max_nodes in config."
            )
        self.wct.load_sequence(
            latents=latents,
            node_levels=node_levels,
            node_indices=node_indices,
            flags=None,
            lengths=lengths,
        )

        latents, node_ids, flags, lengths = self.wct.tensors()
        holdout_mask = self._build_holdout_mask(node_ids, lengths)
        virtual_mask = self._build_virtual_mask(node_ids, lengths, holdout_mask)
        if torch.any(virtual_mask):
            self.wct.mark_virtual(virtual_mask)
            # Refresh tensors to expose updated flags
            latents, node_ids, flags, lengths = self.wct.tensors()

        return {
            "latents": latents,
            "node_ids": node_ids,
            "flags": flags,
            "lengths": lengths,
            "virtual_mask": virtual_mask,
            "holdout_mask": holdout_mask,
        }

    def _build_virtual_mask(
        self,
        node_ids: torch.Tensor,
        lengths: torch.Tensor,
        holdout_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.wct.batch_size is None:
            raise RuntimeError("WorkingContextTree has no nodes to mask.")
        mask = self._allocate_mask(self.wct.batch_size)
        if self.config.virtual_dropout_prob <= 0:
            return mask

        _, _, flags, _ = self.wct.tensors()
        guard = self._must_keep_mask(node_ids, flags, lengths)
        holdout_active = holdout_mask[:, : node_ids.shape[1]]
        guard[:, : holdout_active.shape[1]] |= holdout_active

        max_len = node_ids.shape[1]
        active = (
            torch.arange(max_len, device=node_ids.device)
            .unsqueeze(0)
            .expand(self.wct.batch_size, -1)
        ) < lengths.unsqueeze(1)
        tokens = (node_ids[..., 0] == 0) & active
        eligible = tokens & (~guard[:, :max_len]) & (~holdout_active)
        if not torch.any(eligible):
            return mask

        rand = torch.rand(
            eligible.shape,
            generator=self._dropout_generator,
        ).to(eligible.device)
        drop = (rand < float(self.config.virtual_dropout_prob)) & eligible
        mask[:, :max_len] = drop
        return mask

    def _build_holdout_mask(
        self,
        node_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if self.wct.batch_size is None:
            raise RuntimeError("WorkingContextTree has no nodes to mask.")
        mask = self._allocate_mask(self.wct.batch_size)
        if self.config.holdout_tokens <= 0:
            return mask
        max_len = node_ids.shape[1]
        active = (
            torch.arange(max_len, device=node_ids.device)
            .unsqueeze(0)
            .expand(self.wct.batch_size, -1)
        ) < lengths.unsqueeze(1)
        tokens = (node_ids[..., 0] == 0) & active
        rev_cumsum = torch.cumsum(tokens.flip(-1), dim=-1).flip(-1)
        keep = tokens & (rev_cumsum <= self.config.holdout_tokens)
        mask[:, :max_len] = keep
        return mask


class InferenceContextManager(BaseContextManager):
    """
    Runtime façade: incrementally appends nodes and enforces a size window.
    """

    def __init__(
        self,
        config: InferenceContextManagerConfig,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(config, device=device, dtype=dtype)
        self.config = config
        self.mct = self._new_mct()

    def step(self, lod0_latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Append new LOD0 latents and expose the updated WCT tensors.
        """
        mct = self._ensure_mct()
        mct.append_lod0(lod0_latents)
        latents, node_levels, node_indices, lengths = self._flatten_all_levels()
        if latents.shape[1] > self.wct.max_nodes:
            raise ValueError(
                "WorkingContextTree capacity is insufficient for the mirrored tree."
            )
        self.wct.load_sequence(
            latents=latents,
            node_levels=node_levels,
            node_indices=node_indices,
            flags=None,
            lengths=lengths,
        )
        latents, node_ids, flags, lengths = self.wct.tensors()
        return {
            "latents": latents,
            "node_ids": node_ids,
            "flags": flags,
            "lengths": lengths,
        }

    def maybe_refocus(
        self,
        *,
        focus_override: Optional[torch.Tensor] = None,
        reason: Optional[str] = None,
    ) -> List[ContextOperation]:
        """
        Collapse low-focus leaves when the WCT exceeds the configured window.
        """
        if self.wct.batch_size is None:
            return []
        if self.config.target_node_count <= 0:
            return []

        latents, node_ids, flags, lengths = self.wct.tensors()
        max_len = node_ids.shape[1]
        if focus_override is not None:
            if focus_override.shape != (node_ids.shape[0], max_len):
                raise ValueError(
                    "focus_override must match the `[B, max_len]` tensor returned by WorkingContextTree."
                )
            focus_scores = focus_override
        else:
            focus_scores = self._placeholder_focus_scores(
                node_ids=node_ids,
                lengths=lengths,
                seed=self.config.focus_seed,
            )

        operations = self._collapse_if_needed(
            node_ids=node_ids,
            flags=flags,
            lengths=lengths,
            focus_scores=focus_scores,
            reason=reason,
        )
        return operations

    def force_refocus(
        self,
        *,
        focus_override: Optional[torch.Tensor] = None,
        reason: str = "manual",
    ) -> List[ContextOperation]:
        """Collapse nodes regardless of the current window."""
        if self.wct.batch_size is None:
            return []
        latents, node_ids, flags, lengths = self.wct.tensors()
        max_len = node_ids.shape[1]
        if focus_override is not None:
            if focus_override.shape != (node_ids.shape[0], max_len):
                raise ValueError(
                    "focus_override must match the `[B, max_len]` tensor returned by WorkingContextTree."
                )
            focus_scores = focus_override
        else:
            focus_scores = self._placeholder_focus_scores(
                node_ids=node_ids,
                lengths=lengths,
                seed=self.config.focus_seed,
            )

        return self._collapse_if_needed(
            node_ids=node_ids,
            flags=flags,
            lengths=lengths,
            focus_scores=focus_scores,
            reason=reason,
            force=True,
        )

    # ---------------------------------------------------------------- refocus logic
    def _collapse_if_needed(
        self,
        *,
        node_ids: torch.Tensor,
        flags: torch.Tensor,
        lengths: torch.Tensor,
        focus_scores: torch.Tensor,
        reason: Optional[str],
        force: bool = False,
    ) -> List[ContextOperation]:
        batch = node_ids.shape[0]
        device = node_ids.device
        mask = self._allocate_mask(batch)
        operations: List[ContextOperation] = []
        target = max(0, self.config.target_node_count)
        growth = max(0, self.config.target_node_growth)
        shrink = max(0, self.config.target_node_shrink)
        upper = target + growth
        lower = max(0, target - shrink)
        virtual_bit = int(WCTFlags.VIRTUAL)

        B, max_len = node_ids.shape[:2]
        active = (
            torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        ) < lengths.unsqueeze(1)
        live_mask = active & ((flags[:, :max_len] & virtual_bit) == 0)
        live_count = live_mask.sum(dim=1)

        if force:
            over_budget = live_count > 0
        else:
            over_budget = live_count > upper
        if not torch.any(over_budget):
            return operations

        must_keep = self._must_keep_mask(node_ids, flags, lengths)[:, :max_len]
        tail_tokens = (node_ids[..., 0] == 0) & active
        eligible_mask = tail_tokens & (~must_keep) & live_mask
        available = eligible_mask.sum(dim=1)

        lower_tensor = torch.full_like(live_count, lower)
        target_tensor = torch.full_like(live_count, target)
        desired = torch.where(live_count > lower_tensor, lower_tensor, target_tensor)
        if force:
            desired = target_tensor

        to_remove = torch.clamp(live_count - desired, min=0)
        to_remove = torch.minimum(to_remove, available)
        to_remove = torch.where(over_budget, to_remove, torch.zeros_like(to_remove))
        if not torch.any(to_remove):
            return operations

        eligible_scores = torch.where(
            eligible_mask,
            focus_scores[:, :max_len],
            torch.full_like(focus_scores[:, :max_len], float("inf")),
        )
        order = torch.argsort(eligible_scores, dim=1)
        range_idx = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        select_mask = range_idx < to_remove.unsqueeze(1)

        drop_long = torch.zeros(B, max_len, dtype=torch.long, device=device)
        drop_long.scatter_(1, order, select_mask.long())
        drop_mask = drop_long.to(torch.bool)

        mask[:, :max_len] = drop_mask
        if torch.any(drop_mask):
            self.wct.mark_virtual(mask)
            for b in range(batch):
                indices = torch.nonzero(drop_mask[b], as_tuple=False).flatten()
                if indices.numel() == 0:
                    continue
                operations.append(
                    ContextOperation(
                        op="collapse",
                        batch=b,
                        indices=indices.tolist(),
                        reason=reason,
                    )
                )
        return operations


__all__ = [
    "BaseContextManagerConfig",
    "ContextOperation",
    "InferenceContextManager",
    "InferenceContextManagerConfig",
    "TrainingContextManager",
    "TrainingContextManagerConfig",
]
