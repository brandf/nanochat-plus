import pytest
import torch

from mc.working_context_tree import WCTFlags, WorkingContextTree


def _make_wct(*, max_nodes: int = 8) -> WorkingContextTree:
    return WorkingContextTree(
        max_nodes=max_nodes,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_wct_lazy_init_infers_dims_from_first_append() -> None:
    """Constructor should learn batch/d_model/device from first append."""
    wct = WorkingContextTree(max_nodes=4)
    latents = torch.ones(1, 1, 3, dtype=torch.float64)
    levels = torch.zeros(1, 1, dtype=torch.long)
    idxs = torch.zeros(1, 1, dtype=torch.long)

    wct.append_nodes(latents=latents, node_levels=levels, node_indices=idxs)

    assert wct.batch_size == 1
    assert wct.d_model == 3
    assert wct.latents is not None and wct.latents.dtype == torch.float64


def test_wct_append_nodes_tracks_metadata() -> None:
    """Appending nodes should populate node_ids and leave flags at NONE."""
    torch.manual_seed(0)
    wct = _make_wct()
    latents = torch.randn(2, 2, 4)
    levels = torch.tensor([[0, 1], [0, 2]])
    idxs = torch.tensor([[0, 0], [1, 0]])

    wct.append_nodes(latents=latents, node_levels=levels, node_indices=idxs)

    lat, node_ids, flags, lengths = wct.tensors()
    torch.testing.assert_close(lat, latents)
    torch.testing.assert_close(node_ids[..., 0], levels)
    torch.testing.assert_close(node_ids[..., 1], idxs)
    assert torch.all(flags == int(WCTFlags.NONE))
    assert lengths.tolist() == [2, 2]
    assert not wct.tree_causal_ordering
    assert wct.fully_virtual


def test_wct_multi_append_preserves_metadata() -> None:
    """Multiple appends should accumulate nodes with correct ids and lengths."""
    torch.manual_seed(1)
    wct = _make_wct()
    first_latents = torch.randn(2, 2, 4)
    second_latents = torch.randn(2, 1, 4)
    wct.append_nodes(
        latents=first_latents,
        node_levels=torch.zeros(2, 2, dtype=torch.long),
        node_indices=torch.tensor([[0, 1], [0, 1]]),
    )
    wct.append_nodes(
        latents=second_latents,
        node_levels=torch.ones(2, 1, dtype=torch.long),
        node_indices=torch.tensor([[0], [0]]),
    )

    _, node_ids, _, lengths = wct.tensors()
    assert lengths.tolist() == [3, 3]
    torch.testing.assert_close(node_ids[0, :3, 0], torch.tensor([0, 0, 1]))
    torch.testing.assert_close(node_ids[1, :3, 1], torch.tensor([0, 1, 0]))


def test_wct_append_with_flags_and_zero_nodes() -> None:
    """Explicit flags should be honored and zero-length appends should no-op."""
    wct = _make_wct()
    latents = torch.zeros(2, 1, 4)
    levels = torch.zeros(2, 1, dtype=torch.long)
    idxs = torch.zeros(2, 1, dtype=torch.long)
    flag_mask = torch.full((2, 1), fill_value=int(WCTFlags.VIRTUAL), dtype=torch.long)
    wct.append_nodes(
        latents=latents,
        node_levels=levels,
        node_indices=idxs,
        flags=flag_mask,
    )
    lat, _, flags, lengths = wct.tensors()
    assert torch.all(lat == 0)
    assert torch.all(flags[:, 0] == int(WCTFlags.VIRTUAL))
    assert lengths.tolist() == [1, 1]

    empty_latents = torch.zeros(2, 0, 4)
    empty_meta = torch.zeros(2, 0, dtype=torch.long)
    wct.append_nodes(
        latents=empty_latents,
        node_levels=empty_meta,
        node_indices=empty_meta,
    )
    _, _, _, lengths_after = wct.tensors()
    assert lengths_after.tolist() == lengths.tolist()


def test_wct_mark_and_reinstate_virtual() -> None:
    """Virtual toggles should flip flag bits without changing lengths."""
    wct = _make_wct()
    latents = torch.zeros(2, 3, 4)
    levels = torch.zeros(2, 3, dtype=torch.long)
    idxs = torch.arange(3).repeat(2, 1)
    wct.append_nodes(latents=latents, node_levels=levels, node_indices=idxs)

    mask = torch.zeros(2, wct.max_nodes, dtype=torch.bool)
    mask[0, 1] = True
    wct.mark_virtual(mask)
    _, _, flags, lengths = wct.tensors()
    assert flags[0, 1].item() & int(WCTFlags.VIRTUAL)
    assert lengths.tolist() == [3, 3]

    wct.reinstate_virtual(mask)
    _, _, flags, _ = wct.tensors()
    assert flags[0, 1].item() == int(WCTFlags.NONE)


def test_wct_mask_shape_validation() -> None:
    """mark_virtual should reject masks that do not match [B, max_nodes]."""
    wct = _make_wct()
    latents = torch.zeros(2, 1, 4)
    levels = torch.zeros(2, 1, dtype=torch.long)
    idxs = torch.zeros(2, 1, dtype=torch.long)
    wct.append_nodes(latents=latents, node_levels=levels, node_indices=idxs)

    bad_mask = torch.zeros(1, wct.max_nodes, dtype=torch.bool)
    with pytest.raises(ValueError):
        wct.mark_virtual(bad_mask)


def test_wct_append_shape_validation() -> None:
    """append_nodes should enforce consistent `[B, K]` metadata shapes."""
    wct = _make_wct()
    latents = torch.zeros(1, 1, 4)
    levels = torch.zeros(2, 1, dtype=torch.long)
    idxs = torch.zeros(1, 1, dtype=torch.long)
    with pytest.raises(ValueError):
        wct.append_nodes(latents=latents, node_levels=levels, node_indices=idxs)


def test_wct_omit_nodes_compacts_and_sets_flag() -> None:
    """Omitting nodes should shrink lengths and set fully_virtual False."""
    wct = _make_wct()
    latents = torch.randn(2, 3, 4)
    levels = torch.zeros(2, 3, dtype=torch.long)
    idxs = torch.zeros(2, 3, dtype=torch.long)
    wct.append_nodes(latents=latents, node_levels=levels, node_indices=idxs)

    mask = torch.zeros(2, wct.max_nodes, dtype=torch.bool)
    mask[0, 0] = True
    wct.omit_nodes(mask)

    _, _, _, lengths = wct.tensors()
    assert lengths.tolist() == [2, 3]
    assert not wct.fully_virtual


def test_wct_compact_virtual_drops_nodes_and_resets_flags() -> None:
    """Compacting should drop virtual nodes and clear their flag bits."""
    wct = _make_wct()
    latents = torch.zeros(1, 4, 4)
    levels = torch.zeros(1, 4, dtype=torch.long)
    idxs = torch.arange(4).unsqueeze(0)
    wct.append_nodes(latents=latents, node_levels=levels, node_indices=idxs)

    mask = torch.zeros(1, wct.max_nodes, dtype=torch.bool)
    mask[0, 1] = True
    mask[0, 3] = True
    wct.mark_virtual(mask)
    wct.compact_virtual()

    _, _, flags, lengths = wct.tensors()
    assert lengths.tolist() == [2]
    assert torch.all(flags == int(WCTFlags.NONE))
    assert not wct.fully_virtual


def test_wct_rebuild_causal_order_restores_flag_and_order() -> None:
    """Rebuild should reorder latents per permutation and set causal flag."""
    wct = _make_wct()
    batch = 2
    d_model = 4
    latents = torch.arange(batch * 4 * d_model, dtype=torch.float32).view(
        batch, 4, d_model
    )
    levels = torch.zeros(batch, 4, dtype=torch.long)
    idxs = torch.arange(4).repeat(batch, 1)
    wct.append_nodes(latents=latents, node_levels=levels, node_indices=idxs)

    lat_before, _, _, _ = wct.tensors()
    lat_before = lat_before.clone()
    perm = torch.stack(
        [torch.arange(wct.max_nodes) for _ in range(batch)], dim=0
    )
    perm[:, :4] = torch.arange(3, -1, -1)
    wct.rebuild_causal_order(perm)
    assert wct.tree_causal_ordering
    lat_after, _, _, _ = wct.tensors()
    torch.testing.assert_close(lat_after[:, :4], torch.flip(lat_before[:, :4], dims=[1]))


def test_wct_permutation_validation() -> None:
    """Rebuild should reject permutations with out-of-range indices."""
    wct = _make_wct()
    latents = torch.zeros(2, 1, 4)
    levels = torch.zeros(2, 1, dtype=torch.long)
    idxs = torch.zeros(2, 1, dtype=torch.long)
    wct.append_nodes(latents=latents, node_levels=levels, node_indices=idxs)

    perm = torch.full((2, wct.max_nodes), fill_value=wct.max_nodes, dtype=torch.long)
    with pytest.raises(ValueError):
        wct.rebuild_causal_order(perm)


def test_wct_append_capacity_guard() -> None:
    """Appending beyond max_nodes should raise ValueError."""
    wct = _make_wct(max_nodes=2)
    latents = torch.zeros(2, 2, 4)
    levels = torch.zeros(2, 2, dtype=torch.long)
    idxs = torch.zeros(2, 2, dtype=torch.long)
    wct.append_nodes(latents=latents, node_levels=levels, node_indices=idxs)

    with pytest.raises(ValueError):
        wct.append_nodes(
            latents=torch.zeros(2, 1, 4),
            node_levels=torch.zeros(2, 1, dtype=torch.long),
            node_indices=torch.zeros(2, 1, dtype=torch.long),
        )


def test_wct_tensors_slice_to_max_length() -> None:
    """Returned tensors should truncate to the longest active sequence."""
    wct = _make_wct()
    latents = torch.zeros(2, 3, 4)
    levels = torch.zeros(2, 3, dtype=torch.long)
    idxs = torch.zeros(2, 3, dtype=torch.long)
    wct.append_nodes(latents=latents, node_levels=levels, node_indices=idxs)
    mask = torch.zeros(2, wct.max_nodes, dtype=torch.bool)
    mask[1, 1] = True
    wct.omit_nodes(mask)

    lat, _, _, lengths = wct.tensors()
    assert lat.shape[1] == int(lengths.max().item())
