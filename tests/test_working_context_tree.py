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
    wct = WorkingContextTree(max_nodes=4)
    latents = torch.ones(1, 1, 3, dtype=torch.float64)
    levels = torch.zeros(1, 1, dtype=torch.long)
    idxs = torch.zeros(1, 1, dtype=torch.long)

    wct.append_nodes(latents=latents, node_levels=levels, node_indices=idxs)

    assert wct.batch_size == 1
    assert wct.d_model == 3
    assert wct.latents is not None and wct.latents.dtype == torch.float64


def test_wct_append_nodes_tracks_metadata() -> None:
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


def test_wct_multi_append_preserves_metadata() -> None:
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
    wct = _make_wct()
    latents = torch.zeros(2, 1, 4)
    levels = torch.zeros(2, 1, dtype=torch.long)
    idxs = torch.zeros(2, 1, dtype=torch.long)
    flag_mask = torch.full((2, 1), fill_value=int(WCTFlags.MASKED), dtype=torch.long)
    wct.append_nodes(
        latents=latents,
        node_levels=levels,
        node_indices=idxs,
        flags=flag_mask,
    )
    lat, _, flags, lengths = wct.tensors()
    assert torch.all(lat == 0)
    assert torch.all(flags[:, 0] == int(WCTFlags.MASKED))
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


def test_wct_mark_and_reinstate_masked() -> None:
    wct = _make_wct()
    latents = torch.zeros(2, 3, 4)
    levels = torch.zeros(2, 3, dtype=torch.long)
    idxs = torch.arange(3).repeat(2, 1)
    wct.append_nodes(latents=latents, node_levels=levels, node_indices=idxs)

    mask = torch.zeros(2, wct.max_nodes, dtype=torch.bool)
    mask[0, 1] = True
    wct.mark_masked(mask)
    _, _, flags, lengths = wct.tensors()
    assert flags[0, 1].item() & int(WCTFlags.MASKED)
    assert lengths.tolist() == [3, 3]

    wct.reinstate_masked(mask)
    _, _, flags, _ = wct.tensors()
    assert flags[0, 1].item() == int(WCTFlags.NONE)


def test_wct_mask_shape_validation() -> None:
    wct = _make_wct()
    latents = torch.zeros(2, 1, 4)
    levels = torch.zeros(2, 1, dtype=torch.long)
    idxs = torch.zeros(2, 1, dtype=torch.long)
    wct.append_nodes(latents=latents, node_levels=levels, node_indices=idxs)

    bad_mask = torch.zeros(1, wct.max_nodes, dtype=torch.bool)
    with pytest.raises(ValueError):
        wct.mark_masked(bad_mask)


def test_wct_append_shape_validation() -> None:
    wct = _make_wct()
    latents = torch.zeros(1, 1, 4)
    levels = torch.zeros(2, 1, dtype=torch.long)
    idxs = torch.zeros(1, 1, dtype=torch.long)
    with pytest.raises(ValueError):
        wct.append_nodes(latents=latents, node_levels=levels, node_indices=idxs)


def test_wct_append_capacity_guard() -> None:
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


def test_wct_load_sequence_overwrites_entire_tree() -> None:
    wct = _make_wct()
    latents = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
    levels = torch.tensor([[0, 1, 2]], dtype=torch.long)
    idxs = torch.tensor([[0, 0, 0]], dtype=torch.long)
    flags = torch.full((1, 3), fill_value=int(WCTFlags.MASKED), dtype=torch.long)
    lengths = torch.tensor([2], dtype=torch.long)

    wct.load_sequence(
        latents=latents,
        node_levels=levels,
        node_indices=idxs,
        flags=flags,
        lengths=lengths,
    )

    lat, node_ids, wct_flags, wct_lengths = wct.tensors()
    torch.testing.assert_close(lat[:, :2], latents[:, :2])
    torch.testing.assert_close(node_ids[0, :2, 0], levels[0, :2])
    torch.testing.assert_close(node_ids[0, :2, 1], idxs[0, :2])
    assert torch.all(wct_flags[:, :2] == int(WCTFlags.MASKED))
    assert wct_lengths.tolist() == [2]
