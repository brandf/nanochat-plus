import pytest
import torch

from mc.gistnet import GistNet
from mc.mega_context_tree import MegaContextTree


def _make_gistnet(d_model: int, block_size: int) -> GistNet:
    return GistNet(
        d_model=d_model,
        n_head=2,
        n_kv_head=2,
        sigma_level=1.0,
        block_size=block_size,
    )


def _assert_uniform_count(tree: MegaContextTree, level: int, expected: int) -> None:
    counts = tree.level_counts(level)
    assert counts.tolist() == [expected] * counts.shape[0]


def test_mct_prefill_establishes_levels_counts_and_padding() -> None:
    """Prefill should build every level that has complete blocks and expose dense tensors."""
    torch.manual_seed(0)
    d_model = 8
    block_size = 2
    initial = torch.randn(1, block_size * 2, d_model)
    tree = MegaContextTree(_make_gistnet(d_model, block_size), initial_lod0=initial)

    assert tree.num_levels() == 3
    _assert_uniform_count(tree, 0, 4)
    _assert_uniform_count(tree, 1, 2)
    _assert_uniform_count(tree, 2, 1)

    padded, counts = tree.padded_level(1)
    torch.testing.assert_close(padded, tree.level_latents(1))
    torch.testing.assert_close(counts, torch.tensor([2], dtype=torch.long))


def test_mct_partial_blocks_delay_parent_creation_until_full() -> None:
    """Parents only appear when enough children exist to fill a block."""
    torch.manual_seed(1)
    d_model = 8
    block_size = 3
    tree = MegaContextTree(_make_gistnet(d_model, block_size))

    chunk_partial = torch.randn(1, block_size - 1, d_model)
    tree.append_lod0(chunk_partial)
    assert tree.num_levels() == 1
    _assert_uniform_count(tree, 0, block_size - 1)

    chunk_finish = torch.randn(1, 1, d_model)
    tree.append_lod0(chunk_finish)
    _assert_uniform_count(tree, 0, block_size)
    _assert_uniform_count(tree, 1, 1)

    lod0 = tree.level_latents(0)[0]
    with torch.no_grad():
        expected = tree.gistnet(lod0)
    torch.testing.assert_close(tree.level_latents(1)[0, 0], expected)


def test_mct_batch_appends_stay_in_lockstep() -> None:
    """All batch entries must grow equally and produce identical counts per level."""
    torch.manual_seed(2)
    d_model = 8
    block_size = 2
    tree = MegaContextTree(_make_gistnet(d_model, block_size))

    chunk1 = torch.randn(2, block_size, d_model)
    tree.append_lod0(chunk1)
    _assert_uniform_count(tree, 0, 2)
    _assert_uniform_count(tree, 1, 1)

    chunk2 = torch.randn(2, block_size, d_model)
    tree.append_lod0(chunk2)
    _assert_uniform_count(tree, 0, 4)
    _assert_uniform_count(tree, 1, 2)
    _assert_uniform_count(tree, 2, 1)

    lod1 = tree.level_latents(1)
    with torch.no_grad():
        expected_second = torch.stack(
            [tree.gistnet(chunk2[b]) for b in range(2)], dim=0
        ).unsqueeze(1)
    torch.testing.assert_close(lod1[:, 1:2], expected_second)


def test_mct_respects_max_lods_limit() -> None:
    """max_lods should cap how many LOD tensors are materialized."""
    torch.manual_seed(3)
    d_model = 8
    block_size = 2
    gist = _make_gistnet(d_model, block_size)
    tree = MegaContextTree(gist, max_lods=2)

    chunk = torch.randn(1, block_size * 4, d_model)
    tree.append_lod0(chunk)

    assert tree.num_levels() == 2  # LOD0 + LOD1 only
    assert tree.level_latents(0).shape[1] == 8
    assert tree.level_latents(1).shape[1] == 4


def test_mct_level_queries_before_data_raise() -> None:
    """Accessing levels before any append should fail loudly."""
    tree = MegaContextTree(_make_gistnet(d_model=8, block_size=2))
    with pytest.raises(RuntimeError):
        tree.level_latents(0)
    with pytest.raises(RuntimeError):
        tree.level_counts(0)


def test_mct_accepts_2d_append_for_batch_one() -> None:
    """When batch size is 1, `[N, d]` inputs should be auto-expanded."""
    torch.manual_seed(4)
    d_model = 8
    block_size = 2
    tree = MegaContextTree(_make_gistnet(d_model, block_size))

    chunk = torch.randn(block_size * 3, d_model)
    tree.append_lod0(chunk)

    _assert_uniform_count(tree, 0, 6)
    assert tree.level_latents(0).shape == (1, 6, d_model)


def test_mct_num_levels_tracks_new_parents() -> None:
    """num_levels should grow as new LODs become populated."""
    torch.manual_seed(5)
    d_model = 8
    block_size = 2
    tree = MegaContextTree(_make_gistnet(d_model, block_size))

    tree.append_lod0(torch.randn(1, block_size, d_model))
    assert tree.num_levels() == 2  # LOD0 + first parent level

    tree.append_lod0(torch.randn(1, block_size, d_model))
    assert tree.num_levels() == 3  # LOD2 now exists


def test_mct_rejects_invalid_appends_after_init() -> None:
    """After establishing batch size, ragged appends should be rejected."""
    d_model = 8
    block_size = 2
    tree = MegaContextTree(_make_gistnet(d_model, block_size))

    tree.append_lod0(torch.randn(2, block_size, d_model))

    with pytest.raises(ValueError):
        tree.append_lod0(torch.randn(1, block_size, d_model))

    with pytest.raises(ValueError):
        tree.append_lod0(torch.empty(2, 0, d_model))
