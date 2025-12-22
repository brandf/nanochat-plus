import pytest
import torch

from mc.context_manager import (
    ContextOperation,
    InferenceContextManager,
    InferenceContextManagerConfig,
    TrainingContextManager,
    TrainingContextManagerConfig,
)
from mc.working_context_tree import WCTFlags


def _gistnet_kwargs(d_model: int = 8, block_size: int = 2) -> dict:
    return {
        "d_model": d_model,
        "n_head": 2,
        "n_kv_head": 2,
        "sigma_level": 1.0,
        "block_size": block_size,
    }


def _rand_latents(batch: int, tokens: int, d_model: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, tokens, d_model)


def test_training_context_manager_builds_masks_and_masked_flags() -> None:
    config = TrainingContextManagerConfig(
        max_nodes=32,
        tail_token_keep=1,
        holdout_tokens=1,
        mask_dropout_probs=[1.0, 0.5, 0.0],
        mask_dropout_seed=123,
        gistnet_kwargs=_gistnet_kwargs(),
    )
    manager = TrainingContextManager(config)
    lod0_latents = _rand_latents(1, 4, config.gistnet_kwargs["d_model"])

    bundle = manager.build_batch(lod0_latents)

    lengths = bundle["lengths"]
    assert lengths.tolist() == [7], "Expected tokens+gists to populate the tree."
    assert torch.any(bundle["holdout_mask"])

    masked_bit = int(WCTFlags.MASKED)
    flags = bundle["flags"]
    active_len = flags.shape[1]
    holdout = bundle["holdout_mask"][:, :active_len].bool()
    assert torch.any(flags & masked_bit), "Masked dropout should set flag bits."
    assert torch.all((flags & masked_bit)[holdout] == 0), "Holdout nodes must remain active."


def test_training_masked_dropout_deterministic_per_seed() -> None:
    base_kwargs = dict(
        max_nodes=32,
        tail_token_keep=0,
        holdout_tokens=0,
        mask_dropout_probs=[0.5, 0.25, 0.0],
        gistnet_kwargs=_gistnet_kwargs(),
    )
    latents = _rand_latents(2, 6, base_kwargs["gistnet_kwargs"]["d_model"])

    config_a = TrainingContextManagerConfig(**base_kwargs, mask_dropout_seed=7)
    config_b = TrainingContextManagerConfig(**base_kwargs, mask_dropout_seed=7)
    config_c = TrainingContextManagerConfig(**base_kwargs, mask_dropout_seed=11)

    mask_a = TrainingContextManager(config_a).build_batch(latents)["masked_mask"]
    mask_b = TrainingContextManager(config_b).build_batch(latents)["masked_mask"]
    mask_c = TrainingContextManager(config_c).build_batch(latents)["masked_mask"]

    assert torch.equal(mask_a, mask_b), "Seeds must produce deterministic masked masks."
    assert not torch.equal(mask_a, mask_c), "Different seeds should decorrelate masked masks."


def test_training_holdout_resists_masked_dropout() -> None:
    config = TrainingContextManagerConfig(
        max_nodes=32,
        tail_token_keep=0,
        holdout_tokens=2,
        mask_dropout_probs=[1.0, 0.5, 0.0],
        mask_dropout_seed=123,
        gistnet_kwargs=_gistnet_kwargs(),
    )
    manager = TrainingContextManager(config)
    bundle = manager.build_batch(_rand_latents(1, 6, config.gistnet_kwargs["d_model"]))

    masked_bit = int(WCTFlags.MASKED)
    flags = bundle["flags"]
    active_len = flags.shape[1]
    holdout_mask = bundle["holdout_mask"][:, :active_len].bool()
    assert torch.all((flags & masked_bit)[holdout_mask] == 0)


def _expected_total_nodes(num_tokens: int, block_size: int) -> int:
    total = 0
    level_tokens = num_tokens
    power = 0
    while level_tokens > 0:
        total += level_tokens
        power += 1
        level_tokens //= block_size
    return total


def test_inference_manager_step_flattens_full_tree_each_time() -> None:
    config = InferenceContextManagerConfig(
        max_nodes=64,
        target_node_count=4,
        tail_token_keep=1,
        gistnet_kwargs=_gistnet_kwargs(),
    )
    manager = InferenceContextManager(config)
    d_model = config.gistnet_kwargs["d_model"]

    out1 = manager.step(_rand_latents(1, 2, d_model))
    length1 = int(out1["lengths"][0])
    assert length1 == _expected_total_nodes(2, config.gistnet_kwargs["block_size"])

    out2 = manager.step(_rand_latents(1, 2, d_model))
    length2 = int(out2["lengths"][0])
    assert length2 == _expected_total_nodes(4, config.gistnet_kwargs["block_size"])

    node_levels = out2["node_ids"][0, :length2, 0]
    tokens = 4
    level_counts = []
    remaining = tokens
    level = 0
    while remaining > 0:
        level_counts.append((level, remaining))
        remaining //= config.gistnet_kwargs["block_size"]
        level += 1
    offset = 0
    for lvl, count in level_counts:
        assert torch.all(node_levels[offset : offset + count] == lvl)
        offset += count


def test_inference_manager_refocus_respects_tail_window() -> None:
    config = InferenceContextManagerConfig(
        max_nodes=64,
        target_node_count=3,
        target_node_growth=0,
        target_node_shrink=0,
        tail_token_keep=2,
        gistnet_kwargs=_gistnet_kwargs(),
    )
    manager = InferenceContextManager(config)
    manager.step(_rand_latents(1, 6, config.gistnet_kwargs["d_model"]))

    _ = manager.maybe_refocus()
    _, node_ids, flags, lengths = manager.wct.tensors()
    masked_bit = int(WCTFlags.MASKED)
    active_len = int(lengths[0])
    tail_tokens = torch.nonzero(
        node_ids[0, :active_len, 0] == 0, as_tuple=False
    ).flatten()
    keep = tail_tokens[-config.tail_token_keep :]
    assert torch.all((flags[0, keep] & masked_bit) == 0), "Tail tokens must never collapse."


def test_inference_focus_override_shape_validation() -> None:
    config = InferenceContextManagerConfig(
        max_nodes=32,
        target_node_count=2,
        gistnet_kwargs=_gistnet_kwargs(),
    )
    manager = InferenceContextManager(config)
    manager.step(_rand_latents(1, 2, config.gistnet_kwargs["d_model"]))

    bad_override = torch.zeros(1, 1)
    with pytest.raises(ValueError):
        manager.maybe_refocus(focus_override=bad_override)


def test_inference_force_refocus_uses_focus_override() -> None:
    config = InferenceContextManagerConfig(
        max_nodes=32,
        target_node_count=1,
        tail_token_keep=0,
        gistnet_kwargs=_gistnet_kwargs(),
    )
    manager = InferenceContextManager(config)
    step_out = manager.step(_rand_latents(1, 4, config.gistnet_kwargs["d_model"]))
    length = int(step_out["lengths"][0])
    override = torch.ones(1, step_out["node_ids"].shape[1])
    override[:, : length - 1] = -1.0  # prefer dropping everything except last entry

    ops = manager.force_refocus(focus_override=override, reason="override")
    assert ops and all(isinstance(op, ContextOperation) for op in ops)
    _, node_ids, flags, lengths = manager.wct.tensors()
    masked_bit = int(WCTFlags.MASKED)
    active = int(lengths[0])
    token_mask = node_ids[0, :active, 0] == 0
    live_tokens = int(((flags[0, :active] & masked_bit) == 0)[token_mask].sum().item())
    assert live_tokens <= config.target_node_count


def test_inference_multi_batch_refocus_independent() -> None:
    config = InferenceContextManagerConfig(
        max_nodes=128,
        target_node_count=4,
        target_node_growth=0,
        tail_token_keep=1,
        gistnet_kwargs=_gistnet_kwargs(),
    )
    manager = InferenceContextManager(config)
    latents = _rand_latents(2, 6, config.gistnet_kwargs["d_model"])
    manager.step(latents)

    ops = manager.maybe_refocus()
    assert ops, "Both batch entries should trigger refocus."
    _, node_ids, flags, lengths = manager.wct.tensors()
    masked_bit = int(WCTFlags.MASKED)
    for b in range(2):
        active = int(lengths[b])
        token_mask = node_ids[b, :active, 0] == 0
        live_tokens = int(((flags[b, :active] & masked_bit) == 0)[token_mask].sum().item())
        assert live_tokens <= config.target_node_count
