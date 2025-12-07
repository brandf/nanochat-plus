import copy

import pytest
import torch
import torch.nn as nn

from mc.gistnet import GistNet


class _NoOpRoPE(nn.Module):
    def forward(self, q, k, mu, sigma):
        return q, k


class _SpyRoPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, q, k, mu, sigma):
        self.calls.append(
            {
                "has_q": q is not None,
                "has_k": k is not None,
                "mu": mu.detach().clone(),
                "sigma": sigma.detach().clone(),
            }
        )
        return q, k


def _make_identity_gistnet(d_model: int = 4) -> GistNet:
    net = GistNet(d_model=d_model, n_head=1, n_kv_head=1, sigma_level=1.0)
    with torch.no_grad():
        eye = torch.eye(d_model)
        net.attn.q_proj.weight.copy_(eye)
        net.attn.k_proj.weight.copy_(eye)
        net.attn.v_proj.weight.copy_(eye)
        net.attn.out_proj.weight.copy_(eye)
        net.query_token.copy_(torch.tensor([1.0] + [0.0] * (d_model - 1)))
    net.attn.gaussian_rope = _NoOpRoPE()
    net.mlp = nn.Identity()
    return net


def test_gistnet_forward_shape_and_gradients():
    torch.manual_seed(0)
    net = GistNet(d_model=32, n_head=4, n_kv_head=2, sigma_level=1.5)
    children = torch.randn(4, 32, requires_grad=True)

    output = net(children)

    assert output.shape == (32,)
    loss = output.sum()
    loss.backward()
    assert torch.isfinite(children.grad).all()


def test_gistnet_attention_weights_sum_to_one():
    torch.manual_seed(1)
    net = GistNet(d_model=48, n_head=6, n_kv_head=3, sigma_level=2.0)
    children = torch.randn(5, 48)

    _ = net(children)
    attn = net.last_attention_weights

    assert attn is not None
    weights_sum = attn.sum(dim=-1)
    torch.testing.assert_close(
        weights_sum,
        torch.ones_like(weights_sum),
        rtol=0,
        atol=1e-6,
    )


def test_gistnet_rejects_empty_child_spans():
    net = GistNet(d_model=16, n_head=2, n_kv_head=2, sigma_level=1.0)
    with pytest.raises(ValueError):
        net(torch.empty(0, 16))


def test_gistnet_attention_prefers_similarity():
    net = _make_identity_gistnet(d_model=4)
    children = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
        ]
    )

    _ = net(children)
    attn = net.last_attention_weights
    assert attn is not None
    # Query aligns with first child so it must receive higher attention.
    assert attn[0, 0, 0, 0] > attn[0, 0, 0, 1]


def test_gistnet_passes_expected_mu_sigma_to_rope():
    spy = _SpyRoPE()
    net = GistNet(d_model=8, n_head=2, n_kv_head=2, sigma_level=3.0)
    net.attn.gaussian_rope = spy
    children = torch.randn(3, 8)

    _ = net(children)

    assert len(spy.calls) == 2  # one for query, one for keys
    query_call = next(call for call in spy.calls if call["has_q"])
    key_call = next(call for call in spy.calls if call["has_k"])

    torch.testing.assert_close(
        query_call["mu"], torch.full((1, 1), 2.0)  # G-1 when G=3
    )
    torch.testing.assert_close(
        query_call["sigma"], torch.full((1, 1), net.sigma_level)
    )
    torch.testing.assert_close(
        key_call["mu"], torch.arange(3, dtype=torch.float32).view(1, 3)
    )
    torch.testing.assert_close(
        key_call["sigma"], torch.full((1, 3), net.sigma_level)
    )


def test_gistnet_sigma_level_changes_output():
    torch.manual_seed(2)
    base = GistNet(d_model=32, n_head=4, n_kv_head=2, sigma_level=1.0)
    children = torch.randn(4, 32)

    variant = GistNet(d_model=32, n_head=4, n_kv_head=2, sigma_level=4.0)
    variant.load_state_dict(copy.deepcopy(base.state_dict()))

    out_base = base(children)
    out_variant = variant(children)

    assert not torch.allclose(out_base, out_variant, rtol=1e-8, atol=1e-8)


@pytest.mark.skipif(
    not hasattr(torch, "compile"), reason="torch.compile not available"
)
def test_gistnet_torch_compile_matches_eager():
    net = GistNet(d_model=32, n_head=4, n_kv_head=2, sigma_level=1.0)
    children = torch.randn(4, 32)

    compiled = torch.compile(net, backend="eager")
    eager_out = net(children)
    compiled_out = compiled(children)
    torch.testing.assert_close(compiled_out, eager_out, rtol=0, atol=1e-5)
def test_gistnet_end_to_end_integration_on_toy_tree():
    """Build two LOD1 nodes with known children and verify gists match expectations."""
    torch.manual_seed(3)
    net = GistNet(d_model=16, n_head=4, n_kv_head=2, sigma_level=1.0)
    # span A children, span B children
    span_a = torch.randn(4, 16)
    span_b = torch.randn(4, 16)

    gist_a = net(span_a)
    gist_b = net(span_b)

    # Ensure gists are distinct and remain consistent if spans are reprocessed.
    assert not torch.allclose(gist_a, gist_b)
    torch.testing.assert_close(gist_a, net(span_a), rtol=0, atol=1e-6)
    torch.testing.assert_close(gist_b, net(span_b), rtol=0, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gistnet_handles_multiple_dtypes(dtype):
    net = GistNet(d_model=32, n_head=4, n_kv_head=2, sigma_level=1.0).to(dtype=dtype)
    children = torch.randn(6, 32, dtype=dtype)

    output = net(children)
    assert output.dtype == dtype


def test_gistnet_supports_varied_head_ratios_and_span_sizes():
    torch.manual_seed(4)
    net = GistNet(d_model=48, n_head=6, n_kv_head=3, sigma_level=1.0)
    children = torch.randn(8, 48)

    _ = net(children)
    attn = net.last_attention_weights
    assert attn is not None
    # Ensure each head's attention sums to 1 even with GQA duplication.
    torch.testing.assert_close(
        attn.sum(dim=-1),
        torch.ones_like(attn.sum(dim=-1)),
        rtol=0,
        atol=1e-6,
    )
