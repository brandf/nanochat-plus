import pytest
import torch

from mc.gaussian_rope import GaussianRoPE


def _reference_rotary(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Apply standard RoPE rotation head-by-head for sanity checks."""
    b, t, h, d = x.shape
    num_pairs = d // 2
    x_flat = x.reshape(b * t, h, d).view(b * t, h, num_pairs, 2)
    theta_flat = theta.reshape(b * t, num_pairs)
    cos = theta_flat.cos().to(x.dtype).unsqueeze(1)
    sin = theta_flat.sin().to(x.dtype).unsqueeze(1)
    x1 = x_flat[..., 0]
    x2 = x_flat[..., 1]
    out0 = x1 * cos - x2 * sin
    out1 = x1 * sin + x2 * cos
    out = torch.stack((out0, out1), dim=-1).reshape(b, t, h, d)
    return out


def test_gaussian_rope_identity_when_mu_sigma_zero():
    rope = GaussianRoPE(d_head=64, sigma_base=1.0)
    q = torch.randn(2, 5, 3, 64)
    k = torch.randn(2, 5, 2, 64)
    mu = torch.zeros(2, 5)
    sigma = torch.full((2, 5), rope.sigma_base)

    q_rot, k_rot = rope(q, k, mu, sigma)

    torch.testing.assert_close(q, q_rot, rtol=0, atol=1e-6)
    torch.testing.assert_close(k, k_rot, rtol=0, atol=1e-6)


def test_gaussian_rope_only_scale_changes_second_half():
    d_head = 32
    rope = GaussianRoPE(d_head=d_head, sigma_base=1.0)
    q = torch.randn(1, 4, 2, d_head)
    k = torch.randn(1, 4, 2, d_head)
    mu = torch.zeros(1, 4)
    sigma = torch.tensor([[1.0, 2.0, 4.0, 8.0]])

    q_rot, _ = rope(q, k, mu, sigma)

    half = d_head // 2
    torch.testing.assert_close(
        q[..., :half], q_rot[..., :half], rtol=0, atol=1e-5
    )
    assert not torch.allclose(q[..., half:], q_rot[..., half:])


def test_gaussian_rope_time_rotation_matches_reference():
    d_head = 32
    rope = GaussianRoPE(d_head=d_head, sigma_base=1.0)
    q = torch.randn(1, 6, 2, d_head)
    k = torch.randn(1, 6, 2, d_head)
    mu = torch.arange(6, dtype=torch.float32).view(1, 6)
    sigma = torch.full_like(mu, rope.sigma_base)

    q_rot, k_rot = rope(q, k, mu, sigma)

    inv_freq = rope.time_inv_freq.to(q)
    theta = mu[..., None] * inv_freq
    q_time_ref = _reference_rotary(q[..., : d_head // 2], theta)
    k_time_ref = _reference_rotary(k[..., : d_head // 2], theta)
    torch.testing.assert_close(
        q_time_ref, q_rot[..., : d_head // 2], rtol=0, atol=1e-5
    )
    torch.testing.assert_close(
        k_time_ref, k_rot[..., : d_head // 2], rtol=0, atol=1e-5
    )


def test_gaussian_rope_scale_rotation_matches_reference():
    d_head = 32
    rope = GaussianRoPE(d_head=d_head, sigma_base=1.0)
    q = torch.randn(1, 5, 2, d_head)
    k = torch.randn(1, 5, 2, d_head)
    mu = torch.zeros(1, 5)
    sigma = rope.sigma_base * (2.0 ** torch.arange(5)).view(1, 5)

    q_rot, k_rot = rope(q, k, mu, sigma)

    inv_freq = rope.scale_inv_freq.to(q)
    log_sigma = torch.log2(sigma / rope.sigma_base)[..., None]
    theta = log_sigma * inv_freq
    q_scale_ref = _reference_rotary(q[..., d_head // 2 :], theta)
    k_scale_ref = _reference_rotary(k[..., d_head // 2 :], theta)
    torch.testing.assert_close(
        q_scale_ref, q_rot[..., d_head // 2 :], rtol=0, atol=1e-5
    )
    torch.testing.assert_close(
        k_scale_ref, k_rot[..., d_head // 2 :], rtol=0, atol=1e-5
    )


def test_gaussian_rope_preserves_vector_norms():
    rope = GaussianRoPE(d_head=48)
    q = torch.randn(2, 3, 3, 48)
    k = torch.randn(2, 3, 3, 48)
    mu = torch.linspace(0, 5, steps=6).view(2, 3)
    sigma = torch.linspace(1, 3, steps=6).view(2, 3)

    q_rot, k_rot = rope(q, k, mu, sigma)

    torch.testing.assert_close(
        torch.linalg.norm(q, dim=-1),
        torch.linalg.norm(q_rot, dim=-1),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.linalg.norm(k, dim=-1),
        torch.linalg.norm(k_rot, dim=-1),
        rtol=0,
        atol=1e-6,
    )


def test_gaussian_rope_handles_limited_scale_frequencies():
    rope = GaussianRoPE(d_head=64, max_scale_components=4)
    q = torch.randn(1, 3, 2, 64)
    k = torch.randn(1, 3, 2, 64)
    mu = torch.arange(3, dtype=torch.float32).view(1, 3)
    sigma = torch.tensor([[1.0, 2.0, 4.0]])

    q_rot, k_rot = rope(q, k, mu, sigma)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


@pytest.mark.skipif(
    not hasattr(torch, "compile"), reason="torch.compile not available"
)
def test_gaussian_rope_torch_compile_matches_eager():
    rope = GaussianRoPE(d_head=32)

    def fn(q, k, mu, sigma):
        return rope(q, k, mu, sigma)

    compiled_fn = torch.compile(fn, backend="eager")
    q = torch.randn(1, 4, 2, 32)
    k = torch.randn(1, 4, 2, 32)
    mu = torch.arange(4, dtype=torch.float32).view(1, 4)
    sigma = torch.full_like(mu, rope.sigma_base)

    eager = fn(q, k, mu, sigma)
    compiled = compiled_fn(q, k, mu, sigma)
    torch.testing.assert_close(
        compiled[0], eager[0], rtol=0, atol=1e-5
    )
    torch.testing.assert_close(
        compiled[1], eager[1], rtol=0, atol=1e-5
    )


def test_gaussian_rope_backpropagates_gradients():
    rope = GaussianRoPE(d_head=32)
    q = torch.randn(1, 5, 2, 32, requires_grad=True)
    k = torch.randn(1, 5, 2, 32, requires_grad=True)
    mu = torch.linspace(0, 4, steps=5).view(1, 5)
    sigma = torch.linspace(1.0, 2.0, steps=5).view(1, 5)

    q_rot, k_rot = rope(q, k, mu, sigma)
    loss = q_rot.square().sum() + k_rot.square().sum()
    loss.backward()
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
def test_gaussian_rope_supports_optional_q_or_k():
    rope = GaussianRoPE(d_head=32)
    q = torch.randn(1, 3, 2, 32)
    k = torch.randn(1, 3, 2, 32)
    mu = torch.arange(3, dtype=torch.float32).view(1, 3)
    sigma = torch.full_like(mu, rope.sigma_base)

    q_rot, k_rot = rope(q, None, mu, sigma)
    assert q_rot is not None
    assert q_rot.shape == q.shape
    assert k_rot is None
    q_rot2, k_rot2 = rope(None, k, mu, sigma)
    assert q_rot2 is None
    assert k_rot2 is not None
    assert k_rot2.shape == k.shape
