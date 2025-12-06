import torch

from mc.gaussian_rope import GaussianRoPE


def test_gaussian_rope_identity_when_mu_sigma_zero():
    rope = GaussianRoPE(d_head=64, sigma_base=1.0)
    q = torch.randn(2, 5, 3, 64)
    k = torch.randn(2, 5, 2, 64)
    mu = torch.zeros(2, 5)
    sigma = torch.full((2, 5), rope.sigma_base)

    q_rot, k_rot = rope(q, k, mu, sigma)

    assert torch.allclose(q, q_rot, atol=1e-6)
    assert torch.allclose(k, k_rot, atol=1e-6)


def test_gaussian_rope_only_scale_changes_second_half():
    d_head = 32
    rope = GaussianRoPE(d_head=d_head, sigma_base=1.0)
    q = torch.randn(1, 4, 2, d_head)
    k = torch.randn(1, 4, 2, d_head)
    mu = torch.zeros(1, 4)
    sigma = torch.tensor([[1.0, 2.0, 4.0, 8.0]])

    q_rot, _ = rope(q, k, mu, sigma)

    half = d_head // 2
    assert torch.allclose(q[..., :half], q_rot[..., :half], atol=1e-5)
    assert not torch.allclose(q[..., half:], q_rot[..., half:])


def test_gaussian_rope_preserves_vector_norms():
    rope = GaussianRoPE(d_head=48)
    q = torch.randn(2, 3, 3, 48)
    k = torch.randn(2, 3, 3, 48)
    mu = torch.linspace(0, 5, steps=6).view(2, 3)
    sigma = torch.linspace(1, 3, steps=6).view(2, 3)

    q_rot, k_rot = rope(q, k, mu, sigma)

    q_norm = torch.linalg.norm(q, dim=-1)
    q_rot_norm = torch.linalg.norm(q_rot, dim=-1)
    k_norm = torch.linalg.norm(k, dim=-1)
    k_rot_norm = torch.linalg.norm(k_rot, dim=-1)
    assert torch.allclose(q_norm, q_rot_norm, atol=1e-6)
    assert torch.allclose(k_norm, k_rot_norm, atol=1e-6)


def test_gaussian_rope_handles_limited_scale_frequencies():
    rope = GaussianRoPE(d_head=64, max_scale_components=4)
    q = torch.randn(1, 3, 2, 64)
    k = torch.randn(1, 3, 2, 64)
    mu = torch.arange(3, dtype=torch.float32).view(1, 3)
    sigma = torch.tensor([[1.0, 2.0, 4.0]])

    q_rot, k_rot = rope(q, k, mu, sigma)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
