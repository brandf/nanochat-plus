# IMPLEMENTING - GUIDELINES & BEST PRACTICES

Use this checklist whenever you touch the shared codebase. It keeps new features aligned with nanochat style, fast, and easy to maintain across experiments.

---

## 1. Design Principles
- **Reuse nanochat patterns:** mirror existing APIs, tensor layouts, and naming. Favor small, modular helpers that slot into `nanochat/` or `scripts/` without surprise.
- **Keep it simple:** lean on clear control flow and direct tensor ops; avoid speculative abstractions.
- **GPU-friendly data flows:** store metadata in tensors when possible, minimize Python-side per-token work, and preserve batching.
- **Branch-portable code:** avoid sprinkling experiment-specific assumptions into shared helpers. Gate custom logic behind explicit configuration objects.

---

## 2. Performance Essentials
- **torch.compile ready:** structure code so it works under `torch.compile` (stable shapes, no hidden stateful globals, avoid Python-side shape switching).
- **Static-ish shapes:** preallocate buffers, avoid per-step tensor reallocations, and use masks instead of Python loops for element-wise filtering.
- **Efficient dtypes:** follow nanochat defaults (BF16 activations/embeddings on GPU, FP32 for numerically sensitive paths). Only downgrade if we already benchmarked the baseline.
- **Memory awareness:** reuse buffers, favor inplace ops when safe, and keep intermediate tensors lean to support long sequences.

---

## 3. Tensor Documentation
- **Inline shape comments:** append `# [B, T, d]` style notes to non-trivial tensor assignments.
- **Consistent naming:** reuse canonical names like `tokens`, `positions`, `mu`, `sigma`, `levels`, `cache`, `logits` so components compose cleanly.

---

## 4. Testing & Validation
- **Unit + integration:** follow `TESTING.md` to decide which suites must be updated. Every helper touched gets corresponding tests.
- **CPU-friendly configs:** shrink dimensions/sequence lengths for tests so they run in seconds.
- **Run tests every time:** execute the relevant `pytest` targets before handing off a change; capture the command in commit or PR notes when helpful.

---

## 5. Collaboration Protocol
- **Ask when unclear:** clarify ambiguous requirements or cross-branch impacts early.
- **Act independently otherwise:** once aligned, implement, self-review, and iterate without waiting for further prompts.
- **Document assumptions:** add short comments or commit notes when taking non-obvious paths (precision tweaks, cache eviction rules, etc.).
- **Self-review before handoff:** read your diffs end-to-end to ensure naming, shapes, and style match these guidelines.
- **No silent deviations:** if you must diverge from these rules, note it explicitly and explain why.

---

Following these guidelines keeps nanochat-plus development fast ("training goes brrr"), reliable, and ready for whatever experiment branches build on top.
