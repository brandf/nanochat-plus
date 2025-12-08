# IMPLEMENTING – GUIDELINES & BEST PRACTICES

Use this checklist whenever you touch the codebase. It keeps new features aligned with nanochat style, fast, and easy to maintain.

---

## 1. Design Principles
- **Reuse nanochat patterns:** when adding MegaContext modules, mirror existing nanochat APIs, tensor layouts, and coding style. Favor small, modular helpers.
- **Keep it simple:** avoid unnecessary abstractions. Clear, readable code is easier to compile, optimize, and debug.
- **GPU-friendly data flows:** store metadata (positions, levels, etc.) in tensors whenever possible. Minimize Python objects or per-node data structures that break batching.

---

## 2. Performance Essentials
- **torch.compile everywhere:** plan algorithms so they work with `torch.compile` (no dynamic graph captures, no Python-side control flow that changes shapes). Enable it by default in new modules.
- **Static shapes when possible:** preallocate buffers, avoid shape-dependent branches during forward passes, and use Boolean masks instead of Python loops for per-element decisions.
- **Efficient data types:** follow nanochat’s defaults—BF16 for activations/embeddings on GPUs, FP32 where precision is required (e.g., logits before softmax). Downgrade to FP16 only if the baseline nanochat model already does so or benchmarks show clear gains.
- **Memory-aware code:** prefer in-place operations (when safe), share buffers, and rely on contiguous tensors. Keep intermediate tensors lean to prevent slowdowns in forward/backward passes.

---

## 3. Tensor Documentation
- **Inline shape comments:** append `# [B, T, d]`-style comments to every non-trivial tensor assignment so reviewers immediately see dimensionality.
- **Consistent naming:** use `mu`, `sigma`, `level`, `latent`, `node_ids`, etc., consistently across modules to avoid confusion.

---

## 4. Testing & Validation
- **Unit + integration tests:** follow `TESTING.md` requirements—write focused unit tests for each helper (e.g., Gaussian RoPE) and integration tests for multi-module flows.
- **CPU-friendly test configs:** shrink dimensions/sequence lengths for tests so they run quickly on CPUs without CUDA.
- **Run tests every time:** before finishing any task, run the relevant `pytest` targets locally and ensure they pass.

---

## 5. Collaboration Protocol
- **Ask when unclear:** if requirements are ambiguous, pause and confirm with the human rather than guessing.
- **Otherwise act independently:** once requirements are clear, implement, test, and iterate without waiting for extra prompts.
- **Document assumptions:** note design choices (e.g., head splits, precision tweaks) in code comments or PR summaries so others know the rationale.
- **Self-review before handoff:** re-read your diffs and the touched files end-to-end to ensure they honor these guidelines (naming, shape comments, tensor usage) and don’t leave “turd” leftovers for others to discover.
- **No silent deviations:** Never execute an implementation strategy that differs from explicit instructions without asking first. If a request seems infeasible or unclear, pause and confirm rather than improvising.

---

Following these guidelines keeps MegaContext development fast (“training goes brrr”), reliable, and aligned with nanochat’s minimal style.
