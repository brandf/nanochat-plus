# TESTING - PRINCIPLES & PRACTICES

This document defines how we design, implement, and run tests across the shared nanochat-plus codebase. Treat it as a contract: every contribution to `main` must honor these expectations so experiments remain reliable.

---

## 1. Core Principles

- **Purposeful coverage:** each test validates a specific behavior or invariant; do not duplicate implementation logic.
- **Fail fast, fail local:** catch issues on laptops/CPUs before they leak into multi-hour GPU runs.
- **Blame clarity:** when a test fails, decide whether the issue is in the test, the implementation, or both, then fix the right layer.
- **Continuous validation:** after any code change, run the relevant tests locally and require green status before sharing work.
- **Regression capture:** every discovered bug earns a regression test before the fix lands.

---

## 2. Test Types & Scope

1. **Unit Tests**
   - Scope: single module/class/function (e.g., rotary helpers, token batching utilities, KV cache updates).
   - Expectations:
     - Cover happy-path plus edge cases (empty inputs, boundary values, dtype mismatches).
     - Use deterministic seeds and small tensors so tests stay CPU friendly.
     - Assert shapes, invariants, and numerical properties (attention weights sum to 1, caches round-trip correctly, etc.).

2. **Integration Tests**
   - Scope: multiple modules working together (e.g., dataloader -> model -> loss, engine + KV cache repair, tokenizer + dataset sampling).
   - Expectations:
     - Use toy-sized models/sequences (T<=32, d_model<=64) to keep runtimes in the seconds range.
     - Assert end-to-end properties (finite losses, monotonic metrics, invariant preservation).

3. **Functional/Scenario Tests**
   - Scope: higher-level workflows (single training step, inference loop with cache reuse, CLI entrypoints).
   - Expectations:
     - Mirror real usage with scaled-down configs.
     - Check logs/metrics when appropriate (token throughput, cache hit counts, etc.).

---

## 3. CPU-Only / Toy Config Guidelines

- Default to CPU execution; tests must not require CUDA.
- Prefer tiny configs:
  - `d_model`: 16-64
  - `n_layer`: 2-4
  - `sequence_len`: 8-32
  - Vocabulary: <=256 tokens (synthetic or subset)
- For stochastic components (dropout, sampling), seed RNGs and/or assert statistical bounds across repeated runs.

---

## 4. Writing Effective Tests

- **Names:** `test_<module>_<behavior>()`, e.g., `test_kv_cache_truncates_old_slots`.
- **Purpose statement:** short docstring/comment describing what the test proves.
- **Arrange/Act/Assert:** keep inputs explicit, isolate the behavior, then assert concise outcomes.
- **Edge cases:** include empty inputs, singleton batches, extreme probabilities, truncated caches, etc.
- **Helper assertions:** share invariant checkers (e.g., `assert_tokens_monotonic`) to keep tests short and consistent.
- **Numerical tolerance:** use `torch.testing.assert_close` or equivalent with explicit tolerances for float comparisons.

---

## 5. Running Tests

- Use `pytest` as the default runner; structure tests under `tests/` mirroring module paths.
- Common commands:
  - `pytest tests/unit`
  - `pytest tests/integration`
  - `pytest tests` (full suite)
- Document which command you ran (commit message, PR description) when it adds clarity.

---

## 6. Handling Failures

1. **Reproduce locally** using the failing command/seed.
2. **Categorize** the root cause (test bug, code bug, both).
3. **Fix flow:** add/adjust the test first, apply the code fix, rerun the affected suite.
4. **Document** non-obvious fixes in commit messages or docstrings.

---

## 7. Regression Workflow

When a bug appears in a real run:
1. Capture context (inputs, seeds, configs, logs).
2. Build the smallest reproduction under `tests/regressions/`.
3. Confirm the new test fails without the fix.
4. Implement the fix.
5. Re-run the regression test plus any related suites.
6. Reference the regression in commit notes to track coverage.

---

## 8. Ownership & Automation

- Module owners ensure their surfaces have adequate tests before declaring work complete.
- Agents touching a file extend/adjust its tests as part of the same change; no TODOs for "add tests later."
- CI (when enabled) will run CPU pytest suites, so keep runtimes reasonable (<5 minutes) by sticking to toy configs.

---

Following these practices keeps nanochat-plus iterations lightweight and prevents expensive regressions. When in doubt, write the test first.
