# TESTING – PRINCIPLES & PRACTICES

This document defines how we design, implement, and run tests across the MegaContext stack. Treat it as a contract: every contribution must honor these expectations to keep the system reliable and debuggable.

---

## 1. Core Principles

- **Purposeful coverage:** every test must validate a specific behavior or invariant—never write a test just to inflate coverage or restate code.
- **Fail fast, fail local:** tests should expose bugs on laptops/CPUs before they leak into long GPU runs. If an issue slips into a real run, assume a missing test.
- **Blame clarity:** when a test fails, explicitly determine whether the bug lives in the test, the implementation, or both, and fix accordingly.
- **Continuous validation:** after any code change, agents run all relevant tests locally and ensure they pass before submitting work. Broken tests in a PR are unacceptable.
- **Regression capture:** whenever a bug is discovered (during real runs or manual testing), first reproduce it in a minimal test, land the test, then fix the code.

---

## 2. Test Types & Scope

1. **Unit Tests**
   - Scope: single module/class/function (e.g., `GistNet`, `tree_flatten`, `gaussian_rope`).
   - Expectations:
     - Cover “happy path” + edge cases (e.g., incomplete child groups, dropout boundary conditions, empty WC expansions).
     - Use deterministic seeds and small tensors (CPU friendly).
     - Validate shapes, invariants, and numerical properties (e.g., attention weights sum to 1, rotations preserve norms).

2. **Integration Tests**
   - Scope: multiple modules working together (e.g., build tree → flatten → dropout pipeline; ContextManager + WCState; KV cache repair).
   - Expectations:
     - Use toy-sized models/sequences (e.g., T≤8, d_model≤32) so CPU runtimes stay seconds-level.
     - Assert end-to-end properties (e.g., CE+MSE losses finite and decreasing, WC invariants preserved across iterations).

3. **Functional/Scenario Tests**
   - Scope: higher-level workflows (e.g., one training step, one inference focusing cycle).
   - Expectations:
     - Mirror real usage patterns with scaled-down configs.
     - Capture metrics/log assertions (e.g., node counts, KV repair boundaries).

---

## 3. CPU-Only / Toy Config Guidelines

- Default to CPU execution; avoid CUDA-only dependencies in tests.
- Use tiny configs:
  - `d_model`: 16–64
  - `n_layer`: 2–4
  - `sequence_len`: 8–32
  - Token vocab: ≤256 (synthetic or subset)
- For stochastic components (dropout, sampling), seed RNGs and/or test statistical properties across multiple runs.

---

## 4. Writing Effective Tests

- **Name clarity:** `test_<module>_<behavior>()`, e.g., `test_tree_flatten_children_before_parent`.
- **Purpose docstring/comment:** describe what the test proves (helps avoid duplicated intent).
- **Arrange / Act / Assert pattern:** set up inputs, invoke code, assert outcomes. Keep minimal boilerplate via fixtures/factories when possible.
- **Edge cases:** consider empty inputs, singleton groups, large spans, repeated levels, dropout probability extremes, absent children, etc.
- **Helper assertions:** centralize repeated invariant checks (e.g., “WC forms induced subtree”) to keep tests concise and consistent.
- **Numerical tolerance:** use `torch.testing.assert_close` with explicit tolerances when comparing floats.

---

## 5. Running Tests

- Use `pytest` as the default runner; tests live under `tests/` mirroring the module structure.
- Provide grouped entrypoints:
  - `pytest tests/unit` (fast, run on every edit)
  - `pytest tests/integration` (slightly longer, run before submitting work)
  - `pytest tests` (full suite, run nightly or before major merges)
- Agents must run relevant subsets locally (CPU) and ensure green status before handing off work. Document the commands executed in PR summaries when helpful.

---

## 6. Handling Failures

1. **Reproduce locally** using the same command/seed if possible.
2. **Categorize** the root cause:
   - **Test issue:** incorrect assumption, brittle expectation, obsolete behavior.
   - **Code issue:** bug or regression in implementation.
   - **Both:** test and code need adjustments.
3. **Fix flow:**
   - Update/add tests first to lock in expected behavior.
   - Apply code fix.
   - Re-run the affected suite and confirm green.
4. **Document** tricky fixes in commit messages or docstrings when appropriate.

---

## 7. Regression Workflow

When a bug appears during a “real” run (training/inference):
1. Pause and capture context (inputs, seeds, configs).
2. Build a minimal reproduction in `tests/regressions/` (unit or integration depending on scope).
3. Verify the new test fails without the fix.
4. Implement the fix.
5. Re-run the new regression test + relevant suites.
6. Reference the regression test in the issue/commit to track coverage.

---

## 8. Ownership & Automation Expectations

- **Every module owner** ensures their component has adequate tests before declaring it “done.”
- **Agents** automatically extend/update tests for any code they touch; never leave TODOs for “add tests later.”
- **CI integration** (planned) will run `pytest` on CPU; agents should keep runtimes reasonable (<5 min total) by using toy configs.

---

Following these practices keeps MegaContext iterations lightweight and prevents expensive regressions. When in doubt, write the test first. !*** End Patch
