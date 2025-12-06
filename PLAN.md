# PLAN – MEGACONTEXT IMPLEMENTATION ROADMAP

Use the sections below as a living checklist to guide discussion, coding, reviews, and testing. Each block moves from design alignment → implementation → review/refinement → verification before the next dependency starts.

---

## 1. Core Tree & Gist Infrastructure
- [x] Confirm `GistNet` spec (Gaussian RoPE-only inputs, per-LOD sigma, fixed-span cross-attn).
- [ ] Implement `GistNet` module with unit tests.
- [ ] Design `Node` data model + `build_lod_tree()` flow.
- [ ] Implement tree builder + span math sanity tests.
- [ ] Define flattening order rules and metadata needed downstream.
- [ ] Implement `tree_flatten` and validate children-before-parent ordering.
- [ ] Specify dropout constraints + sampling approach.
- [ ] Implement `tree_dropout` with unit tests for invariants.

## 2. Positional Encoding & Transformer Integration
- [x] Finalize Gaussian 2D RoPE spec/simplifications.
- [ ] Implement `gaussian_rope` helper + tests for norm preservation.
- [ ] Draft MegaContext-aware transformer wrapper (model inputs/outputs).
- [ ] Integrate Gaussian RoPE, level embeddings, metadata plumbing.
- [ ] Wire CE + MSE heads (token logits + gist predictors).
- [ ] Review/iterate on model API (batch handling, teacher gist generation).

## 3. POC Training Loop & Validation
- [ ] Outline `training.py` flow (data prep, tree building, batching).
- [ ] Implement trainer with configurable LOD/group sizes + loss weights.
- [ ] Create synthetic dataset or adapter for existing loader.
- [ ] Add logging/metrics hooks (loss curves, node counts).
- [ ] Write unit/integration tests (forward/backward, loss decrease).
- [ ] Run initial end-to-end training sanity pass and capture findings.

## 4. WorkingContext Manager Foundations
- [ ] Align on WC invariants + must-keep policy specifics.
- [ ] Implement `WCNode`, `WCState` utilities + tests.
- [ ] Implement `FocusScorer` using attention maps (EMA, tail queries).
- [ ] Draft `ContextManager` expand/collapse policy (heuristics only).
- [ ] Unit-test invariants and synthetic update sequences.
- [ ] Review interplay between WC updates and MegaContext model outputs.

## 5. LensNet Routing Head
- [ ] Validate mid-layer tap strategy and data needed per gist.
- [ ] Implement LensNet head module + target builder per TDD.
- [ ] Integrate into training loop (loss term, logging).
- [ ] Create focused tests (high child attention ⇒ positive score, etc.).
- [ ] Review thresholds/configuration for inference consumers.

## 6. MegaContext KV Strategies
- [ ] Decide initial KV mode (dirty-suffix vs append-window) for MVP.
- [ ] Implement shared KV data structures + interfaces.
- [ ] Implement chosen strategy (+ optional hooks for the other).
- [ ] Build verification tests (post-update logits match full recompute).
- [ ] Integrate with inference engine wrapper and WC manager events.
- [ ] Document recovery/fallback workflows.

## 7. End-to-End Integration & Tooling
- [ ] Define CLI/config entrypoints for MegaContext training/inference.
- [ ] Hook context manager + LensNet into inference loop.
- [ ] Run scenario tests (WC zooming under budget, KV repairs).
- [ ] Collect metrics/dashboards for debugging (node counts, focus scores, KV timings).
- [ ] Polish documentation (README sections, AGENTS/PLAN updates).
- [ ] Plan for future refinements (learned dropout, draft modes).

---

Update checkboxes as we progress; add sub-tasks under each section when diving into detailed workstreams. Order can be revisited, but favor completing upstream blocks before downstream dependencies. !*** End Patch
