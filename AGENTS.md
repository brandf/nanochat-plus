# AGENTS GUIDE - NANOCHAT-PLUS

Welcome! This document orients new agents who are joining any nanochat-plus experiment. The `main` branch hosts the shared foundation (model, scripts, docs) that all experiment branches build on, so stewardship here keeps every project moving quickly.

---

## 1. Mission Overview

- **Goal:** keep nanochat-plus a lightweight, well-tested playground for long-context and inference experiments.
- **What lives on main:** core GPT model, training/inference harnesses, shared utilities, cross-experiment docs (this file plus `PLAN.md`, `IMPLEMENTING.md`, `TESTING.md`).
- **Branch etiquette:** experiment-specific product docs or spikes (e.g., MegaContext, LensNet ideas, KV strategies) live on dedicated branches; main only carries reusable, branch-agnostic guidance.

---

## 2. Documents You Must Read

| Doc | Why it matters |
| --- | --- |
| `PLAN.md` | Active roadmap for shared infra, including blockers that would impact multiple branches. Update status as tasks land. |
| `IMPLEMENTING.md` | Coding style, shape-comment expectations, torch.compile guidance, and performance norms. |
| `TESTING.md` | Testing philosophy, CPU-first constraints, and failure-handling workflow. |
| `docs/` | Houses PRDs/TDDs that are generally useful (tokenizer notes, engine notes, historical experiments). Keep new experiment-specific specs next to the code that consumes them. |

If a branch introduces new canonical conventions, promote the relevant docs into main once they are broadly applicable.

---

## 3. Existing Codebase Primer

### 3.1 Core Model (`nanochat/gpt.py`)
- Decoder-only transformer with rotary embeddings, grouped-Q attention, KV-cache-aware attention, and CE loss.
- Reference points:
  - `CausalSelfAttention` shows attention wiring, rotary application, and cache usage.
  - `GPT.forward` illustrates the baseline token workflow that experiments extend.

### 3.2 Training Script (`scripts/base_train.py`)
- Sets up `GPT`, optimizers, logging, eval loops, and checkpointing.
- Use it as the template for experiment trainers so metrics/logging stay compatible.

### 3.3 Data Loading & Tokenizer
- `nanochat/dataloader.py` streams `(B, T)` token batches.
- `nanochat/dataset.py` owns shard download/globbing utilities.
- `nanochat/tokenizer.py` exposes BPE helpers used throughout experiments.

### 3.4 Inference & Engine (`nanochat/engine.py`)
- Contains the reference KV cache implementation and streaming `Engine.generate` loop.
- Serve new inference ideas (routing, KV patching, context schedulers) through this surface so downstream tools remain consistent.

---

## 4. Workflow Expectations

1. **Start from docs.** Before touching a module, read the matching PRD/TDD (for general pieces) or the branch-specific design doc.
2. **Prototype responsibly.** Use `dev/` or branch-local folders for scratch work; only graduate stable, shared helpers into `nanochat/` or `scripts/` on main.
3. **Align APIs.** Mirror existing nanochat naming and tensor layouts so experiments can share components.
4. **Keep history clean.** Land atomic commits on main; keep large experiments rebased frequently to pick up shared fixes.
5. **Document invariants sparingly.** When behavior is subtle, add a short comment or docstring explaining the invariant rather than duplicating the design doc.

---

## 5. Testing Snapshot

- Write unit tests whenever you add helpers or touch math-heavy code.
- Ensure integration tests cover realistic flows (data -> model -> loss -> metrics) with CPU-friendly shapes.
- Every change on main must pass the relevant `pytest` targets locally; see `TESTING.md` for the complete checklist.

---

## 6. Collaboration Tips

1. **Clarify requirements early.** If a request crosses branches or lacks detail, sync with the requester before coding.
2. **Act independently once aligned.** Implement, self-review, and test without waiting for additional prompts.
3. **Record assumptions.** Briefly capture non-obvious choices in commit messages or PR descriptions so other branches can reuse the reasoning.
4. **Share learnings back.** When an experiment surfaces a reusable pattern, port the generalized version into main (plus docs/tests) so future work starts with a better baseline.

---

## 7. Quick Reference Checklist

- [ ] Familiar with `PLAN.md`, `IMPLEMENTING.md`, and `TESTING.md`.
- [ ] Understand current model/training/engine surfaces in `nanochat/` and `scripts/`.
- [ ] Keep experiment docs with their branch; upstream only the generalizable pieces.
- [ ] Run targeted tests before handing off any change.
- [ ] Update shared docs when conventions shift.

Welcome aboard! Treat `main` as the common toolbox for every nanochat-plus experiment.
