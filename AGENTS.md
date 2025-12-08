# AGENTS GUIDE – NANOCHAT-PLUS

Welcome! This document is the quick-start for new agents working on the MegaContext initiative on top of nanochat. It lists the essential specifications, code anchors, and expected workflows so you can jump in without re-deriving the project intent.

---

## 1. Mission Overview

- **Goal:** extend the minimal nanochat GPT stack into a hierarchical, multi-level-of-detail (LOD) context system with dynamic focusing. The end state is a research-friendly playground for long-context modeling and inference tricks.
- **Key Deliverables:** the four PRD/TDD pairs under `docs/` plus matching code in `nanochat/` and `scripts/`.

---

## 2. Documents You Must Read

| Doc | What it covers | When to reference |
| --- | --- | --- |
| `docs/1. MegaContext_PRD.md` | High-level product goals for multi-LOD trees, GistNet, Gaussian RoPE, multi-scale losses. | Anytime you touch tree construction, flattening, dropout, transformer inputs, or loss wiring. |
| `docs/1. MegaContext_TDD.md` | Module-level APIs: `GistNet`, `lod_tree`, `tree_flatten`, `tree_dropout`, `gaussian_rope`, `model`, `training`. | Before writing/reading implementation code for the core pipeline. |
| `docs/2. ContextManager_PRD.md` | Requirements for the WorkingContext (WC) manager that zooms into the MegaContext stored on disk. | Needed for inference-time sparse context maintenance. |
| `docs/2. ContextManager_TDD.md` | Concrete classes (`WCNode`, `WCState`, `FocusScorer`, `ContextManager`). | When implementing/inspecting WC logic or writing tests. |
| `docs/3. LensNet_Signed_PRD.md` | Why we need the LensNet routing head and the semantics of the signed score. | Used when training or consuming LensNet outputs to drive context updates. |
| `docs/3. LensNet_Signed_TDD.md` | Exact module definition, target computation, loss wiring. | Before adding the routing head to the transformer or trainer. |
| `docs/4. MegaContext_KV_PRD.md` | KV caching goals + strategy comparison (dirty-suffix vs append-window). | For inference-mode KV updates after WC changes. |
| `docs/4. MegaContext_KV_TDD.md` | Data structures (`KVSlot`, `KVCache` variants) and algorithms for each strategy. | When coding/testing KV repair logic. |
| `TESTING.md` | Project-wide testing philosophy, CPU-only requirements, and failure-handling workflow. | Before writing or updating any tests, and whenever a regression is discovered. |
| `PLAN.md` | Active roadmap with dependency-ordered work items and progress checkboxes. | Keep this file up to date—add new tasks as they surface, check off completed work, and reprioritize before declaring any milestone “done.” |
| `IMPLEMENTING.md` | Coding style, performance expectations, and workflow guidelines for new features. | Review before implementing modules; ensure torch.compile compatibility, shape comments, and GPU-friendly data flows. |

Keep the PDF-like downloads in sync with `docs/`—the repo copies are canonical for agents.

---

## 3. Existing Codebase Primer

### 3.1 Core Model (`nanochat/gpt.py`)
- Minimal decoder-only transformer with rotary embeddings, GQA, KV-cache-aware attention, and standard CE loss.
- Useful references:
  - `CausalSelfAttention` for how rotary embeddings are currently applied (we’ll replace/extend with Gaussian RoPE).
  - `GPT.forward` to see the vanilla token workflow we’re generalizing.

### 3.2 Training Script (`scripts/base_train.py`)
- Sets up `GPT`, data loaders, optimizers, and training loop. Use it as a blueprint for `training.py` in the MegaContext POC:
  - Hook point for swapping `GPT` with a tree-aware model.
  - Shows how checkpointing, logging, and evaluation are wired today.

### 3.3 Dataloading & Tokenizer
- `nanochat/dataloader.py` streams `(B, T)` token batches from parquet shards.
- `nanochat/dataset.py` knows how to download shards.
- `nanochat/tokenizer.py` (not shown above) provides BPE utilities; MegaContext training still starts from token IDs before building trees.

### 3.4 Inference Engine & KV Cache (`nanochat/engine.py`)
- Contains the current `KVCache` and streaming `Engine.generate`.
- These pieces will host the MegaContext KV strategies and future WC manager hooks.

---

## 4. MegaContext Module Roadmap

> The following entries map the TDD sections to expected implementation files. Use this as a checklist when navigating or adding code.

1. **`gistnet.py`** – Implements `GistNet(nn.Module)` exactly as specified (single query cross-attention over child latents, span-aware query construction). Required for both tree construction and teacher gists.
2. **`lod_tree.py`** – Provides the `Node` dataclass and `build_lod_tree()` function. Handles token-to-gist hierarchy assembly and span metadata (`mu`, `sigma`).
3. **`tree_flatten.py`** – Emits children-before-parent sequences and maintains node indices for loss masking.
4. **`tree_dropout.py`** – Applies level-aware dropout, enforces ancestor constraints, and repacks flattened sequences.
5. **`gaussian_rope.py`** – Extends rotary embeddings to accept `(mu, sigma)` per node. Even a simplified two-axis RoPE is acceptable for the POC.
6. **`model.py`** – Wrapper around nanochat blocks that consumes flattened nodes and node metadata, applies Gaussian RoPE, and produces both token logits and gist predictions.
7. **`training.py`** – Mini trainer that:
   - Builds trees per sequence.
   - Computes CE on LOD0 and MSE on higher LODs.
   - Logs sanity metrics (loss curves, gist norms).

---

## 5. Context Manager & LensNet Integration Plan

1. **WC data model (`WCNode`, `WCState`)** from `docs/2.*`.
2. **Focus scoring (`FocusScorer`)** uses attention heatmaps from the tree-aware transformer; keep the mapping from flattened indices to nodes handy.
3. **Context manager loop (`ContextManager`)** coordinates collapse/expand operations, calling `MegaContextStore.load_children()` when needed.
4. **LensNet routing head** slots into a mid layer of the transformer to emit signed scores per gist. Training targets rely on attention decomposition between a gist and its children.
5. **KV strategy** selection (dirty-suffix vs append-window) must align with the context manager’s update cadence—see `docs/4.*` for the two APIs.

Implementation order typically goes: core MegaContext model → POC trainer/tests → WC manager heuristics → LensNet targets/head → KV strategies.

---

## 6. Testing Expectations

- Follow the unit/integration tests listed in each TDD.
- Priority checks:
  1. **Tree builder** yields correct spans and levels.
  2. **Flattening** obeys strict children-before-parent ordering.
  3. **Dropout** never removes required ancestors/last nodes.
  4. **Gaussian RoPE** keeps norms stable while reacting to `(mu, sigma)`.
  5. **Training loop** runs forward/backward on a toy dataset with decreasing CE/MSE.
  6. **Context manager** maintains WC invariants during synthetic expand/collapse simulations.
  7. **KV strategies** reproduce baseline logits after focus operations.

---

## 7. Workflow Tips for Agents

1. **Read the matching PRD/TDD before editing a module.** Most questions about behavior vs implementation are already captured there.
2. **Prototype in `dev/` or `tasks/`** if you need notebooks or ad-hoc experiments; keep production modules under `nanochat/` or `scripts/`.
3. **Reference existing nanochat patterns** (e.g., optimizer setup, logging, CLI overrides) so MegaContext code feels native.
4. **Document invariants in code comments** sparingly—only when non-obvious (e.g., why a parent can’t be dropped).
5. **Run lightweight tests frequently** (unit tests or tiny training runs) to avoid debugging giant diffs.

---

## 8. Quick Reference Checklist

- [ ] Understand multi-LOD tree pipeline (`docs/1.*`).
- [ ] Know WC manager rules (`docs/2.*`).
- [ ] Plan for LensNet routing (`docs/3.*`).
- [ ] Pick KV strategy (`docs/4.*`).
- [ ] Familiarize with current GPT/engine/dataloader code.
- [ ] Align new modules with TDD APIs.
- [ ] Write/extend tests before major refactors.

Welcome aboard—ping the docs listed above whenever you’re unsure which component owns a concept. This AGENTS guide should be your starting compass.
