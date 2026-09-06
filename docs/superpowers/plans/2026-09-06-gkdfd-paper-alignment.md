# GK-DFD Paper Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the GK-DFD implementation with the paper's class-balanced sampling, shared teacher topology, UEM/CAC graph construction, and signed DFD eigenspace objective.

**Architecture:** Keep the existing RepDistiller training entry point, but make `GKDFDLoss` expose paper-faithful graph and DFD primitives. The teacher logits define one detached adjacency shared by both graph branches; DFD fits centered PCA on teacher graph features, selects class-aware local neighbors, assembles signed local matrices into a global sample-space matrix, and applies a detached shared eigenbasis to both branches.

**Tech Stack:** Python, PyTorch, torchvision, unittest/pytest.

---

### Task 1: Encode paper defaults and sampling constraints

**Files:**
- Modify: `train_student.py`
- Modify: `dataset/cifar100.py`
- Test: `tests/test_gkdfd.py`

- [x] Add an explicit `--gkdfd_k2` option and validate `1 <= k1 <= K-1`, `1 <= k2 <= N-K` for the configured balanced batch.
- [x] Pass `k2` into `GKDFDLoss` and keep the paper defaults (`C=8`, `K=8`, `k1=7`, fixed valid `k2`).
- [x] Add tests that balanced batches satisfy `N=C*K` and that invalid neighbor configurations fail early.

### Task 2: Make graph construction paper-faithful

**Files:**
- Modify: `distiller_zoo/GKDFD.py`
- Test: `tests/test_gkdfd.py`

- [x] Implement UEM exactly as normalized cosine minus `1-exp(-euclidean)` on teacher logits.
- [x] Apply CAC from ground-truth labels and retain only positive edges in the shared topology.
- [x] Build both graph branches with the same detached teacher adjacency; keep graph consistency as normalized Frobenius MSE between global graph representations.
- [x] Test symmetry, positivity filtering, cross-class attenuation, shared-topology behavior, and finite gradients.

### Task 3: Implement signed local alignment and shared DFD basis

**Files:**
- Modify: `distiller_zoo/GKDFD.py`
- Test: `tests/test_gkdfd.py`

- [x] Fit centered teacher PCA and project teacher/student into the same feature coordinates without whitening.
- [x] Select exactly `k1` nearest same-class and `k2` nearest different-class neighbors per anchor.
- [x] Construct each signed local matrix `L_i` using weights `[1...1,-theta...-theta]`, `theta=k1/k2`, and assemble `L=sum_i S_i L_i S_i^T`.
- [x] Solve the symmetric eigenproblem in feature coordinates and use the detached eigenvectors with smallest eigenvalues for both branches.
- [x] Compute DFD as the mean squared teacher/student projected-feature distance, with stable fallback for degenerate batches.
- [x] Add tests for matrix symmetry, signed weights, selected-neighbor counts, detached basis, and nonzero student gradients.

### Task 4: Align training loss wiring and documentation

**Files:**
- Modify: `helper/loops.py`
- Modify: `README.md`

- [x] Wire `Loss_total = alpha*Loss_GC + beta*L_DFD + Loss_Class` for GK-DFD, leaving optional KL disabled by default.
- [x] Update the run command and option descriptions to document `k2`, shared topology, and the paper defaults.
- [x] Run the focused tests and a CPU smoke forward/backward pass.

### Verification

- [x] Run the focused test suite with the available PyTorch environment (`python -m unittest tests.test_gkdfd -v`).
- [x] Run a direct 64-sample CPU `GKDFDLoss` forward/backward smoke test and confirm all losses and gradients are finite.
- [x] Review the changed files for accidental changes outside GK-DFD.
