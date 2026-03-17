# Finding 014: Shared Vocabulary Transfer Learning

**Date:** 2026-03-16
**Phase:** 7
**Datasets:** All 7 BEELINE datasets
**Ground truth:** STRING (primary), ChIP-seq, Non-ChIP
**Baseline:** Solo edge+diff (GAT+SEM+edge_features+diff_attn) per dataset
**Depends on:** Finding 008 (original transfer), Finding 011 (edge+diff)

## Motivation

Finding 008 showed that joint training with per-dataset gene embeddings was
roughly neutral (-0.004 AUROC). Root causes: (1) near-zero gene overlap
across HVG selections, (2) sequential per-dataset gradient updates, (3)
broken checkpoint loading in finetune mode.

This finding documents a series of architectural fixes and their impact.

## Gene Overlap (All BEELINE Genes Already Uppercase)

| Pair | Shared genes | % of smaller set |
|------|-------------|------------------|
| mHSC-E & mHSC-GM | 618 | 54.6% |
| hESC & hHep | 401 | 28.4% |
| mESC & hHep | 333 | 20.6% |
| mESC & mDC | 299 | 18.5% |
| mESC & hESC | 289 | 17.8% |

Shared vocabulary size: 5805 unique genes across 7 datasets.

## Experiment 1: Shared Vocab + Gradient Accumulation

**Changes vs Finding 008:**
1. Shared gene vocabulary (`nn.Embedding` over union of all gene names)
2. Gradient accumulation (1 optimizer step per epoch)
3. Per-dataset `a_scale`
4. Fixed checkpoint loading bug

### STRING AUROC

| Dataset | Solo | Joint (accum) | Delta |
|---------|------|---------------|-------|
| hESC | 0.6639 | 0.6652 | +0.001 |
| hHep | 0.6712 | 0.6743 | +0.003 |
| mDC | 0.5539 | 0.5314 | **-0.023** |
| mESC | 0.6175 | 0.6153 | -0.002 |
| mHSC-E | 0.7311 | 0.7367 | +0.006 |
| mHSC-GM | 0.7626 | 0.7722 | **+0.010** |
| mHSC-L | 0.7010 | 0.6957 | -0.005 |
| **Mean** | **0.6716** | **0.6701** | **-0.002** |

4/7 datasets improved (vs 1/5 in Finding 008). mDC is a clear outlier
(-0.023), dragging down the mean. High-overlap pairs (mHSC-E/GM) benefit
most.

## Experiment 2: Shared Low-Rank Adjacency (Failed)

**Additional change:** Decompose A_k = B[indices] + R_k where B is a
shared low-rank base (rank=32) over the union vocabulary and R_k is a
per-dataset full-rank residual with L2 penalty (alpha=0.01).

### STRING AUROC

| Dataset | Solo | Joint + SharedAdj |
|---------|------|--------------------|
| **Mean** | **0.6716** | **0.5418 (-0.130)** |

**Catastrophic failure.** This confirms Finding 006: low-rank adjacency
factorization fundamentally fails for GRN inference, even with a full-rank
residual. The low-rank base dominates early training (residual initialized
10x smaller + L2 penalty), shaping the gradient landscape into a basin the
model cannot escape. Conflicting regulatory signals from 7 datasets
average out into noise in the shared base rather than useful structure.

**Lesson:** Share gene identities (embeddings), not regulatory structure
(adjacency). A is inherently dataset-specific and high-dimensional.

## Experiment 3: Mixed Dataloader + Consistency Regularization

**Changes vs Experiment 1:**
1. **Mixed dataloader** (default) — randomly interleave datasets each step
   instead of accumulating all gradients before stepping. Same number of
   optimizer steps as the original sequential loop (28/epoch vs 1/epoch).
2. **Consistency regularization** (alpha=0.01) — soft L2 penalty on
   divergent A entries for shared genes:
   `loss += alpha * ||A_k1[shared] - A_k2[shared]||²`

### STRING AUROC

| Dataset | Solo | Accum | Mixed+Consist | Delta vs Solo |
|---------|------|-------|---------------|---------------|
| hESC | 0.6639 | 0.6652 | 0.6655 | **+0.002** |
| hHep | 0.6712 | 0.6743 | 0.6698 | -0.001 |
| mDC | 0.5539 | 0.5314 | **0.5624** | **+0.009** |
| mESC | 0.6175 | 0.6153 | 0.6177 | +0.000 |
| mHSC-E | 0.7311 | 0.7367 | 0.7309 | -0.000 |
| mHSC-GM | 0.7626 | 0.7722 | 0.7529 | -0.010 |
| mHSC-L | 0.7010 | 0.6957 | **0.7017** | **+0.001** |
| **Mean** | **0.6716** | **0.6701** | **0.6716** | **+0.000** |

### AUPR

| Dataset | Solo | Mixed+Consist | Delta |
|---------|------|---------------|-------|
| hESC | 0.0508 | 0.0520 | +0.0012 |
| hHep | 0.0538 | 0.0521 | -0.0017 |
| mDC | 0.0496 | 0.0509 | +0.0013 |
| mESC | 0.0521 | 0.0526 | +0.0005 |
| mHSC-E | 0.1251 | 0.1151 | -0.0100 |
| mHSC-GM | 0.2358 | 0.2242 | -0.0116 |
| mHSC-L | 0.2920 | 0.3120 | **+0.0200** |

## Analysis

### Mixed dataloader fixed the mDC catastrophe

The biggest improvement from accumulation → mixed dataloader is mDC:
-0.023 → +0.009. With accumulation (1 step/epoch), Adam only got 500
updates total (vs 14,000 with mixed). The 28x increase in optimizer steps
gives Adam better adaptive learning rate estimates, and random dataset
interleaving eliminates systematic ordering bias.

### The tradeoff: mDC vs mHSC-GM

| | Accumulation | Mixed+Consist |
|---|---|---|
| mDC | -0.023 | **+0.009** |
| mHSC-GM | **+0.010** | -0.010 |

Accumulation benefited mHSC-GM (consensus gradient from 618 shared genes
with mHSC-E), but the mixed dataloader hurt it. The consistency
regularization may be the key variable — it constrains A entries for
shared genes, which helps low-overlap datasets (mDC) but may over-constrain
high-overlap ones (mHSC-GM). Sweep pending.

### Training loop comparison

| Approach | Steps/epoch | Adam updates (500 epochs) | Mean dAUROC |
|----------|------------|--------------------------|-------------|
| Sequential (Finding 008) | 20 | 10,000 | -0.004 |
| Accumulation (Exp 1) | 1 | 500 | -0.002 |
| Mixed dataloader (Exp 3) | 28 | 14,000 | **+0.000** |

### What didn't work

- **Shared low-rank adjacency** (Exp 2): catastrophic. Low-rank A is
  fundamentally incompatible with GRN inference (Finding 006).

## Experiment 4: Consistency Alpha Sweep

Sweep over `alpha_consistency ∈ {0, 0.001, 0.005, 0.01, 0.05, 0.1}`, all
using mixed dataloader. α=0 isolates the mixed dataloader contribution.

### STRING AUROC (dAUROC vs Solo)

| Dataset | Solo | α=0 | α=0.001 | α=0.005 | α=0.01 | α=0.05 | α=0.1 |
|---------|------|-----|---------|---------|--------|--------|-------|
| hESC | 0.6639 | +0.000 | +0.001 | -0.001 | **+0.002** | -0.000 | -0.003 |
| hHep | 0.6712 | **+0.002** | -0.005 | -0.003 | -0.001 | -0.004 | -0.004 |
| mDC | 0.5539 | -0.010 | +0.005 | +0.004 | **+0.009** | +0.003 | +0.005 |
| mESC | 0.6175 | +0.000 | -0.001 | +0.000 | +0.000 | -0.000 | -0.002 |
| mHSC-E | 0.7311 | -0.002 | -0.005 | -0.008 | -0.000 | -0.013 | -0.011 |
| mHSC-GM | 0.7626 | **+0.007** | -0.003 | -0.006 | -0.010 | -0.005 | -0.005 |
| mHSC-L | 0.7010 | -0.006 | -0.005 | -0.001 | +0.001 | -0.018 | +0.001 |
| **Mean** | **0.6716** | -0.001 | -0.002 | -0.002 | **+0.000** | -0.005 | -0.003 |

### Analysis

**Mixed dataloader alone (α=0) is already strong.** Without consistency,
mean dAUROC is only -0.001, and mHSC-GM achieves +0.007. The dataloader
is the primary improvement over accumulation.

**α=0.01 is the optimal consistency strength.** It's the only alpha that
matches solo on the mean (0.6716) and gives the best mDC improvement
(+0.009).

**Fundamental tension: mDC vs mHSC-GM.**

| Alpha | mDC dAUROC | mHSC-GM dAUROC |
|-------|-----------|---------------|
| 0 (no consist) | -0.010 | **+0.007** |
| 0.01 (optimal) | **+0.009** | -0.010 |

Consistency helps mDC (low overlap, hard dataset) by forcing its A to
borrow structure from other datasets. But it hurts mHSC-GM (high overlap)
by over-constraining its cell-type-specific edges. No single alpha helps
both simultaneously.

**Higher alphas (0.05, 0.1) are too aggressive.** Mean drops to -0.005 and
-0.003. mHSC-L especially suffers at α=0.05 (-0.018).

## Experiment 5: Overlap-Adaptive Consistency (α=0.01)

Replaced uniform consistency with overlap-adaptive weighting:
`w_ij = 1 - overlap_fraction_ij`. High-overlap pairs get weak consistency,
low-overlap pairs get strong consistency.

### STRING AUROC

| Dataset | Solo | Uniform α=0.01 | Adaptive α=0.01 | Adaptive Δ |
|---------|------|----------------|-----------------|-----------|
| hESC | 0.6639 | 0.6655 | 0.6639 | +0.000 |
| hHep | 0.6712 | 0.6698 | 0.6661 | -0.005 |
| mDC | 0.5539 | 0.5624 | 0.5567 | +0.003 |
| mESC | 0.6175 | 0.6177 | 0.6188 | +0.001 |
| mHSC-E | 0.7311 | 0.7309 | 0.7227 | -0.008 |
| mHSC-GM | 0.7626 | 0.7529 | **0.7609** | **-0.002** |
| mHSC-L | 0.7010 | 0.7017 | 0.6981 | -0.003 |
| **Mean** | **0.6716** | **0.6716** | **0.6696** | **-0.002** |

### Analysis

**mHSC-GM recovered** from -0.010 (uniform) to -0.002 (adaptive) — the
weighting worked as intended for high-overlap datasets. But the mean
dropped to -0.002 because adaptive weighting effectively reduces the
average consistency pressure by ~30%. The α=0.01 was tuned for uniform
weighting; adaptive at α=0.01 is effectively like uniform at ~α=0.007.

**Best worst-case:** Adaptive has the smallest worst-case regression
(-0.008 vs -0.010), suggesting the weighting stabilizes performance
across datasets even if the mean is slightly lower.

**Next:** Sweep higher base alphas (0.015–0.03) with adaptive weighting
to compensate for the reduced average pressure.

## Summary Across All Configurations

| Config | Mean dAUROC | mDC | mHSC-GM | Worst |
|--------|-------------|-----|---------|-------|
| Mixed, α=0 | -0.001 | -0.010 | **+0.007** | -0.010 |
| Mixed, uniform α=0.01 | **+0.000** | **+0.009** | -0.010 | -0.010 |
| Mixed, adaptive α=0.01 | -0.002 | +0.003 | -0.002 | -0.008 |

## Progression Across All Experiments

| | Finding 008 | Exp 1 (accum) | Exp 3 (uniform α=0.01) | Exp 5 (adaptive α=0.01) |
|---|---|---|---|---|
| Mean dAUROC | -0.004 | -0.002 | **+0.000** | -0.002 |
| Worst dataset | -0.011 | -0.023 | -0.010 | **-0.008** |
| Best dataset | +0.002 | +0.010 | +0.009 | +0.003 |
| Datasets improved | 1/5 | 4/7 | 4/7 | 2/7 |

## Ongoing: Adaptive Alpha Sweep

Sweep adaptive weighting with higher base alphas (0.015, 0.02, 0.03) to
compensate for the reduced average consistency pressure from overlap
weighting.

## Potential Further Improvements

### 1. Multi-seed validation

All results are single-seed (42). The α=0.01 exactly-tied result could
shift positive with seed averaging. Need 3+ seeds to get confidence
intervals and determine if any configuration is genuinely better.

### 3. Dataset-size-weighted sampling

The mixed dataloader currently gives equal probability to all datasets.
Larger datasets (mHSC-E: 1071 cells) have more training signal than
smaller ones (mDC: 383 cells). Weighting by dataset size could help
under-represented datasets.

### 4. Few-shot with fixed checkpoint

The checkpoint loading bug is fixed. Re-run pretrain→finetune to test
genuine transfer. The shared vocabulary + consistency should give better
pretrained representations than Finding 008's broken pipeline.

## Architectural Summary

| Component | Solo | Joint (current best) |
|-----------|------|---------------------|
| Gene embedding | Per-dataset nn.Parameter | Shared nn.Embedding (union vocab) |
| Training loop | N/A | Mixed dataloader (random interleave) |
| a_scale | N/A | Per-dataset |
| Adjacency A | Independent FullRankParam | Independent FullRankParam + consistency (α=0.01) |
| Checkpoint | N/A | Fixed: saves/loads shared weights + vocab |

## Trace References

- Solo baselines: `traces/*_combo_edge+diff_*_s42/`
- Exp 1 (accum, shared vocab): `traces/transfer/20260316_124238_CDT/`
- Exp 2 (shared adj, rank=32): `traces/transfer/20260316_134533_CDT/`
- Exp 3 (mixed+consist α=0.01): `traces/transfer/20260316_143900_CDT/`
- Exp 4 sweep — α=0: `traces/transfer/20260316_152914_CDT/`
- Exp 4 sweep — α=0.001: `traces/transfer/20260316_155931_CDT/`
- Exp 4 sweep — α=0.005: `traces/transfer/20260316_163156_CDT/`
- Exp 4 sweep — α=0.05: `traces/transfer/20260316_170419_CDT/`
- Exp 4 sweep — α=0.1: `traces/transfer/20260316_173640_CDT/`
- Exp 5 (adaptive α=0.01): `traces/transfer/20260316_183223_CDT/`
