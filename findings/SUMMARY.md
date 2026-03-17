# LatentFlowGRN: Findings Summary

**Date:** 2026-03-15
**Authors:** (project team)

## Abstract

LatentFlowGRN embeds a learnable adjacency matrix A into a conditional flow
matching model for gene regulatory network (GRN) inference from single-cell
RNA-seq data. Through systematic investigation across architecture variants,
coupling mechanisms, and regularization strategies, we identified a GAT+SEM
configuration that outperforms the RegDiffusion baseline on AUROC across all
BEELINE benchmark ground truth types. This document summarizes the key
findings from Phase 1 (foundation), Phase 2 (GAT architecture), and Phase 3
(full BEELINE benchmark).

---

## 1. Problem Setting

**Task:** Infer gene regulatory networks from scRNA-seq expression data.

**Approach:** Learn a velocity field v(x_t, t; A) via conditional flow matching,
where A is a learnable G x G adjacency matrix embedded in the model architecture.
After training, |A| is read off as the predicted GRN.

**Baseline:** RegDiffusion — a diffusion-based method with the same (I-A) mixing
strategy but using DDPM noise prediction instead of flow matching.

**Evaluation:** BEELINE benchmark — 7 datasets, 3 ground truth types (STRING,
ChIP-seq, Non-ChIP), 1000 genes. Metrics: AUROC (overall ranking), AUPR
(precision-recall), AUPRR (AUPR ratio vs random), EPR (early precision ratio).

---

## 2. Phase 1: Foundation and the Weight Decay Discovery

### Architecture

MLP velocity field with per-gene embeddings, sinusoidal time conditioning,
stacked VelocityBlocks, and post-hoc (I-A) adjacency mixing. Directly ports
RegDiffusion's architecture to flow matching.

### Critical finding: weight decay kills A (Finding 001)

The adjacency matrix collapsed to zero during training, producing random
AUROC (~0.50). Three rounds of investigation:

1. **L1 sweep** — L1 regularization was not the cause (A collapsed even with
   alpha_l1=0)
2. **Adjacency survival sweep** — Weight decay (`wd_adj=0.01`) on the A
   parameter group was the culprit. The implicit L2 penalty overwhelmed A's
   weak gradient signal from the flow-matching loss.
3. **Fix:** Set `wd_adj=0.0`. AUROC immediately rose to **0.616**, matching
   the RegDiffusion baseline (0.613).

### OT vs independent coupling (Finding 002)

No significant difference between OT-CFM and independent CFM for GRN
inference (AUROC 0.6155 vs 0.6153). The coupling mode is not a bottleneck.

### Phase 1 outcome

| Model | AUROC | AUPR |
|-------|-------|------|
| RegDiffusion | 0.6126 | 0.0522 |
| MLP (wd_adj=0) | 0.6155 | 0.0440 |

AUROC matches baseline. AUPR gap remains (0.044 vs 0.052).

---

## 3. Phase 2: GAT Architecture

### Motivation

The MLP's post-hoc (I-A) mixing is a weak coupling — the network can learn
to predict velocity gene-independently, making A's gradient vanishingly small.
A GAT architecture integrates A directly into attention logits, creating
tighter gradient flow.

### GAT design (Finding 003)

Custom dense multi-head attention (not PyG GATConv) with A as additive bias:

```
logit[h, g] = (q_h . k_g) / sqrt(d) + a_scale * A[g, h]
```

**Key discoveries:**

- **a_scale_init must be ~100** — A values (~0.003) are invisible to
  attention logits (~1-10) without amplification. At a_scale=1, AUROC=0.534.
  At a_scale=100, AUROC=0.607.

- **Lower LR helps** — Transformers benefit from lr=2e-4 (vs MLP's 1e-3).
  Best GAT: lr=2e-4, warmup=50, hidden_dim=128, spe=4, batch=32.

- **GAT trades AUROC for precision** — AUROC 0.607 (below MLP's 0.616) but
  AUPR 0.050 (above MLP's 0.044). Attention concentrates gradient on the
  most important edges.

### SEM residual: the breakthrough (Finding 005)

Adding a direct linear pathway `v += lambda_adj * (x_t @ A)` alongside the
attention mechanism dramatically improved GAT:

| Config | AUROC | AUPR | AUPRR |
|--------|-------|------|-------|
| GAT baseline | 0.6071 | 0.0500 | 2.353 |
| GAT + SEM(0.2) | 0.6154 | 0.0524 | 2.465 |
| **GAT + SEM(0.3)** | **0.6164** | **0.0526** | **2.472** |
| RegDiffusion | 0.6126 | 0.0522 | 2.453 |

**GAT + SEM(0.3) beats RegDiffusion on AUROC, AUPR, and AUPRR** on the
development dataset (mESC/STRING). The SEM residual provides a first-order
gradient pathway (`dL/dA = dL/dv * x_t^T`) that complements the attention
bias, creating triple A-coupling: attention routing + SEM linear + learnable
a_scale.

**Lambda peak at 0.3** — above this, the linear term dominates and the
network degenerates toward a linear SEM.

### What didn't work (Findings 004, 006)

- **Known-edge supervision** — data leakage (trains on evaluation edges)
- **NOTEARS acyclicity** — no effect at tested scales
- **Low-rank A factorization** — catastrophic (AUROC ~0.49). Sparse GRNs
  need full-rank parameterization, not low-dimensional structure.
- **Co-expression correlation bias** — slightly harmful. Correlation != regulation.

---

## 4. Phase 3: Full BEELINE Benchmark (Finding 007)

### Aggregated results (mean across 7 datasets)

| Model | STRING AUROC | STRING AUPR | ChIP AUROC | ChIP AUPR | Non-ChIP AUROC |
|-------|-------------|-------------|------------|-----------|---------------|
| **MLP+SEM** | **0.6743** | 0.1280 | 0.5205 | 0.3818 | 0.5822 |
| **GAT+SEM** | 0.6706 | 0.1237 | **0.5218** | **0.3845** | **0.5851** |
| Baseline | 0.6612 | **0.1330** | 0.5116 | 0.3796 | 0.5767 |

### Key results

1. **Both models beat RegDiffusion on AUROC** across all ground truth types.
   MLP leads on STRING (+0.013 avg), GAT leads on ChIP-seq and Non-ChIP.

2. **Baseline retains AUPR advantage on STRING** (0.133 vs 0.128/0.124).
   The advantage observed on mESC does not fully generalize.

3. **MLP and GAT have complementary strengths:**
   - MLP wins on larger datasets (mHSC-E, mHSC-GM) — benefits from batch=256
   - GAT wins on smaller datasets (hESC, mESC, mHSC-L) — attention helps
     with limited data
   - Per-dataset STRING AUROC wins: MLP 3, GAT 3, Baseline 1

4. **GAT excels on ChIP-seq ground truth** — best on both AUROC and AUPR.
   Cell-type-specific ChIP-seq may favor attention-based routing.

5. **mDC is the hardest dataset** — baseline wins; all models below 0.58.

### Per-dataset results (STRING ground truth)

| Dataset | Cells | Baseline | MLP | GAT | Winner |
|---------|-------|----------|-----|-----|--------|
| hESC | ~758 | 0.653 | 0.662 | **0.664** | GAT |
| hHep | ~425 | 0.653 | **0.677** | 0.670 | MLP |
| mDC | ~1700 | **0.573** | 0.551 | 0.556 | Baseline |
| mESC | ~421 | 0.613 | 0.616 | **0.616** | GAT |
| mHSC-E | ~1656 | 0.708 | **0.744** | 0.727 | MLP |
| mHSC-GM | ~1656 | 0.738 | **0.783** | 0.764 | MLP |
| mHSC-L | ~1656 | 0.690 | 0.687 | **0.698** | GAT |

---

## 5. Best Configurations

### GAT + SEM + edge+diff (best for STRING AUROC, Finding 011)

```json
{
    "model": {
        "architecture": "gat", "hidden_dim": 128, "n_heads": 4,
        "n_layers": 2, "a_scale_init": 100.0, "lambda_adj": 0.3,
        "edge_features": true, "diff_attn": true
    },
    "train": {
        "batch_size": 32, "lr": 2e-4, "lr_warmup": 50,
        "steps_per_epoch": 4, "wd_adj": 0.0, "epochs": 500
    }
}
```

### GAT + SEM + diff_attn (best for ChIP-seq and Non-ChIP)

```json
{
    "model": {
        "architecture": "gat", "hidden_dim": 128, "n_heads": 4,
        "n_layers": 2, "a_scale_init": 100.0, "lambda_adj": 0.3,
        "diff_attn": true
    },
    "train": {
        "batch_size": 32, "lr": 2e-4, "lr_warmup": 50,
        "steps_per_epoch": 4, "wd_adj": 0.0, "epochs": 500
    }
}
```

### MLP + SEM (best for STRING AUROC on large datasets)

```json
{
    "model": {"architecture": "mlp", "n_layers": 3, "lambda_adj": 0.1},
    "train": {"batch_size": 256, "lr": 1e-3, "wd_adj": 0.0, "epochs": 500}
}
```

---

## 6. Technical Lessons

| Lesson | Impact | Finding |
|--------|--------|---------|
| Weight decay on A must be zero | Critical — A collapses otherwise | 001 |
| OT vs independent coupling doesn't matter | No effect on GRN quality | 002 |
| GAT needs a_scale ~100 for A visibility | 0.534 -> 0.607 AUROC | 003 |
| SEM residual + GAT is synergistic | 0.607 -> 0.616 AUROC | 005 |
| Lambda_adj peak at 0.3 (GAT), 0.1 (MLP) | Higher values degrade | 005 |
| Low-rank A factorization fails | Sparse GRNs need full rank | 006 |
| Co-expression bias doesn't help | Correlation != regulation | 006 |
| Supervision on eval edges = leakage | AUROC=1.0 (trivially) | 004 |
| L1 regularization is ineffective | ~0.00003 vs ~1.5 flow loss | 001 |
| Joint training neutral with per-dataset embeddings | Near-zero gene overlap | 008 |
| Few-shot is stable (10% ≈ 100%) but below solo | Frozen blocks constrain | 008 |
| Shared gene vocab + mixed dataloader matches solo | +0.009 best, mean tied at 0.6716 | 014 |

---

## 5. Phase 4: Transfer Learning (Finding 008, superseded by Finding 014)

### Architecture

Weight sharing via module references: GAT blocks (attention, FFN, time
projection) shared across all datasets; gene embeddings and A_k per-dataset.
Different gene counts supported naturally since blocks operate on hidden dim D.

### Joint training (5 mouse datasets)

| Dataset | Solo AUROC | Joint AUROC | Delta |
|---------|-----------|-------------|-------|
| mDC | 0.556 | 0.544 | -0.011 |
| mESC | 0.616 | **0.618** | +0.002 |
| mHSC-E | 0.727 | 0.727 | 0.000 |
| mHSC-GM | 0.764 | 0.759 | -0.005 |
| mHSC-L | 0.698 | 0.691 | -0.006 |
| **Mean** | **0.672** | **0.668** | **-0.004** |

**Roughly neutral.** No dataset shows significant benefit or harm.

### Few-shot titration (mESC, pretrained on 4 other datasets)

| Cells | AUROC | vs Solo (0.616) |
|-------|-------|-----------------|
| 421 (100%) | 0.599 | -0.017 |
| 210 (50%) | 0.599 | -0.017 |
| 84 (20%) | 0.601 | -0.016 |
| 42 (10%) | 0.602 | -0.015 |

**Stable across fractions** — pretrained blocks prevent overfitting. But
**absolute performance is 0.015 below solo** due to domain mismatch.

### Root cause: gene overlap

HVG selection picks different genes per dataset. Only 13 genes overlap
across all 5 mouse datasets (out of 4223 union). The shared blocks learn
generic hidden-state processing but can't transfer gene-specific regulatory
knowledge. To see real transfer benefit, datasets need shared gene panels
or ortholog-based mapping.

---

## 6. Phase 5: GRNFormer-Inspired Improvements (Finding 009)

Evaluated 6 ideas from GRNFormer (a supervised graph transformer for GRN
inference) adapted to our unsupervised framework. All tested on
mESC/STRING.

| Feature | dAUROC | dAUPR | Verdict |
|---------|--------|-------|---------|
| corr_init (A from correlation) | -0.013 | -0.003 | Harmful |
| edge_feat (learned edge projection) | **+0.003** | -0.000 | Marginal |
| var_embed (variational gene embeddings) | +0.001 | -0.001 | Neutral |
| tf_mask (TF-only rows at inference) | 0.000 | 0.000 | No-op |
| leaky_relu (LeakyReLU throughout) | +0.001 | -0.001 | Neutral |
| arcsinh (arcsinh normalization) | -0.001 | -0.008 | Harmful |

**No feature produced a meaningful improvement.** The key insight is that
GRNFormer's ideas are tied to its supervised paradigm (BCE on known edges,
subgraph sampling, variational graph autoencoder). These don't transfer to
an unsupervised velocity-prediction approach where A is learned purely from
flow matching.

Two additional features (balanced_neg_sampling, ground_truth_union) could
not be validly tested in the unsupervised setting — they only apply to the
supervision pathway which is data leakage on BEELINE (Finding 004).

---

## 7. Phase 6: scDFM-Inspired Improvements (Finding 010)

Evaluated 4 ideas from scDFM (ICLR 2026), a distributional flow matching
model for perturbation prediction. Tested on mESC + hESC against all three
ground truth types.

| Feature | STRING dAUROC | ChIP dAUROC | Non-ChIP dAUROC | Verdict |
|---------|--------------|-------------|----------------|---------|
| diff_attn | +0.000 | **+0.004** | **+0.002** | Mild positive |
| knn_mask (k=30) | **-0.038** | +0.004 | -0.011 | Strong negative |
| adaln_zero | -0.001 | -0.001 | -0.003 | Negative |
| mmd_loss (α=0.5) | -0.000 | +0.000 | +0.000 | Neutral |

**diff_attn is the only positive signal** — small AUROC gains on experimental
GTs by suppressing noisy attention patterns. All other features hurt or are
irrelevant to adjacency learning:

- **knn_mask** destroyed STRING/Non-ChIP performance (EPR −61% on mESC).
  Hard masking based on correlation prevents A from exploring the full
  regulatory space. Reinforces the lesson from Finding 006: correlation ≠
  regulation.
- **adaln_zero** zero-initialized gates waste training budget warming up.
- **mmd_loss** is orthogonal to A learning — distributional alignment
  doesn't strengthen A's gradient signal.

The core issue: scDFM's innovations target perturbation prediction (matching
cell distributions), not GRN inference (learning a sparse adjacency). The
bottleneck in our model is the gradient signal to A, and only diff_attn
marginally helps by denoising attention.

### TF mask re-evaluation

A separate TF mask ablation (Finding 009b) revealed:
1. The `GRNEvaluator` already filters evaluation to TF→target edges
   regardless of our model's TF mask setting
2. TF masking applies during training (not just inference), constraining
   the SEM residual and attention bias — uniformly hurting performance
3. The STRING ground truth metrics were incorrectly reported as zeros due to
   a key naming bug in `MultiEvaluator` (bare `AUROC` vs `STRING/AUROC`),
   now fixed

---

## 8. Phase 6b: Best Feature Combination — Full BEELINE (Finding 011)

Tested the three features with positive signal (edge_feat, diff_attn, and
their combination) across all 7 BEELINE datasets. SEM residual (λ=0.3) is
always on as part of the baseline.

### Aggregated STRING AUROC (mean across 7 datasets)

| Condition | AUROC | dAUROC |
|-----------|-------|--------|
| baseline (GAT+SEM) | 0.6704 | — |
| edge_feat | 0.6691 | -0.001 |
| diff_attn | 0.6709 | +0.001 |
| **edge+diff** | **0.6716** | **+0.001** |

**edge+diff is the new best GAT config** for STRING AUROC. The combination
outperforms either feature alone — a genuine synergy where diff_attn's
noise suppression helps edge features learn cleaner correlation biases.

Per-dataset STRING winners: edge+diff 3/7, diff_attn 2/7, edge_feat 1/7,
baseline 1/7. No feature ever catastrophically hurts (worst: -0.008).

For experimental GTs, **diff_attn alone is best**: ChIP-seq +0.003,
Non-ChIP +0.002. Adding edge_feat dilutes the gain on these GTs.

### Updated best configurations

- **STRING AUROC**: GAT + SEM(0.3) + edge_features + diff_attn
- **ChIP-seq / Non-ChIP**: GAT + SEM(0.3) + diff_attn

---

## 8b. Phase 6c: Gene Embedding Style Comparison (Finding 012)

Tested whether replacing the RegDiffusion concat embedding (`[x_i ; emb_i]`)
with projected alternatives (scDFM or MLP style: `MLP(x_i) + emb_i`) improves
GRN inference on top of the best edge+diff config.

| Embed style | STRING dAUROC | ChIP dAUROC | Non-ChIP dAUROC |
|------------|--------------|-------------|----------------|
| concat (default) | — | — | — |
| scdfm | -0.003 | +0.001 | -0.002 |
| mlp | -0.002 | -0.001 | -0.001 |

**Both projected styles hurt STRING AUROC.** The raw expression scalar in a
dedicated dimension (1/128) provides a cleaner signal than projecting to
all D dimensions via an MLP. The concat approach preserves direct access
to expression values for all downstream layers, while projection distributes
information without adding any.

The edge+diff config with concat embedding remains the best.

---

## 8c. Phase 6d: Pretrained Gene Embeddings (Finding 013)

Tested whether pretrained gene embeddings (GenePT: GPT-3.5 text embeddings
of gene summaries, 1536d; scGPT: learned tokens from 33M cells, 512d) can
improve GRN inference on top of the best edge+diff config. Embeddings are
projected to 128d and added to the gene identity vector.

### Aggregated STRING AUROC (mean across mESC, hESC)

| Condition | AUROC | dAUROC vs edge+diff |
|-----------|-------|---------------------|
| edge+diff (no pretrained) | 0.6386 | — |
| ed+genept (frozen, linear) | 0.6399 | +0.001 |
| ed+scgpt (frozen, linear) | 0.6389 | +0.000 |
| ed+genept_mlpproj | 0.6374 | -0.001 |
| ed+genept_finetune | 0.6385 | -0.000 |
| ed+scgpt_mlpproj | 0.6342 | -0.004 |
| ed+scgpt_finetune | 0.6368 | -0.002 |

**Pretrained embeddings provide at best marginal gains** (+0.001 AUROC).
GenePT is slightly better than scGPT on STRING; scGPT is slightly better
on ChIP-seq and Non-ChIP. All ablations (MLP projection, finetuning) hurt
or match the frozen linear default.

The gains don't justify the added complexity: the model already learns
effective gene identities from flow matching, pretrained embeddings encode
per-gene properties rather than the pairwise regulatory relationships that
GRN inference requires, and the 1000 HVG genes are well-characterized
enough that random initialization works well.

**The edge+diff config from Finding 011 remains the best configuration.**

---

## 8d. Phase 7: Shared Vocabulary Transfer Learning (Finding 014)

Three rounds of architectural improvements to joint training:

**Experiment 1 — Shared vocab + gradient accumulation:** Shared gene
vocabulary (`nn.Embedding` over union), gradient accumulation (1 step/epoch),
per-dataset `a_scale`, fixed checkpoint bug. Result: 4/7 datasets improved
(vs 1/5 in Finding 008), mean dAUROC -0.002. mHSC-GM +0.010 (best), mDC
-0.023 (worst, outlier).

**Experiment 2 — Shared low-rank adjacency:** Decompose A_k = B + R_k with
shared low-rank base. **Catastrophic failure** (mean AUROC 0.542, -0.130
vs solo). Confirms Finding 006: low-rank A is incompatible with GRN inference.

**Experiment 3 — Mixed dataloader + consistency regularization:**
Replaced gradient accumulation with mixed dataloader (random interleaving,
28 steps/epoch). Added soft consistency loss on shared gene edges.

| Dataset | Solo | Joint (mixed+consist) | Delta |
|---------|------|-----------------------|-------|
| hESC | 0.6639 | 0.6655 | +0.002 |
| hHep | 0.6712 | 0.6698 | -0.001 |
| mDC | 0.5539 | **0.5624** | **+0.009** |
| mESC | 0.6175 | 0.6177 | +0.000 |
| mHSC-E | 0.7311 | 0.7309 | -0.000 |
| mHSC-GM | 0.7626 | 0.7529 | -0.010 |
| mHSC-L | 0.7010 | 0.7017 | +0.001 |
| **Mean** | **0.6716** | **0.6716** | **+0.000** |

**Mean now matches solo exactly.** Mixed dataloader fixed the mDC
catastrophe (-0.023 → +0.009) by giving Adam 28x more updates.

**Experiment 4 — Consistency alpha sweep:** Tested α ∈ {0, 0.001, 0.005,
0.01, 0.05, 0.1}. α=0.01 is optimal (mean ties solo at 0.6716). α=0
(mixed dataloader only, no consistency) gives mean -0.001. Higher alphas
(0.05, 0.1) are too aggressive.

Key tension: consistency helps low-overlap datasets (mDC: +0.009 at α=0.01)
but hurts high-overlap ones (mHSC-GM: -0.010). No single alpha resolves
this because the penalty is uniform across all dataset pairs. Pair-weighted
consistency (weaker for high-overlap pairs, stronger for low-overlap) is
the likely next step.

---

## 9. Technical Lessons

| Lesson | Impact | Finding |
|--------|--------|---------|
| Weight decay on A must be zero | Critical — A collapses otherwise | 001 |
| OT vs independent coupling doesn't matter | No effect on GRN quality | 002 |
| GAT needs a_scale ~100 for A visibility | 0.534 -> 0.607 AUROC | 003 |
| SEM residual + GAT is synergistic | 0.607 -> 0.616 AUROC | 005 |
| Lambda_adj peak at 0.3 (GAT), 0.1 (MLP) | Higher values degrade | 005 |
| Low-rank A factorization fails | Sparse GRNs need full rank | 006 |
| Co-expression bias doesn't help | Correlation != regulation | 006 |
| Supervision on eval edges = leakage | AUROC=1.0 (trivially) | 004 |
| L1 regularization is ineffective | ~0.00003 vs ~1.5 flow loss | 001 |
| Joint training is neutral with HVG gene sets | Near-zero gene overlap | 008 |
| Few-shot is stable (10% ≈ 100%) but below solo | Frozen blocks constrain | 008 |
| Supervised GRN ideas don't transfer to unsupervised | Paradigm mismatch | 009 |
| Correlation-based A init is harmful | Biases away from regulation | 009 |
| Hard correlation masking destroys A learning | Correlation ≠ regulation (again) | 010 |
| Distributional alignment (MMD) is orthogonal to A | Velocity ≠ distribution quality | 010 |
| Differential attention mildly helps GRN inference | Denoises attention → cleaner A gradients | 010 |
| TF mask hurts during training, is redundant at eval | Evaluator already filters to TF edges | 010 |
| edge_feat + diff_attn synergize | Combo > either alone on STRING AUROC | 011 |
| diff_attn is best single feature for experimental GTs | +0.003 ChIP-seq, +0.002 Non-ChIP | 011 |
| Projected gene embeddings don't help GRN inference | Raw scalar in 1 dim > MLP projection to D dims | 012 |
| Pretrained embeddings (GenePT/scGPT) provide negligible gains | +0.001 AUROC, not worth the complexity | 013 |
| Frozen linear projection is best for pretrained embeddings | MLP proj and finetuning both hurt | 013 |
| GRN inference needs pairwise signals, not per-gene features | Pretrained embeddings encode gene properties, not interactions | 013 |
| Shared gene vocabulary enables genuine transfer | 4/7 datasets improve, +0.010 best | 014 |
| High gene overlap predicts transfer benefit | mHSC-E/GM (618 genes, 55%) benefit most | 014 |
| Shared low-rank adjacency fails catastrophically | Confirms Finding 006: low-rank A kills GRN | 014 |
| Mixed dataloader >> gradient accumulation | 28x more Adam updates, fixes mDC outlier | 014 |
| Consistency regularization softly shares adj structure | L2 penalty on shared gene edges across datasets | 014 |

---

## 10. Limitations and Future Work

1. **AUPR gap on STRING at scale** — baseline retains slight AUPR advantage
   on non-development datasets. May reflect DDPM vs CFM objective differences
   or hyperparameter over-specialization to mESC.

2. **Batch size constraint** — GAT is limited to batch=32 by O(G^2) memory.
   This 8x throughput disadvantage vs MLP may explain MLP's superiority on
   larger datasets.

3. **Single seed** — results are from seed=42 only. Multi-seed runs needed
   for confidence intervals.

4. **Transfer learning ties solo but doesn't yet beat it** — shared
   vocabulary + mixed dataloader + consistency (α=0.01) matches solo mean
   exactly (0.6716). The mDC-vs-mHSC-GM tension (consistency helps
   low-overlap, hurts high-overlap) needs pair-weighted consistency to
   resolve. Multi-seed validation needed to confirm any positive signal.

5. **No latent space** — flow matching operates directly on expression vectors.
   An encoder-decoder architecture could improve scaling and enable transfer
   by mapping variable gene sets into a shared latent space.

6. **Single seed** — all results from seed=42 only. Multi-seed validation
   needed for the transfer learning gains to be conclusive.

---

## Appendix: Experiment Trace References

| Phase | Experiment | Traces |
|-------|-----------|--------|
| 1 | L1 sweep | `traces/mESC_1000_STRING_l1sweep_*/` |
| 1 | Adj survival sweep | `traces/mESC_1000_STRING_adjsweep_*/` |
| 1 | OT ablation | `traces/mESC_1000_STRING_ablation_*/` |
| 2 | GAT sweep | `traces/mESC_1000_STRING_gatsweep_*/` |
| 2 | GAT tuning | `traces/mESC_1000_STRING_gattune_*/` |
| 2 | Architecture comparison | `traces/mESC_1000_STRING_arch_*/` |
| 2 | Coupling mechanisms | `traces/mESC_1000_STRING_coupling_*/` |
| 2 | SEM sweep | `traces/mESC_1000_STRING_sem_*/` |
| 2 | GRN coupling (low-rank) | `traces/mESC_1000_STRING_grncoup_*/` |
| 3 | Full BEELINE benchmark | `traces/bench_*_*/` |
| 4 | Joint training | `traces/transfer/20260315_010953_CDT/` |
| 4 | Pretrain (4 datasets) | `traces/transfer/20260315_011825_CDT/` |
| 4 | Few-shot titration | `traces/transfer/20260315_01*_CDT/` |
| 5 | GRNFormer ideas ablation | `traces/*_p5_*/` |
| 6 | TF mask ablation | `traces/*_tf_mask_*/` |
| 6 | scDFM ideas ablation | `traces/*_scdfm_*/` |
| 6 | Best combo (full BEELINE) | `traces/*_combo_*/` |
| 6 | Embed style comparison | `traces/*_combo_*+scdfm_*/`, `traces/*_combo_*+mlp_*/` |
| 6 | Pretrained embeddings | `traces/*_combo_ed+genept_*/`, `traces/*_combo_ed+scgpt_*/` |
| 7 | Shared vocab + accum | `traces/transfer/20260316_124238_CDT/` |
| 7 | Shared low-rank adj (failed) | `traces/transfer/20260316_134533_CDT/` |
| 7 | Mixed dataloader + consist | `traces/transfer/20260316_143900_CDT/` |
| 7 | Consistency sweep (α=0–0.1) | `traces/transfer/20260316_15*_CDT/`, `traces/transfer/20260316_17*_CDT/` |
