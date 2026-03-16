# Finding 010: scDFM-Inspired Improvements

**Date:** 2026-03-15
**Phase:** 6
**Datasets:** mESC, hESC / STRING, ChIP-seq, Non-ChIP (all three GTs evaluated)
**Baseline:** GAT+SEM (mESC AUROC 0.616/EPR 4.24, hESC AUROC 0.664/EPR 4.75)

## Motivation

scDFM (ICLR 2026) is a distributional flow matching model for single-cell
perturbation prediction that introduces several architectural innovations.
We evaluated 4 ideas adapted to our GRN inference framework to see if any
improve adjacency learning.

## Features Tested

| # | Feature | Config | What it does |
|---|---------|--------|-------------|
| 1 | diff_attn | `model.diff_attn: true` | Differential attention: softmax(Q1K1^T) - λ·softmax(Q2K2^T) |
| 2 | knn_mask | `model.knn_mask_k: 30` | Sparse KNN correlation mask on attention logits |
| 3 | adaln_zero | `model.adaln_zero: true` | Adaptive layer norm time conditioning (scale, shift, gate) |
| 4 | mmd_loss | `train.alpha_mmd: 0.5` | MMD distributional alignment auxiliary loss |

## Results

### Individual features (delta vs baseline, mean across mESC + hESC)

**STRING ground truth:**

| Feature | dAUROC | dAUPR | dAUPRR | dEPR |
|---------|--------|-------|--------|------|
| diff_attn | +0.0003 | -0.0006 | -0.028 | +0.015 |
| knn_mask | **-0.0375** | **-0.0170** | **-1.102** | **-1.948** |
| adaln_zero | -0.0007 | -0.0026 | -0.124 | -0.188 |
| mmd_loss | -0.0002 | -0.0003 | -0.013 | -0.025 |

**ChIP-seq ground truth:**

| Feature | dAUROC | dAUPR | dAUPRR | dEPR |
|---------|--------|-------|--------|------|
| diff_attn | **+0.0035** | +0.0005 | -0.002 | -0.023 |
| knn_mask | +0.0043 | -0.0038 | -0.000 | -0.005 |
| adaln_zero | -0.0007 | -0.0032 | -0.018 | -0.040 |
| mmd_loss | +0.0001 | -0.0003 | -0.001 | +0.000 |

**Non-ChIP ground truth:**

| Feature | dAUROC | dAUPR | dAUPRR | dEPR |
|---------|--------|-------|--------|------|
| diff_attn | **+0.0020** | +0.0002 | +0.014 | +0.053 |
| knn_mask | -0.0113 | -0.0028 | -0.229 | -0.957 |
| adaln_zero | -0.0033 | -0.0006 | -0.047 | -0.126 |
| mmd_loss | +0.0002 | +0.0000 | +0.002 | +0.045 |

### Combination: diff_attn + knn_mask

| GT | dAUROC | dEPR |
|----|--------|------|
| STRING | -0.019 | -1.373 |
| ChIP-seq | +0.007 | +0.007 |
| Non-ChIP | -0.027 | -0.415 |

knn_mask damage dominates; diff_attn cannot recover.

## Analysis

### diff_attn: the only positive signal

Differential attention produces small but consistent AUROC gains on
experimental ground truths (ChIP-seq +0.0035, Non-ChIP +0.0020). The
mechanism suppresses spurious attention patterns by subtracting a learned
"background" distribution, which slightly sharpens the gradient signal
flowing back to A. The effect is modest — comparable to noise at single-seed
level — but it's the only feature that doesn't hurt anywhere.

### knn_mask: hard masking destroys A learning

The KNN mask was the strongest negative result. On mESC STRING, EPR collapsed
from 4.24 to 1.66 (−61%). The fundamental problem: **correlation ≠ regulation**.
A hard mask based on co-expression forces zero attention between weakly
correlated gene pairs, but many true regulatory edges connect genes that are
not strongly co-expressed (e.g., repressors, indirect regulation). With k=30
and G=1000, only ~6% of gene pairs are unmasked — too restrictive for the
adjacency to explore the full regulatory space.

This is consistent with Finding 006 (co-expression correlation bias was
harmful) and reinforces the lesson: static co-expression structure should
not be used to constrain the learnable adjacency.

### adaln_zero: zero-init gates too conservative

adaLN-Zero consistently hurt across all GTs (STRING EPR −0.19, Non-ChIP
EPR −0.13). The zero-initialized gates start the attention and FFN
sub-blocks as identity functions, which means the model must first learn to
"open" the gates before it can learn useful representations. With only 500
epochs and 4 steps/epoch (2000 total gradient steps), this warmup cost is
not recouped. The feature might work with longer training or non-zero gate
initialization, but is not worth pursuing at current training budgets.

### mmd_loss: irrelevant to adjacency learning

MMD loss was effectively neutral (|dAUROC| < 0.001 everywhere). The MMD
gradient flows through the velocity field to model parameters, but A's
gradient from velocity prediction is already provided by the CFM loss.
Adding distributional alignment between one-step predictions and targets
doesn't strengthen A's learning signal — it just adds another source of
gradients for the neural network weights, which already train well.

scDFM uses MMD because its task (perturbation prediction) requires matching
target distributions. Our task (learning A) is fundamentally different —
we extract A as a byproduct of velocity prediction, so the quality of the
generated distribution is not the optimization target.

## Bug discovered: knn_mask wiring

The initial experiment showed knn_mask having zero effect (all deltas
exactly 0.0000). Investigation revealed that `train.py` computed the
correlation matrix but didn't pass it to the model when only `knn_mask_k`
was set. The `corr_for_model` guard only checked `corr_bias` and
`edge_features`, silently producing `corr_matrix=None` and skipping mask
construction. Fixed by adding `knn_mask_k > 0` to the guard.

## Conclusion

The scDFM innovations were designed for perturbation prediction (generating
realistic cell distributions), not GRN inference (learning a sparse adjacency
matrix). The bottleneck in our model is the gradient signal to A, and only
diff_attn marginally helps by denoising attention patterns. The other three
features either constrain A's learning (knn_mask), slow down training
(adaln_zero), or are orthogonal to A learning (mmd_loss).

**diff_attn** is the only candidate worth keeping, with the caveat that the
effect is small and needs multi-seed/multi-dataset validation.

## Configuration Reference

All features default to off (preserving original behavior):

```json
{
    "model": {
        "diff_attn": false,
        "knn_mask_k": 0,
        "adaln_zero": false
    },
    "train": {
        "alpha_mmd": 0.0
    }
}
```
