# Finding 008: Transfer Learning — Joint Training and Few-Shot

**Date:** 2026-03-15
**Status:** Complete
**Depends on:** Finding 005 (GAT+SEM), Finding 007 (BEELINE benchmark)

## Setup

Transfer learning via weight sharing: GAT blocks (attention, FFN, time
projection) are shared across datasets; gene embeddings and adjacency
matrices A_k are per-dataset. Best GAT+SEM(0.3) config used throughout.

5 mouse BEELINE datasets: mDC (383 cells, 1321 genes), mESC (421 cells,
1620 genes), mHSC-E (1071 cells, 1204 genes), mHSC-GM (889 cells, 1132
genes), mHSC-L (847 cells, 692 genes). STRING ground truth, 1000 HVG
setting.

**Critical constraint:** Gene overlap across datasets is near-zero (13 genes
in the 5-way intersection out of 4223 union genes). Each dataset's HVG
selection picks almost entirely different genes.

## 4b: Joint Training Results

| Dataset | Solo AUROC | Joint AUROC | Delta | Solo AUPR | Joint AUPR |
|---------|-----------|-------------|-------|-----------|------------|
| mDC | 0.5556 | 0.5442 | -0.011 | 0.0496 | 0.0477 |
| mESC | 0.6164 | 0.6182 | +0.002 | 0.0526 | 0.0524 |
| mHSC-E | 0.7274 | 0.7274 | 0.000 | 0.1189 | 0.1092 |
| mHSC-GM | 0.7637 | 0.7586 | -0.005 | 0.2421 | 0.2390 |
| mHSC-L | 0.6976 | 0.6913 | -0.006 | 0.3012 | 0.2922 |
| **Mean** | **0.6721** | **0.6679** | **-0.004** | | |

**Joint training is roughly neutral.** mESC shows a tiny improvement; other
datasets show small degradation. On average, joint training slightly hurts
(-0.004 AUROC). No dataset shows dramatic benefit or harm.

## 4c: Few-Shot Titration (mESC)

Pretrained on 4 datasets (mDC, mHSC-E, mHSC-GM, mHSC-L), then finetune
only A_mESC with frozen shared blocks at varying cell fractions.

| Cells | Fraction | AUROC | AUPR | AUPRR | EPR |
|-------|----------|-------|------|-------|-----|
| 421 | 100% | 0.5988 | 0.0330 | 1.553 | 2.651 |
| 210 | 50% | 0.5988 | 0.0326 | 1.533 | 2.673 |
| 84 | 20% | 0.6007 | 0.0328 | 1.541 | 2.573 |
| 42 | 10% | 0.6018 | 0.0337 | 1.584 | 2.618 |

**Performance is remarkably stable across cell fractions** — 10% of cells
achieves the same AUROC as 100%. The pretrained shared blocks provide
strong regularization that prevents overfitting even with 42 cells.

**However, absolute performance is far below solo training:**

| Setup | AUROC | AUPR |
|-------|-------|------|
| Solo GAT+SEM(0.3) | **0.6164** | **0.0526** |
| Joint training | 0.6182 | 0.0524 |
| Few-shot (100% cells) | 0.5988 | 0.0330 |
| Few-shot (10% cells) | 0.6018 | 0.0337 |

The frozen shared blocks (trained without mESC) constrain the model too
much — AUPR drops by 37% (0.053 → 0.033) compared to solo training.

## Analysis

### Why joint training doesn't help

1. **Near-zero gene overlap.** Only 13 genes are shared across all 5 datasets.
   The gene embeddings are entirely per-dataset, so the shared blocks learn
   "how to process hidden states" but can't transfer gene-specific regulatory
   knowledge.

2. **Different data scales.** mHSC-E has 1071 cells while mDC has 383. The
   shared blocks are biased toward larger datasets.

3. **The shared blocks are already general.** The velocity blocks (linear
   layers operating on hidden dim D) are simple enough that they converge to
   similar weights whether trained on one or five datasets.

### Why few-shot is stable but weak

The pretrained blocks provide a strong prior on the velocity field structure
(how to process time-conditioned gene features). This prevents overfitting
with few cells — but the prior was learned on different gene sets, so it
can't capture mESC-specific regulatory dynamics. Only A_mESC is being
learned, and A alone can't compensate for the mismatched shared blocks.

### The gene overlap problem

This is the fundamental limitation. With HVG selection, each dataset
independently picks its top variable genes. The resulting gene sets barely
overlap:

| Pair | Overlap | Union |
|------|---------|-------|
| mESC & mHSC-E | 188 | 2636 |
| mESC & mHSC-GM | 171 | 2581 |
| mHSC-E & mHSC-GM | 618 | 1718 |
| All 5 mouse | 13 | 4223 |

For transfer to work, the model would need to share knowledge about
**specific genes** across datasets. With different gene sets, the only
transferable knowledge is the generic architecture weights.

## What Would Help

1. **Fixed gene panel** instead of HVG — use the same 1000 genes across all
   datasets. Gene embeddings and A structure become directly transferable.

2. **Gene-name-aware embeddings** — map genes to a shared namespace (e.g.,
   Ensembl IDs) and use a shared embedding lookup. Genes appearing in
   multiple datasets would share embeddings.

3. **Cross-species transfer with ortholog mapping** — mouse→human with
   explicit gene correspondence via orthologs. The proposal outlines this
   as Phase 4d.

4. **Encoder-based latent space** — an encoder that maps variable-size
   gene sets into a fixed-size latent representation. Flow matching in
   latent space would be naturally transferable.

## Conclusion

The transfer learning infrastructure works correctly — weight sharing,
gradient isolation, checkpoint saving, and few-shot finetuning all function
as designed. However, the near-zero gene overlap across BEELINE HVG
datasets makes this an extremely hard transfer scenario.

**Joint training is roughly neutral** (no significant help or harm). **Few-
shot finetuning shows impressive stability** (10% cells ≈ 100% cells) but
**absolute performance falls well short of solo training** due to the
domain mismatch in shared blocks.

The transfer hypothesis — that regulatory grammar transfers across cell
types — is not invalidated, but it requires shared gene vocabularies to
be properly tested. The current HVG setup makes this impossible.

## Trace References

- Joint training: `traces/transfer/20260315_010953_CDT/`
- Pretrain (4 datasets): `traces/transfer/20260315_011825_CDT/`
- Finetune 100%: `traces/transfer/20260315_012402_CDT/`
- Finetune 50%: `traces/transfer/20260315_012632_CDT/`
- Finetune 20%: `traces/transfer/20260315_012903_CDT/`
- Finetune 10%: `traces/transfer/20260315_013133_CDT/`
- Few-shot summary: `traces/fewshot_summary.json`
