# Finding 007: Full BEELINE Benchmark — Phase 3 Results

**Date:** 2026-03-15
**Status:** Complete
**Depends on:** Finding 005 (SEM residual), Finding 003 (GAT)

## Setup

Three models on all 7 BEELINE datasets x 3 ground truth types at 1000 genes:
- **Baseline**: RegDiffusion (1000 steps)
- **MLP**: MLP + SEM(0.1), n_layers=3, batch=256
- **GAT**: GAT + SEM(0.3), hdim=128, lr=2e-4, warmup=50, spe=4, batch=32

## Aggregated Results (mean across 7 datasets)

| Model | Ground Truth | AUROC | AUPR | AUPRR | EPR |
|-------|-------------|-------|------|-------|-----|
| baseline | STRING | 0.6612 | 0.1330 | 4.057 | 5.697 |
| **MLP** | **STRING** | **0.6743** | 0.1280 | 3.906 | 5.408 |
| GAT | STRING | 0.6706 | 0.1237 | 3.763 | 5.450 |
| baseline | ChIP-seq | 0.5116 | 0.3796 | 1.029 | 0.987 |
| MLP | ChIP-seq | 0.5205 | 0.3818 | 1.035 | 1.058 |
| **GAT** | **ChIP-seq** | **0.5218** | **0.3845** | **1.046** | **1.075** |
| baseline | Non-ChIP | 0.5767 | **0.0554** | **2.290** | **4.275** |
| MLP | Non-ChIP | 0.5822 | 0.0536 | 2.223 | 4.218 |
| **GAT** | **Non-ChIP** | **0.5851** | 0.0512 | 2.158 | 4.122 |

## Per-Dataset Analysis: STRING Ground Truth

| Dataset | Baseline | MLP | GAT | Winner |
|---------|----------|-----|-----|--------|
| hESC | 0.6529 | 0.6623 | **0.6635** | GAT |
| hHep | 0.6534 | **0.6768** | 0.6700 | MLP |
| mDC | **0.5734** | 0.5514 | 0.5556 | Baseline |
| mESC | 0.6132 | 0.6159 | **0.6164** | GAT |
| mHSC-E | 0.7075 | **0.7439** | 0.7274 | MLP |
| mHSC-GM | 0.7384 | **0.7827** | 0.7637 | MLP |
| mHSC-L | 0.6898 | 0.6868 | **0.6976** | GAT |

**MLP wins 3, GAT wins 3, Baseline wins 1** (STRING AUROC).

## Key Findings

### 1. Both models beat RegDiffusion on AUROC (STRING)

MLP (+0.013) and GAT (+0.009) outperform the baseline on average AUROC.
The improvement is consistent — only mDC shows baseline superiority.

### 2. Baseline retains AUPR advantage on STRING and Non-ChIP

Despite higher AUROC, both MLP and GAT show slightly lower AUPR on STRING
(0.128/0.124 vs 0.133) and Non-ChIP (0.054/0.051 vs 0.055). The baseline's
DDPM objective may produce sharper top-k predictions at scale.

**Exception**: On mESC (our development dataset), GAT matches baseline AUPR
(0.0526 vs 0.0521). This is where we tuned hyperparameters.

### 3. GAT excels on ChIP-seq ground truth

GAT achieves the best AUROC (0.5218) and AUPR (0.3845) on ChIP-seq,
beating both baseline and MLP. ChIP-seq ground truths are cell-type-specific,
suggesting the attention mechanism captures cell-type-relevant regulatory
patterns better.

### 4. MLP excels on hematopoietic datasets (STRING)

MLP achieves the highest AUROC on mHSC-E (0.744) and mHSC-GM (0.783),
substantially outperforming both baseline and GAT. These datasets have
more cells (~1656) and more defined trajectories, which may favor the
MLP's higher data throughput (batch=256 vs GAT's 32).

### 5. mDC is the hardest dataset

All models struggle on mDC (AUROC 0.55-0.57), with baseline performing
best. This dataset may have noisier expression or less informative
regulatory structure for flow-matching approaches.

### 6. Overfitting to mESC

GAT+SEM(0.3) was tuned on mESC/STRING where it matched the baseline on
all metrics. On other datasets, the advantage is inconsistent — GAT
wins on some (hESC, mHSC-L) but loses on others (mHSC-E, mHSC-GM).
The hyperparameters may be over-specialized to mESC's characteristics.

## Summary Table: Wins by Model (AUROC, STRING)

| Model | Datasets Won | Avg AUROC | Avg AUPR |
|-------|-------------|-----------|----------|
| MLP | hHep, mHSC-E, mHSC-GM | **0.6743** | 0.1280 |
| GAT | hESC, mESC, mHSC-L | 0.6706 | 0.1237 |
| Baseline | mDC | 0.6612 | **0.1330** |

## Recommendations

1. **Report both MLP and GAT** — they have complementary strengths.
   MLP is better for larger datasets (mHSC), GAT for smaller ones (hESC, mESC).

2. **The AUPR gap vs baseline on non-mESC datasets** suggests the SEM
   lambda_adj may need per-dataset tuning, or the flow-matching objective
   inherently produces less precise top-k rankings than DDPM at scale.

3. **Phase 4 (transfer learning)** could address the per-dataset variation
   by training a shared model across datasets.

## Trace References

- Benchmark results: `traces/bench_*_*/`
- Summary: `traces/benchmark_summary.json`
