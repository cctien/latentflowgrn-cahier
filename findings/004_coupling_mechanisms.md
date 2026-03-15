# Finding 004: Coupling Mechanisms — SEM Residual, Supervision, NOTEARS

**Date:** 2026-03-14
**Status:** Complete
**Depends on:** Finding 001 (wd_adj=0), Finding 003 (GAT architecture)

## Question

Can three new coupling mechanisms improve GRN inference beyond the (I-A)
mixing (MLP) and attention bias (GAT) approaches?

1. SEM residual: `v += lambda_adj * (x_t @ A)`
2. Known-edge supervision: BCE loss on ground-truth edges
3. NOTEARS acyclicity: `tr(e^{A*A}) - G` penalty

## Results

| Tag | AUROC | AUPR | AUPRR | EPR | Notes |
|-----|-------|------|-------|-----|-------|
| sup_0.1 | 1.0000 | 0.9989 | 46.97 | 46.31 | DATA LEAKAGE |
| sup_0.01 | 0.9955 | 0.9242 | 43.46 | 40.74 | DATA LEAKAGE |
| sem1_sup0.1 | 0.8670 | 0.0774 | 3.64 | 3.30 | Partially leaked |
| **sem_0.1** | **0.6159** | **0.0472** | **2.22** | **3.93** | **Best valid** |
| mlp_base | 0.6155 | 0.0440 | 2.07 | 3.88 | Reference |
| dag_0.001 | 0.6155 | 0.0440 | 2.07 | 3.88 | No effect |
| dag_0.01 | 0.6155 | 0.0440 | 2.07 | 3.88 | No effect |
| sem_1.0 | 0.5925 | 0.0431 | 2.03 | 3.48 | SEM too strong |
| sem_10.0 | 0.5087 | 0.0248 | 1.17 | 1.77 | SEM destroys |

## Analysis

### 1. Supervision is data leakage (INVALID)

The supervision loss trains on the exact edges used for evaluation (AUROC,
AUPR). Achieving AUROC=1.0 is trivially expected — the model memorizes the
evaluation answers. This approach is **not a valid improvement** without a
proper train/test edge split.

To make supervision valid, one would need:
- Split ground truth edges into train (e.g., 50%) and held-out test (50%)
- Train with `alpha_sup` on the train edges only
- Evaluate on the held-out edges

This is a semi-supervised setup and changes the experimental protocol.
Not pursued further in the current unsupervised evaluation framework.

### 2. SEM residual works at low strength

`lambda_adj=0.1` provides a small improvement:
- AUROC: 0.6159 vs 0.6155 (+0.0004)
- AUPR: 0.0472 vs 0.0440 (+0.0032, 7% relative improvement)

The direct pathway `v += 0.1 * (x_t @ A)` gives A a first-order effect on
velocity, creating stronger gradient (`dL/dA = dL/dv * x_t^T`). At this
strength, it supplements the (I-A) mixing without dominating.

Higher values hurt because the linear `A @ x_t` term dominates the velocity
prediction, preventing the neural network from learning rich nonlinear
features. The model degenerates toward a linear SEM.

### 3. NOTEARS has no effect

Both `alpha_dag=0.001` and `0.01` produce results identical to the baseline.
Possible reasons:
- The DAG penalty value is tiny relative to flow loss (~1.5)
- The soft-thresholded A may already be approximately acyclic
- The matrix exponential gradient may not propagate useful signal at this scale

Higher `alpha_dag` was not tested but risks interfering with flow matching.

## Recommended Configuration

For unsupervised GRN inference (MLP):
```json
{
    "model": {"architecture": "mlp", "lambda_adj": 0.1, "n_layers": 3},
    "train": {"wd_adj": 0.0, "alpha_sup": 0.0, "alpha_dag": 0.0}
}
```

## Updated Performance Table

| Model | AUROC | AUPR | AUPRR | EPR |
|-------|-------|------|-------|-----|
| RegDiffusion baseline | 0.6126 | **0.0522** | **2.453** | **4.220** |
| MLP + SEM(0.1) | **0.6159** | 0.0472 | 2.220 | 3.926 |
| MLP baseline | 0.6155 | 0.0440 | 2.069 | 3.882 |
| GAT best (lr2e4) | 0.6071 | 0.0500 | 2.353 | 3.987 |

The SEM residual closes some of the AUPR gap with RegDiffusion
(0.047 vs 0.052) while maintaining the AUROC advantage (0.616 vs 0.613).

## Trace References

- Coupling sweep: `traces/mESC_1000_STRING_coupling_*/`
- Summary: `traces/coupling_sweep_summary.json`
