# Finding 001: Adjacency Matrix Collapse — Root Cause and Fix

**Date:** 2026-03-14
**Status:** Resolved

## Problem

Phase 1 LatentFlowGRN model achieved AUROC ~0.50 (random) on mESC/1000/STRING,
far below the RegDiffusion baseline (AUROC 0.613). The learned adjacency matrix A
collapsed to zero during training, eliminating all gene-gene regulatory structure.

## Investigation

### Round 1: L1 Sweep (9 configurations)

Swept `alpha_l1 ∈ {0, 0.0001, 0.001}`, `l1_delay ∈ {100, 200}`, `l1_ramp ∈ {0, 100}`.

**Result:** All 9 configurations produced identical results (best AUROC 0.5801 at
epoch 49). A collapsed to zero in all cases — including `alpha_l1=0` (no L1 at all).

**Conclusion:** L1 regularization is not the cause of A's collapse.

### Round 2: Adjacency Survival Sweep (9 configurations)

Swept `wd_adj ∈ {0.0, 0.001, 0.01}` × `adj_mixing ∈ {pre, post, both}`, with L1
disabled to isolate effects.

**Results (ranked by best AUROC):**

| wd_adj | adj_mixing | Best AUROC | A status at epoch 300 |
|--------|------------|------------|----------------------|
| 0.0    | both       | **0.6157** | alive (A_mean 99%)   |
| 0.0    | post       | **0.6155** | alive (A_mean 99%)   |
| 0.001  | post       | 0.5979     | collapsed (4%)       |
| 0.001  | both       | 0.5963     | collapsed (4%)       |
| 0.01   | post       | 0.5814     | dead (0.3%)          |
| 0.01   | both       | 0.5736     | dead (0.3%)          |
| 0.0    | pre        | 0.5463     | frozen (no gradient)  |
| 0.001  | pre        | 0.5304     | dead                 |
| 0.01   | pre        | 0.5013     | dead                 |

### Round 3: L1 Re-validation with wd_adj=0 (9 configurations)

With the fix applied (`wd_adj=0`), re-tested L1 sparsity:
`alpha_l1 ∈ {0, 0.0001, 0.001, 0.01}`, `l1_delay ∈ {100, 200}`, `l1_ramp=100`.

**Result:** All configurations achieve identical metrics:

| alpha_l1 | AUROC  | AUPR   | AUPRR  | EPR    |
|----------|--------|--------|--------|--------|
| 0.0      | 0.6155 | 0.0440 | 2.0687 | 3.8820 |
| 0.0001   | 0.6155 | 0.0440 | 2.0687 | 3.8820 |
| 0.001    | 0.6155 | 0.0440 | 2.0687 | 3.8820 |
| 0.01     | 0.6155 | 0.0440 | 2.0687 | 3.8820 |

**Key observation:** A now learns naturally with `wd_adj=0`:
- A_mean stays stable at ~0.0027 throughout training (no decay)
- A_grad_norm grows from 0.018 → 0.17 (strong, active gradient)
- A_sparsity increases naturally: 0.001 → 0.085 → 0.268 by epoch 499
- Flow loss drops from 1.89 → 1.48 (substantial learning, unlike the ~1.88 plateau before)

**Conclusion:** L1 has no measurable effect at these magnitudes. The L1 loss
(~0.00003 at `alpha_l1=0.01`) is negligible vs flow loss (~1.48). Natural sparsity
emerges without explicit L1 pressure. L1 regularization is unnecessary for Phase 1.

## Root Cause

**Weight decay on the adjacency parameter group** (`wd_adj=0.01` in Adam) was
the primary cause. The implicit L2 penalty `wd * A` applied every step overwhelmed
the weak gradient signal from the flow-matching loss on A (grad norm ~0.003 by
epoch 50 with weight decay; grows to ~0.17 without it).

### Why weight decay killed A

With `wd_adj=0.01`, weight decay applied `-0.01 * A` to the gradient each step.
Early in training, A's flow gradient (~0.01) barely counteracted this. Once the
network learned to predict velocity gene-independently, A's gradient dropped to
~0.003, and weight decay dominated — pulling A to zero exponentially.

### Why removing weight decay fixes everything

With `wd_adj=0.0`, A is free to evolve based purely on the flow-matching signal.
The gradient grows over training (0.018 → 0.17), A learns meaningful cross-gene
structure, and natural sparsity emerges (~27% zero entries by epoch 500) without
any explicit regularization.

### Why pre-mixing fails

With `adj_mixing="pre"`, `(I-A)` is applied *before* the velocity blocks. The
velocity blocks learn to invert the mixing, and A's gradient drops to zero by
epoch 25. A freezes at its initial uniform value.

## Fix Applied

- Set `wd_adj=0.0` (remove weight decay on A)
- Keep `adj_mixing="post"` (simpler, equivalent to `both`)
- Keep `alpha_l1=0.001` with `l1_delay=100, l1_ramp=100` (harmless, may help at scale)

## Final Performance

| Model              | AUROC  | AUPR   | AUPRR  | EPR    |
|--------------------|--------|--------|--------|--------|
| RegDiffusion       | 0.6126 | 0.0522 | 2.4531 | 4.2203 |
| LatentFlowGRN (v2) | 0.6155 | 0.0440 | 2.0687 | 3.8820 |

LatentFlowGRN matches baseline AUROC (0.616 vs 0.613) but lags on precision
metrics (AUPR 0.044 vs 0.052). This gap may be addressable in Phase 2 with the
GAT architecture providing stronger A coupling.

## Trace References

- L1 sweep (round 1): `traces/mESC_1000_STRING_l1sweep_*/` (with wd_adj=0.01)
- Adj sweep: `traces/mESC_1000_STRING_adjsweep_*/`
- L1 revalidation (round 3): `traces/mESC_1000_STRING_l1sweep_*/` (with wd_adj=0.0)
- Summaries: `traces/l1_sweep_summary.json`, `traces/adj_sweep_summary.json`
