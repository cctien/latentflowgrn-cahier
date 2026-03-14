# Finding 002: OT vs Independent Coupling — No Significant Difference

**Date:** 2026-03-14
**Status:** Complete
**Depends on:** Finding 001 (wd_adj=0 fix)

## Question

Does OT-CFM (optimal transport conditional flow matching) improve GRN inference
over standard independent CFM?

## Setup

Ran both coupling modes with the fixed configuration (`wd_adj=0.0`) on
mESC/1000/STRING for 500 epochs. All other settings identical.

## Results

| Coupling    | AUROC  | AUPR   | AUPRR  | EP  | EPR    | Final flow_loss |
|-------------|--------|--------|--------|-----|--------|-----------------|
| OT          | 0.6155 | 0.0440 | 2.0687 | 700 | 3.8820 | 1.484           |
| Independent | 0.6153 | 0.0444 | 2.0888 | 693 | 3.8432 | 1.578           |

## Analysis

**No meaningful difference.** AUROC is identical (0.6155 vs 0.6153). AUPR is
marginally better for independent (0.0444 vs 0.0440) but within noise.

### Training dynamics differ

OT coupling converges to a lower flow loss (1.484 vs 1.578), indicating it
finds straighter transport paths. However, this doesn't translate to better
GRN inference — the adjacency matrix A evolves similarly in both cases:

| Metric        | OT     | Independent |
|---------------|--------|-------------|
| A_sparsity@500| 0.268  | 0.295       |
| A_grad@200    | 0.167  | 0.158       |
| A_grad@400    | 0.102  | 0.087       |

Independent coupling produces slightly more natural sparsity (29.5% vs 26.8%)
and slightly higher early AUROC at epoch 24 (0.587 vs 0.579), but both converge
to the same final performance.

## Conclusion

For Phase 1 with this architecture, **OT vs independent coupling makes no
practical difference**. Both achieve ~0.616 AUROC and ~0.044 AUPR. The choice
of coupling mode is not a bottleneck — architecture changes (Phase 2 GAT) are
more likely to close the remaining precision gap vs RegDiffusion.

Default remains `ot_coupling="ot"` since it achieves lower flow loss and is
the theoretically motivated choice, even if the empirical difference is negligible.

## Comparison to RegDiffusion Baseline

| Model              | AUROC  | AUPR   | AUPRR  | EPR    |
|--------------------|--------|--------|--------|--------|
| RegDiffusion       | 0.6126 | 0.0522 | 2.4531 | 4.2203 |
| LatentFlowGRN (OT) | 0.6155 | 0.0440 | 2.0687 | 3.8820 |
| LatentFlowGRN (ind) | 0.6153 | 0.0444 | 2.0888 | 3.8432 |

LatentFlowGRN matches or slightly exceeds baseline AUROC but trails on precision
(AUPR, EPR). The AUPR gap (0.044 vs 0.052) suggests the model ranks edges well
overall but the top-ranked predictions are less precise.

## Trace References

- OT run: `traces/mESC_1000_STRING_ablation_ot/`
- Independent run: `traces/mESC_1000_STRING_ablation_independent/`
