# Finding 006: Low-Rank A and Co-Expression Bias — Both Fail

**Date:** 2026-03-14
**Status:** Complete (negative result)
**Depends on:** Finding 003 (GAT), Finding 005 (SEM residual)

## Question

Can GRNFormer-inspired improvements help?
1. Low-rank A factorization: `A = Z_src @ Z_tgt^T` (reduce G² to 2·G·rank params)
2. Co-expression correlation bias: add Pearson correlations to GAT attention

## Results

| Tag | AUROC | AUPR | vs reference |
|-----|-------|------|-------------|
| mlp_full (ref) | 0.6159 | 0.0472 | — |
| gat_full (ref) | 0.6071 | 0.0500 | — |
| gat_corr | 0.6030 | 0.0497 | -0.004 AUROC |
| mlp_lr16 | 0.5141 | 0.0235 | -0.102 AUROC |
| mlp_lr32 | 0.5153 | 0.0236 | -0.101 AUROC |
| mlp_lr64 | 0.5134 | 0.0235 | -0.102 AUROC |
| gat_lr16 | 0.4862 | 0.0204 | -0.121 AUROC |
| gat_lr32 | 0.4864 | 0.0207 | -0.121 AUROC |
| gat_lr64 | 0.4870 | 0.0208 | -0.120 AUROC |
| gat_lr32_corr | 0.4922 | 0.0217 | -0.115 AUROC |

**Both features harm performance. Low-rank A is catastrophic (near random).**

## Analysis

### Low-rank A: wrong inductive bias for sparse networks

The factorization `A = Z_src @ Z_tgt^T` constrains A to a rank-d subspace
where entries are correlated across rows and columns. Real GRNs are **sparse
but full-rank** — a few strong edges scattered across the full G×G space
with no low-dimensional structure.

Rank has minimal effect (16/32/64 all ~0.51 AUROC), confirming the problem
is the low-rank constraint itself, not insufficient capacity.

### Co-expression bias: correlation ≠ regulation

Fixed Pearson correlations added to attention logits slightly hurt AUROC
(0.603 vs 0.607). Co-expressed genes are not necessarily regulatory — both
may be driven by a common upstream factor. The fixed bias adds noise to the
learned attention pattern.

## Conclusion

Keep the full-rank `adj_A` parameterization with soft thresholding. The
GRNFormer-inspired ideas don't transfer to our flow-matching framework
because GRNFormer reconstructs adjacency as output (where low-rank is the
learned representation), while we embed A as a structural parameter (where
full flexibility is needed).

## Trace References

- GRN coupling sweep: `traces/mESC_1000_STRING_grncoup_*/`
- Summary: `traces/grn_coupling_sweep_summary.json`
