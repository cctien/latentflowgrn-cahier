# Finding 009: GRNFormer-Inspired Improvements

**Date:** 2026-03-15
**Phase:** 5
**Dataset:** mESC / STRING (development set)
**Baseline:** GAT+SEM (AUROC 0.6159, AUPR 0.0529)

## Motivation

GRNFormer (Hegde & Cheng, 2026) is a supervised graph transformer for GRN
inference that achieves strong results on BEELINE benchmarks. We evaluated
8 ideas from GRNFormer adapted to our unsupervised flow-matching framework
to see if any improve performance.

## Features Tested

| # | Feature | Config | What it does |
|---|---------|--------|-------------|
| 1 | corr_init | `model.adj_init: "corr"` | Initialize A from correlation matrix |
| 2 | edge_feat | `model.edge_features: true` | Learned edge feature projection in attention |
| 3 | var_embed | `model.variational_embed: true` | Variational gene embeddings + KL loss |
| 4 | tf_mask | `model.tf_mask: true` | Zero non-TF rows in A at inference |
| 5 | leaky_relu | `model.activation: "leaky_relu"` | LeakyReLU throughout |
| 6 | arcsinh | `data.normalization: "arcsinh"` | Arcsinh + z-score normalization |

Two additional features (balanced_neg_sampling, ground_truth_union) were not
validly testable in the unsupervised setting — see Analysis section.

## Results

| Feature | AUROC | dAUROC | AUPR | dAUPR | AUPRR | dAUPRR |
|---------|-------|--------|------|-------|-------|--------|
| baseline | 0.6159 | — | 0.0529 | — | 2.488 | — |
| corr_init | 0.6033 | -0.013 | 0.0497 | -0.003 | 2.336 | -0.153 |
| edge_feat | 0.6192 | **+0.003** | 0.0526 | -0.000 | 2.473 | -0.016 |
| var_embed | 0.6168 | +0.001 | 0.0515 | -0.001 | 2.421 | -0.067 |
| tf_mask | 0.6159 | 0.000 | 0.0529 | 0.000 | 2.488 | 0.000 |
| leaky_relu | 0.6166 | +0.001 | 0.0519 | -0.001 | 2.442 | -0.047 |
| arcsinh | 0.6146 | -0.001 | 0.0446 | -0.008 | 2.099 | -0.390 |

## Analysis

### No feature produced a meaningful improvement

The best result was **edge_feat** at +0.003 AUROC, which is within noise for
a single-seed experiment. All other features were neutral or harmful.

### Why these ideas don't transfer from GRNFormer

The core issue is that GRNFormer is a **supervised** model (BCE loss on known
edges) while LatentFlowGRN is **unsupervised** (flow matching loss only).
Ideas that work in a supervised setting don't necessarily help when the
adjacency matrix is learned purely from velocity prediction:

1. **corr_init** (-0.013): Correlation != regulation (consistent with
   Finding 006). The uniform initialization lets A learn freely from gradient
   signal. A correlation-biased start pushes A toward co-expression structure
   that the model then has to unlearn.

2. **edge_feat** (+0.003): The only mildly positive result. A learned
   per-head projection of correlation values is slightly better than the
   fixed `corr_bias` (which was harmful in Finding 006). The learned
   projection can potentially filter out irrelevant correlations. However,
   the effect is marginal and needs multi-dataset validation.

3. **var_embed** (+0.001): The variational regularization at alpha_kl=0.001
   is too weak to matter. The point-estimate gene embeddings already work
   well — GRNFormer needs variational embeddings because it operates on
   small subgraphs with high variance, while our model sees the full gene
   set every batch.

4. **tf_mask** (0.000): The TF mask zeros non-TF rows at inference, but
   the BEELINE evaluator already focuses on edges present in the ground
   truth. If non-TF genes have near-zero A values (they should — no
   regulatory gradient signal drives them), the mask changes nothing.

5. **leaky_relu** (+0.001): Activation choice doesn't matter at this scale.
   GRNFormer uses LeakyReLU to preserve negative co-expression gradients,
   but in our architecture A is the primary carrier of regulatory information,
   not the activation patterns.

6. **arcsinh** (-0.001 AUROC, -0.008 AUPR): The existing min-max + z-score
   normalization works better. Arcsinh is designed for raw count data;
   BEELINE data may already be preprocessed.

### Features that couldn't be validly tested

**balanced_neg_sampling**: Only applies when supervision (`alpha_sup > 0`)
is enabled. The experiment config mistakenly turned on supervision to test
it, producing +0.13 AUROC from data leakage (same issue as Finding 004).
The balanced sampling itself is a minor refinement to the supervision
pathway and cannot be tested independently.

**ground_truth_union**: Had a bug where the unioned edges replaced the
evaluation ground truth, making metrics incomparable. Even with the bug
fixed, gt_union is only useful for semi-supervised training (supervise with
ChIP-seq edges, evaluate on STRING). Since our best model is unsupervised,
this is a protocol change rather than a feature improvement.

Both features remain in the codebase as infrastructure for potential
semi-supervised experiments but are not part of the unsupervised comparison.

### How GRNFormer avoids leakage in supervised training

GRNFormer handles the supervision/evaluation boundary by:
1. Splitting edges into train/test sets within each subgraph
2. Evaluating on entirely held-out cell types (cross-dataset generalization)
3. Constructing clean negative pools excluding all training edges

Our BEELINE setup doesn't split edges because unsupervised methods don't
need to. Adding proper supervised training would require edge splitting
infrastructure — a separate research direction.

## Conclusion

None of the GRNFormer-inspired ideas produced meaningful improvements when
adapted to the unsupervised flow-matching framework. The key architectural
ideas from GRNFormer (subgraph sampling, graph VAE, supervised edge
prediction) are fundamentally tied to its supervised paradigm and don't
transfer to an unsupervised velocity-prediction approach.

**edge_feat** (+0.003 AUROC) is the only candidate worth multi-dataset
validation, but the expected improvement is small.

## Configuration Reference

All features are configurable and default to off (preserving original
behavior):

```json
{
    "data": {
        "normalization": "zscore",
        "ground_truth_union": false
    },
    "model": {
        "adj_init": "default",
        "edge_features": false,
        "variational_embed": false,
        "tf_mask": false,
        "activation": "tanh",
        "ffn_activation": "gelu"
    },
    "train": {
        "alpha_kl": 0.0,
        "balanced_neg_sampling": false
    }
}
```
