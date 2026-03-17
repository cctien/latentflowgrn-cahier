# Finding 012: Gene Embedding Style Comparison

**Date:** 2026-03-16
**Phase:** 6
**Datasets:** All 7 BEELINE datasets
**Ground truths:** STRING, ChIP-seq, Non-ChIP
**Baseline:** edge+diff (GAT+SEM+edge_features+diff_attn, concat embedding)

## Motivation

The default gene embedding (from RegDiffusion) concatenates a raw expression
scalar with a (D-1)-dim learned identity vector: `h_i = [x_i ; emb_i]`. This
confines expression information to 1/128 dimensions. scDFM and standard MLP
alternatives project the scalar to full D dimensions via a learned MLP and
add the gene identity: `h_i = MLP(x_i) + emb_i`.

We tested whether richer expression encoding improves GRN inference when
combined with the best feature set (edge+diff).

## Embedding Styles Tested

| Style | Expression encoding | Combination | Source |
|-------|-------------------|-------------|--------|
| concat (default) | Raw scalar (1 dim) | Concatenation | RegDiffusion |
| scdfm | Linear→ReLU→Linear→LN→Dropout (D dims) | Addition | scDFM (ICLR 2026) |
| mlp | Linear→SiLU→Linear (D dims) | Addition | Standard |

## Aggregated Results (mean across 7 datasets)

### STRING

| Condition | AUROC | dAUROC vs edge+diff |
|-----------|-------|---------------------|
| **edge+diff** (concat) | **0.6716** | — |
| edge+diff+mlp | 0.6698 | -0.0018 |
| edge+diff+scdfm | 0.6685 | -0.0031 |

### ChIP-seq

| Condition | AUROC | dAUROC |
|-----------|-------|--------|
| edge+diff | 0.5202 | — |
| **edge+diff+scdfm** | **0.5211** | +0.0009 |
| edge+diff+mlp | 0.5194 | -0.0008 |

### Non-ChIP

| Condition | AUROC | dAUROC |
|-----------|-------|--------|
| **edge+diff** | **0.5818** | — |
| edge+diff+mlp | 0.5810 | -0.0008 |
| edge+diff+scdfm | 0.5804 | -0.0015 |

## Per-Dataset Winners (STRING AUROC)

| Dataset | Species | Winner | Margin |
|---------|---------|--------|--------|
| hESC | human | scdfm | +0.0007 |
| hHep | human | mlp | +0.0027 |
| mDC | mouse | edge+diff | — |
| mESC | mouse | edge+diff | — |
| mHSC-E | mouse | edge+diff | — |
| mHSC-GM | mouse | mlp | +0.0029 |
| mHSC-L | mouse | edge+diff | — |

edge+diff (concat) wins 4/7 datasets. Projected styles only win on human
datasets (hESC, hHep) and one mouse dataset (mHSC-GM).

## Analysis

### Projected embeddings don't improve GRN inference

Both scdfm and mlp styles hurt STRING AUROC on the aggregate (-0.003 and
-0.002 respectively). The gains on individual human datasets are within
single-seed noise.

### Why the raw scalar works best

1. **Expression is the only per-cell varying signal.** Gene identity
   embeddings are static across all cells in a batch. In concat style, the
   expression scalar occupies a dedicated dimension that every downstream
   layer can directly read. Projecting it through an MLP distributes the
   information but doesn't add any — the model just has to work harder to
   extract it.

2. **Post-embedding activation compresses projected values.** Both MLP and
   GAT paths apply `Tanh(gene_emb(x))` immediately after embedding. For
   concat, Tanh on a single scalar is well-behaved. For projected styles,
   Tanh is applied after a nonlinear projection of the scalar across D
   dimensions, potentially over-compressing the expression range.

3. **scDFM's encoder was designed for a different task.** scDFM encodes
   control and perturbed cell states separately, then cross-attends between
   them. The `ContinuousValueEncoder` needs rich representations because
   the downstream model compares two cell states. Our model only has one
   expression input per gene — the extra projection capacity is wasted.

4. **MLP skip connections rely on the raw scalar.** In the MLP velocity
   field, between-block skip connections re-concatenate the raw expression
   value. With projected embeddings, these skips are disabled (expression
   is already distributed), removing a useful architectural feature.

## Conclusion

The RegDiffusion concat embedding remains the best gene encoding for GRN
inference. The raw expression scalar in a dedicated dimension provides a
cleaner signal than projected alternatives despite occupying only 1/128 of
the representation.

**The edge+diff config from Finding 011 stands as the current best.**

## Configuration Reference

```json
{
    "model": {
        "embed_style": "concat",   // default, best for GRN inference
        "embed_style": "scdfm",    // Linear→ReLU→Linear→LN→Dropout + add
        "embed_style": "mlp"       // Linear→SiLU→Linear + add
    }
}
```
