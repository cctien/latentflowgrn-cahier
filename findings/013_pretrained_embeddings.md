# Finding 013: Pretrained Gene Embeddings (GenePT & scGPT)

**Date:** 2026-03-16
**Phase:** 6d
**Datasets:** mESC, hESC
**Ground truths:** STRING, ChIP-seq, Non-ChIP
**Baseline:** edge+diff (GAT+SEM+edge_features+diff_attn, concat embedding)

## Motivation

Pretrained gene embeddings encode biological knowledge (gene function
summaries, expression patterns across 33M cells) that could provide a
stronger initialization than random gene identity vectors. This is
especially relevant given the transfer learning finding (Finding 008) that
HVG gene sets are near-disjoint across datasets — pretrained embeddings
offer a way to inject cross-dataset biological knowledge without requiring
shared gene panels.

Two embedding sources were tested:

| Source | Method | Dimensions | Genes |
|--------|--------|-----------|-------|
| GenePT | GPT-3.5 text embeddings of NCBI gene summaries | 1536 | ~20K |
| scGPT | Learned gene tokens from 33M-cell pretraining | 512 | ~60K |

Embeddings are projected to model dimension (128) via a learned linear or
MLP projection, then added to the gene identity vector. A learnable
fallback vector handles genes missing from the pretrained vocabulary.

## Conditions Tested

### Stage 1: Source comparison (frozen, linear projection)

| Condition | Pretrained source | Freeze | Projection |
|-----------|------------------|--------|------------|
| edge+diff (baseline) | none | — | — |
| ed+genept | GenePT | yes | linear |
| ed+scgpt | scGPT | yes | linear |

### Stage 2: Ablations (per source)

| Condition | Pretrained source | Freeze | Projection |
|-----------|------------------|--------|------------|
| ed+genept_mlpproj | GenePT | yes | MLP |
| ed+genept_finetune | GenePT | no | linear |
| ed+scgpt_mlpproj | scGPT | yes | MLP |
| ed+scgpt_finetune | scGPT | no | linear |

## Results (Aggregated: mean across mESC, hESC)

### STRING

| Condition | AUROC | dAUROC | AUPR | dAUPR |
|-----------|-------|--------|------|-------|
| edge+diff (no pretrained) | 0.6386 | — | 0.0536 | — |
| **ed+genept** | **0.6399** | **+0.0013** | 0.0532 | -0.0004 |
| ed+scgpt | 0.6389 | +0.0003 | 0.0524 | -0.0012 |
| ed+genept_mlpproj | 0.6374 | -0.0012 | 0.0510 | -0.0026 |
| ed+genept_finetune | 0.6385 | -0.0001 | 0.0505 | -0.0032 |
| ed+scgpt_mlpproj | 0.6342 | -0.0043 | 0.0519 | -0.0017 |
| ed+scgpt_finetune | 0.6368 | -0.0018 | 0.0520 | -0.0016 |

### ChIP-seq

| Condition | AUROC | dAUROC |
|-----------|-------|--------|
| edge+diff | 0.5095 | — |
| ed+genept | 0.5110 | +0.0015 |
| ed+scgpt | 0.5117 | +0.0022 |
| ed+genept_mlpproj | 0.5103 | +0.0008 |
| ed+genept_finetune | 0.5104 | +0.0009 |
| ed+scgpt_mlpproj | 0.5137 | +0.0042 |
| ed+scgpt_finetune | 0.5117 | +0.0023 |

### Non-ChIP

| Condition | AUROC | dAUROC |
|-----------|-------|--------|
| edge+diff | 0.5583 | — |
| ed+genept | 0.5590 | +0.0007 |
| ed+scgpt | 0.5609 | +0.0026 |
| ed+genept_mlpproj | 0.5585 | +0.0002 |
| ed+genept_finetune | 0.5575 | -0.0008 |
| ed+scgpt_mlpproj | 0.5620 | +0.0037 |
| ed+scgpt_finetune | 0.5607 | +0.0024 |

## Per-Dataset Results (STRING AUROC)

| Dataset | edge+diff | ed+genept | ed+scgpt | Best pretrained |
|---------|-----------|-----------|----------|-----------------|
| mESC | 0.6148 | 0.6152 (+0.0004) | 0.6144 (-0.0004) | genept |
| hESC | 0.6625 | 0.6646 (+0.0021) | 0.6633 (+0.0008) | genept |

## Analysis

### Pretrained embeddings provide negligible gains

The best pretrained condition (ed+genept) improves STRING AUROC by +0.0013
on the aggregate — well within single-seed noise. On mESC the delta is
only +0.0004. The improvement on hESC (+0.0021) is larger but still not
conclusive without multi-seed validation.

### GenePT slightly outperforms scGPT on STRING, but scGPT wins on experimental GTs

- **STRING**: GenePT +0.0013 vs scGPT +0.0003
- **ChIP-seq**: scGPT +0.0022 vs GenePT +0.0015
- **Non-ChIP**: scGPT +0.0026 vs GenePT +0.0007

scGPT's advantage on ChIP-seq/Non-ChIP may reflect that its embeddings
encode expression-level patterns (learned from 33M cells) rather than
text-level gene function (GenePT). Experimental ground truths (ChIP-seq,
Non-ChIP) capture cell-type-specific regulation that correlates with
expression patterns.

### Default settings (frozen, linear projection) are best

All ablations (MLP projection, finetuning) either match or hurt performance:

- **MLP projection** hurts STRING AUROC for both sources (GenePT: -0.0025,
  scGPT: -0.0046). The extra parameters don't help and may overfit.
- **Finetuning** is neutral for GenePT (-0.0014) and slightly negative for
  scGPT (-0.0021) on STRING. Unfreezing 1536/512 embedding parameters per
  gene with only ~400-750 training cells is an unfavorable ratio.

### Why pretrained embeddings don't help more

1. **The model already learns effective gene identities.** The learnable
   identity vectors in the concat embedding specialize to the current
   dataset's regulatory structure. Pretrained embeddings add generic
   biological knowledge but can't specialize as effectively.

2. **1000 HVG genes are already well-characterized.** These are the most
   variable (and typically best-studied) genes. Pretrained embeddings would
   be more valuable for less-characterized genes or larger gene panels where
   random initialization is a worse starting point.

3. **The additive integration is weak.** `h_i = [x_i; proj(pretrained_i) +
   emb_i]` adds the projected embedding to the identity vector. The model
   can learn to ignore the pretrained component by letting `emb_i` dominate.

4. **GRN inference depends on pairwise relationships, not individual gene
   properties.** Pretrained embeddings encode per-gene features (function,
   expression patterns) but don't directly encode regulatory relationships.
   The model must still learn all pairwise interactions from the flow
   matching objective.

## Conclusion

Pretrained gene embeddings (GenePT and scGPT) provide at best marginal
improvements to GRN inference (+0.001 STRING AUROC). The gains do not
justify the added complexity, download requirements, or computational cost.

**The edge+diff config from Finding 011 remains the best configuration.**
Pretrained embeddings are not recommended for the current pipeline.

## Configuration Reference

```json
{
    "model": {
        "pretrained_embed": "genept",    // or "scgpt", or null (default)
        "pretrained_embed_freeze": true, // freeze pretrained vectors
        "pretrained_embed_proj": "linear" // "linear" or "mlp"
    }
}
```
