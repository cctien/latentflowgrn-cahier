# Finding 003: GAT Architecture — Attention Bias and Tuning

**Date:** 2026-03-14
**Status:** Complete
**Depends on:** Finding 001 (wd_adj=0 fix)

## Question

Does replacing the MLP's post-hoc (I-A) mixing with GAT attention-biased
A coupling improve GRN inference?

## Investigation

### Round 1: Initial GAT (a_scale=1.0)

GAT with default `a_scale=1.0` achieved AUROC 0.534 — far worse than MLP's
0.616. Diagnostics revealed A received near-zero gradient (A_grad_norm ~0.0000002
by epoch 100). A values (~0.003) were invisible to attention logits (~1-10).

### Round 2: GAT Sweep (12 configurations)

Swept `a_scale_init ∈ {1, 10, 100}` × `hidden_dim ∈ {64, 128}` ×
`steps_per_epoch ∈ {1, 4}`.

**`a_scale_init` is the dominant factor:**

| a_scale | Best AUROC | Best AUPR |
|---------|-----------|-----------|
| 1       | 0.590     | 0.030     |
| 10      | 0.593     | 0.031     |
| 100     | 0.607     | 0.046     |

`hidden_dim=128` and `steps_per_epoch=4` provided incremental gains.

### Round 3: GAT Tuning (9 configurations)

From the best sweep config (a_scale=100, hdim=128, spe=4), tested four
directions:

| Direction | Tag | AUROC | AUPR |
|-----------|-----|-------|------|
| Baseline | base | 0.6067 | 0.0461 |
| More epochs (1000) | epochs1000 | 0.6068 | 0.0474 |
| More data (spe=8) | spe8 | 0.6066 | 0.0470 |
| Lower LR (5e-4) | lr5e4 | 0.6073 | 0.0491 |
| Lower LR + warmup | lr5e4_warmup50 | 0.6072 | 0.0490 |
| Even lower LR + warmup | lr2e4_warmup50 | 0.6071 | 0.0500 |
| Deeper (3 layers, hd64) | 3layers_hd64 | 0.6043 | 0.0419 |
| Combo: spe8 + lr5e4 + warmup | spe8_lr5e4_warmup50 | 0.6078 | 0.0489 |
| Combo: spe8 + epochs1000 | spe8_epochs1000 | 0.6066 | 0.0470 |

**Lower LR is the key improvement.** `lr=2e-4` with warmup achieves AUPR
0.050, nearly matching RegDiffusion baseline (0.052).

## Results

### Final comparison

| Model | AUROC | AUPR | AUPRR | EPR |
|-------|-------|------|-------|-----|
| RegDiffusion baseline | 0.6126 | **0.0522** | **2.453** | **4.220** |
| MLP (Phase 1) | **0.6155** | 0.0440 | 2.069 | 3.882 |
| GAT best (lr2e4) | 0.6071 | 0.0500 | 2.353 | 3.987 |

### MLP vs GAT trade-off

- **MLP wins on AUROC** (0.616 vs 0.607): better overall edge ranking
- **GAT wins on precision** (AUPR 0.050 vs 0.044, EPR 3.99 vs 3.88):
  sharper top-k predictions

The attention mechanism concentrates gradient signal on the most important
regulatory edges, improving precision at the cost of diffuse ranking quality.

## Technical Details

### Why a_scale matters

A values (~0.003) are negligible compared to QK^T attention logits (~1-10).
`a_scale` amplifies A's contribution: `logit = QK^T/√d + a_scale * A`.
At `a_scale=100`, the bias becomes ~0.3, meaningful relative to attention.

### Why lower LR helps GAT but not MLP

Transformers are known to benefit from lower learning rates than MLPs.
The attention mechanism creates sharper loss landscapes that benefit from
smaller steps. The warmup helps but isn't critical.

### Why deeper GAT hurts

3 layers with hdim=64 underperforms 2 layers with hdim=128. The reduced
hidden dimension limits per-gene representation capacity, and additional
attention layers add noise without sufficient data to train them
(only 32 samples per batch).

## Trace References

- GAT sweep: `traces/mESC_1000_STRING_gatsweep_*/`
- GAT tuning: `traces/mESC_1000_STRING_gattune_*/`
- Architecture comparison: `traces/mESC_1000_STRING_arch_*/`
- Summaries: `traces/gat_sweep_summary.json`, `traces/gat_tuning_summary.json`
