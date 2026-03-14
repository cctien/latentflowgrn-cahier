# Phase 1 Implementation Status

Implements the foundation from the [implementation plan](plans/LatentFlowGRN_Implementation_Plan.md): a minimal CFM + parameterized A model on BEELINE, matching RegDiffusion's adjacency strategy with a flow-matching velocity field.

## What's Implemented

### Core modules (`src/latentflowgrn/`)

| Module | Contents |
|--------|----------|
| `adjacency.py` | `AdjacencyMatrix` — learnable (G,G) parameter with soft thresholding, inverted dropout, L1 sparse loss. Ported from RegDiffusion. |
| `model.py` | `MLPVelocityField` — sinusoidal time embedding (t×1000), per-gene learnable embedding, stacked `VelocityBlock`s with skip connections, (I-A) mixing via einsum, final linear→scalar. |
| `data.py` | `load_beeline()` wraps `regdiffusion.data.beeline`, applies cell min-max then gene z-score normalization. `make_dataloader()` with replacement sampling (one batch = one epoch). |
| `eval.py` | `Evaluator` wrapping `regdiffusion.evaluator.GRNEvaluator` (AUROC, AUPR, AUPRR, EP, EPR). |
| `train.py` | Full training loop: CFM or OT-CFM via TorchCFM, dual-lr Adam (A at ~gene_reg_norm/50), delayed L1 sparsity (epoch>10), periodic eval, ClearML logging, results saved to `traces/`. |
| `config.py` | Pydantic schemas: `DataConfig`, `ModelConfig` (+adj_dropout, init_coef), `TrainConfig` (+lr_adj), `TraceConfig`. OmegaConf merging with CLI overrides. |
| `cli.py` | `latentflowgrn` CLI via cyclopts. |

### Experiment scripts (`src/latentflowgrn/experiments/`)

| Script | Purpose |
|--------|---------|
| `run_baseline.py` | Run RegDiffusion out-of-box on BEELINE, log to ClearML, save to `traces/`. |
| `run_single.py` | Run LatentFlowGRN on one dataset with CLI overrides. |
| `run_ablation.py` | OT coupling ablation: runs independent vs OT and compares. |

### Key design decisions

1. **A as standalone module** — `AdjacencyMatrix` is separate from the velocity field. Reusable for GAT (Phase 2) and shared/private transfer (Phase 4).
2. **hidden_dim=16 for Phase 1** — matches RegDiffusion's `[16,16,16]`. The config default of 256 is for Phase 2 GAT.
3. **Dual learning rates** — A learns ~100x slower than the network. Critical for stability.
4. **sigma=0.0** — deterministic OT-CFM interpolation paths.
5. **Time scaling ×1000** — maps continuous t∈[0,1] to frequency range comparable to RegDiffusion's discrete T=5000 schedule.

## What's NOT implemented yet (later phases)

- GAT velocity field (Phase 2)
- Jacobian-based GRN extraction ablation (Phase 2)
- Dropout-robust distance / geodesic OT coupling (Phase 3)
- Full BEELINE benchmark across all 7 datasets × 3 ground truths (Phase 3)
- Transfer learning: multi-task, pretrain-finetune, cross-species (Phase 4)

## How to run

```bash
just debug                    # quick smoke test (500 genes, 5 epochs)
just train --config configs/default.json  # full run (1000 genes, 500 epochs)
just baseline                 # RegDiffusion baseline for comparison
just run-single --dataset mESC --epochs 1000
just ablation --epochs 500    # OT vs independent comparison
```

All results go to `traces/` and are logged to ClearML (project: `latentflowgrn`).
