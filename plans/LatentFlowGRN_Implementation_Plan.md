# LatentFlowGRN — Implementation Plan

## Hardware: Single desktop, RTX 4090 (24GB VRAM)

## Goal: From zero to transfer results on BEELINE as fast as possible

---

## Codebases to Build On

### RegDiffusion (primary skeleton)

- **Repo:** `TuftsBCB/RegDiffusion` — `pip install regdiffusion`
- **Use for:** BEELINE data loading (`rd.data.load_beeline`), evaluation (`GRNEvaluator` — AUPRC, EPR, AUROC with proper edge matching), adjacency matrix parameterization pattern, ground truth handling, local network visualization
- **Study:** How `RegDiffusionTrainer` parameterizes A in MLP blocks, how `get_adj()` extracts the final adjacency, how sparsity is controlled (L1/L2 + regulation norm initialization), convergence detection
- **Don't use:** Their DDPM training loop (we replace with CFM)

### TorchCFM (flow matching engine)

- **Repo:** `atong01/conditional-flow-matching` — `pip install torchcfm`
- **Use for:** `ConditionalFlowMatcher` class, `ExactOptimalTransportConditionalFlowMatcher` for OT coupling, minibatch OT solver, time sampling
- **Study:** How `sample_location_and_conditional_flow` works — it gives you x_t and the target velocity u_t in one call
- **Don't use:** Their example models (we write our own GAT velocity field)

### FlowGRN-Tong (specific components only)

- **Repo:** `1250326/FlowGRN`
- **Use for:** Dropout-robust d_raw formula (Eq. 4 in their paper — simple to reimplement), kNN graph + Dijkstra geodesic distance pattern
- **Study:** Their [SF]²M training loop for reference, how they handle Slingshot pseudotime and lineage-aware trajectory reconstruction
- **Don't use:** dynGENIE3 integration (R/CPU), Slingshot preprocessing (not needed for noise→data flow matching), their model architecture (plain MLP, no parameterized A)

### PyTorch Geometric (GAT layers)

- **Repo:** `pyg-team/pytorch_geometric` — `pip install torch-geometric`
- **Use for:** `GATConv` or `GATv2Conv` as starting point for the attention layer, sparse graph operations
- **Alternative:** Write a custom GAT layer (~50 lines) if you want full control over the attention bias mechanism — `A_{ij}` as additive bias in softmax is not standard in PyG's GATConv and may be easier to implement from scratch

### What NOT to use

- Any R packages (dynGENIE3, Slingshot R version)
- DeepSEM's matrix inversion code
- Any CPU-bound external GRN inference tools
- Heavy preprocessing pipelines that add latency to the dev loop

---

## Phase 1: Foundation (Get to first AUPRC number)

### 1a: Environment + RegDiffusion Baseline (~2 hours)

- Set up conda/venv with: torch, regdiffusion, torchcfm, torch-geometric, scanpy, pot
- Run RegDiffusion on BEELINE mESC (smallest dataset, 421 cells, 1620 genes) out of the box
- Record AUPRC/EPR numbers — this is the target to match
- Inspect their code: how `RegDiffusionTrainer` parameterizes A, how `get_adj()` extracts it, how `GRNEvaluator` works
- **Checkpoint:** You have a working eval pipeline and a baseline number

### 1b: Minimal CFM + Parameterized A (~1-2 days)

- Fork RegDiffusion's data loading and evaluation code into your own repo
- Write a minimal model: same A parameterization as RegDiffusion (linear mixing in MLP blocks), but replace DDPM noise prediction with CFM velocity prediction
- Use TorchCFM's `ConditionalFlowMatcher` for the training loop and OT coupling
- Training objective: `‖v_θ(x_t, t) − (x_1 − x_0)‖²` with independent coupling first (simpler, get it working)
- Extract A the same way RegDiffusion does
- Train on mESC, evaluate AUPRC
- **Checkpoint:** First AUPRC number from your own model. Compare vs RegDiffusion. Even if worse, the pipeline works.

### 1c: Add OT Coupling (~half day)

- Switch from independent coupling to minibatch OT coupling (TorchCFM provides this)
- Re-run on mESC, compare AUPRC: independent vs OT coupling
- **Checkpoint:** Confirmed whether OT coupling helps (it should reduce training variance at minimum)

---

## Phase 2: Architecture (GAT velocity field)

### 2a: Replace MLP with GAT (~1-2 days)

- Implement GAT velocity field where A is an attention bias:
  ```
  α_{ij} = softmax(e_{ij} + λ·A_{ij})
  v_i = Σ_j α_{ij} · MLP_msg(z_j, t) + MLP_self(z_i, t)
  ```
- Keep everything else identical (same data, same OT coupling, same A extraction, same eval)
- Train on mESC, compare: MLP-A (Phase 1b) vs GAT-A (now)
- **Checkpoint:** GAT architecture working. Is GAT-A better than MLP-A?

### 2b: Extraction Ablation (~half day)

- From the trained GAT model, extract GRN two ways:
  - (a) |A\_{ij}| readoff (our method)
  - (b) |∂v_i/∂x_j| Jacobian (what FlowGRN-Tong tested and rejected)
- Compare AUPRC for both
- Also train a plain-MLP model (no A) and extract via Jacobian — replicate FlowGRN-Tong's finding
- **Checkpoint:** Validated that A extraction >> Jacobian. This is a key result for the paper.

### 2c: Add Sparsity + Regularization (~half day)

- Add L1 penalty on A: tune α ∈ {0.001, 0.01, 0.1}
- Optionally add NOTEARS acyclicity constraint
- Find best regularization on mESC
- **Checkpoint:** Tuned single-dataset model

---

## Phase 3: Full BEELINE Benchmark (Single-Dataset)

### 3a: Run All 7 Experimental Datasets (~1 day)

- Run LatentFlowGRN-Solo on all 7 BEELINE experimental datasets
- 10 seeds each, TFs + 500 genes AND TFs + 1000 genes
- All 4 ground truth types: STRING, non-specific ChIP-seq, cell-type-specific ChIP-seq, LOF/GOF
- Parallelize across seeds on the 4090 (or run sequentially — each should be minutes)
- **Checkpoint:** Full comparison table vs RegDiffusion, GENIE3, GRNBoost2, DeepSEM, DAZZLE

### 3b: Run Synthetic + Curated Datasets (~half day)

- Run on 6 synthetic + 4 curated BEELINE datasets
- Include dropout variants (50%, 70%) for curated datasets
- **Checkpoint:** Complete BEELINE benchmark results

### 3c: Add Dropout-Robust d_knn (~1 day)

- Implement FlowGRN-Tong's d_raw (only nonzero gene intersections) and d_knn (geodesic on kNN graph)
- Replace Euclidean OT cost with d_knn
- Re-run on all experimental datasets
- Compare: Euclidean OT vs dropout-robust OT
- **Checkpoint:** Ablation result for dropout-robust coupling. Full single-dataset results finalized.

---

## Phase 4: Transfer Learning (Primary Contribution)

### 4a: Multi-Task Training Infrastructure (~1 day)

- Implement multi-task data loader: interleave batches from multiple BEELINE datasets
- Handle gene set alignment across datasets (intersection of gene sets, or union with masking)
- Implement shared/private parameter separation: θ_shared updated by all datasets, A_k updated only by dataset k
- Verify gradient isolation: A_k receives no gradient from dataset j≠k

### 4b: Joint Training Experiment (~1 day)

- Train LatentFlowGRN-Joint on all 7 experimental datasets simultaneously
- Compare per-dataset AUPRC: Joint vs Solo (from Phase 3a)
- Also try Joint-Cond (with dataset embedding e_k)
- **Checkpoint:** First transfer result. Does joint training help?

### 4c: Few-Shot / Data Titration (~1 day)

- Leave-one-out: for each dataset k, pretrain on 6 others → freeze θ_shared → train only A_k
- Data titration: subsample target to {100%, 50%, 20%, 10%, 5%} cells
- Compare Transfer vs Solo at each data level
- Plot AUPRC vs number of cells — find the crossing point
- **Checkpoint:** Few-shot result. At what data fraction does transfer match full solo?

### 4d: Cross-Species Transfer (~1 day)

- Get mouse-human orthologs from Ensembl BioMart
- Train on 5 mouse datasets (mDC, mESC, mHSC-E/GM/L) → θ_shared_mouse
- Transfer to human (hESC, hHEP): freeze θ_shared, train A_human
- Warm-start (O·A_mouse·Oᵀ) vs cold-start (A=0) vs Solo
- **Checkpoint:** Cross-species result

---

## Phase 5: Analysis + Paper

### 5a: Shared vs Specific Regulation (~1 day)

- After joint training, compare A_k across datasets
- Identify conserved edges (high in all A_k) vs context-specific edges
- Validate conserved edges against known housekeeping regulators
- Visualize shared latent space (UMAP of encoded cells from all datasets)

### 5b: Scalability Benchmarks (~half day)

- Runtime vs genes: {100, 500, 1000, 2000, 5000} on 4090
- Memory usage at each scale
- Compare against RegDiffusion's reported numbers

### 5c: External Validation (if time permits)

- ChIP-seq validation from ENCODE
- Perturbation prediction on Perturb-seq data
- These strengthen the paper but are not blocking for a first submission

### 5d: Paper Writing (parallel with experiments from Phase 3 onward)

- Draft figures as results come in
- Introduction + related work can start during Phase 2
- Methods section can be written during Phase 1

---

## Critical Path

```
1a → 1b → 1c → 2a → 2b → 3a → 4a → 4b → 4c
         (2-3 days)   (1-2 days) (1 day) (1 day)(1 day)(1 day)
```

Total critical path: ~8-10 working days from environment setup to transfer results.

Phases 2c, 3b, 3c, 4d, 5a-c are important but can run in parallel or after the critical path.

---

## Decision Points

After **Phase 1b**: If AUPRC is catastrophically bad (>50% worse than RegDiffusion), debug before proceeding. Likely causes: A initialization, learning rate, sparsity penalty too strong/weak.

After **Phase 2a**: If GAT-A is significantly worse than MLP-A, the GAT architecture may be overcomplicating things. Option: fall back to MLP-A (closer to RegDiffusion) for the backbone and still pursue transfer (the primary contribution doesn't depend on GAT vs MLP).

After **Phase 2b**: If A extraction ≈ Jacobian extraction, the architectural argument weakens but the method still works. Adjust paper framing.

After **Phase 4b**: If joint training hurts all datasets (negative transfer everywhere), pivot to analyzing _why_ — which is itself a publishable finding. Also try: (a) only sharing across same-species datasets, (b) only sharing across same-lineage (the 3 mHSC datasets), (c) reducing shared capacity.

---

## Files to Create First

```
latentflowgrn/
├── train.py                # Main training script
├── model.py                # Velocity field (start MLP, then GAT)
├── adjacency.py            # A parameterization, extraction, sparsity
├── data.py                 # Wraps RegDiffusion's load_beeline + custom loader
├── eval.py                 # Wraps RegDiffusion's GRNEvaluator
├── config.py               # Hyperparameters (dataclass or yaml)
└── experiments/
    ├── run_single.py       # Phase 3: single-dataset benchmark
    ├── run_transfer.py     # Phase 4: multi-task + few-shot
    └── run_ablation.py     # Phase 2b: extraction ablation
```

Start with <100 lines in model.py. Get it training. Then iterate.
