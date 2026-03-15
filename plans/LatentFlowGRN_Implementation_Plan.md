# LatentFlowGRN — Implementation Plan v2

## Hardware: Single desktop, RTX 4090 (24GB VRAM)
## Goal: From zero to transfer results on BEELINE as fast as possible

---

## Codebases to Build On

### RegDiffusion (primary skeleton)
- **Repo:** `TuftsBCB/RegDiffusion` — `pip install regdiffusion`
- **Use for:** BEELINE data loading (`rd.data.load_beeline`), evaluation (`GRNEvaluator` — AUPRC, EPR, AUROC with proper edge matching), adjacency matrix parameterization pattern (study their `I_minus_A()`, `get_adj()`, soft thresholding), ground truth handling, TF lists (BEELINE provides these)
- **Study:** Their `GeneEmbeddings` class (d_gene=16: 15 learnable + 1 expression), how A is applied as `einsum('ogd,gh->ohd', h_x, I-A)`, sparsity control (L1/L2 + regulation norm init), convergence detection via adjacency matrix change
- **Don't use:** Their DDPM training loop (replace with CFM), their MLP-A mixing pattern (replace with GAT)

### TorchCFM (flow matching engine)
- **Repo:** `atong01/conditional-flow-matching` — `pip install torchcfm`
- **Use for:** `ConditionalFlowMatcher` class, `ExactOptimalTransportConditionalFlowMatcher` for OT coupling, minibatch OT solver, time sampling
- **Study:** How `sample_location_and_conditional_flow` works — gives x_t and target velocity u_t in one call
- **Don't use:** Their example models (we write our own GAT velocity field)

### FlowGRN-Tong (specific components only)
- **Repo:** `1250326/FlowGRN`
- **Use for:** Dropout-robust d_raw formula (Eq. 4 — simple to reimplement), kNN graph + Dijkstra geodesic distance pattern
- **Study:** Their [SF]²M training loop for reference, how they handle OT coupling with custom distance
- **Don't use:** dynGENIE3 (R/CPU), Slingshot preprocessing, their MLP architecture

### GRNFormer (design ideas only)
- **Repo:** `BioinfoMachineLearning/GRNformer`
- **Study:** TF-Walker subgraph sampling strategy, how they construct co-expression graphs and use edge features in message passing, their evaluation protocol (clean negative pool construction)
- **Don't use:** Their supervised training pipeline, GraViTAE architecture (we have our own GAT)

### PyTorch Geometric (GAT layers)
- **Repo:** `pyg-team/pytorch_geometric` — `pip install torch-geometric`
- **Use for:** Sparse graph operations, `Data`/`Batch` classes for subgraph handling
- **Alternative:** Write custom TF-centric GAT layer (~80 lines) with A bias + co-expression edge features — likely easier than adapting PyG's GATConv which doesn't natively support additive bias + edge features in the way we need

### What NOT to use
- Any R packages (dynGENIE3, Slingshot R version)
- DeepSEM's matrix inversion code
- Any CPU-bound external GRN inference tools
- Heavy preprocessing pipelines that add latency to the dev loop

---

## Phase 1: Foundation (Get to first AUPRC number)

### 1a: Environment + RegDiffusion Baseline (~2 hours)

- Set up conda/venv: torch, regdiffusion, torchcfm, torch-geometric, scanpy, pot
- Run RegDiffusion on BEELINE mESC out of the box
- Record AUPRC/EPR — this is the target to match
- Inspect code: `GeneEmbeddings`, `Block`, `I_minus_A()`, `get_adj()`
- **Also extract:** TF list from BEELINE (BEELINE provides TFs.csv per dataset), compute co-expression matrix C (Spearman correlation on expression matrix)
- **Checkpoint:** Working eval pipeline, baseline number, TF list and C matrix ready

### 1b: Minimal CFM + MLP-A (simplest possible, ~1-2 days)

- Fork RegDiffusion's data loading and evaluation into your own repo
- Write minimal model: same A parameterization as RegDiffusion (linear mixing), but replace DDPM noise prediction with CFM velocity prediction
- Use TorchCFM's `ConditionalFlowMatcher` for training loop
- Training: ‖v_θ(x_t, t) − (x_1 − x_0)‖² with independent coupling first
- Extract A same way as RegDiffusion: `get_adj()`
- Train on mESC, evaluate AUPRC
- **Checkpoint:** First AUPRC from your own model. Pipeline works end-to-end.

### 1c: Add OT Coupling (~half day)

- Switch to `ExactOptimalTransportConditionalFlowMatcher`
- Re-run on mESC, compare: independent vs OT coupling
- **Checkpoint:** OT coupling working. Does it help?

---

## Phase 2: Architecture (TF-Centric GAT)

### 2a: TF-Centric A + Simple Attention (~1 day)

Key structural change: reshape A from g×g to |TFs|×g.

- Load TF list from BEELINE for mESC
- Reshape adjacency: A ∈ ℝ^{|TFs| × g} instead of g×g
- Implement simple TF→target attention (no full GAT yet):
  ```
  For each TF f:
    α_{fj} = softmax_j(h_f^T · W · h_j + λ·A[f,j])
    v_j += α_{fj} · MLP_msg(h_f, t)
  v_j += MLP_self(h_j, t)
  ```
- This is a TF-centric attention without co-expression features yet
- Train on mESC, evaluate AUPRC
- Compare: TF-centric A (|TFs|×g) vs full A (g×g from Phase 1b)
- **Checkpoint:** TF-centric architecture working. Is it better? (It should be at least comparable, with much smaller A.)

### 2b: Add Co-Expression Edge Features (~half day)

- Compute Spearman correlation C_{fj} between each TF f and gene j from expression data (one-time, ~seconds)
- Add C_{fj} as input to attention:
  ```
  e_{fj} = LeakyReLU(a^T · [W_Q h_f ‖ W_K h_j ‖ γ(t) ‖ C_{fj}])
  α_{fj} = softmax(e_{fj} + λ·A[f,j])
  ```
- Re-run on mESC, compare: with vs without co-expression features
- **Checkpoint:** Co-expression features integrated. Do they help convergence?

### 2c: Add Gene Module Regularization (~half day)

- Compute gene modules: threshold co-expression matrix → graph → Louvain clustering
- Implement L_module:
  ```
  L_module = Σ_{module M} Σ_{(i,j) ∈ M} ‖A[:,i] − A[:,j]‖²
  ```
- Add to loss: L = L_CFM + α‖A‖₁ + γ·L_module
- Tune γ ∈ {0, 0.001, 0.01, 0.1}
- **Checkpoint:** Gene module regularization working. Does it help?

### 2d: Extraction Ablation (~half day)

- From the trained model, extract GRN two ways:
  (a) |A[f,j]| readoff (our method)
  (b) |∂v_j/∂h_f| Jacobian (what FlowGRN-Tong tested and rejected)
- Also train plain-MLP model (no A, no GAT) and extract via Jacobian
- Compare AUPRC for all
- **Checkpoint:** Validated that A extraction >> Jacobian. Key paper result.

### 2e: Sparsity + Regularization Tuning (~half day)

- Tune: α (L1 on A) ∈ {0.001, 0.01, 0.1}
- Tune: λ (A bias strength) ∈ {0.1, 1.0, 10.0}
- Optionally add NOTEARS acyclicity constraint
- **Checkpoint:** Tuned single-dataset model on mESC

---

## Phase 3: Full BEELINE Benchmark

### 3a: All 7 Experimental Datasets (~1 day)

- Run LatentFlowGRN-Solo on all 7 experimental datasets
- 10 seeds, TFs + 500 genes AND TFs + 1000 genes
- All ground truth types: STRING, ChIP-seq, LOF/GOF
- **Checkpoint:** Full comparison table vs RegDiffusion, FlowGRN-Tong, GENIE3, GRNBoost2

### 3b: Synthetic + Curated Datasets (~half day)

- 6 synthetic + 4 curated, including dropout variants (50%, 70%)
- **Checkpoint:** Complete BEELINE benchmark

### 3c: Add Dropout-Robust d_knn (~1 day)

- Implement d_raw (only nonzero gene intersections) and d_knn (Dijkstra on kNN graph)
- Replace Euclidean OT cost with d_knn
- Re-run experimental datasets
- **Checkpoint:** Dropout-robust coupling ablation. Full single-dataset results finalized.

---

## Phase 4: Transfer Learning (Primary Contribution)

### 4a: Multi-Task Infrastructure (~1 day)

- Multi-task data loader: interleave batches from multiple datasets
- Gene set alignment: use intersection of gene sets across datasets, or union with masking
- TF set alignment: use union of TFs (some datasets share TFs, some don't)
- Shared/private parameter separation: θ_shared trained by all, A_k by dataset k only
- Verify gradient isolation
- **Checkpoint:** Multi-task training loop running without errors

### 4b: Joint Training (~1 day)

- LatentFlowGRN-Joint: all 7 datasets, shared θ, per-dataset A_k
- Compare per-dataset AUPRC: Joint vs Solo (Phase 3a)
- Try Joint-Cond (+ dataset embedding e_k)
- **Checkpoint:** First transfer result. Does sharing help?

### 4c: Few-Shot / Data Titration (~1 day)

- Leave-one-out: pretrain on 6 → freeze θ_shared → train A_k on held-out
- Data titration: {100%, 50%, 20%, 10%, 5%} cells
- Plot AUPRC vs cells: Transfer vs Solo
- **Checkpoint:** Few-shot crossing point

### 4d: Cross-Species Transfer (~1 day)

- Mouse-human orthologs from Ensembl BioMart
- Train on 5 mouse datasets → transfer to human (hESC, hHEP)
- Warm-start vs cold-start vs Solo
- **Checkpoint:** Cross-species result

---

## Phase 5: Analysis + Paper

### 5a: Shared vs Specific Regulation (~1 day)

- Compare A_k across datasets after joint training
- Conserved edges (high in all A_k) vs context-specific
- Validate conserved edges against known housekeeping regulators
- UMAP of shared latent space

### 5b: Scalability (~half day)

- Runtime/memory vs genes {100, 500, 1000, 2000, 5000}
- TF-centric A is smaller than g×g: quantify savings
- Compare vs RegDiffusion's numbers

### 5c: External Validation (if time)

- ChIP-seq (ENCODE), Perturb-seq

### 5d: Paper Writing (parallel from Phase 3 onward)

---

## Critical Path

```
1a → 1b → 1c → 2a → 2b → 3a → 4a → 4b → 4c
(2hr) (1-2d) (½d) (1d)  (½d) (1d)  (1d) (1d) (1d)
```

Total: ~8-10 working days to transfer results.

Phases 2c, 2d, 2e, 3b, 3c, 4d, 5a-c important but off critical path.

---

## Decision Points

After **Phase 1b**: If AUPRC catastrophically bad, debug. Check: A init, learning rate, L1 penalty.

After **Phase 2a**: If TF-centric A worse than full g×g, check TF list quality and subgraph size k. Option: keep full g×g and mask non-TF rows to zero.

After **Phase 2b**: If co-expression features don't help (or hurt), drop them. They add model complexity — only keep if they improve results.

After **Phase 2d**: If A extraction ≈ Jacobian, the architectural argument weakens but method still works. Adjust framing.

After **Phase 4b**: If negative transfer everywhere, try: (a) same-species only, (b) same-lineage only (3 mHSC datasets), (c) reduce shared capacity.

---

## Files to Create First

```
latentflowgrn/
├── train.py                # Main training script
├── model.py                # TF-centric GAT velocity field (~100 lines)
├── adjacency.py            # A parameterization (TF×g), extraction, sparsity
├── preprocessing.py        # Co-expression C, gene modules, TF-Walker subgraphs
├── data.py                 # Wraps RegDiffusion's load_beeline + TF lists
├── eval.py                 # Wraps RegDiffusion's GRNEvaluator
├── config.py               # Hyperparameters
└── experiments/
    ├── run_single.py       # Phase 3
    ├── run_transfer.py     # Phase 4
    └── run_ablation.py     # Phase 2d
```

Start with model.py < 100 lines. Get it training. Then iterate.

---

## Key Hyperparameters to Track

| Param | Range | Phase introduced |
|-------|-------|-----------------|
| d_gene (per-gene embedding dim) | {1, 8, 16} | 1b |
| OT coupling | {independent, exact OT} | 1c |
| A shape | {g×g, |TFs|×g} | 2a |
| λ (A bias strength) | {0.1, 1.0, 10.0} | 2a |
| Co-expression features | {on, off} | 2b |
| γ (module regularization) | {0, 0.001, 0.01, 0.1} | 2c |
| α (L1 sparsity) | {0.001, 0.01, 0.1} | 2e |
| Subgraph size k (TF-Walker) | {20, 50, 100, all} | 2a |
| Dataset embedding dim | {0, 8, 16} | 4b |
| Learning rate | {1e-3, 1e-4} | 1b |
| Batch size | {32, 64, 128} | 1b |
