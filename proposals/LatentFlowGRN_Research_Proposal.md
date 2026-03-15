# LatentFlowGRN: Transferable Gene Regulatory Network Inference via Latent Flow Matching with Shared Regulatory Dynamics

## A Detailed Research Proposal — v6

---

## 1. Architecture Design

### 1.1 Background: Generative Models for GRN Inference

Three strategies connect generative models to GRN extraction:

#### 1.1.1 Strategy A: Parameterized A inside a generative model (DeepSEM → RegDiffusion)

Embed a learnable adjacency matrix A directly in the model. GRN = |A| readoff.

**DeepSEM/DAZZLE (VAE):** A in encoder/decoder via SEM. Matrix inversion O(g³). Unstable training.

**RegDiffusion (DDPM):** A as linear mixing in 3 MLP blocks. No matrix inversion, O(g²). Scales to 40k+ genes in <5 min. Each gene gets a learnable d_gene=16 embedding (15 learnable + 1 expression scalar), concatenated and processed through MLP blocks. A applied as post-hoc matrix multiply: `hz = einsum('ogd,gh->ohd', h_x, I-A)`.

Shared limitations: No transfer, no OT coupling, no velocity field, MLP architecture, |A| extraction only.

#### 1.1.2 Strategy B: Flow matching for trajectories → post-hoc GRN (FlowGRN-Tong)

FlowGRN (Tong & Pang, ACM BCB 2025): [SF]²M for trajectory reconstruction with dropout-robust OT coupling → dynGENIE3 for GRN. Key finding: Jacobian extraction from unconstrained MLP fails (indirect effects). Limitations: dynGENIE3 is CPU/R-bound (128 CPUs), no transfer, two-stage pipeline, ~1,783 gene scale.

#### 1.1.3 Strategy C: Supervised graph transformer (GRNFormer)

GRNFormer (Hegde & Cheng, 2025): A supervised variational graph transformer trained on known TF-target interactions. Three key innovations:
- **TF-Walker:** TF-anchored subgraph sampling — processes local neighborhoods around each TF rather than the full gene graph. Biologically motivated (GRNs are TF→target directed) and computationally efficient.
- **Gene-Transcoder + GraViTAE:** Transformer encoder for gene expression + variational graph transformer with pairwise attention updating both node and edge embeddings.
- **Edge features:** Co-expression weights used alongside node features during message passing.

Achieves 0.90-0.98 AUROC/AUPRC on BEELINE. Limitation: Requires ground-truth GRN labels for supervised training.

#### 1.1.4 Dual-encoder with gene modules (HyperG-VAE)

HyperG-VAE (2025): Two encoders — a cell encoder (SEM for GRN) and a gene encoder (hypergraph self-attention for gene modules). Key insight: gene modules (clusters of co-regulated genes) improve GRN inference — if genes A, B, C always co-express and TF X regulates A, then X likely regulates B and C. The dual encoder jointly optimizes cell heterogeneity and gene module structure.

#### 1.1.5 The Gap

No method supports cross-dataset transfer. GRNFormer generalizes across cell types but requires supervised labels. No unsupervised method transfers regulatory dynamics.

### 1.2 Proposed Architecture: LatentFlowGRN

LatentFlowGRN unifies the best ideas from Strategies A-C plus HyperG-VAE, while adding cross-dataset transfer:

| Source | Idea adopted | How adapted |
|--------|-------------|-------------|
| RegDiffusion (Strategy A) | Parameterized A in generative model, end-to-end | A as GAT attention bias (not MLP mixing) |
| FlowGRN-Tong (Strategy B) | OT-CFM backbone, dropout-robust OT coupling | Adopted directly |
| GRNFormer (Strategy C) | TF-Walker subgraph sampling, co-expression edge features | Adapted for unsupervised setting |
| HyperG-VAE | Gene module regularization | Lightweight L_module penalty |
| **Novel** | Shared-private decomposition for transfer | GAT architecture enables clean separation |

#### Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                      LatentFlowGRN Pipeline                           │
│                                                                        │
│  Input: X ∈ ℝ^{c × g}, TF list, co-expression matrix C              │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Preprocessing (one-time)                                        │  │
│  │  • Compute co-expression C_{ij} = corr(gene_i, gene_j)          │  │
│  │  • Identify TF-centered subgraphs via TF-Walker                 │  │
│  │  • Cluster genes into modules (Louvain on co-expression graph)  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─────────────┐    ┌───────────────┐    ┌────────────────────────┐   │
│  │  Per-gene    │───▶│ Gene features │───▶│  OT-CFM Flow Matching  │   │
│  │  Embedding   │    │ h_i ∈ ℝ^d    │    │  with dropout-robust   │   │
│  │  [SHARED]    │    │              │    │  coupling               │   │
│  └─────────────┘    └───────────────┘    └──────────┬─────────────┘   │
│                                                      │                  │
│  ┌───────────────────────────────────────────────────┘                  │
│  │  GAT Velocity Field on TF-centered subgraphs:                       │
│  │                                                                      │
│  │  For each TF f, sample subgraph S_f (f + top-k co-expressed genes) │
│  │                                                                      │
│  │  SHARED: MLP_msg, MLP_self, W_Q, W_K, a                            │
│  │  PER-DATASET: A_k ∈ ℝ^{|TFs| × g}  (TF→target only)              │
│  │                                                                      │
│  │  Attention with A bias + co-expression edge features:               │
│  │  e_{fj} = LeakyReLU(a^T · [W_Q h_f ‖ W_K h_j ‖ γ(t) ‖ C_{fj}]) │
│  │  α_{fj} = softmax(e_{fj} + λ·A_{fj})                              │
│  │  v_j = Σ_f α_{fj} · MLP_msg(h_f, t) + MLP_self(h_j, t)           │
│  │                                                                      │
│  │  GRN: A_k → TF→target regulatory edges                             │
│  └──────────────────────────────────────────────────────────────────────┘
└──────────────────────────────────────────────────────────────────────────┘
```

#### Component Details

**A) Per-Gene Embedding**

Each gene's scalar expression is concatenated with a learnable gene identity embedding (same pattern as RegDiffusion's d_gene=16):

```
h_i = MLP_enc([x_i; gene_embed_i]) ∈ ℝ^{d_gene}
```

The encoder weights and gene embeddings are **shared across datasets** in the transfer setting.

**B) TF-Centric Subgraph Sampling (adapted from GRNFormer's TF-Walker)**

Instead of a full g×g attention matrix, we restrict to **TF→target edges only**:

- BEELINE provides a list of TFs for each dataset
- For each TF f, define its subgraph S_f = {f} ∪ {top-k genes by co-expression with f}
- A_k ∈ ℝ^{|TFs| × g} — only TF→target entries, not arbitrary gene→gene
- During training, sample subgraphs and run GAT on each

**Why this matters:**
1. **Biologically correct:** GRNs are TF-centric — TFs regulate targets, not the reverse. A_{TF→target} is the biologically meaningful structure.
2. **Scalable:** For 100 TFs × 1000 targets, A has 100k entries instead of 1M. Attention is O(|TFs| × k) per subgraph.
3. **Transfer-friendly:** The shared velocity field learns "how a TF regulates its targets" — a universal pattern that transfers. The per-dataset A_k specifies *which* TF-target edges are active.

**C) GAT Velocity Field with Co-Expression Edge Features**

For each gene j in TF f's subgraph, at flow time t:

```
e_{fj} = LeakyReLU(a^T · [W_Q h_f^t ‖ W_K h_j^t ‖ γ(t) ‖ C_{fj}])
α_{fj} = exp(e_{fj} + λ·A_{fj}) / Σ_m exp(e_{fm} + λ·A_{fm})
v_j(h_t, t) = Σ_{f ∈ TFs(S)} α_{fj} · MLP_msg(h_f^t, t) + MLP_self(h_j^t, t)
```

where:
- A_{fj} is the learnable adjacency bias (TF f → gene j)
- C_{fj} is the pre-computed co-expression between TF f and gene j (Spearman correlation)
- γ(t) is sinusoidal time embedding

**Co-expression as edge features (from GRNFormer):** C_{fj} gives the model a warm signal about which gene pairs co-vary, complementing the learned A. C is computed once from the expression matrix and is **not** a learnable parameter. It serves as an input feature to the attention, not a substitute for A.

**Why A extraction works here (vs FlowGRN-Tong's Jacobian problem):**
- A enters only as attention bias — it controls gating (which messages reach gene j)
- MLPs control message content — indirect effects propagate through MLP nonlinearities
- The Jacobian ∂v_j/∂h_f captures both (gating + content). A captures gating only.
- RegDiffusion validates this |A| readoff strategy empirically.

**D) OT-CFM Training with Dropout-Robust Coupling**

Adopt FlowGRN-Tong's dropout-robust distance as OT cost:
```
d_raw(x,y) = (1/|S_{x,y}|)·Σ_{i∈S_{x,y}} |x_i − y_i|
```
Build kNN graph, compute geodesic d_knn, use as minibatch OT cost.

**E) Full Loss Function**

```
L = L_CFM + α·‖A‖₁ + β·R(A) + γ·L_module
```

where:
- L_CFM: OT-conditional flow matching loss
- α·‖A‖₁: sparsity on adjacency
- R(A): optional acyclicity (NOTEARS) or degree constraint
- **L_module: gene module regularization (inspired by HyperG-VAE)**

**Gene module regularization:**

Pre-compute gene modules M_1, ..., M_K by Louvain clustering on the co-expression graph. Then:

```
L_module = Σ_{module M} Σ_{(i,j) ∈ M, i≠j} ‖A_{:,i} − A_{:,j}‖²
```

This encourages genes in the same co-expression module to have similar incoming regulatory profiles. If TF X regulates gene A, and gene B is in the same module as A, then TF X likely regulates B too. This is the core insight from HyperG-VAE's gene encoder, implemented as a simple regularization penalty instead of a separate encoder.

**F) GRN Extraction**

Direct readoff from A_k:
- Edge score(TF f → gene j) = |A_k[f,j]|
- Edge sign = sign(A_k[f,j])
- Rank all TF-gene pairs by score

No external tools. GPU-native. End-to-end.

### 1.3 Transfer Learning via Shared-Private Decomposition

**Primary contribution.** The architecture cleanly separates:

| Component | Shared or Private |
|-----------|-------------------|
| Gene embeddings, MLP_enc | **Shared** |
| MLP_msg, MLP_self, W_Q, W_K, a | **Shared** |
| Co-expression C (precomputed per dataset) | Per-dataset input (not learned) |
| Adjacency A_k | **Private** |
| Dataset embedding e_k (optional) | **Private** |

**Transfer variants:**

**T1: Joint Multi-Task** — Shared θ trained on all K datasets, per-dataset A_k.
**T2: Pretrain-then-Finetune** — Pretrain θ_shared, freeze, train A_target on new data.
**T3: Cross-Species** — Mouse→human via ortholog mapping. Warm-start A_human.
**T4: Foundation Model Backbone** — Frozen scGPT/Geneformer as shared encoder.

### 1.4 Architecture Comparison

| Feature | RegDiffusion | FlowGRN-Tong | GRNFormer | HyperG-VAE | **LatentFlowGRN** |
|---------|-------------|-------------|-----------|-----------|-------------------|
| Backbone | DDPM | [SF]²M | Graph Transformer | VAE | **OT-CFM** |
| A in model? | Yes (MLP mix) | No | Supervised link pred | Yes (SEM) | **Yes (GAT bias)** |
| TF-centric? | No (g×g) | No | **Yes (TF-Walker)** | No (g×g) | **Yes (adapted)** |
| Edge features | No | No | **Yes (co-expr)** | No | **Yes (co-expr)** |
| Gene modules | No | No | No | **Yes (gene encoder)** | **Yes (L_module)** |
| Transfer | None | None | Supervised only | None | **Unsupervised** |
| End-to-end GPU? | Yes | No (CPU dynGENIE3) | Yes | Yes | **Yes** |

---

## 2. Mathematical Formulation

### 2.1 OT-CFM with Dropout-Robust Coupling

CFM loss: L_CFM = E_{t,z,x} ‖v_θ(x,t) - u_t(x|z)‖². OT coupling with d_knn as cost. Conditional paths: x_t = (1-t)x_0 + tx_1.

### 2.2 TF-Centric GAT Velocity Field

For TF f's subgraph S_f at flow time t:

```
e_{fj} = LeakyReLU(a^T · [W_Q h_f^t ‖ W_K h_j^t ‖ γ(t) ‖ C_{fj}])
α_{fj} = exp(e_{fj} + λ·A_{fj}) / Σ_m exp(e_{fm} + λ·A_{fm})
v_j = Σ_{f ∈ TFs(S)} α_{fj} · MLP_msg(h_f^t, t) + MLP_self(h_j^t, t)
```

A ∈ ℝ^{|TFs| × g} — only TF→target entries.

### 2.3 Full Loss

```
L = L_CFM + α·‖A‖₁ + β·R(A) + γ·L_module
```

Gene module regularization:
```
L_module = Σ_{module M} Σ_{(i,j) ∈ M} ‖A_{:,i} − A_{:,j}‖²
```

### 2.4 Multi-Task Loss for Transfer

```
L_total = Σ_k w_k · L^{(k)}(θ_shared, A_k, e_k)
```

Gradient isolation: ∂L^{(k)}/∂A_j = 0 for j≠k. Transfer: fix θ_shared*, train A_target.

### 2.5 GRN Extraction

Direct: Edge(TF f → gene j) = |A[f,j]|, sign = sign(A[f,j]).

---

## 3. Experimental Setup

### 3.1 BEELINE Benchmark

Synthetic (6), Curated (4), Experimental (7). Ground truths: STRING, ChIP-seq, LOF/GOF, plus DoRothEA/CollecTRI.

### 3.2 Metrics

AUPRC ratio (primary), EPR, AUROC.

### 3.3 Baselines

**Tier 1:** RegDiffusion, FlowGRN-Tong, DeepSEM, DAZZLE, HyperG-VAE
**Tier 2:** GENIE3, GRNBoost2, PIDC, PPCOR, LEAP, dynGENIE3
**Tier 3:** GRNFormer, scKAN, scRegNet, GRANGER
**Tier 4 (transfer):** GRNPT, Meta-TGLink, scMTNI, LINGER

### 3.4 Experiments

**Exp 1: BEELINE single-dataset.** LatentFlowGRN-Solo vs all baselines.

**Exp 2: Extraction ablation.** (a) |A| readoff from GAT, (b) Jacobian from GAT, (c) Jacobian from plain MLP. Validates A extraction vs FlowGRN-Tong's finding.

**Exp 3: Architecture ablations.**
- Backbone: OT-CFM vs DDPM vs VAE (all with same GAT+A)
- Velocity field: GAT-with-A vs MLP-with-A (RegDiffusion-style)
- TF-centric subgraphs vs full g×g attention
- With/without co-expression edge features
- With/without gene module regularization L_module
- OT coupling: Euclidean vs dropout-robust d_knn

**Exp 4: External biological validation.** ChIP-seq, perturbation prediction.

**Exp 5: Multi-task joint training.** Solo vs Joint vs Joint-Cond.

**Exp 6: Few-shot GRN inference.** Leave-one-out + data titration.

**Exp 7: Cross-species transfer.** Mouse → human.

**Exp 8: Shared vs specific regulation analysis.**

**Exp 9: Scalability.** Runtime/memory vs genes and cells. Compare vs RegDiffusion (40k+), FlowGRN-Tong (~1783), GRNFormer.

### 3.5 Implementation Plan

**Libraries:** PyTorch, TorchCFM, PyTorch Geometric (or custom GAT), Scanpy, POT, pybiomart

**No R dependencies. Entire pipeline GPU-native on RTX 4090.**

**Code structure:**
```
latentflowgrn/
├── models/
│   ├── encoder.py              # Per-gene embedding (shared)
│   ├── velocity_field.py       # TF-centric GAT with co-expr edge features
│   ├── flow_matching.py        # OT-CFM with dropout-robust coupling
│   ├── grn_extraction.py       # |A| readoff + Jacobian (ablation)
│   └── transfer.py             # Multi-task trainer, freeze/finetune
├── data/
│   ├── beeline_loader.py       # BEELINE datasets + TF lists
│   ├── coexpression.py         # Compute C_{ij}, gene modules (Louvain)
│   ├── tf_walker.py            # TF-centered subgraph sampling
│   ├── dropout_similarity.py   # d_raw and d_knn
│   ├── multitask_loader.py     # Cross-dataset batching
│   └── ortholog_mapper.py      # Cross-species mapping
├── evaluation/
│   ├── metrics.py              # AUPRC, EPR, AUROC
│   └── transfer_analysis.py    # Conserved vs specific edges
└── experiments/
    ├── exp1_benchmark.py
    ├── exp2_extraction.py
    ├── exp3_ablations.py       # Includes TF-centric, co-expr, module ablations
    ├── exp4_bio_validation.py
    ├── exp5_joint_training.py
    ├── exp6_few_shot.py
    ├── exp7_cross_species.py
    ├── exp8_shared_analysis.py
    └── exp9_scalability.py
```

### 3.6 Expected Outcomes and Risks

**Success:**
- Competitive with RegDiffusion/FlowGRN-Tong on single-dataset BEELINE
- TF-centric subgraphs + co-expression features improve over baseline GAT
- Gene module regularization helps on datasets with clear co-expression structure
- Joint training ≥ solo on majority of datasets
- Few-shot: pretrained at 20% data matches solo at 100%
- Cross-species transfer outperforms solo when target data limited

**Risks:**

| Risk | Mitigation |
|------|-----------|
| Single-dataset doesn't beat RegDiffusion/FlowGRN-Tong | Primary contribution is transfer. Competitive + transfer is the story. |
| TF-centric sampling misses non-TF regulators | Include full g×g as ablation baseline. Most regulatory interactions are TF-mediated. |
| Co-expression features dominate A (A becomes redundant) | Monitor A sparsity during training. Ablate C_{fj} to confirm A carries unique signal. |
| Gene module regularization too strong → all edges identical | Tune γ carefully. Include γ=0 baseline. |
| Negative transfer | Report honestly. Analyze which pairs help/hurt. |

---

## 4. Novelty Claims and Positioning

### Contribution hierarchy:

**Primary: Transferable unsupervised GRN inference.** First method to share regulatory dynamics across datasets, cell types, and species without ground-truth labels.

**Secondary: Unified architecture combining best practices.** OT-CFM backbone (FlowGRN-Tong) + embedded A (RegDiffusion) + TF-centric subgraphs and co-expression edge features (GRNFormer) + gene module regularization (HyperG-VAE) — in a single end-to-end GPU-native model with clean shared-private decomposition.

**Tertiary: TF-centric A in a flow matching model.** By restricting A to TF→target edges and using TF-anchored subgraph sampling, the adjacency is both biologically meaningful and computationally efficient.

### Positioning:

1. **vs RegDiffusion:** Same strategy (A in generative model), but: GAT instead of MLP mixing; OT-CFM instead of DDPM; TF-centric A instead of g×g; co-expression features; gene module regularization; transfer learning.

2. **vs FlowGRN-Tong:** Same backbone family (CFM), but: A embedded in model (not post-hoc dynGENIE3); TF-centric subgraphs; end-to-end GPU-native; transfer capable.

3. **vs GRNFormer:** We adopt TF-Walker and co-expression features but in an unsupervised setting. GRNFormer requires ground-truth labels; we don't. GRNFormer doesn't use flow matching dynamics; we do.

4. **vs HyperG-VAE:** We adopt the gene module insight as a regularization penalty. HyperG-VAE uses a full second encoder (complex); we use a simple L_module term (lightweight). HyperG-VAE is a VAE; we use OT-CFM. No transfer in HyperG-VAE.

### Paper framing:

"We unify recent advances — parameterized adjacency from RegDiffusion, flow matching dynamics from FlowGRN, TF-centric subgraph sampling and co-expression features from GRNFormer, and gene module structure from HyperG-VAE — into a single end-to-end architecture. Critically, the GAT-based velocity field with these components enables a shared-private decomposition for cross-dataset transfer learning, a capability no existing unsupervised GRN method provides."

### Target venues:
- Nature Methods, Genome Biology (full scope with transfer + biology)
- Bioinformatics (method focus)
- NeurIPS / ICML (ML: transferable generative models for biological networks)
