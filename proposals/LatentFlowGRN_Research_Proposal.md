# LatentFlowGRN: Transferable Gene Regulatory Network Inference via Latent Flow Matching with Shared Regulatory Dynamics

## A Detailed Research Proposal

---

## 1. Architecture Design

### 1.1 Background: Generative Models for GRN Inference

Recent GRN inference methods leverage generative models trained on scRNA-seq data. Two distinct strategies have emerged for connecting the generative model to GRN extraction:

#### 1.1.1 Strategy A: Embed a learnable adjacency A in the generative model (DeepSEM → RegDiffusion)

This family embeds a **parameterized adjacency matrix A** directly inside the generative model. A is jointly optimized with the model, and GRN extraction is a simple readoff: |A\_{ij}| = regulatory edge score.

**DeepSEM / DAZZLE (VAE backbone):**

- Encoder: `Z = X(I − Aᵀ)`, Decoder: `X = Z(I − Aᵀ)⁻¹`
- Limitations: Matrix inversion O(g³), training instability, static reconstruction

**RegDiffusion (DDPM backbone, Zhu & Slonim 2024):**

- Forward: add Gaussian noise; Reverse: neural network with A in 3 MLP blocks predicts noise
- Eliminates matrix inversion → O(g²); scales to 40k+ genes in <5 min
- Outperforms DeepSEM/DAZZLE on most BEELINE datasets; very stable across seeds

**Shared limitations of Strategy A:** (1) No cross-dataset transfer; (2) No OT-structured transport; (3) No velocity field / dynamics interpretation; (4) MLP-only architecture; (5) |A| extraction only — no directionality beyond sign

#### 1.1.2 Strategy B: Use flow matching for trajectory reconstruction, extract GRN post-hoc (FlowGRN)

FlowGRN (Tong & Pang, ACM BCB 2025) takes a fundamentally different approach — the GRN is NOT embedded in the generative model:

1. Define dropout-robust cell similarity d_raw using only nonzero gene intersections; compute geodesic distances on a kNN graph
2. Train [SF]²M (CFM + score matching) to learn velocity field v_θ, using dropout-robust distance as OT cost
3. Reconstruct per-cell trajectories by integrating v_θ forward/backward
4. Feed trajectories to **dynGENIE3** (temporal random forest) for GRN extraction

**Key finding:** Extracting GRN directly from the velocity field Jacobian (∇v_θ) performs significantly worse than dynGENIE3 on reconstructed trajectories. The reason: indirect regulatory effects propagate through the chain rule in an unconstrained MLP, making the Jacobian a poor direct-GRN estimator.

**Limitations:** (1) No cross-dataset transfer; (2) Two-stage pipeline — flow model and GRN extractor are not jointly optimized; (3) dynGENIE3 is CPU-only, R-dependent, and scales poorly (their paper used 128 CPUs for this step); (4) No shared parameters between stages — transfer is structurally impossible; (5) Scalability limited to ~1,783 genes

#### 1.1.3 The Gap

All methods — both strategies plus classical methods (GENIE3, GRNBoost2, PIDC) — train from scratch per dataset. No regulatory dynamics, no representations, no structural knowledge transfers across cell types, tissues, or species.

### 1.2 Proposed Architecture: LatentFlowGRN

LatentFlowGRN combines Strategy A (embedded A for end-to-end GRN learning) with Strategy B's insight (flow matching for dynamics modeling), while adding a new capability: **cross-dataset transfer via shared-private decomposition.**

**Design principles:**

1. **From Strategy A:** Embed a learnable A directly in the generative model → end-to-end, GPU-native, no external tools
2. **From Strategy B:** Use OT-conditional flow matching as backbone, with dropout-robust OT coupling
3. **New:** GAT-structured velocity field where A is an attention bias — enables clean shared-private decomposition
4. **New:** Latent-space operation for scalability and noise robustness

**Why we don't need dynGENIE3:** FlowGRN-Tong needed dynGENIE3 because their velocity field is an unconstrained MLP — its Jacobian captures both direct and indirect regulatory effects indiscriminately. In LatentFlowGRN, A enters the velocity field _only_ as an attention bias that gates which genes' messages reach gene i. A captures direct regulatory gating by construction. This is the same parameterized-A strategy validated by RegDiffusion, which achieves SOTA results with |A| extraction alone — no external GRN inference needed.

#### Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    LatentFlowGRN Pipeline                          │
│                    (end-to-end, GPU-native)                        │
│                                                                    │
│  Input: Gene expression matrix X ∈ ℝ^{c × g}                     │
│                                                                    │
│  ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   Encoder    │───▶│  Latent Space    │───▶│  OT-CFM Flow     │  │
│  │  (per-gene   │    │  z ∈ ℝ^{c × d}  │    │  Matching in     │  │
│  │   MLP, shared│    │                  │    │  Latent Space    │  │
│  │   weights)   │    │                  │    │                  │  │
│  └─────────────┘    └──────────────────┘    └───────┬──────────┘  │
│   [SHARED θ_enc]                                     │             │
│                                                      │             │
│  ┌───────────────────────────────────────────────────┘             │
│  │  Velocity Field v_θ(z_t, t, k) parameterized by:              │
│  │                                                                 │
│  │  ┌───────────────────────────────────────────────┐             │
│  │  │  Graph Attention Network (GAT)                │             │
│  │  │                                               │             │
│  │  │  SHARED: MLP_msg, MLP_self, W_Q, W_K, a      │             │
│  │  │  (regulatory grammar — how signals propagate) │             │
│  │  │                                               │             │
│  │  │  DATASET-SPECIFIC: A_k ∈ ℝ^{g×g}             │             │
│  │  │  (which edges are active in cell type k)      │             │
│  │  │                                               │             │
│  │  │  v_i(z_t, t) = Σ_j α_{ij}^k · f(z_t^j, t)  │             │
│  │  │  where α_{ij}^k = softmax(A_k_{ij})          │             │
│  │  └───────────────────────────────────────────────┘             │
│  │                                                                 │
│  │  GRN: A_k_{ij} → regulatory edge j→i in cell type k           │
│  │  (direct readoff, no external tools)                            │
│  └─────────────────────────────────────────────────────────────────┘
└────────────────────────────────────────────────────────────────────┘
```

#### Component Details

**A) Encoder: Per-Gene MLP with Shared Weights**

```
h_i = MLP_enc(x_i)  for each gene i ∈ {1, ..., g}
z = [h_1, h_2, ..., h_g]  ∈ ℝ^{c × g × d}
```

Shared across all datasets in the transfer setting. Alternatively, a frozen **scGPT/Geneformer** encoder (Variant T4).

**B) GRN-Structured Velocity Field**

```
v_i(z_t, t) = Σ_{j=1}^{g} α_{ij} · MLP_msg(z_t^j, t) + MLP_self(z_t^i, t)
```

Attention with adjacency bias:

```
e_{ij} = LeakyReLU(a^T · [W_Q z_t^i ‖ W_K z_t^j ‖ γ(t)])
α_{ij} = exp(e_{ij} + λ·A_{ij}) / Σ_k exp(e_{ik} + λ·A_{ik})
```

**Why A extraction works (and why Jacobian doesn't):**

FlowGRN-Tong found that Jacobian extraction from an unconstrained MLP fails because the chain rule propagates indirect effects. In LatentFlowGRN:

- **A controls gating:** A\_{ij} determines whether gene j's message is _allowed to reach_ gene i. This is a direct regulatory relationship.
- **MLPs control content:** MLP_msg transforms the message content. Multi-hop effects propagate through stacked GAT layers and MLP nonlinearities.
- **The Jacobian ∂v_i/∂x_j captures both:** gating (A) + content (MLP indirect effects). This is why Jacobian extraction is unreliable.
- **A alone captures direct gating only:** Reading off A avoids indirect effects entirely.

This is analogous to transformer architectures: the attention matrix captures direct token-to-token relevance, while the full Jacobian through residual connections and FFNs captures everything. We read the attention prior, not the Jacobian.

RegDiffusion validates this strategy empirically — it uses parameterized-A readoff and achieves SOTA on BEELINE without any external GRN inference tool.

**C) OT-CFM Training with Dropout-Robust Coupling**

Following FlowGRN-Tong, we replace standard Euclidean OT cost with dropout-robust geodesic distance:

1. d*raw(x,y) = (1/|S*{x,y}|)·Σ*{i∈S*{x,y}} |x*i − y_i| where S*{x,y} = {i: x_i≠0, y_i≠0}
2. Build kNN graph G_knn with d_raw
3. d_knn = shortest path on G_knn (Dijkstra)
4. Minibatch OT with d_knn as cost

**Training loss:**

```
L = L_CFM + α·‖A‖₁ + β·L_DAG
```

**D) GRN Extraction**

Direct readoff from A:

- Edge score(j → i) = |A\_{ij}|
- Edge sign(j → i) = sign(A\_{ij}) (activation/repression)
- Rank all pairs by score → evaluated against ground truth

No external tools. No R dependencies. No CPU bottleneck. Entire pipeline: GPU-native, PyTorch-only.

### 1.3 Latent Flow Matching Variant

For genome-scale operation:

1. Pretrain autoencoder: E: ℝ^g → ℝ^d, D: ℝ^d → ℝ^g
2. Flow matching in latent space: z_0 ~ N(0,I_d), z_1 = E(x)
3. GRN from A (operates on gene-level attention, not latent dimensions)

### 1.4 Primary Innovation: Transfer Learning via Shared-Private Decomposition

**This is the central contribution.** No existing method supports this.

#### Biological Rationale

1. **Conserved grammar** — how TFs bind DNA, how signals propagate — shared across cell types / species
2. **Context-specific wiring** — which TFs active, which edges on — differs per cell type

LatentFlowGRN separates these:

| Component                      | Captures                    | Shared or Private |
| ------------------------------ | --------------------------- | ----------------- |
| Encoder (MLP_enc)              | Expression → latent         | **Shared**        |
| Velocity MLPs                  | How signals propagate       | **Shared**        |
| Attention params (W_Q, W_K, a) | Influence computation       | **Shared**        |
| Adjacency A_k                  | Active edges in cell type k | **Private**       |
| Dataset embedding e_k          | Cell-type context           | **Private**       |

**Why neither RegDiffusion nor FlowGRN-Tong can do this:**

- **RegDiffusion:** MLP blocks with A as linear mixing — weights and A entangled, not separable
- **FlowGRN-Tong:** GRN extraction in dynGENIE3 is entirely separate from the flow model — no shared parameters to transfer

#### Transfer Variants

**T1: Joint Multi-Task Training**

```
L_total = Σ_k L_CFM^{(k)}(θ_shared, A_k) + α·Σ_k ‖A_k‖₁
```

**T2: Pretrain-then-Finetune** — Pretrain θ_shared on K datasets, freeze, train only A_target on new data.

**T3: Cross-Species** — Mouse→human via ortholog mapping O. Warm-start: A_human ← O·A_mouse·Oᵀ.

**T4: Foundation Model Backbone** — Frozen scGPT/Geneformer encoder as shared encoder.

**Dataset conditioning (optional):**

```
v_i(z_t, t, k) = Σ_j α_{ij}^k · MLP_msg([z_t^j; γ(t); e_k]) + MLP_self([z_t^i; γ(t); e_k])
```

### 1.5 Architecture Comparison

| Feature         | DeepSEM/DAZZLE  | RegDiffusion    | FlowGRN (Tong)              | **LatentFlowGRN**                |
| --------------- | --------------- | --------------- | --------------------------- | -------------------------------- |
| Backbone        | VAE             | DDPM            | [SF]²M                      | OT-CFM                           |
| A in model?     | Yes             | Yes             | No                          | **Yes**                          |
| GRN extraction  | \|A\| readoff   | \|A\| readoff   | dynGENIE3 (CPU, R)          | **\|A\| readoff (GPU, PyTorch)** |
| End-to-end?     | Yes             | Yes             | No (two-stage)              | **Yes**                          |
| GPU-native?     | Yes             | Yes             | Partially (dynGENIE3 = CPU) | **Yes**                          |
| Velocity field  | No              | No              | Yes                         | **Yes**                          |
| OT coupling     | No              | No              | Yes (dropout-robust)        | **Yes (dropout-robust)**         |
| Graph structure | A in linear SEM | A in MLP mixing | Plain MLP                   | **A as GAT attention bias**      |
| Transfer        | None            | None            | None                        | **Yes**                          |
| Few-shot GRN    | No              | No              | No                          | **Yes**                          |
| Cross-species   | No              | No              | No                          | **Yes**                          |
| Scalability     | O(g³n)          | O(g²), 40k+     | ~1,783 genes                | O(g·k), TBD                      |
| Single-GPU dev? | Yes             | Yes             | No (128 CPUs for dynGENIE3) | **Yes (RTX 4090)**               |

---

## 2. Mathematical Formulation

### 2.1 Conditional Flow Matching

CNF: dψ*t/dt = v*θ(ψ_t, t). CFM loss:

```
L_CFM = E_{t, z, x} ‖v_θ(x, t) - u_t(x|z)‖²
```

### 2.2 OT-CFM with Dropout-Robust Coupling

Replace Euclidean OT cost with d_knn (geodesic on kNN graph built with d_raw). Conditional paths: x_t = (1−t)x_0 + tx_1, u_t = x_1 − x_0.

### 2.3 GRN-Structured Velocity Field

```
v_i(z_t, t) = Σ_j α_{ij} · MLP_msg(z_t^j, t) + MLP_self(z_t^i, t)
α_{ij} = softmax_j(e_{ij} + λ·A_{ij})
```

Full loss: L = L_CFM + α‖A‖₁ + βR(A)

Regularizer options:

- Acyclicity (NOTEARS): R(A) = tr(e^{A∘A}) − g
- Degree constraint: R(A) = Σ*i max(0, Σ_j|A*{ij}| − k)
- Prior mask: R(A) = ‖A∘M‖\_F² (TF motif data)

### 2.4 Multi-Task Loss for Transfer

```
L_total = Σ_k w_k · L^{(k)}(θ_shared, A_k, e_k) + α·Σ_k ‖A_k‖₁
```

Gradient isolation: ∂L^{(k)}/∂A_j = 0 for j≠k.

Transfer: A_target* = argmin L^{(target)}(θ_shared*, A_target) with θ_shared\* frozen.

Cross-species: O·A_mouse·Oᵀ warm-start.

### 2.5 GRN Extraction

Direct: Edge(j→i) = |A*{ij}|, sign = sign(A*{ij}).

Transfer-enriched: In multi-task setting, A_k reflects dataset-specific wiring learned in the context of shared dynamics from all datasets — potentially more accurate than isolated A.

### 2.6 Biological Interpretation

- v_i(x,t) > 0: gene i upregulated
- α\_{ij} large: gene j gates influence on gene i
- Shared MLPs: universal propagation logic
- A_k comparison: conserved (all A_k) vs context-specific (one A_k) edges

---

## 3. Experimental Setup

### 3.1 BEELINE Benchmark

**Synthetic (6):** dyn-LI, CY, LL, BF, TF, BFC. 100–5,000 cells × 10 samples.
**Curated (4):** HSC, mCAD, VSC, GSD. 2,000 cells × 10 samples + dropout variants.
**Experimental (7):** hESC, hHEP, mDC, mESC, mHSC-E/GM/L.

**Ground truths:** STRING, non-specific ChIP-seq, cell-type-specific ChIP-seq, LOF/GOF. Additionally DoRothEA/CollecTRI (per FlowGRN-Tong's updated references).

### 3.2 Metrics

AUPRC ratio (primary), EPR, AUROC.

### 3.3 Baselines

**Tier 1: Generative + GRN (direct competitors)**

- **RegDiffusion** — DDPM + parameterized A (Strategy A, DDPM)
- **FlowGRN-Tong** — CFM + dynGENIE3 (Strategy B)
- DeepSEM, DAZZLE, HyperG-VAE

**Tier 2: Classical**

- GENIE3, GRNBoost2, PIDC, PPCOR, LEAP

**Tier 3: Recent SOTA**

- GRNFormer, scKAN, scRegNet, GRANGER

**Tier 4: Transfer-aware**

- GRNPT, Meta-TGLink, scMTNI, LINGER

### 3.4 Experimental Protocol

#### Experiment 1: BEELINE Single-Dataset

Standard BEELINE evaluation. LatentFlowGRN-Solo vs all baselines. 10 seeds.

Key comparisons:

- vs RegDiffusion: same extraction strategy (|A|), different backbone (OT-CFM vs DDPM) and architecture (GAT vs MLP)
- vs FlowGRN-Tong: same backbone family (flow matching), different extraction (|A| readoff vs dynGENIE3)

#### Experiment 2: GRN Extraction Ablation (Addresses FlowGRN-Tong's Finding)

From the _same trained LatentFlowGRN model_, compare:

- **(a) A extraction:** |A\_{ij}| readoff from GAT attention bias (our method)
- **(b) Jacobian extraction:** |∂v_i/∂x_j| averaged over cells and time (what FlowGRN-Tong tested and rejected)

Additionally, train a **plain-MLP velocity field** (no A, no GAT — same as FlowGRN-Tong's architecture) and compare:

- **(c) MLP Jacobian:** |∂v_i/∂x_j| from unconstrained MLP (replicating FlowGRN-Tong's setup)

**Expected:** (a) >> (b) ≈ (c). This validates that A extraction avoids the indirect-effect problem while Jacobian extraction fails regardless of architecture.

If (a) >> (b): confirms our architectural argument — attention bias A captures direct regulation, Jacobian captures everything.
If (a) ≈ (b): GAT structure is sufficient to make even Jacobian extraction work — bonus finding.
If (b) >> (a): our architectural argument is wrong — we revise the extraction strategy (but this contradicts RegDiffusion's success with |A|).

#### Experiment 3: Architecture Ablations

- **Backbone:** OT-CFM vs DDPM (RegDiffusion-style) vs VAE (DeepSEM-style) — all with same GAT + A
- **Velocity field:** GAT vs MLP-with-A-mixing (RegDiffusion-style) vs plain MLP (FlowGRN-Tong-style)
- **OT coupling:** Euclidean vs dropout-robust d_knn vs Sinkhorn
- **Latent vs gene space**
- **With/without dropout augmentation (DAZZLE's DA)**

#### Experiment 4: External Biological Validation

- ChIP-seq (ENCODE): TF-target edge validation
- Perturbation prediction (Perturb-seq/CRISPRi)

#### Experiment 5: Multi-Task Joint Training (Primary Contribution)

- LatentFlowGRN-Solo: independent per dataset
- LatentFlowGRN-Joint: all 7 datasets, shared θ, per-dataset A_k
- LatentFlowGRN-Joint-Cond: + dataset embeddings e_k
- RegDiffusion-Solo, FlowGRN-Tong-Solo: transfer-unaware controls

Per-dataset AUPRC. Which datasets benefit? Does joint training hurt any?

#### Experiment 6: Few-Shot GRN Inference

**Leave-One-Out:** Pretrain on 6 → finetune A on held-out.

**Data Titration:** {100%, 50%, 20%, 10%, 5%} cells.

- LatentFlowGRN-Transfer (pretrained θ_shared)
- LatentFlowGRN-Solo
- RegDiffusion-Solo
- FlowGRN-Tong-Solo

#### Experiment 7: Cross-Species Transfer (Mouse → Human)

5 mouse datasets → human (hESC, hHEP). Warm-start vs cold-start vs solo.

#### Experiment 8: Shared vs Specific Regulation Analysis

Conserved edges across A_k, context-specific edges, latent space visualization.

#### Experiment 9: Scalability

Runtime/memory vs genes (100–10,000) and cells (100–20,000).

- vs RegDiffusion: 40k+ genes, <5 min (A100)
- vs FlowGRN-Tong: ~1,783 genes, <10 min (V100) + hours for dynGENIE3 (128 CPUs)
- All LatentFlowGRN experiments on single RTX 4090

### 3.5 Implementation Plan

**Libraries:** PyTorch, TorchCFM, PyTorch Geometric, Scanpy, POT, pybiomart

**No R dependencies. No CPU-bound external tools. Entire pipeline on RTX 4090.**

**Code structure:**

```
latentflowgrn/
├── models/
│   ├── encoder.py              # Per-gene MLP encoder (shared)
│   ├── velocity_field.py       # GAT-based velocity network
│   ├── flow_matching.py        # OT-CFM with dropout-robust coupling
│   ├── grn_extraction.py       # |A| readoff + Jacobian (ablation only)
│   └── transfer.py             # Multi-task trainer, freeze/finetune
├── data/
│   ├── beeline_loader.py       # BEELINE datasets
│   ├── dropout_similarity.py   # d_raw and d_knn computation
│   ├── multitask_loader.py     # Cross-dataset batching
│   └── ortholog_mapper.py      # Cross-species mapping
├── evaluation/
│   ├── metrics.py              # AUPRC, EPR, AUROC
│   └── transfer_analysis.py    # Conserved vs specific edges
├── baselines/
│   ├── run_regdiffusion.py     # RegDiffusion comparison
│   └── run_beeline.sh          # BEELINE Docker baselines
└── experiments/
    ├── exp1_benchmark.py       # Single-dataset BEELINE
    ├── exp2_extraction.py      # A vs Jacobian ablation
    ├── exp3_ablations.py       # Architecture ablations
    ├── exp4_bio_validation.py  # ChIP-seq, perturbation
    ├── exp5_joint_training.py  # Multi-task transfer
    ├── exp6_few_shot.py        # Leave-one-out, data titration
    ├── exp7_cross_species.py   # Mouse → human
    ├── exp8_shared_analysis.py # Conserved vs specific
    └── exp9_scalability.py     # Runtime benchmarks
```

**Development cycle on RTX 4090:**

```
Edit code → Train (minutes) → Extract A → Evaluate AUPRC → Iterate
No CPU bottleneck. No R. No context switching.
```

### 3.6 Expected Outcomes and Risk Mitigation

**What success looks like:**

_Single-dataset (supporting evidence):_

- Competitive with RegDiffusion and FlowGRN-Tong on BEELINE
- A extraction validated vs Jacobian in ablation

_Transfer (primary contribution):_

- Joint training ≥ solo on majority of datasets
- Few-shot: pretrained at 20% data matches solo at 100%
- Cross-species transfer outperforms solo when target data limited
- Biologically meaningful conserved vs context-specific edges

**Risks and mitigations:**

| Risk                                                         | Mitigation                                                                          |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------- | --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LatentFlowGRN-Solo doesn't beat RegDiffusion or FlowGRN-Tong | Acceptable — primary contribution is transfer. Competitive + transfer is the story. |
| A extraction underperforms FlowGRN-Tong's dynGENIE3 results  | (a) RegDiffusion validates                                                          | A   | extraction works; (b) different evaluation setup (they use DoRothEA/CollecTRI, we compare both); (c) if gap is large, add dynGENIE3 as a one-time reviewer-response experiment, not core pipeline. |
| Negative transfer                                            | Report honestly. Analyze which pairs help/hurt.                                     |
| Cross-species fails                                          | Same-species transfer as control.                                                   |
| Scalability worse than RegDiffusion                          | Focus on 1k-5k gene range. Use sparse attention.                                    |
| Reviewer: "why not use dynGENIE3 like FlowGRN-Tong?"         | (a) RegDiffusion proves                                                             | A   | readoff works; (b) dynGENIE3 is CPU-only, R-dependent, breaks end-to-end GPU pipeline; (c) cannot transfer — no shared parameters; (d) ablation Exp 2 directly compares extraction methods.        |
| Name confusion with FlowGRN-Tong                             | "LatentFlowGRN" clearly distinct. Always cite Tong & Pang explicitly.               |

---

## 4. Novelty Claims and Related Work Positioning

### Contribution hierarchy:

**Primary: Transferable GRN inference.** First unsupervised method to share regulatory dynamics across datasets, cell types, and species.

**Secondary: Unified end-to-end flow matching + embedded A architecture.** FlowGRN-Tong showed flow matching works for trajectory reconstruction (Strategy B). RegDiffusion showed embedded A works for GRN extraction (Strategy A). LatentFlowGRN unifies both — OT-CFM with A as GAT attention bias — in a single end-to-end GPU-native model.

**Tertiary: Dropout-robust OT coupling in latent flow matching.** Adopts and extends FlowGRN-Tong's d_raw measure.

### Detailed positioning:

1. **vs FlowGRN-Tong:** They use CFM for trajectories + dynGENIE3 for GRN — two stages, no shared parameters, CPU-bound GRN step, no transfer possible. We embed A in the GAT velocity field — one stage, end-to-end, GPU-native, transfer-capable. We adopt their dropout-robust similarity measure (credited).

2. **vs RegDiffusion:** They embed A in DDPM with MLP mixing. We embed A in OT-CFM with GAT attention. Key differences: (a) GAT enables clean shared-private decomposition for transfer; (b) OT-structured transport; (c) velocity field with biological dynamics interpretation; (d) dropout-robust coupling.

3. **vs DeepSEM/DAZZLE:** Both RegDiffusion and LatentFlowGRN improve on VAE backbone. We add transfer.

4. **vs GRNPT/Meta-TGLink:** Supervised. We are unsupervised.

5. **vs LINGER/scMTNI:** LINGER uses external bulk data. scMTNI restricted to single lineage. We transfer across arbitrary datasets and species.

### Recommended paper framing:

"RegDiffusion demonstrated that embedding a learnable adjacency matrix in a generative model enables fast, accurate GRN inference. FlowGRN showed that flow matching can reconstruct cellular trajectories for improved GRN analysis. We unify these insights — embedding A in a flow matching velocity field via graph attention — and show this architecture uniquely enables cross-dataset transfer of shared regulatory dynamics, a capability no existing GRN inference method provides."

### Target venues:

- Nature Methods, Genome Biology (full scope with biological transfer analysis)
- Bioinformatics, Briefings in Bioinformatics (method focus)
- NeurIPS / ICML (ML: transferable generative models for biological networks)
