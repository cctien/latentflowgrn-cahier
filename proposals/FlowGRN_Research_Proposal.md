# FlowGRN: Transferable Gene Regulatory Network Inference via Latent Flow Matching

## A Detailed Research Proposal

---

## 1. Architecture Design

### 1.1 Background: Generative Models with Parameterized Adjacency for GRN Inference

A family of recent methods share a common architecture: embed a **learnable adjacency matrix A** inside a generative model, train the model on scRNA-seq data, and extract A as the inferred GRN. These methods differ in their choice of generative backbone.

#### 1.1.1 DeepSEM / DAZZLE (VAE backbone)

The **Structural Equation Model (SEM)** framework uses a VAE where A appears in both encoder and decoder:

**Linear SEM assumption:**

```
X = X·Aᵀ + Z
```

where:

- `X ∈ ℝ^{c × g}` is the gene expression matrix (c cells, g genes)
- `A ∈ ℝ^{g × g}` is the **weighted adjacency matrix** (the GRN to infer)
- `Z` is a random noise term

Rearranging into a VAE:

- **Encoder:** `Z = X(I − Aᵀ)` — the GRN layer
- **Decoder:** `X = Z(I − Aᵀ)⁻¹` — the inverse GRN layer

The loss function is:

```
L = -E_{z~q(Z|X)}[log p(X|Z)] + β·D_KL(q(Z|X) || p(Z)) + α·‖A‖₁
```

**Limitations:** (1) Training instability — networks degrade past convergence due to overfitting dropout noise; (2) Matrix inversion `(I − Aᵀ)⁻¹` is numerically unstable and O(g³) in runtime; (3) Static reconstruction only; (4) Directionality lost through |A| averaging over runs.

#### 1.1.2 RegDiffusion (DDPM backbone)

RegDiffusion (Zhu & Slonim, 2024) replaced the VAE with a **denoising diffusion probabilistic model (DDPM)**, achieving strong results:

- **Forward process:** Add Gaussian noise to gene expression data following a diffusion schedule over T steps
- **Reverse process:** A neural network with parameterized adjacency matrix A predicts the added noise
- **Architecture:** Gene expression is embedded, then processed through 3 MLP training blocks. In each block, gene features are mixed with time step embeddings and cell type features. A is integrated as a learnable mixing matrix
- **GRN extraction:** Same as DeepSEM — read off |A\_{ij}| as edge scores

**Key advances over DeepSEM/DAZZLE:**

- Eliminates matrix inversion entirely → runtime drops from O(m³n) to O(m²)
- Superior stability across runs (small std over 10 seeds)
- Outperforms DeepSEM, DAZZLE, GENIE3, GRNBoost2 on most BEELINE datasets
- Scales to 15,000+ genes in under 5 minutes on A100; 40,000+ genes in v0.2
- Can integrate into SCENIC pipeline as a drop-in replacement for GENIE3/GRNBoost2

**Remaining limitations (shared with all prior methods):**

1. **No knowledge sharing:** Every dataset is trained from scratch — no transfer of regulatory logic across cell types, tissues, or species
2. **No OT-structured transport:** DDPM uses standard Gaussian noise schedules; the noise-to-data mapping is not structured by optimal transport
3. **No velocity field / dynamics:** Noise prediction doesn't yield a biologically interpretable velocity field — you can't ask "how fast is gene i changing and why?"
4. **MLP-only architecture:** No explicit graph message-passing structure in the noise prediction network; adjacency A acts as a linear mixing matrix, not as attention weights in a GNN
5. **|A| extraction only:** No Jacobian or integrated-gradient-based GRN extraction; directionality still limited

### 1.2 Proposed Architecture: FlowGRN

We propose **FlowGRN**, which advances beyond both the VAE family (DeepSEM/DAZZLE) and the DDPM family (RegDiffusion) in two key ways:

1. **OT-conditional flow matching** as the generative backbone — providing structured transport, a biologically interpretable velocity field, and training stability advantages over both VAE and DDPM
2. **Shared-private decomposition** for cross-dataset transfer learning — the primary novel contribution, enabled by the latent-space flow matching formulation

#### Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                       FlowGRN Pipeline                            │
│                                                                    │
│  Input: Gene expression matrix X ∈ ℝ^{c × g}                     │
│                                                                    │
│  ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   Encoder    │───▶│  Latent Space    │───▶│  Flow Matching   │  │
│  │  (per-gene   │    │  z ∈ ℝ^{c × d}  │    │  in Latent Space │  │
│  │   MLP, shared│    │                  │    │                  │  │
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
│  │  │  OPTIONAL: dataset embedding e_k              │             │
│  │  │  (cell-type context conditioning)             │             │
│  │  │                                               │             │
│  │  │  v_i(z_t, t) = Σ_j α_{ij}^k · f(z_t^j, t)  │             │
│  │  │  where α_{ij}^k = softmax(A_k_{ij})          │             │
│  │  └───────────────────────────────────────────────┘             │
│  │                                                                 │
│  │  GRN extraction: A_k_{ij} → regulatory edge i←j in cell type k │
│  └─────────────────────────────────────────────────────────────────┘
└────────────────────────────────────────────────────────────────────┘
```

#### Component Details

**A) Encoder: Per-Gene MLP with Shared Weights**

Following DeepSEM's design principle, use a small MLP that processes one gene at a time with shared weights:

```
h_i = MLP_enc(x_i)  for each gene i ∈ {1, ..., g}
z = [h_1, h_2, ..., h_g]  ∈ ℝ^{c × g × d}  (or ℝ^{c × g} if d=1)
```

The shared-weight design ensures that gene-gene interactions are NOT embedded in the encoder — they are exclusively captured by the adjacency structure in the velocity field.

Alternatively, you could use a **pretrained encoder** from scGPT or Geneformer, extracting gene embeddings that already capture biological context. This would be the "latent" in latent flow matching.

The encoder weights θ_enc are **shared across all datasets** in the multi-task / transfer setting. This forces the encoder to learn a universal gene expression → latent mapping, rather than one tuned to a single cell type.

**B) GRN-Structured Velocity Field (Core Architectural Difference vs RegDiffusion)**

Where RegDiffusion uses 3 MLP blocks with A as a linear mixing matrix, FlowGRN uses a **Graph Attention Network** where A modulates attention weights:

For each gene i, at flow time t:

```
v_i(z_t, t) = Σ_{j=1}^{g} α_{ij} · MLP_msg(z_t^j, t) + MLP_self(z_t^i, t)
```

where:

- `α_{ij} = softmax_j(LeakyReLU(a^T · [W·z_t^i || W·z_t^j] + b_A · A_{ij}))`
- `A ∈ ℝ^{g × g}` is the **learnable adjacency matrix** (the GRN)
- `b_A` is a learnable bias that controls how much the adjacency prior influences attention
- `MLP_msg` transforms incoming "messages" from regulator genes
- `MLP_self` captures gene-autonomous dynamics

**Why GAT over MLP:** In RegDiffusion, A linearly mixes gene features — each gene's representation is a weighted sum of all genes' features via A. This is a single-hop linear interaction. The GAT formulation allows: (a) nonlinear, state-dependent attention that varies with the current expression state, (b) multi-hop message passing via stacked GAT layers, and (c) natural decomposition into shared (MLP weights) and private (A) components for transfer learning.

**C) Flow Matching Training (No Simulation Required)**

Instead of the VAE's reconstruction + KL loss (DeepSEM) or DDPM's noise prediction loss (RegDiffusion), we train with the **conditional flow matching (CFM) objective**:

```
L_CFM = E_{t, x_0, x_1} ‖v_θ(x_t, t) - u_t(x_t | x_0, x_1)‖²
```

where:

- `x_0 ~ p_noise` (e.g., standard Gaussian or gene expression from a reference state)
- `x_1 ~ p_data` (actual scRNA-seq expression vectors)
- `x_t = (1-t)·x_0 + t·x_1` (linear interpolation, OT conditional path)
- `u_t(x_t | x_0, x_1) = x_1 - x_0` (the target velocity for the linear path)

**Why CFM over DDPM (RegDiffusion):** (a) Single-step velocity regression vs T-step noise prediction — simpler training; (b) OT minibatch coupling produces straighter, non-crossing flows with lower training variance; (c) The learned velocity field v_θ is directly interpretable as "how fast and in what direction is each gene changing?"; (d) Deterministic ODE integration at inference vs stochastic reverse diffusion — faster sampling.

**D) Full Loss Function**

```
L = L_CFM + α·‖A‖₁ + β·L_DAG
```

where:

- `L_CFM`: conditional flow matching loss
- `α·‖A‖₁`: L1 sparsity penalty on adjacency matrix
- `L_DAG = tr(e^{A ∘ A}) - g`: acyclicity constraint (from NOTEARS), optional, penalizes cyclic structures

**E) GRN Extraction**

After training:

1. Take the learned adjacency matrix A
2. |A\_{ij}| represents the regulatory strength of gene j → gene i
3. sign(A\_{ij}) indicates activation (+) or repression (−)
4. Rank all gene pairs by |A\_{ij}| to produce a ranked edge list for evaluation

Additionally, FlowGRN enables **velocity-field-based GRN extraction** not possible with RegDiffusion or DeepSEM:

- **Jacobian analysis:** J\_{ij}(x\*, t) = ∂v_i/∂x_j captures both direct A and indirect effects
- **Integrated Gradients:** IG\_{ij} = ∫_0^1 (∂v_i/∂x_j)·(x_1^j - x_0^j) dt for attribution over the full flow

### 1.3 Variant: Latent Flow Matching

For better noise handling, operate the flow in a compressed latent space:

1. **Pretrain an autoencoder** on scRNA-seq data:
   - Encoder: `E: ℝ^g → ℝ^d` (d << g, e.g., d=64-256)
   - Decoder: `D: ℝ^d → ℝ^g`

2. **Run flow matching in latent space:**
   - `z_0 ~ N(0, I_d)`
   - `z_1 = E(x)` for each cell x
   - Train velocity field on z_t trajectories

3. **Extract GRN from velocity field Jacobian:**
   - Compute ∂v_i/∂z_j at representative points
   - Map back to gene space via the decoder Jacobian: ∂x/∂z

This is more scalable for genome-wide inference but adds complexity to GRN extraction.

### 1.4 Primary Innovation: Transfer Learning via Shared-Private Decomposition

**This is the central contribution that no prior GRN method supports** — not DeepSEM, not DAZZLE, not RegDiffusion, not GENIE3, not GRNBoost2. All current methods train from scratch on each dataset independently.

#### The Biological Rationale

Gene regulation has two distinct layers:

1. **Conserved regulatory grammar** — the _mechanism_ by which TFs bind DNA, how signaling cascades propagate, how chromatin remodeling works. These are broadly shared across cell types within an organism and partially conserved across species.

2. **Context-specific regulatory wiring** — _which_ TFs are active, _which_ enhancers are accessible, _which_ feedback loops are engaged. These differ between hESC and mDC and hHEP.

FlowGRN's architecture naturally separates these:

| Component                          | What it captures                                 | Shared or private         |
| ---------------------------------- | ------------------------------------------------ | ------------------------- |
| Encoder (MLP_enc)                  | How to embed gene expression into latent space   | **Shared**                |
| Velocity MLPs (MLP_msg, MLP_self)  | How regulatory signals propagate — the "grammar" | **Shared**                |
| Attention parameters (W_Q, W_K, a) | How to compute regulatory influence strength     | **Shared**                |
| Adjacency matrix A_k               | Which edges are active in cell type k            | **Private (per-dataset)** |
| Dataset embedding e_k (optional)   | Cell-type-specific context                       | **Private (per-dataset)** |

**Why RegDiffusion cannot do this:** RegDiffusion's MLP blocks use A as a linear mixing matrix interleaved with MLP layers. The MLP weights and A are not cleanly separable — the MLPs learn dataset-specific patterns entangled with the adjacency structure. In FlowGRN's GAT formulation, A only enters through the attention bias, making the separation between "how signals propagate" (shared MLPs) and "which edges exist" (private A_k) architecturally explicit.

#### Transfer Variants

**Variant T1: Joint Multi-Task Training (simplest)**

Train one FlowGRN on all K datasets simultaneously with shared weights:

```
L_total = Σ_{k=1}^{K} L_CFM^{(k)}(θ_shared, A_k) + α·Σ_k ‖A_k‖₁
```

- θ_shared = {θ_enc, θ_MLP_msg, θ_MLP_self, W_Q, W_K, a} — trained on all datasets
- A_k — separate adjacency matrix per dataset, only trained on data from dataset k
- Batches alternate or mix cells from different datasets

**Variant T2: Pretrain-then-Finetune**

Two-stage training:

Stage 1 (Pretrain): Train FlowGRN jointly on a large collection of datasets → get θ_shared\*.

Stage 2 (Finetune): For a new target dataset, freeze θ_shared\* and only train a new A_target from scratch. This is **few-shot GRN inference** — you need far less data because the regulatory dynamics are already learned.

**Variant T3: Cross-Species Transfer**

For organisms with well-established ortholog mappings (e.g., mouse ↔ human):

1. Train on source species datasets (e.g., all 5 mouse BEELINE datasets)
2. Map gene identities via ortholog table (~16,000 one-to-one orthologs for mouse-human)
3. Transfer: freeze θ_shared, train A_target on human datasets
4. Warm-start option: A_human ← O · A_mouse · Oᵀ (project mouse GRN via ortholog matrix O)

**Variant T4: Foundation Model Backbone**

Use a frozen pretrained encoder from scGPT (33M cells) or Geneformer (30M cells):

1. The foundation model already provides a universal latent space
2. Only train the velocity field (shared MLPs + per-dataset A_k)
3. Most parameter-efficient variant

#### Optional: Dataset Conditioning

Condition the velocity field on a dataset identity:

```
v_i(z_t, t, k) = Σ_j α_{ij}^k · MLP_msg(z_t^j, t, e_k) + MLP_self(z_t^i, t, e_k)
```

where e_k ∈ ℝ^{d_e} is a learnable embedding for dataset k. This lets the shared MLPs modulate their behavior slightly per dataset, beyond just the adjacency difference.

### 1.5 Architecture Comparison

| Feature                 | DeepSEM/DAZZLE        | **RegDiffusion** | GRNFormer                   | **FlowGRN (ours)**                   |
| ----------------------- | --------------------- | ---------------- | --------------------------- | ------------------------------------ |
| Generative backbone     | VAE                   | DDPM             | N/A (discriminative)        | OT-CFM                               |
| GRN parameterization    | A in enc/dec          | A as MLP mixing  | Graph transformer attention | A as GAT attention bias              |
| Training objective      | Reconstruction + KL   | Noise prediction | Supervised (needs labels)   | Velocity regression (unsupervised)   |
| Matrix inversion        | Yes, O(g³)            | **No**           | No                          | **No**                               |
| Dynamics / velocity     | No                    | No               | No                          | **Yes**                              |
| OT-structured transport | No                    | No               | No                          | **Yes**                              |
| Directionality          | Weak (averaged \|A\|) | Weak (\|A\|)     | Via TF subgraphs            | **Strong** (directed velocity field) |
| GRN extraction methods  | \|A\| only            | \|A\| only       | Attention weights           | **\|A\| + Jacobian + IG**            |
| Cross-dataset transfer  | None                  | None             | None                        | **Yes** (shared-private)             |
| Few-shot GRN inference  | Not possible          | Not possible     | Requires labels             | **Yes** (freeze shared, train A)     |
| Cross-species transfer  | Not possible          | Not possible     | Not possible                | **Yes** (ortholog mapping)           |
| Scalability             | O(g³n)                | **O(g²)**        | O(g·k)                      | O(g·k) sparse attention              |
| Runtime (15k genes)     | >4 hours              | **<5 min**       | N/A                         | Est. 5-15 min                        |

---

## 2. Mathematical Formulation

### 2.1 Conditional Flow Matching: Foundations

**Continuous Normalizing Flow (CNF):** A CNF defines a time-dependent velocity field v_θ: ℝ^d × [0,1] → ℝ^d that generates a flow ψ_t via the ODE:

```
dψ_t(x)/dt = v_θ(ψ_t(x), t),    ψ_0(x) = x
```

The flow transports a source distribution p_0 to a target distribution p_1:

```
p_t = (ψ_t)_# p_0
```

**The Flow Matching Objective (Lipman et al., 2022):**

The FM loss regresses v_θ against a target velocity field u_t:

```
L_FM = E_{t~U[0,1], x~p_t} ‖v_θ(x, t) - u_t(x)‖²
```

This is intractable since p_t and u_t are unknown. The key insight of FM is to condition on endpoints.

**Conditional Flow Matching (CFM):**

Define conditional probability paths p_t(x | z) with known conditional velocity fields u_t(x | z) such that:

```
p_t(x) = ∫ p_t(x | z) q(z) dz
```

Then the CFM loss:

```
L_CFM = E_{t~U[0,1], z~q(z), x~p_t(·|z)} ‖v_θ(x, t) - u_t(x | z)‖²
```

has the same gradients as L_FM with respect to θ.

### 2.2 OT-CFM with Minibatch Couplings (Tong et al., 2024)

**Standard CFM** uses independent coupling: z = (x_0, x_1) where x_0 ~ p_0 and x_1 ~ p_1 are sampled independently. This leads to crossing paths and higher training variance.

**OT-CFM** instead uses an optimal transport coupling:

```
π* = argmin_{π ∈ Π(p_0, p_1)} ∫ ‖x_0 - x_1‖² dπ(x_0, x_1)
```

In practice, **minibatch OT** approximates it:

1. Sample a minibatch of B noise samples {x_0^i} and B data samples {x_1^j}
2. Solve the assignment problem within the batch:
   ```
   σ* = argmin_{σ ∈ S_B} Σ_{i=1}^B ‖x_0^i - x_1^{σ(i)}‖²
   ```
3. Form paired samples (x_0^i, x_1^{σ\*(i)})

**The conditional paths for OT-CFM:**

```
x_t = (1 - t)·x_0 + t·x_1         (linear interpolation)
u_t(x | x_0, x_1) = x_1 - x_0     (constant velocity along straight line)
```

**Why this matters for GRN inference:** OT coupling produces straighter, non-crossing flows. Since gene expression distributions have complex multimodal structure (different cell types), OT coupling ensures the velocity field captures biologically meaningful transitions rather than arbitrary noise-to-data mappings. RegDiffusion's DDPM uses isotropic Gaussian noise schedules, which don't have this structure.

### 2.3 Applying CFM to GRN Inference

**Setup for single-cell data:**

- Source distribution p_0: Standard Gaussian N(0, I_g) (or empirical distribution of a reference cell state)
- Target distribution p_1: Empirical distribution of scRNA-seq expression vectors
- Each sample x ∈ ℝ^g represents a cell's expression profile across g genes

**GRN-structured velocity field:**

```
v_θ,i(x_t, t) = Σ_{j ∈ N(i)} α_{ij}(x_t, t) · φ(x_t^j, t) + ψ(x_t^i, t)
```

**Attention mechanism with adjacency bias:**

```
e_{ij} = LeakyReLU(a^T · [W_Q x_t^i ‖ W_K x_t^j ‖ γ(t)])
α_{ij} = exp(e_{ij} + λ·A_{ij}) / Σ_k exp(e_{ik} + λ·A_{ik})
```

**Full training loss (single-dataset):**

```
L = E_{t~U[0,1], (x_0,x_1)~π_OT} ‖v_θ(x_t, t) - (x_1 - x_0)‖² + α·‖A‖₁ + β·R(A)
```

where R(A) is an optional structural regularizer:

- **Acyclicity (NOTEARS):** R(A) = tr(e^{A∘A}) - g
- **Degree constraint:** R(A) = Σ*i max(0, Σ_j |A*{ij}| - k)
- **Prior incorporation:** R(A) = ‖A ∘ M‖\_F² where M is a mask from TF motif data

### 2.4 Multi-Task Flow Matching for Transfer Learning

#### Formulation

Given K datasets {D_1, ..., D_K}, decompose parameters into:

- θ_shared = {θ_enc, θ_MLP_msg, θ_MLP_self, W_Q, W_K, a, b_A, λ} — shared across datasets
- θ_private = {A_1, ..., A_K, e_1, ..., e_K} — per-dataset adjacency matrices and embeddings

**Multi-task loss:**

```
L_total = Σ_{k=1}^{K} w_k · L^{(k)}(θ_shared, A_k, e_k) + α·Σ_k ‖A_k‖₁ + β·Σ_k R(A_k)
```

where:

```
L^{(k)} = E_{t, (x_0, x_1^{(k)})~π_OT^{(k)}} ‖v_{θ_shared}(x_t, t; A_k, e_k) - (x_1^{(k)} - x_0)‖²
```

**Gradient flow property:** Gradients from L^{(k)} flow into θ_shared (shared weights learn from all datasets) but are blocked from A_j for j ≠ k (each adjacency learns only from its own data).

#### Transfer to New Datasets

```
A_target* = argmin_{A_target} L^{(target)}(θ_shared*, A_target, e_target) + α·‖A_target‖₁
```

where θ_shared\* is frozen from pretraining.

#### Cross-Species Formulation

Introduce ortholog mapping matrix O ∈ {0,1}^{g_h × g_m}. Warm-start: A_human ← O · A_mouse · Oᵀ. Finetune A_human on human data with θ_shared frozen.

### 2.5 GRN Extraction and Scoring

**Method 1: Direct adjacency** — Edge score(j → i) = |A*{ij}|, sign = sign(A*{ij})

**Method 2: Jacobian analysis** — J\_{ij}(x*, t) = ∂v_i(x*, t)/∂x_j, averaged over t and representative cells. Captures both direct A and indirect effects from shared MLPs.

**Method 3: Integrated Gradients** — IG\_{ij} = ∫_0^1 (∂v_i(x_t, t)/∂x_j)·(x_1^j - x_0^j) dt

**Transfer-specific note:** In the multi-task setting, Methods 2 and 3 capture effects from both A_k (dataset-specific) and θ_shared (universal dynamics). The Jacobian-based GRN may be more informative than raw A_k, since it incorporates universal regulatory logic learned from all datasets.

### 2.6 Biological Interpretation of the Flow

- **v_i(x, t) > 0:** Gene i is being upregulated at state x, flow time t
- **α\_{ij} large:** Gene j strongly influences the rate of change of gene i → regulatory edge
- **Flow trajectories:** Integrating v_θ from noise to data traces cell state transitions
- **Cell-type-specific GRNs:** Evaluate J\_{ij}(x\*, t) at different cluster centroids
- **Shared vs specific regulation:** Compare A_k across datasets — edges in all A_k represent conserved regulation; edges unique to one A_k represent context-specific wiring

---

## 3. Experimental Setup

### 3.1 BEELINE Benchmark: Datasets and Ground Truths

**BEELINE consists of three dataset categories:**

#### A) Synthetic Networks (6 datasets)

Generated from toy networks using BoolODE: dyn-LI (linear), dyn-CY (cyclic), dyn-LL (long linear), dyn-BF (bifurcating), dyn-TF (trifurcating), dyn-BFC (bifurcating converging). Each has sub-datasets with 100–5,000 cells × 10 samples.

#### B) Curated Boolean Models (4 datasets)

HSC (hematopoietic), mCAD (cortical), VSC (spinal cord), GSD (gonadal). Each 2,000 cells × 10 samples, plus 50% and 70% dropout variants.

#### C) Experimental scRNA-seq (7 datasets)

hESC, hHEP, mDC, mESC, mHSC-E, mHSC-GM, mHSC-L.

**Ground truths:** Non-specific ChIP-seq, STRING, Cell-type-specific ChIP-seq, LOF/GOF.

Gene sets: All significantly varying TFs + top 500 or 1000 most varying genes.

### 3.2 Evaluation Metrics

1. **AUPRC ratio** = AUPRC / baseline (primary, handles class imbalance)
2. **EPR** (Early Precision Ratio) — precision among top K predicted edges
3. **AUROC** (secondary)

### 3.3 Baselines to Compare Against

**Tier 1: Direct competitors (generative model with parameterized A, unsupervised)**

- **RegDiffusion** (J. Comp. Biol., 2024) — DDPM backbone, current best unsupervised DL on BEELINE
- DeepSEM (Nature Computational Science, 2021) — VAE backbone
- DAZZLE (PLOS Computational Biology, 2025) — VAE + dropout augmentation
- GRN-VAE (simplified DeepSEM)
- HyperG-VAE (hypergraph VAE with SEM cell encoder)

**Tier 2: Classical methods (still competitive)**

- GENIE3, GRNBoost2, PIDC, PPCOR

**Tier 3: Recent SOTA**

- GRNFormer, scKAN, scRegNet, GRANGER

**Tier 4: Transfer-aware methods (for transfer experiments)**

- GRNPT (LLM embeddings + Transformer, cross-cell-type)
- Meta-TGLink (graph meta-learning, few-shot, supervised)
- scMTNI (multi-task across lineages)
- LINGER (lifelong learning with atlas-scale bulk data)

### 3.4 Experimental Protocol

#### Experiment 1: BEELINE Benchmark — Single-Dataset

Standard BEELINE evaluation. FlowGRN-Solo vs all Tier 1-3 baselines. 10 random seeds, report mean ± std.

**Key comparison:** FlowGRN-Solo vs RegDiffusion — isolates the effect of flow matching (OT-CFM) vs diffusion (DDPM) with matched adjacency-matrix extraction.

#### Experiment 2: Ablation Studies

- **Generative backbone:** OT-CFM vs DDPM (RegDiffusion-style) vs VAE (DeepSEM-style) — all with same GAT velocity/noise network, isolating only the training objective
- **OT coupling:** None (independent) vs minibatch OT vs Sinkhorn-regularized OT
- **Velocity field architecture:** GAT vs MLP (RegDiffusion-style) vs Transformer
- **GRN extraction method:** Direct A vs Jacobian vs Integrated Gradients
- **Latent vs gene space:** Flow matching in original gene space vs latent space
- **Dropout robustness:** Performance on curated datasets with 0%, 50%, 70% dropout
- **With/without dropout augmentation:** Can DA (from DAZZLE) further improve FlowGRN?

#### Experiment 3: External Biological Validation

**A) ChIP-seq validation:** Predict TF-target edges, validate against ENCODE ChIP-seq.
**B) Perturbation prediction:** Use GRN to predict TF knockout effects, compare against Perturb-seq/CRISPRi data.

#### Experiment 4: Scalability Analysis

Runtime and memory vs genes (100–10,000) and cells (100–20,000). Compare directly against RegDiffusion's reported timings.

#### Experiment 5: Multi-Task Joint Training (Transfer — Primary Contribution)

**Setup:**

- FlowGRN-Solo: Independent training per dataset (baseline, same as Experiment 1)
- FlowGRN-Joint: All 7 datasets, shared θ, per-dataset A_k
- FlowGRN-Joint-Cond: Same + dataset conditioning embeddings e_k
- RegDiffusion-Solo: Independent RegDiffusion per dataset (transfer-unaware control)

**Evaluation:** Per-dataset AUPRC, EPR, AUROC. Report which datasets benefit from sharing.

#### Experiment 6: Few-Shot GRN Inference (Transfer — Data Efficiency)

**Leave-One-Out:** For each dataset k: pretrain on 6 others → freeze θ_shared → finetune only A_k.

**Data Titration:** Subsample to {100%, 50%, 20%, 10%, 5%} of cells. Compare: (a) FlowGRN-Solo from scratch, (b) FlowGRN-Transfer with pretrained θ_shared, (c) RegDiffusion from scratch.

**Expected result:** Transfer should help most in low-data regime. Crossing point (where transfer matches full-data solo) is the key result.

#### Experiment 7: Cross-Species Transfer (Mouse → Human)

Train on 5 mouse datasets → transfer to human (hESC, hHEP) via ortholog mapping. Compare warm-start (projected A_mouse) vs cold-start (A=0) vs FlowGRN-Solo on human data vs RegDiffusion on human data.

#### Experiment 8: Analysis of Shared vs Specific Regulation

After joint training: (A) identify conserved edges (high |A\_{ij}| across all A_k), (B) identify context-specific edges, (C) visualize shared latent space, (D) inspect what shared MLPs learn.

### 3.5 Implementation Plan

**Libraries:** PyTorch, TorchCFM, PyTorch Geometric, Scanpy, POT, pybiomart

**Code structure:**

```
flowgrn/
├── models/
│   ├── encoder.py              # Per-gene MLP encoder (shared weights)
│   ├── velocity_field.py       # GAT-based velocity network
│   ├── flow_matching.py        # OT-CFM training loop
│   ├── grn_extraction.py       # |A| / Jacobian / IG methods
│   └── transfer.py             # Multi-task trainer, freeze/finetune
├── data/
│   ├── beeline_loader.py       # BEELINE dataset interface
│   ├── multitask_loader.py     # Interleaved batching across datasets
│   └── ortholog_mapper.py      # Cross-species gene mapping
├── evaluation/
│   ├── metrics.py              # AUPRC, EPR, AUROC
│   └── transfer_analysis.py    # Conserved vs specific edge analysis
├── baselines/
│   ├── run_baselines.sh        # BEELINE Docker baselines
│   └── run_regdiffusion.py     # RegDiffusion head-to-head comparison
└── experiments/
    ├── exp1_benchmark.py       # Single-dataset BEELINE
    ├── exp2_ablations.py       # Architecture ablations (incl. CFM vs DDPM)
    ├── exp3_bio_validation.py  # ChIP-seq, perturbation validation
    ├── exp4_scalability.py     # Runtime / memory (incl. vs RegDiffusion)
    ├── exp5_joint_training.py  # Multi-task within BEELINE
    ├── exp6_few_shot.py        # Leave-one-out, data titration
    ├── exp7_cross_species.py   # Mouse → human transfer
    └── exp8_shared_analysis.py # Conserved vs specific regulation
```

### 3.6 Expected Outcomes and Risk Mitigation

**What success looks like:**

_Flow matching backbone (secondary contribution):_

- Competitive or better AUPRC vs RegDiffusion on BEELINE (not guaranteed to be large margin — RegDiffusion is already strong)
- More stable training (lower variance across seeds)
- Richer GRN extraction via Jacobian/IG methods
- Biologically interpretable velocity field

_Transfer learning (primary contribution):_

- Joint training matches or beats single-dataset training on majority of BEELINE datasets
- Few-shot: pretrained FlowGRN with 20% data matches RegDiffusion-Solo with 100% data
- Cross-species transfer outperforms solo training when target data is limited
- Biologically interpretable conserved vs context-specific edges

**Risks and mitigations:**

| Risk                                                      | Mitigation                                                                                                                                                                                                                                                  |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FlowGRN-Solo doesn't clearly beat RegDiffusion on BEELINE | This is acceptable — the primary contribution is transfer, not single-dataset SOTA. Report honestly; competitive performance + transfer capability is the story.                                                                                            |
| OT computation too expensive for large gene sets          | Sinkhorn regularization; sparse attention limiting to top-k neighbors                                                                                                                                                                                       |
| Negative transfer in joint training                       | Report honestly with Solo baseline. Analyze which dataset pairs help/hurt — negative transfer findings are publishable.                                                                                                                                     |
| Cross-species transfer doesn't work                       | Include same-species transfer as control. Even within-species transfer alone is novel.                                                                                                                                                                      |
| Gene overlap across BEELINE datasets too low              | Focus on TF-centric subnetwork where overlap is higher. Report overlap statistics.                                                                                                                                                                          |
| Shared latent space collapses                             | Monitor with UMAP/silhouette. Add contrastive auxiliary loss if needed.                                                                                                                                                                                     |
| Reviewers question novelty given RegDiffusion             | Emphasize: (1) transfer is the primary contribution, no prior method does this; (2) CFM vs DDPM is a principled choice with velocity field interpretation; (3) GAT architecture enables clean shared-private decomposition that RegDiffusion's MLPs cannot. |

---

## 4. Novelty Claims and Related Work Positioning

### Contribution hierarchy (explicit):

**Primary contribution: Transferable GRN inference.** No existing method — RegDiffusion, DeepSEM, DAZZLE, GENIE3, GRNBoost2, or any other — supports transfer of regulatory dynamics across datasets, cell types, or species. FlowGRN's shared-private decomposition enables this for the first time in an unsupervised generative framework.

**Secondary contribution: OT-conditional flow matching for GRN inference.** RegDiffusion demonstrated that diffusion-family models outperform VAEs for GRN inference. We advance this further with flow matching, which provides OT-structured transport, a biologically interpretable velocity field, and cleaner architectural separation for transfer learning.

**Tertiary contribution: Velocity-field-based GRN extraction.** Jacobian and integrated gradient analysis of the learned velocity field provides richer, more directional, and more context-aware GRN scores than the |A| extraction used by all prior methods.

### Detailed positioning:

1. **vs RegDiffusion (most direct competitor):** RegDiffusion uses DDPM with parameterized A in MLP blocks. FlowGRN uses OT-CFM with A as GAT attention bias. The key difference is not just the generative backbone (flow matching vs diffusion) but the architectural separation: FlowGRN's GAT cleanly separates shared dynamics (MLP weights) from dataset-specific wiring (A_k), enabling transfer learning. RegDiffusion's entangled MLP-A architecture cannot support this decomposition. Additionally, FlowGRN provides velocity-field-based GRN extraction and OT-structured transport, neither available in RegDiffusion.

2. **vs DeepSEM/DAZZLE:** Both RegDiffusion and FlowGRN improve on the VAE backbone. FlowGRN additionally enables transfer learning, which neither DeepSEM/DAZZLE nor RegDiffusion support.

3. **vs TrajectoryNet/TIGON/CellOT:** These use OT/flow for trajectory inference with GRN as downstream byproduct. FlowGRN embeds the GRN directly into the velocity field as the primary training objective.

4. **vs CycleGRN:** Restricted to oscillatory/cell-cycle genes. FlowGRN is general-purpose.

5. **vs GRNFormer/scRegNet:** Supervised methods requiring ground-truth labels. FlowGRN is fully unsupervised.

6. **vs GRNPT:** Uses LLM text embeddings and supervised training for cross-cell-type generalization. FlowGRN achieves transfer via shared generative dynamics, unsupervised, no external text data.

7. **vs Meta-TGLink:** Supervised meta-learning for few-shot GRN. FlowGRN achieves few-shot via pretrained generative dynamics without labels during pretraining.

8. **vs scMTNI:** Shares information across cell types on a single lineage via multi-task regularization. FlowGRN applies to any collection of datasets, and shared components are explicitly interpretable.

9. **vs LINGER:** Uses external bulk data via lifelong learning. FlowGRN transfers from other single-cell datasets — complementary approaches that could be combined.

10. **vs scGPT/Geneformer:** Foundation models extract gene embeddings for post-hoc GRN similarity inference. FlowGRN learns the GRN as a structural component of the generative model, but can use foundation model encoders as its backbone (Variant T4).

### Paper framing (revised for RegDiffusion context):

**Recommended framing:** "Transferable GRN inference via latent flow matching." The narrative: "Recent work (RegDiffusion) showed that diffusion models outperform VAEs for GRN inference. We extend this line of inquiry with two contributions: (1) we advance the generative backbone from DDPM to OT-conditional flow matching, gaining interpretable velocity fields and OT-structured transport; (2) more importantly, we show that the latent flow matching formulation uniquely enables a shared-private decomposition for cross-dataset transfer learning — the first unsupervised GRN method to share regulatory dynamics across cell types and species."

### Target venues:

- Nature Methods, Nature Computational Science, Genome Biology (full scope with transfer)
- Bioinformatics, Briefings in Bioinformatics (method + transfer)
- NeurIPS / ICML (ML angle: transferable generative models for biological networks)
