# FlowGRN: Latent Flow Matching for Gene Regulatory Network Inference

## A Detailed Research Proposal

---

## 1. Architecture Design

### 1.1 Background: The DeepSEM/DAZZLE Paradigm (Current SOTA on BEELINE)

The current leading deep learning approach on BEELINE is based on the **Structural Equation Model (SEM)** framework, used by DeepSEM and its successor DAZZLE. The key idea:

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

The adjacency matrix `A` is a **learnable parameter**, jointly optimized with the encoder/decoder networks. The loss function is:

```
L = -E_{z~q(Z|X)}[log p(X|Z)] + β·D_KL(q(Z|X) || p(Z)) + α·‖A‖₁
```

The L1 penalty on A encourages sparsity (real GRNs are sparse).

**Key limitations of this approach:**

1. **Training instability:** DeepSEM's inferred networks degrade as training continues past convergence, likely due to overfitting dropout noise
2. **Matrix inversion:** The `(I − Aᵀ)⁻¹` operation is numerically unstable and computationally expensive
3. **Static reconstruction:** The model reconstructs static snapshots; it doesn't model the _dynamics_ that generate gene expression
4. **Undirected edges in practice:** While A is asymmetric in principle, the model averages |A| over runs, partially losing directionality
5. **No knowledge sharing:** Every dataset is trained from scratch — no transfer of regulatory logic across cell types, tissues, or species

### 1.2 Proposed Architecture: FlowGRN

We propose replacing the VAE generative backbone with **conditional flow matching (CFM)**, while preserving the core insight of embedding the GRN structure into the generative model. Critically, the latent-space formulation enables a natural decomposition into **shared** (transferable) and **dataset-specific** (GRN) components.

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

**B) GRN-Structured Velocity Field (Core Innovation)**

The velocity field `v_θ(z_t, t)` that drives the flow is parameterized as a **Graph Attention Network** operating over genes:

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

**Key insight:** The velocity at gene i depends on the states of other genes j, weighted by α*{ij}. The learned attention weights—specifically the adjacency biases A*{ij}—directly encode whether gene j regulates gene i. High |A\_{ij}| means gene j has strong regulatory influence on gene i's dynamics.

**C) Flow Matching Training (No Simulation Required)**

Instead of the VAE's reconstruction + KL loss, we train with the **conditional flow matching (CFM) objective**:

```
L_CFM = E_{t, x_0, x_1} ‖v_θ(x_t, t) - u_t(x_t | x_0, x_1)‖²
```

where:

- `x_0 ~ p_noise` (e.g., standard Gaussian or gene expression from a reference state)
- `x_1 ~ p_data` (actual scRNA-seq expression vectors)
- `x_t = (1-t)·x_0 + t·x_1` (linear interpolation, OT conditional path)
- `u_t(x_t | x_0, x_1) = x_1 - x_0` (the target velocity for the linear path)

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

### 1.4 Extension: Transfer Learning via Shared-Private Decomposition

The latent-space formulation enables a natural decomposition of the model into **shared** (transferable) and **private** (dataset-specific) components. This is a key architectural innovation — current GRN methods (DeepSEM, DAZZLE, GENIE3, GRNBoost2) all train from scratch on each dataset independently, discarding any regulatory logic that could transfer.

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

#### Transfer Variants

**Variant T1: Joint Multi-Task Training (simplest)**

Train one FlowGRN on all K datasets simultaneously with shared weights:

```
L_total = Σ_{k=1}^{K} L_CFM^{(k)}(θ_shared, A_k) + α·Σ_k ‖A_k‖₁
```

- θ_shared = {θ_enc, θ_MLP_msg, θ_MLP_self, W_Q, W_K, a} — trained on all datasets
- A_k — separate adjacency matrix per dataset, only trained on data from dataset k
- Batches alternate or mix cells from different datasets

This forces the shared components to learn universal regulatory dynamics, while each A_k specializes to its cell type.

**Variant T2: Pretrain-then-Finetune**

Two-stage training:

Stage 1 (Pretrain): Train FlowGRN jointly on a large collection of datasets (e.g., 6 of 7 BEELINE datasets, or external atlas data). Learn θ_shared and {A_1, ..., A_6}.

Stage 2 (Finetune): For a new target dataset, freeze θ_shared and only train a new A_target from scratch (or from a warm initialization). This is **few-shot GRN inference** — you need far less data because the encoder and velocity dynamics are already learned.

```
Stage 1: min_{θ_shared, {A_k}} Σ_{k=1}^{K} L^{(k)}
Stage 2: min_{A_target} L^{(target)}  with θ_shared frozen
```

**Variant T3: Cross-Species Transfer**

For organisms with well-established ortholog mappings (e.g., mouse ↔ human):

1. Train on source species datasets (e.g., all 5 mouse BEELINE datasets: mDC, mESC, mHSC-E/GM/L)
2. Map gene identities via ortholog table (well-established for mouse-human, ~16,000 one-to-one orthologs)
3. Transfer: freeze θ_shared, train A_target on human datasets (hESC, hHEP)

The latent space is key here — in raw gene expression space, mouse genes and human genes are different features entirely. But the shared encoder learns to map orthologous regulatory programs into similar latent regions, enabling the velocity field MLPs to apply cross-species.

**Variant T4: Foundation Model Backbone**

Use a frozen pretrained encoder from scGPT (33M cells) or Geneformer (30M cells):

1. The foundation model's encoder already provides a universal latent space across cell types and species
2. Only train the velocity field (shared MLPs + per-dataset A_k)
3. The pretrained gene embeddings capture coexpression and regulatory structure from massive pretraining data

This is the most parameter-efficient variant and directly leverages the investment in foundation model pretraining.

#### Optional: Dataset Conditioning

For Variants T1 and T2, optionally condition the velocity field on a dataset identity:

```
v_i(z_t, t, k) = Σ_j α_{ij}^k · MLP_msg(z_t^j, t, e_k) + MLP_self(z_t^i, t, e_k)
```

where e_k ∈ ℝ^{d_e} is a learnable embedding for dataset k (analogous to a "cell-type token"). This lets the shared MLPs modulate their behavior slightly per dataset, beyond just the adjacency difference. At inference time for a new dataset, e_k can be:

- Initialized as the mean of existing embeddings
- Learned alongside A_target in the finetune stage
- Derived from dataset-level statistics (mean expression, cell count, etc.)

### 1.5 Architecture Comparison

| Feature                | DeepSEM/DAZZLE           | GRNFormer                   | **FlowGRN (ours)**            | **FlowGRN + Transfer**           |
| ---------------------- | ------------------------ | --------------------------- | ----------------------------- | -------------------------------- |
| Generative model       | VAE                      | N/A (discriminative)        | Flow Matching                 | Flow Matching                    |
| GRN parameterization   | Explicit A in enc/dec    | Graph transformer attention | Adjacency-biased GAT velocity | Same, per-dataset A_k            |
| Training objective     | Reconstruction + KL      | Supervised (needs labels)   | CFM regression (unsupervised) | Multi-task CFM                   |
| Handles dynamics       | No (static)              | No                          | Yes (velocity field)          | Yes                              |
| Directionality         | Weak (averaged \|A\|)    | Via TF-anchored subgraphs   | Strong (directed velocity)    | Strong                           |
| Dropout robustness     | DAZZLE: DA augmentation  | N/A                         | OT coupling                   | OT + more data via sharing       |
| Cross-dataset transfer | None                     | None                        | None (single-task)            | **Yes** (shared dynamics)        |
| Few-shot GRN           | Not possible             | Requires labels             | Not possible                  | **Yes** (freeze shared, train A) |
| Cross-species          | Not possible             | Not possible                | Not possible                  | **Yes** (via ortholog mapping)   |
| Scalability            | O(g²) for matrix inverse | O(g·k) per subgraph         | O(g·k) sparse attention       | Same + amortized pretraining     |

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

In practice, the exact OT plan is intractable for large datasets. **Minibatch OT** approximates it:

1. Sample a minibatch of B noise samples {x*0^i}*{i=1}^B and B data samples {x*1^j}*{j=1}^B
2. Solve the assignment problem within the batch:
   ```
   σ* = argmin_{σ ∈ S_B} Σ_{i=1}^B ‖x_0^i - x_1^{σ(i)}‖²
   ```
   (This is the Hungarian algorithm, O(B³), or Sinkhorn approximation)
3. Form paired samples (x_0^i, x_1^{σ\*(i)})

**The conditional paths for OT-CFM:**

```
x_t = (1 - t)·x_0 + t·x_1         (linear interpolation)
u_t(x | x_0, x_1) = x_1 - x_0     (constant velocity along straight line)
```

**Why this matters for GRN inference:** OT coupling produces straighter, non-crossing flows. Since gene expression distributions often have complex multimodal structure (different cell types), OT coupling ensures the velocity field captures biologically meaningful transitions (e.g., stem cell → differentiated cell) rather than arbitrary noise-to-data mappings.

### 2.3 Applying CFM to GRN Inference

**Setup for single-cell data:**

- Source distribution p_0: Standard Gaussian N(0, I_g) (or empirical distribution of a reference cell state)
- Target distribution p_1: Empirical distribution of scRNA-seq expression vectors
- Each sample x ∈ ℝ^g represents a cell's expression profile across g genes

**GRN-structured velocity field:**

For a cell at state x_t at flow time t, the velocity for gene i is:

```
v_θ,i(x_t, t) = Σ_{j ∈ N(i)} α_{ij}(x_t, t) · φ(x_t^j, t) + ψ(x_t^i, t)
```

where:

- N(i) is the neighborhood of gene i (all genes if dense, top-k if sparse)
- α\_{ij} are attention weights incorporating the learnable adjacency A
- φ, ψ are shared MLPs for message and self-update functions

**Attention mechanism with adjacency bias:**

```
e_{ij} = LeakyReLU(a^T · [W_Q x_t^i ‖ W_K x_t^j ‖ γ(t)])
α_{ij} = exp(e_{ij} + λ·A_{ij}) / Σ_k exp(e_{ik} + λ·A_{ik})
```

where:

- A ∈ ℝ^{g×g} is the learnable adjacency (initialized to zeros or from prior)
- λ controls the strength of the adjacency prior vs data-driven attention
- γ(t) is a sinusoidal time embedding

**Full training loss (single-dataset):**

```
L = E_{t~U[0,1], (x_0,x_1)~π_OT} ‖v_θ(x_t, t) - (x_1 - x_0)‖² + α·‖A‖₁ + β·R(A)
```

where R(A) is an optional structural regularizer:

- **Acyclicity (NOTEARS):** R(A) = tr(e^{A∘A}) - g (forces DAG structure)
- **Degree constraint:** R(A) = Σ*i max(0, Σ_j |A*{ij}| - k) (limits in-degree to k)
- **Prior incorporation:** R(A) = ‖A ∘ M‖\_F² where M is a mask from TF motif data (penalize edges inconsistent with known motif binding)

### 2.4 Multi-Task Flow Matching for Transfer Learning

#### Formulation

Given K datasets {D_1, ..., D_K} (e.g., different BEELINE cell types), we decompose the model parameters into:

- θ_shared = {θ_enc, θ_MLP_msg, θ_MLP_self, W_Q, W_K, a, b_A, λ} — shared across datasets
- θ_private = {A_1, ..., A_K, e_1, ..., e_K} — per-dataset adjacency matrices and optional embeddings

**Multi-task loss:**

```
L_total = Σ_{k=1}^{K} w_k · L^{(k)}(θ_shared, A_k, e_k) + α·Σ_k ‖A_k‖₁ + β·Σ_k R(A_k)
```

where:

```
L^{(k)} = E_{t, (x_0, x_1^{(k)})~π_OT^{(k)}} ‖v_{θ_shared}(x_t, t; A_k, e_k) - (x_1^{(k)} - x_0)‖²
```

and w_k are per-dataset weights (e.g., proportional to dataset size, or uniform).

The velocity field with dataset conditioning:

```
v_i(z_t, t; A_k, e_k) = Σ_j α_{ij}^{(k)} · MLP_msg([z_t^j; γ(t); e_k]) + MLP_self([z_t^i; γ(t); e_k])
```

where [;] denotes concatenation and α\_{ij}^{(k)} uses A_k in the attention computation.

#### Gradient Flow Analysis

The key mathematical property: gradients from L^{(k)} flow into θ_shared through the velocity field MLPs but are blocked from other datasets' adjacency matrices by construction. Specifically:

```
∂L^{(k)}/∂θ_shared ≠ 0     (shared weights learn from all datasets)
∂L^{(k)}/∂A_k ≠ 0          (A_k learns from dataset k)
∂L^{(k)}/∂A_j = 0, j ≠ k   (A_j is not updated by dataset k's loss)
```

This ensures each A_k reflects only the GRN of its corresponding cell type, while θ_shared captures cross-dataset regulatory patterns.

#### Transfer to New Datasets

For a new dataset D_target not seen during pretraining:

```
A_target* = argmin_{A_target} L^{(target)}(θ_shared*, A_target, e_target) + α·‖A_target‖₁
```

where θ_shared\* is frozen from pretraining. The dataset embedding e_target is either:

- Jointly optimized with A_target
- Set to the mean of pretrained embeddings: e_target = (1/K)Σ_k e_k
- Computed from a lightweight projection of dataset-level statistics

#### Cross-Species Formulation

For cross-species transfer (e.g., mouse → human), introduce an ortholog mapping matrix O ∈ {0,1}^{g*h × g_m} where O*{ij} = 1 if human gene i is orthologous to mouse gene j. Then:

1. Shared encoder operates on species-independent latent features
2. The gene-level indexing in the adjacency matrix is remapped via O
3. Warm-start: A_human ← O · A_mouse · Oᵀ (project mouse GRN into human gene space)
4. Finetune A_human on human data with θ_shared frozen

### 2.5 GRN Extraction and Scoring

After training, extract the GRN:

**Method 1: Direct adjacency extraction**

- Edge score(j → i) = |A\_{ij}|
- Edge sign(j → i) = sign(A\_{ij})

**Method 2: Jacobian analysis of velocity field**

- For representative cells x\* (e.g., cluster centroids):
  ```
  J_{ij}(x*, t) = ∂v_i(x*, t) / ∂x_j
  ```
- Average over t and representative cells:
  ```
  GRN_{ij} = E_{t, x*} [|J_{ij}(x*, t)|]
  ```
- This captures both the direct adjacency AND the indirect effects learned by the MLPs

**Method 3: Integrated Gradients**

- Compute attribution of each gene j to the velocity of gene i over the full flow:
  ```
  IG_{ij} = ∫_0^1 ∂v_i(x_t, t)/∂x_j · (x_1^j - x_0^j) dt
  ```

**Transfer-specific extraction note:** In the multi-task setting, Methods 2 and 3 capture effects from both A_k (dataset-specific) and θ_shared (universal dynamics). This means the Jacobian-based GRN may actually be _more informative_ than the raw A_k, since it incorporates universal regulatory logic learned from all datasets.

### 2.6 Biological Interpretation of the Flow

The trained velocity field has a natural biological interpretation:

- **v_i(x, t) > 0:** Gene i is being upregulated at state x, flow time t
- **α\_{ij} large:** Gene j strongly influences the rate of change of gene i → regulatory edge
- **Flow trajectories:** Integrating v_θ from noise to data traces paths through gene expression space that correspond to cell state transitions
- **Cell-type-specific GRNs:** Evaluate J\_{ij}(x*, t) at different cell-type centroids x* to get context-dependent regulatory networks
- **Shared vs specific regulation:** Compare A_k across datasets — edges that appear in all A_k represent conserved regulation; edges unique to one A_k represent context-specific wiring

---

## 3. Experimental Setup

### 3.1 BEELINE Benchmark: Datasets and Ground Truths

**BEELINE consists of three dataset categories:**

#### A) Synthetic Networks (6 datasets)

Generated from toy networks using BoolODE (stochastic simulation):

- **dyn-LI:** Linear trajectory
- **dyn-CY:** Cyclic trajectory
- **dyn-LL:** Long linear trajectory
- **dyn-BF:** Bifurcating trajectory
- **dyn-TF:** Trifurcating trajectory
- **dyn-BFC:** Bifurcating converging trajectory

Each has sub-datasets with 100, 200, 500, 2000, 5000 cells × 10 samples.

**Ground truth:** The exact network used to generate the data (known perfectly).

#### B) Curated Boolean Models (4 datasets)

From literature-curated Boolean models:

- **HSC:** Hematopoietic stem cell differentiation
- **mCAD:** Mammalian cortical area development
- **VSC:** Ventral spinal cord development
- **GSD:** Gonadal sex determination

Each has 2000 cells × 10 samples, plus variants with 50% and 70% dropout.

**Ground truth:** The Boolean model structure.

#### C) Experimental scRNA-seq (7 datasets)

Real datasets:

- **hESC:** Human embryonic stem cells (GSE75748)
- **hHEP:** Human hepatocytes (GSE81252)
- **mDC:** Mouse dendritic cells (GSE48968)
- **mESC:** Mouse embryonic stem cells (GSE98664)
- **mHSC-E, mHSC-GM, mHSC-L:** Mouse hematopoietic stem cells, three lineages (GSE81682)

**Ground truths (approximate):**

1. **Non-specific ChIP-seq** networks (from ENCODE, broad TF binding)
2. **STRING** protein-protein interaction database
3. **Cell-type-specific ChIP-seq** (matched to cell type)
4. **LOF/GOF** (Loss-of-function / Gain-of-function perturbation studies)

Gene sets: All significantly varying TFs + top 500 or 1000 most varying genes.

### 3.2 Evaluation Metrics

Following BEELINE protocol:

1. **AUPRC (Area Under Precision-Recall Curve):**
   - Primary metric for imbalanced binary classification (GRNs are very sparse)
   - Report as **AUPRC ratio** = AUPRC / baseline (where baseline = edge density of ground truth)

2. **EPR (Early Precision Ratio):**
   - Precision among the top K predicted edges, where K = number of edges in ground truth
   - EPR ratio = EP / random EP
   - Measures: "How good are your best predictions?"

3. **AUROC (Area Under ROC Curve):**
   - Secondary metric; less informative for sparse networks but commonly reported

### 3.3 Baselines to Compare Against

**Tier 1: Direct competitors (deep learning, unsupervised, on BEELINE)**

- DeepSEM (Nature Computational Science, 2021)
- DAZZLE (PLOS Computational Biology, 2025) — current SOTA on BEELINE for unsupervised DL
- GRN-VAE (simplified DeepSEM)
- HyperG-VAE (hypergraph VAE with SEM cell encoder)

**Tier 2: Classical methods (still competitive)**

- GENIE3 (Random Forest regression)
- GRNBoost2 (Gradient Boosting, used in SCENIC)
- PIDC (Partial Information Decomposition)
- PPCOR (Partial Correlation)

**Tier 3: Recent SOTA (may not all use BEELINE)**

- GRNFormer (Graph Transformer, 2025-2026)
- scKAN (Kolmogorov-Arnold Networks, 2025)
- scRegNet (Foundation model + GNN, 2025)
- GRANGER (Recurrent VAE with Granger causality, 2025)

**Tier 4: Transfer-aware methods (for transfer experiments)**

- GRNPT (Transformer with LLM embeddings, cross-cell-type generalization)
- Meta-TGLink (Graph meta-learning, few-shot GRN, supervised)
- scMTNI (Multi-task network inference across lineages)
- LINGER (Lifelong learning with atlas-scale external bulk data)

### 3.4 Experimental Protocol

#### Experiment 1: BEELINE Benchmark — Single-Dataset (Primary)

**Setup:**

1. Use all three BEELINE dataset categories
2. For experimental datasets: TFs + 500 genes AND TFs + 1000 genes
3. Run FlowGRN with 10 random seeds (report mean ± std)
4. Use identical preprocessing as BEELINE Docker containers

**Hyperparameter search:**

- Latent dimension d ∈ {32, 64, 128}
- Sparsity penalty α ∈ {0.001, 0.01, 0.1}
- Adjacency bias strength λ ∈ {0.1, 1.0, 10.0}
- OT batch size B ∈ {64, 128, 256}
- Number of GAT attention heads ∈ {1, 4, 8}
- Flow integration steps at inference ∈ {10, 20, 50, 100}

**Key comparisons:**

- FlowGRN vs DAZZLE (same input, same evaluation) — is flow matching better than VAE?
- FlowGRN with OT coupling vs without — does OT improve inference?
- FlowGRN with DAG constraint vs without — does acyclicity help?

#### Experiment 2: Ablation Studies

- **Velocity field architecture:** GAT vs simple MLP vs Transformer
- **OT coupling:** None (independent) vs minibatch OT vs Sinkhorn-regularized OT
- **GRN extraction method:** Direct A vs Jacobian vs Integrated Gradients
- **Latent vs gene space:** Flow matching in original gene space vs latent space
- **Dropout robustness:** Performance on curated datasets with 0%, 50%, 70% dropout
- **With/without dropout augmentation:** Can DA (from DAZZLE) further improve FlowGRN?

#### Experiment 3: External Biological Validation (Critical for Impact)

BEELINE ground truths are imperfect. Validate beyond BEELINE:

**A) ChIP-seq validation (following GRNFormer and DeepSEM):**

- Predict TF-target edges for specific TFs in specific cell types
- Validate against matched ChIP-seq binding data from ENCODE
- Report AUC and AUPR for each TF

**B) Perturbation prediction (following SCENIC+ and CellOracle):**

- Use inferred GRN to predict downstream effects of TF knockout
- Compare predicted differentially expressed genes against actual perturbation data
- Datasets: Perturb-seq (Dixit et al., 2016), CRISPRi screens

**C) Cross-species transfer (optional, high novelty):**

- Train on well-annotated species (e.g., mouse hematopoiesis)
- Evaluate GRN accuracy on related but less-annotated species
- Tests generalization of flow-based representations

#### Experiment 4: Scalability Analysis

- Runtime vs number of genes (100, 500, 1000, 5000, 10000)
- Runtime vs number of cells (100, 500, 2000, 5000, 20000)
- GPU memory usage
- Compare against DeepSEM, DAZZLE, GENIE3, GRNBoost2

#### Experiment 5: Multi-Task Joint Training (Transfer — Within BEELINE)

Test whether sharing regulatory dynamics across BEELINE datasets helps:

**Setup:**

- FlowGRN-Solo: Train independently on each of the 7 experimental datasets (baseline)
- FlowGRN-Joint: Train on all 7 datasets simultaneously (shared θ, per-dataset A_k)
- FlowGRN-Joint-Cond: Same as Joint but with dataset conditioning embeddings e_k

**Evaluation:**

- Compare AUPRC, EPR, AUROC on each dataset: Joint vs Solo
- Report per-dataset results (does transfer help hESC? hHEP? mDC? etc.)
- Analyze which datasets benefit most from sharing — hypothesis: smaller datasets and datasets from the same organism/lineage benefit most

**What to look for:**

- If Joint > Solo on most datasets: shared regulatory grammar helps
- If Joint > Solo only on small datasets: transfer is a data-efficiency story
- If Joint < Solo on some datasets: negative transfer — the datasets are too different
- If Joint-Cond > Joint: conditioning helps the shared MLPs specialize

#### Experiment 6: Few-Shot GRN Inference (Transfer — Data Efficiency)

Test whether pretrained shared weights reduce the data requirement:

**Setup — Leave-One-Out:**
For each dataset k ∈ {1, ..., 7}:

1. Pretrain FlowGRN on the other 6 datasets → get θ_shared\*
2. Finetune only A_k on dataset k (θ_shared frozen)
3. Compare against FlowGRN-Solo trained on dataset k from scratch

**Setup — Data Titration:**
For each dataset k:

1. Subsample to {100%, 50%, 20%, 10%, 5%} of cells
2. Compare: (a) Solo from scratch vs (b) Transfer with pretrained θ_shared\*
3. Plot AUPRC vs number of cells for both conditions

**Expected result:** Transfer should help most in the low-data regime. If pretrained FlowGRN with 10% of cells matches Solo with 100%, that is a strong result for rare cell types and understudied organisms.

**Additional baseline:** Compare against Meta-TGLink (the recent few-shot GRN method) in matched conditions, noting that Meta-TGLink is supervised while FlowGRN is unsupervised.

#### Experiment 7: Cross-Species Transfer (Transfer — Mouse → Human)

The most ambitious transfer experiment:

**Setup:**

1. **Source:** Train FlowGRN jointly on all 5 mouse datasets (mDC, mESC, mHSC-E/GM/L) → θ_shared_mouse
2. **Ortholog mapping:** Use Ensembl BioMart one-to-one orthologs (~16,000 genes) to map mouse gene IDs → human gene IDs
3. **Target:** Transfer to human datasets (hESC, hHEP):
   - Freeze θ_shared_mouse
   - Initialize A_human = O · A_mouse_avg · Oᵀ (warm start from projected mouse GRN)
   - Alternatively: initialize A_human = 0 (cold start)
   - Finetune A_human on human data
4. **Compare against:** FlowGRN-Solo on human data, GENIE3 on human data, LINGER (which uses external bulk data)

**Control:** Also test the reverse direction (human → mouse) and same-species transfer (mouse subset → mouse held-out) to disentangle the effects of species barrier vs more pretraining data.

**What to look for:**

- Warm start (projected A) vs cold start (A=0): does the mouse GRN structure help?
- Cross-species transfer vs same-species transfer: how much does the species barrier cost?
- Which gene pairs transfer well? Hypothesis: TFs with conserved binding motifs (e.g., core pluripotency factors, hematopoietic TFs) should transfer best

#### Experiment 8: Analysis of Shared vs Specific Regulation

Beyond performance metrics, analyze _what_ the model learns:

**A) Conserved edge analysis:**

- After joint training, compare A_k across datasets
- Edges with high |A\_{ij}| in all datasets → conserved regulatory interactions
- Validate against known housekeeping regulatory interactions

**B) Context-specific edge analysis:**

- Edges with high |A\_{ij}| in one dataset but low in others → context-specific regulation
- Validate against cell-type-specific ChIP-seq ground truths

**C) Shared weight inspection:**

- Analyze what the shared MLP_msg learns: does it capture general activation/repression dynamics?
- Visualize attention patterns before and after the adjacency bias: how much does A_k modulate the universal attention?

**D) Latent space structure:**

- Visualize the shared latent space: do cells from different datasets form meaningful clusters?
- Do cells undergoing similar biological processes (e.g., differentiation) cluster together across datasets?

### 3.5 Implementation Plan

**Libraries:**

- PyTorch for model implementation
- TorchCFM (https://github.com/atong01/conditional-flow-matching) for flow matching
- PyTorch Geometric for graph attention layers
- Scanpy for scRNA-seq preprocessing
- POT (Python Optimal Transport) for OT solvers
- Ensembl BioMart API (pybiomart) for ortholog mapping

**Hardware:** Single GPU (A100/H100), models should be trainable in minutes per dataset

**Code structure:**

```
flowgrn/
├── models/
│   ├── encoder.py              # Per-gene MLP encoder (shared weights)
│   ├── velocity_field.py       # GAT-based velocity network (shared + A_k)
│   ├── flow_matching.py        # OT-CFM training loop
│   ├── grn_extraction.py       # Adjacency / Jacobian / IG methods
│   └── transfer.py             # Multi-task trainer, freeze/finetune logic
├── data/
│   ├── beeline_loader.py       # BEELINE dataset interface
│   ├── preprocessing.py        # log1p transform, gene selection
│   ├── multitask_loader.py     # Interleaved batching across datasets
│   └── ortholog_mapper.py      # Cross-species gene mapping via BioMart
├── evaluation/
│   ├── metrics.py              # AUPRC, EPR, AUROC
│   ├── beeline_eval.py         # Full BEELINE evaluation pipeline
│   └── transfer_analysis.py    # Conserved vs specific edge analysis
├── baselines/
│   └── run_baselines.sh        # Docker commands for BEELINE baselines
└── experiments/
    ├── exp1_benchmark.py       # Single-dataset BEELINE
    ├── exp2_ablations.py       # Architecture ablations
    ├── exp3_bio_validation.py  # ChIP-seq, perturbation validation
    ├── exp4_scalability.py     # Runtime and memory analysis
    ├── exp5_joint_training.py  # Multi-task within BEELINE
    ├── exp6_few_shot.py        # Leave-one-out, data titration
    ├── exp7_cross_species.py   # Mouse → human transfer
    └── exp8_shared_analysis.py # Conserved vs specific regulation
```

### 3.6 Expected Outcomes and Risk Mitigation

**What success looks like:**

_Core FlowGRN:_

- AUPRC ratio improvement of 5-15% over DAZZLE on BEELINE experimental datasets
- Significantly more stable results across random seeds (lower variance)
- Better performance on high-dropout curated datasets (50%, 70%)
- Stronger directionality in predicted edges (verified by ChIP-seq directional validation)
- Competitive or better runtime vs DeepSEM/DAZZLE

_Transfer extension:_

- Joint training matches or beats single-dataset training on majority of BEELINE datasets
- Few-shot setting: pretrained FlowGRN with 20% data matches Solo with 100% data
- Cross-species: mouse→human transfer outperforms training on human data alone when human data is limited
- Biologically interpretable conserved vs context-specific regulatory edges

**Risks and mitigations:**

| Risk                                                                        | Mitigation                                                                                                                                                                                 |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Flow matching doesn't outperform VAE on static snapshots                    | Add time-series BEELINE datasets (synthetic with temporal ordering); use pseudotime to define biological source→target distributions                                                       |
| OT computation too expensive for large gene sets                            | Use Sinkhorn regularization with ε=0.1; reduce batch OT to top-k genes per batch                                                                                                           |
| Adjacency matrix doesn't converge                                           | Initialize A from GENIE3 output (warm start); anneal sparsity penalty                                                                                                                      |
| BEELINE improvements are marginal                                           | Focus story on biological validation (Exp 3) and transfer novelty (Exp 5-8); marginal BEELINE + strong transfer story is publishable                                                       |
| Model is too complex for reviewers                                          | Provide clear ablations showing which components matter; release clean code                                                                                                                |
| **Negative transfer in joint training**                                     | **Report honestly; include Solo baseline always. Analyze which dataset pairs help/hurt. Negative transfer is itself an interesting finding about regulatory conservation.**                |
| **Cross-species transfer doesn't work**                                     | **Include same-species transfer as control to isolate the species barrier. Even if cross-species fails, within-species transfer (e.g., mouse cell types helping each other) may succeed.** |
| **Gene overlap between BEELINE datasets is too low for meaningful sharing** | **Report the gene overlap statistics. Use the intersection gene set for transfer experiments. If overlap is small, focus on the TF-centric subnetwork where overlap is higher.**           |
| **Shared latent space collapses**                                           | **Monitor latent space quality during training (UMAP, silhouette scores). Add a contrastive or reconstruction auxiliary loss if needed to maintain structure.**                            |

---

## 4. Novelty Claims and Related Work Positioning

### What makes this different from existing work:

1. **vs TrajectoryNet/TIGON/CellOT:** These use OT/flow for _trajectory inference_ (cell dynamics), with GRN as a downstream byproduct extracted post-hoc. FlowGRN embeds the GRN _directly into the velocity field architecture_, making GRN inference the primary training objective.

2. **vs CycleGRN:** Uses invariant flows but restricted to oscillatory/cell-cycle genes. FlowGRN is general-purpose and applies to any gene set and trajectory topology.

3. **vs DeepSEM/DAZZLE:** Replaces the VAE with flow matching, which provides: (a) no need for matrix inversion, (b) OT-based noise handling instead of heuristic dropout augmentation, (c) a velocity field with natural biological interpretation as gene expression dynamics. Additionally, the shared-private architecture enables transfer learning, which is impossible in the DeepSEM/DAZZLE framework.

4. **vs GRNFormer:** Supervised method requiring ground-truth labels for training. FlowGRN is fully unsupervised, requiring only expression data.

5. **vs scGPT/Geneformer for GRN:** Foundation models extract gene embeddings and use similarity for GRN inference (post-hoc). FlowGRN learns the GRN as an integral structural component of the generative model. However, FlowGRN can _use_ foundation model encoders as its shared encoder backbone (Variant T4).

6. **vs GRNPT:** GRNPT uses LLM text embeddings and supervised training for cross-cell-type generalization. FlowGRN achieves transfer via shared generative dynamics in an entirely unsupervised manner — no labels, no external text data.

7. **vs Meta-TGLink:** Meta-TGLink is a supervised meta-learning approach for few-shot GRN inference. FlowGRN achieves few-shot capability through pretrained generative dynamics rather than meta-learning, and does not require ground-truth labels during pretraining.

8. **vs scMTNI:** scMTNI shares information across cell types on a single lineage via multi-task regularization. FlowGRN shares through a generative model's learned dynamics, applies to any collection of datasets (not just lineages), and the shared components are explicitly interpretable (what does the velocity field learn universally?).

9. **vs LINGER:** LINGER uses external bulk data via lifelong learning to improve single-cell GRN inference. FlowGRN transfers from other single-cell datasets, which is complementary — both external bulk data (LINGER's approach) and cross-dataset single-cell transfer (FlowGRN's approach) could be combined.

### Paper framing options:

**Option A — Full paper (Nature Methods / Genome Biology):**
FlowGRN as a complete method: flow matching backbone + transfer learning. Experiments 1-8. The narrative arc: "Flow matching provides a better generative backbone for GRN inference, and its latent-space formulation uniquely enables cross-dataset transfer learning."

**Option B — Two papers:**

- Paper 1 (Methods): FlowGRN core (Experiments 1-4). Focus: flow matching for GRN inference.
- Paper 2 (Analysis/Application): Transfer extension (Experiments 5-8). Focus: when and why does transfer help GRN inference? What regulatory logic is conserved?

**Option C — Conference + journal:**

- Conference (NeurIPS/ICML workshop or main): FlowGRN core with transfer as an ablation
- Journal follow-up: Full biological analysis with cross-species transfer

### Target venues:

- Nature Methods, Nature Computational Science, Genome Biology (full scope, Options A or B)
- Bioinformatics, Briefings in Bioinformatics (core method only)
- NeurIPS / ICML (machine learning angle with biological validation)
