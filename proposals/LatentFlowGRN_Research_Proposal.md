# LatentFlowGRN: Transferable Gene Regulatory Network Inference via Latent Flow Matching with Shared Regulatory Dynamics

## A Detailed Research Proposal

---

## 1. Architecture Design

### 1.1 Background: The Evolving Landscape of Generative Models for GRN Inference

Recent GRN inference methods increasingly leverage generative models trained on scRNA-seq data. Three distinct strategies have emerged for connecting the generative model to GRN extraction:

#### 1.1.1 Strategy A: Parameterized adjacency inside a VAE (DeepSEM / DAZZLE)

The **Structural Equation Model (SEM)** framework embeds a learnable adjacency matrix A directly in the encoder and decoder of a VAE:

- **Encoder:** `Z = X(I − Aᵀ)` — the GRN layer
- **Decoder:** `X = Z(I − Aᵀ)⁻¹` — the inverse GRN layer
- **GRN extraction:** Read off |A\_{ij}| as regulatory edge scores

**Limitations:** (1) Training instability — networks degrade past convergence; (2) Matrix inversion `(I − Aᵀ)⁻¹` is O(g³) and numerically unstable; (3) Static reconstruction only; (4) Directionality lost through |A| averaging over runs; (5) No cross-dataset transfer.

#### 1.1.2 Strategy B: Parameterized adjacency inside a DDPM (RegDiffusion)

RegDiffusion (Zhu & Slonim, J. Comp. Biol. 2024) replaced the VAE with a **denoising diffusion probabilistic model (DDPM)**:

- **Forward:** Add Gaussian noise following a diffusion schedule
- **Reverse:** Neural network with parameterized A predicts added noise via 3 MLP blocks
- **GRN extraction:** Same as DeepSEM — read off |A\_{ij}|

**Advances over VAE:** Eliminates matrix inversion (O(g²) runtime); superior stability; scales to 40k+ genes in <5 min; outperforms DeepSEM/DAZZLE on most BEELINE datasets.

**Remaining limitations:** (1) No cross-dataset transfer; (2) No OT-structured transport; (3) No velocity field / dynamics interpretation; (4) MLP-only architecture (A as linear mixing, not graph attention); (5) |A| extraction only.

#### 1.1.3 Strategy C: Flow matching for trajectory reconstruction → post-hoc GRN (FlowGRN)

FlowGRN (Tong & Pang, ACM BCB 2025) takes a fundamentally different approach — the GRN is **NOT** embedded in the generative model:

- **Step 1:** Define a dropout-robust cell similarity d_raw using only nonzero gene intersections, then compute geodesic distances on a kNN graph
- **Step 2:** Train [SF]²M (CFM + score matching with Schrödinger bridge formulation) to learn a velocity field v*θ and score function s*θ, using the dropout-robust distance as the OT cost function
- **Step 3:** Reconstruct per-cell trajectories by integrating the learned velocity field forward/backward in time
- **Step 4:** Feed reconstructed trajectories to **dynGENIE3** (temporal random forest) for GRN inference

**Key findings:**

- Achieves average rank 1.5 on simulated BEELINE datasets (best overall), top-tier on experimental data
- Scales to ~1,783 genes, runs in <10 minutes
- **Critical ablation:** Extracting GRN directly from the velocity field Jacobian (∇v_θ) performs _significantly worse_ than using dynGENIE3 on reconstructed trajectories. The reason: the chain rule causes indirect regulatory effects to appear in the Jacobian, making it hard to distinguish direct from indirect regulation
- Dropout-robust similarity is essential for high-dimensional datasets

**Remaining limitations:** (1) No cross-dataset transfer; (2) GRN not learned jointly with dynamics — two-stage pipeline loses end-to-end optimization; (3) dynGENIE3 is a separate model that doesn't benefit from the neural network's learned representations; (4) Scalability limited to ~1,783 genes (less than RegDiffusion's 40k+)

#### 1.1.4 The Gap: No Existing Method Supports Transfer Learning

All three strategies — and all classical methods (GENIE3, GRNBoost2, PIDC, etc.) — train from scratch on each dataset independently. No regulatory dynamics, no learned representations, no inferred structural knowledge transfers across cell types, tissues, or species. This is the gap LatentFlowGRN fills.

### 1.2 Proposed Architecture: LatentFlowGRN

LatentFlowGRN combines the strengths of Strategies A-C while addressing each strategy's limitations, and adds a fundamentally new capability: **cross-dataset transfer of shared regulatory dynamics.**

The design principles:

1. **From Strategy B (RegDiffusion):** Embed a parameterized adjacency A directly in the generative model for end-to-end GRN learning
2. **From Strategy C (FlowGRN-Tong):** Use OT-conditional flow matching as the generative backbone, with dropout-robust OT coupling
3. **New:** Use a **Graph Attention Network** (not MLP) to parameterize the velocity field, with A as attention bias — this enables clean shared-private decomposition for transfer learning
4. **New:** Operate in **latent space** for scalability and noise robustness

#### Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    LatentFlowGRN Pipeline                          │
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
│  │  │  v_i(z_t, t) = Σ_j α_{ij}^k · f(z_t^j, t)  │             │
│  │  │  where α_{ij}^k = softmax(A_k_{ij})          │             │
│  │  └───────────────────────────────────────────────┘             │
│  │                                                                 │
│  │  GRN extraction options:                                        │
│  │   (a) Direct: A_k → regulatory edges (primary)                 │
│  │   (b) Hybrid: reconstructed trajectories → dynGENIE3           │
│  └─────────────────────────────────────────────────────────────────┘
└────────────────────────────────────────────────────────────────────┘
```

#### Component Details

**A) Encoder: Per-Gene MLP with Shared Weights**

A small MLP processes one gene at a time with shared weights:

```
h_i = MLP_enc(x_i)  for each gene i ∈ {1, ..., g}
z = [h_1, h_2, ..., h_g]  ∈ ℝ^{c × g × d}
```

The encoder weights θ_enc are **shared across all datasets** in the transfer setting. Alternatively, a **pretrained encoder** from scGPT or Geneformer can serve as the shared encoder (Variant T4).

**B) GRN-Structured Velocity Field**

The velocity field uses a **Graph Attention Network** where A modulates attention weights:

```
v_i(z_t, t) = Σ_{j=1}^{g} α_{ij} · MLP_msg(z_t^j, t) + MLP_self(z_t^i, t)
```

with attention:

```
e_{ij} = LeakyReLU(a^T · [W_Q z_t^i ‖ W_K z_t^j ‖ γ(t)])
α_{ij} = exp(e_{ij} + λ·A_{ij}) / Σ_k exp(e_{ik} + λ·A_{ik})
```

**Why GAT attention bias A ≠ raw Jacobian (addressing FlowGRN-Tong's finding):**

FlowGRN-Tong showed that extracting GRN from the Jacobian ∇v*θ fails because indirect effects propagate through the chain rule. In our architecture, A enters the velocity field *only* through the attention bias — it controls the *gating* of which genes' messages reach gene i, not the content of the messages themselves. The Jacobian ∂v_i/∂x_j captures both the attention gating (A) and the MLP transformations (indirect effects). But A itself captures only the *direct regulatory gating* — gene j's message reaches gene i with weight proportional to A*{ij}, regardless of what multi-hop paths exist through the MLPs.

This is analogous to how in a transformer, the attention matrix captures direct token-to-token relationships even though the full Jacobian includes residual connections, layer norms, and FFN interactions. We read off A (the attention prior), not the full Jacobian.

We include a direct ablation (Experiment 2) comparing: (a) A extraction, (b) Jacobian extraction, and (c) hybrid trajectory→dynGENIE3 extraction to validate this claim.

**C) OT-CFM Training with Dropout-Robust Coupling**

We adopt and extend FlowGRN-Tong's dropout-robust similarity measure as our OT cost function:

```
d_raw(x, y) = (1/|S_{x,y}|) · Σ_{i ∈ S_{x,y}} |x_i - y_i|
```

where S\_{x,y} = {i : x_i ≠ 0 and y_i ≠ 0}. We then compute geodesic distances d_knn on a kNN graph built with d_raw. The OT plan π uses d_knn as the cost function for minibatch coupling.

**Training loss:**

```
L = L_CFM + α·‖A‖₁ + β·L_DAG
```

where L_CFM uses the OT-coupled conditional flow matching objective.

**D) GRN Extraction: Two Pathways**

Given FlowGRN-Tong's finding that raw Jacobian extraction underperforms, we provide two extraction pathways:

**Pathway 1 — Direct A extraction (primary):**

- Edge score(j → i) = |A\_{ij}|
- sign(A\_{ij}) indicates activation (+) or repression (−)
- This is _not_ the same as Jacobian extraction — A is the attention bias, not the full Jacobian

**Pathway 2 — Hybrid trajectory→dynGENIE3 (alternative):**

- Integrate the learned velocity field to reconstruct trajectories (as in FlowGRN-Tong)
- Feed trajectories to dynGENIE3 for GRN extraction
- This pathway leverages our shared-private flow matching for trajectory quality while using the proven dynGENIE3 extraction
- In the transfer setting: shared velocity field produces better trajectories (learned from multiple datasets) → better dynGENIE3 input

We compare both pathways in experiments to determine which is superior.

### 1.3 Primary Innovation: Transfer Learning via Shared-Private Decomposition

**This is the central contribution.** No existing method — RegDiffusion, FlowGRN-Tong, DeepSEM, DAZZLE, GENIE3, or any other — supports transfer of regulatory dynamics across datasets.

#### Biological Rationale

Gene regulation has two layers:

1. **Conserved regulatory grammar** — _how_ TFs bind DNA, _how_ signals propagate — broadly shared across cell types and partially across species
2. **Context-specific regulatory wiring** — _which_ TFs are active, _which_ edges are on — differs between cell types

LatentFlowGRN separates these:

| Component                          | What it captures                      | Shared or private |
| ---------------------------------- | ------------------------------------- | ----------------- |
| Encoder (MLP_enc)                  | Gene expression → latent mapping      | **Shared**        |
| Velocity MLPs (MLP_msg, MLP_self)  | How regulatory signals propagate      | **Shared**        |
| Attention parameters (W_Q, W_K, a) | How to compute influence strength     | **Shared**        |
| Adjacency matrix A_k               | Which edges are active in cell type k | **Private**       |
| Dataset embedding e_k (optional)   | Cell-type context                     | **Private**       |

**Why RegDiffusion and FlowGRN-Tong cannot do this:**

- **RegDiffusion:** MLP blocks with A as linear mixing — weights and A are entangled, not cleanly separable
- **FlowGRN-Tong:** GRN extraction happens in dynGENIE3, completely separate from the flow model — there's nothing to share. The flow model learns trajectories but doesn't parameterize regulatory structure

#### Transfer Variants

**T1: Joint Multi-Task Training**

```
L_total = Σ_k L_CFM^{(k)}(θ_shared, A_k) + α·Σ_k ‖A_k‖₁
```

**T2: Pretrain-then-Finetune** — Pretrain θ_shared on K datasets, freeze, train new A_target on target data.

**T3: Cross-Species Transfer** — Train on mouse, transfer to human via ortholog mapping O. Warm-start: A_human ← O·A_mouse·Oᵀ.

**T4: Foundation Model Backbone** — Use frozen scGPT/Geneformer encoder as shared backbone.

**T5: Hybrid Transfer (incorporating FlowGRN-Tong's insight)**

- Use shared-private flow matching for trajectory reconstruction across datasets
- The shared velocity field, trained on multiple datasets, produces higher-quality trajectories than single-dataset training
- Feed per-dataset trajectories to dynGENIE3 for GRN extraction
- This variant tests whether transfer helps _even with the proven dynGENIE3 extraction_

### 1.4 Architecture Comparison

| Feature                   | DeepSEM/DAZZLE     | RegDiffusion      | FlowGRN (Tong)            | **LatentFlowGRN (ours)**                |
| ------------------------- | ------------------ | ----------------- | ------------------------- | --------------------------------------- |
| Generative backbone       | VAE                | DDPM              | [SF]²M                    | OT-CFM                                  |
| GRN in model?             | Yes (A in enc/dec) | Yes (A in MLP)    | No (post-hoc dynGENIE3)   | **Yes (A as GAT bias)**                 |
| GRN extraction            | \|A\|              | \|A\|             | dynGENIE3 on trajectories | **\|A\| or hybrid traj→dynGENIE3**      |
| Jacobian for GRN?         | N/A                | N/A               | Tested, rejected          | Addressed via attention bias ≠ Jacobian |
| Matrix inversion          | Yes, O(g³)         | No                | No                        | **No**                                  |
| Dropout handling          | DA augmentation    | Implicit          | **Dropout-robust d_raw**  | **Adopts d_raw for OT coupling**        |
| Velocity field / dynamics | No                 | No                | **Yes**                   | **Yes**                                 |
| OT coupling               | No                 | No                | **Yes** (dropout-robust)  | **Yes** (dropout-robust)                |
| Cross-dataset transfer    | None               | None              | None                      | **Yes (primary contribution)**          |
| Few-shot GRN              | No                 | No                | No                        | **Yes**                                 |
| Cross-species transfer    | No                 | No                | No                        | **Yes**                                 |
| Scalability               | O(g³n)             | O(g²), 40k+ genes | ~1,783 genes              | O(g·k) sparse attention, TBD            |

---

## 2. Mathematical Formulation

### 2.1 Conditional Flow Matching: Foundations

A CNF defines a velocity field v_θ: ℝ^d × [0,1] → ℝ^d generating a flow via:

```
dψ_t(x)/dt = v_θ(ψ_t(x), t),    ψ_0(x) = x
```

The CFM loss conditions on endpoints:

```
L_CFM = E_{t~U[0,1], z~q(z), x~p_t(·|z)} ‖v_θ(x, t) - u_t(x | z)‖²
```

### 2.2 OT-CFM with Dropout-Robust Coupling

Standard minibatch OT uses Euclidean distance. Following FlowGRN-Tong, we replace it with the dropout-robust geodesic distance d_knn:

1. Compute d*raw(x, y) = (1/|S*{x,y}|)·Σ*{i∈S*{x,y}} |x_i − y_i| for each cell pair
2. Build kNN graph G_knn with d_raw
3. Compute d_knn as shortest-path distance on G_knn (Dijkstra)
4. Solve minibatch OT with d_knn as cost: σ\* = argmin Σ_i d_knn(x_0^i, x_1^{σ(i)})

Conditional paths:

```
x_t = (1-t)·x_0 + t·x_1,     u_t = x_1 - x_0
```

### 2.3 GRN-Structured Velocity Field

```
v_i(z_t, t) = Σ_j α_{ij} · MLP_msg(z_t^j, t) + MLP_self(z_t^i, t)
```

Attention with adjacency bias:

```
e_{ij} = LeakyReLU(a^T · [W_Q z_t^i ‖ W_K z_t^j ‖ γ(t)])
α_{ij} = exp(e_{ij} + λ·A_{ij}) / Σ_k exp(e_{ik} + λ·A_{ik})
```

Full single-dataset loss:

```
L = L_CFM + α·‖A‖₁ + β·R(A)
```

### 2.4 Multi-Task Flow Matching for Transfer Learning

Decompose parameters: θ_shared (encoder, MLPs, attention params) vs θ_private (A_k, e_k per dataset).

**Multi-task loss:**

```
L_total = Σ_k w_k · L^{(k)}(θ_shared, A_k, e_k) + α·Σ_k ‖A_k‖₁ + β·Σ_k R(A_k)
```

**Gradient isolation:** ∂L^{(k)}/∂A_j = 0 for j ≠ k. Each A_k learns only from its own data; θ_shared learns from all.

**Transfer to new dataset:** Fix θ_shared\*, optimize A_target only.

**Cross-species:** Ortholog mapping O, warm-start A_human ← O·A_mouse·Oᵀ.

### 2.5 GRN Extraction

**Pathway 1 — Direct A:** Edge(j→i) = |A*{ij}|, sign = sign(A*{ij}).

**Pathway 2 — Hybrid trajectory→dynGENIE3:**

1. Integrate v_θ to reconstruct trajectories x(t) per cell
2. Feed to dynGENIE3: dx_j/dt = f_j(x(t)) − α_j·x_j(t)
3. Use random forest importance I_j(i) as edge scores

**Pathway 3 — Jacobian (included for ablation only):**
J\_{ij}(x\*,t) = ∂v_i/∂x_j. Included to replicate FlowGRN-Tong's finding and validate that Pathway 1 avoids the indirect-effect problem.

**Transfer note:** In Pathway 2, the shared velocity field (trained on multiple datasets) should produce better trajectories than single-dataset training → better dynGENIE3 input → better GRN. This is tested in Experiment 5.

### 2.6 Biological Interpretation

- **v_i(x, t) > 0:** Gene i upregulated at state x
- **α\_{ij} large:** Gene j strongly gates influence on gene i → direct regulatory edge
- **Shared MLPs:** Universal regulatory propagation logic
- **A_k comparison across datasets:** Conserved (all A_k) vs context-specific (unique to one A_k) edges

---

## 3. Experimental Setup

### 3.1 BEELINE Benchmark

**Synthetic (6):** dyn-LI, CY, LL, BF, TF, BFC. Each 100-5,000 cells × 10 samples.

**Curated (4):** HSC, mCAD, VSC, GSD. 2,000 cells × 10 samples + dropout variants.

**Experimental (7):** hESC, hHEP, mDC, mESC, mHSC-E/GM/L.

**Ground truths:** STRING, non-specific ChIP-seq, cell-type-specific ChIP-seq, LOF/GOF. Additionally, we follow FlowGRN-Tong's updated reference networks from **DoRothEA** and **CollecTRI** for more current ground truth.

### 3.2 Evaluation Metrics

- **AUPRC** (ratio, primary)
- **EPR** (Early Precision Ratio)
- **AUROC** (secondary)

### 3.3 Baselines

**Tier 1: Direct competitors (generative + GRN, recent)**

- **RegDiffusion** (DDPM + parameterized A) — J. Comp. Biol. 2024
- **FlowGRN-Tong** (CFM + dynGENIE3) — ACM BCB 2025
- DeepSEM, DAZZLE, GRN-VAE, HyperG-VAE

**Tier 2: Classical methods**

- GENIE3, GRNBoost2, dynGENIE3, PIDC, PPCOR, LEAP

**Tier 3: Recent SOTA**

- GRNFormer, scKAN, scRegNet, GRANGER

**Tier 4: Transfer-aware methods**

- GRNPT, Meta-TGLink, scMTNI, LINGER

### 3.4 Experimental Protocol

#### Experiment 1: BEELINE Benchmark — Single-Dataset

Standard evaluation. LatentFlowGRN-Solo (Pathway 1: A extraction) vs all baselines. 10 seeds.

Key head-to-head comparisons:

- vs RegDiffusion: same strategy (parameterized A in generative model), different backbone
- vs FlowGRN-Tong: same backbone family (flow matching), different GRN extraction strategy

#### Experiment 2: GRN Extraction Ablation (Critical — Addresses FlowGRN-Tong's Finding)

Compare three extraction pathways from the _same trained model_:

- **Pathway 1:** Direct A extraction (our primary method)
- **Pathway 2:** Hybrid trajectory→dynGENIE3 (FlowGRN-Tong's approach applied to our model)
- **Pathway 3:** Raw Jacobian ∇v_θ (expected to underperform, per FlowGRN-Tong's finding)

If Pathway 1 ≈ Pathway 2 >> Pathway 3, this validates our attention-bias-A design.
If Pathway 2 > Pathway 1 >> Pathway 3, dynGENIE3 extraction is superior and we adopt Pathway 2 as default.

#### Experiment 3: Architecture Ablations

- **Generative backbone:** OT-CFM vs DDPM (RegDiffusion-style) vs VAE (DeepSEM-style) — all with same GAT network
- **Velocity field:** GAT vs MLP (RegDiffusion-style) vs MLP (FlowGRN-Tong-style)
- **OT coupling:** Euclidean vs dropout-robust d_knn (FlowGRN-Tong's measure) vs Sinkhorn
- **Latent vs gene space**
- **With/without dropout augmentation (DAZZLE)**

#### Experiment 4: External Biological Validation

- ChIP-seq validation (ENCODE)
- Perturbation prediction (Perturb-seq/CRISPRi)

#### Experiment 5: Multi-Task Joint Training (Transfer — Primary Contribution)

**Setup:**

- LatentFlowGRN-Solo: Independent per dataset
- LatentFlowGRN-Joint: All 7 datasets, shared θ, per-dataset A_k (Pathway 1 extraction)
- LatentFlowGRN-Joint-Hybrid: Same shared θ, but trajectories→dynGENIE3 (Pathway 2 extraction)
- RegDiffusion-Solo, FlowGRN-Tong-Solo: Transfer-unaware controls

**Evaluation:** Per-dataset AUPRC. Does sharing help? Which extraction pathway benefits more from transfer?

#### Experiment 6: Few-Shot GRN Inference

**Leave-One-Out:** Pretrain on 6 datasets → finetune A_k on held-out dataset.

**Data Titration:** {100%, 50%, 20%, 10%, 5%} cells. Compare:

- LatentFlowGRN-Transfer (pretrained θ_shared)
- LatentFlowGRN-Solo (from scratch)
- RegDiffusion-Solo
- FlowGRN-Tong-Solo

Expected crossing point: transfer matches full-data solo at ~20% data.

#### Experiment 7: Cross-Species Transfer (Mouse → Human)

Train on 5 mouse datasets → transfer to human (hESC, hHEP). Warm-start vs cold-start vs solo baselines.

#### Experiment 8: Analysis of Shared vs Specific Regulation

Conserved edges, context-specific edges, latent space visualization, shared MLP inspection.

#### Experiment 9: Scalability

Runtime/memory vs genes and cells. Compare against:

- RegDiffusion (benchmark: 40k+ genes, <5 min)
- FlowGRN-Tong (benchmark: ~1,783 genes, <10 min)
- DeepSEM, GENIE3, GRNBoost2

### 3.5 Implementation Plan

**Libraries:** PyTorch, TorchCFM, PyTorch Geometric, Scanpy, POT, pybiomart, dynGENIE3 (R/Python wrapper)

**Code structure:**

```
latentflowgrn/
├── models/
│   ├── encoder.py              # Per-gene MLP encoder (shared)
│   ├── velocity_field.py       # GAT-based velocity network
│   ├── flow_matching.py        # OT-CFM with dropout-robust coupling
│   ├── grn_extraction.py       # Pathway 1 (A), 2 (traj→dynGENIE3), 3 (Jacobian)
│   └── transfer.py             # Multi-task trainer, freeze/finetune
├── data/
│   ├── beeline_loader.py       # BEELINE datasets
│   ├── dropout_similarity.py   # FlowGRN-Tong's d_raw and d_knn
│   ├── multitask_loader.py     # Cross-dataset batching
│   └── ortholog_mapper.py      # Cross-species mapping
├── evaluation/
│   ├── metrics.py              # AUPRC, EPR, AUROC
│   └── transfer_analysis.py    # Conserved vs specific edges
├── baselines/
│   ├── run_regdiffusion.py     # RegDiffusion comparison
│   ├── run_flowgrn_tong.py     # FlowGRN-Tong comparison
│   └── run_beeline.sh          # BEELINE Docker baselines
└── experiments/
    ├── exp1_benchmark.py       # Single-dataset BEELINE
    ├── exp2_extraction.py      # A vs traj→dynGENIE3 vs Jacobian
    ├── exp3_ablations.py       # Architecture ablations
    ├── exp4_bio_validation.py  # ChIP-seq, perturbation
    ├── exp5_joint_training.py  # Multi-task transfer
    ├── exp6_few_shot.py        # Leave-one-out, data titration
    ├── exp7_cross_species.py   # Mouse → human
    ├── exp8_shared_analysis.py # Conserved vs specific
    └── exp9_scalability.py     # Runtime benchmarks
```

### 3.6 Expected Outcomes and Risk Mitigation

**What success looks like:**

_Single-dataset (supporting evidence):_

- Competitive with RegDiffusion and FlowGRN-Tong on BEELINE
- Pathway 1 (A extraction) validates that GAT attention bias avoids FlowGRN-Tong's Jacobian problem
- If Pathway 2 is stronger, adopt hybrid approach — this is still a contribution (shared flow + proven extraction)

_Transfer (primary contribution):_

- Joint training matches or beats solo on majority of datasets
- Few-shot: pretrained LatentFlowGRN at 20% data matches solo at 100%
- Cross-species transfer outperforms solo when target data is limited
- Biologically meaningful conserved vs context-specific edges

**Risks and mitigations:**

| Risk                                                                               | Mitigation                                                                                                                                                                                                                            |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LatentFlowGRN-Solo doesn't beat RegDiffusion or FlowGRN-Tong                       | Acceptable — primary contribution is transfer. Competitive single-dataset + novel transfer is the story.                                                                                                                              |
| Pathway 1 (A extraction) underperforms, confirming FlowGRN-Tong's Jacobian concern | Switch to Pathway 2 (hybrid) as default. The transfer contribution is independent of extraction method.                                                                                                                               |
| Negative transfer in joint training                                                | Report honestly. Analyze which dataset pairs help/hurt.                                                                                                                                                                               |
| Cross-species transfer fails                                                       | Same-species transfer as control. Within-species is still novel.                                                                                                                                                                      |
| Scalability worse than RegDiffusion (40k+ genes)                                   | Focus on moderate scale (1k-5k genes) where biological validation is strongest. Use sparse attention to improve.                                                                                                                      |
| Reviewers say "just FlowGRN-Tong + transfer"                                       | Emphasize: (1) parameterized A in velocity field is architecturally distinct from post-hoc dynGENIE3; (2) GAT vs MLP is a real architectural difference; (3) latent space operation; (4) transfer is a major standalone contribution. |
| Name confusion with FlowGRN-Tong                                                   | LatentFlowGRN is distinct — "Latent" signals latent-space operation and transfer capability. Always cite FlowGRN-Tong explicitly.                                                                                                     |

---

## 4. Novelty Claims and Related Work Positioning

### Contribution hierarchy:

**Primary: Transferable GRN inference via shared-private decomposition.** No existing method supports transfer of regulatory dynamics. LatentFlowGRN is the first unsupervised GRN method to share regulatory dynamics across cell types and species.

**Secondary: End-to-end GRN learning in a flow matching model with GAT-structured velocity field.** FlowGRN-Tong uses flow matching for trajectory reconstruction but extracts GRN post-hoc with dynGENIE3 (Strategy C). RegDiffusion embeds A in a diffusion model (Strategy B). LatentFlowGRN uniquely combines flow matching (from C) with embedded A (from B) via a GAT architecture that cleanly separates shared dynamics from dataset-specific wiring.

**Tertiary: Dropout-robust OT coupling in a latent flow matching framework.** We adopt and integrate FlowGRN-Tong's dropout-robust similarity into OT-CFM, extending it to latent-space operation.

### Detailed positioning:

1. **vs FlowGRN-Tong (most architecturally similar):** FlowGRN-Tong uses CFM for trajectory reconstruction and dynGENIE3 for GRN extraction — a two-stage pipeline where the GRN is not embedded in the generative model. LatentFlowGRN embeds A directly in the GAT velocity field, enabling end-to-end GRN learning. More critically, the two-stage design of FlowGRN-Tong cannot support transfer: dynGENIE3 runs independently per dataset with no shared parameters. LatentFlowGRN's shared-private architecture is only possible because A is part of the neural network. We also provide a hybrid variant (Pathway 2 / T5) that combines our shared flow matching with dynGENIE3, testing whether transfer helps even with their extraction method.

2. **vs RegDiffusion:** RegDiffusion embeds A in DDPM with MLP blocks. LatentFlowGRN embeds A in OT-CFM with GAT attention. Key differences: (a) OT-structured transport vs standard noise schedules; (b) GAT enables clean shared-private decomposition that RegDiffusion's entangled MLP-A cannot support; (c) velocity field interpretation; (d) dropout-robust OT coupling.

3. **vs DeepSEM/DAZZLE:** Both RegDiffusion and LatentFlowGRN improve on the VAE backbone. LatentFlowGRN additionally enables transfer.

4. **vs GRNPT / Meta-TGLink:** Supervised methods for cross-cell-type generalization. LatentFlowGRN is fully unsupervised.

5. **vs LINGER / scMTNI:** LINGER uses external bulk data. scMTNI shares across a single lineage. LatentFlowGRN shares across arbitrary dataset collections, species, and conditions.

### Recommended paper framing:

"FlowGRN-Tong recently demonstrated that flow matching can reconstruct cellular trajectories to improve GRN inference, and RegDiffusion showed that embedding a parameterized adjacency matrix in a diffusion model enables fast, accurate GRN learning. We unify these insights — embedding a learnable adjacency in a flow matching velocity field via graph attention — and show that this architecture uniquely enables a shared-private decomposition for cross-dataset transfer learning, the first of its kind for unsupervised GRN inference."

### Target venues:

- Nature Methods, Genome Biology (full scope with transfer + biology)
- Bioinformatics, Briefings in Bioinformatics (method + transfer)
- NeurIPS / ICML (ML angle: transferable generative models for biological networks)
- ACM BCB (direct venue match with FlowGRN-Tong, could be a strong follow-up)
