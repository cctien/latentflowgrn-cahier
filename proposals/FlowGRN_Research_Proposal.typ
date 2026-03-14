// FlowGRN Research Proposal — Typst source
#set document(title: "FlowGRN: Latent Flow Matching for Gene Regulatory Network Inference", author: "Research Proposal")
#set page(margin: (x: 1.2in, y: 1in), numbering: "1")
#set text(font: "Libertinus Serif", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  v(0.8em)
  text(size: 16pt, weight: "bold", it)
  v(0.4em)
}
#show heading.where(level: 2): it => {
  v(0.6em)
  text(size: 13pt, weight: "bold", it)
  v(0.3em)
}
#show heading.where(level: 3): it => {
  v(0.4em)
  text(size: 11.5pt, weight: "bold", it)
  v(0.2em)
}

// Title block
#align(center)[
  #text(size: 22pt, weight: "bold")[FlowGRN: Latent Flow Matching for\ Gene Regulatory Network Inference]
  #v(0.6em)
  #text(size: 13pt, style: "italic")[A Detailed Research Proposal]
  #v(0.3em)
  #line(length: 60%, stroke: 0.5pt + gray)
]

#v(1em)

= Architecture Design

== Background: The DeepSEM/DAZZLE Paradigm

The current leading deep learning approach on BEELINE is based on the *Structural Equation Model (SEM)* framework, used by DeepSEM and its successor DAZZLE.

*Linear SEM assumption:*
$ bold(X) = bold(X) bold(A)^top + bold(Z) $
where $bold(X) in RR^(c times g)$ is the gene expression matrix ($c$ cells, $g$ genes), $bold(A) in RR^(g times g)$ is the weighted adjacency matrix (the GRN to infer), and $bold(Z)$ is a random noise term.

Rearranging into a VAE:
$ "Encoder:" quad bold(Z) = bold(X)(bold(I) - bold(A)^top) quad quad quad "Decoder:" quad bold(X) = bold(Z)(bold(I) - bold(A)^top)^(-1) $

The adjacency matrix $bold(A)$ is a learnable parameter, jointly optimized with the encoder/decoder networks via:
$ cal(L) = -EE_(bold(z) tilde q(bold(Z)|bold(X)))[log p(bold(X)|bold(Z))] + beta dot D_"KL"(q(bold(Z)|bold(X)) || p(bold(Z))) + alpha dot ||bold(A)||_1 $

The $ell_1$ penalty on $bold(A)$ encourages sparsity (real GRNs are sparse).

*Key limitations:*
+ *Training instability:* DeepSEM's inferred networks degrade as training continues past convergence, likely due to overfitting dropout noise.
+ *Matrix inversion:* The $(bold(I) - bold(A)^top)^(-1)$ operation is numerically unstable and computationally expensive.
+ *Static reconstruction:* The model reconstructs static snapshots; it doesn't model the _dynamics_ that generate gene expression.
+ *Undirected edges in practice:* While $bold(A)$ is asymmetric in principle, the model averages $|bold(A)|$ over multiple runs, partially losing directionality.
+ *No knowledge sharing:* Every dataset is trained from scratch---no transfer of regulatory logic across cell types, tissues, or species.


== Proposed Architecture: FlowGRN

We propose replacing the VAE generative backbone with *conditional flow matching (CFM)*, while preserving the core insight of embedding the GRN structure into the generative model. Critically, the latent-space formulation enables a natural decomposition into *shared* (transferable) and *dataset-specific* (GRN) components.

=== Encoder: Per-Gene MLP with Shared Weights

Following DeepSEM's design, a small MLP processes one gene at a time with shared weights:
$ h_i = "MLP"_"enc"(x_i) quad "for each gene" i in {1, dots, g} $
$ bold(z) = [h_1, h_2, dots, h_g] in RR^(c times g times d) $

The shared-weight design ensures gene-gene interactions are _not_ embedded in the encoder---they are exclusively captured by the adjacency structure in the velocity field. Alternatively, a *pretrained encoder* from scGPT or Geneformer can be used. The encoder weights $theta_"enc"$ are *shared across all datasets* in the multi-task/transfer setting.

=== GRN-Structured Velocity Field (Core Innovation)

The velocity field $v_theta (bold(z)_t, t)$ is parameterized as a *Graph Attention Network* operating over genes. For gene $i$ at flow time $t$:

$ v_i (bold(z)_t, t) = sum_(j=1)^g alpha_(i j) dot "MLP"_"msg"(z_t^j, t) + "MLP"_"self"(z_t^i, t) $

The attention weights incorporate a learnable adjacency bias:
$ e_(i j) = "LeakyReLU"(bold(a)^top dot [bold(W) z_t^i || bold(W) z_t^j || gamma(t)]) $
$ alpha_(i j) = frac(exp(e_(i j) + lambda dot A_(i j)), sum_k exp(e_(i k) + lambda dot A_(i k))) $

where:
- $bold(A) in RR^(g times g)$ is the *learnable adjacency matrix* (the GRN),
- $lambda$ controls the strength of the adjacency prior vs.\ data-driven attention,
- $gamma(t)$ is a sinusoidal time embedding.

*Key insight:* The velocity at gene $i$ depends on other genes $j$ weighted by $alpha_(i j)$. High $|A_(i j)|$ means gene $j$ has strong regulatory influence on gene $i$'s dynamics.

=== Flow Matching Training

Instead of the VAE's reconstruction + KL loss, we train with the *conditional flow matching (CFM) objective*:
$ cal(L)_"CFM" = EE_(t, bold(x)_0, bold(x)_1) ||v_theta (bold(x)_t, t) - u_t (bold(x)_t | bold(x)_0, bold(x)_1)||^2 $

where $bold(x)_0 tilde p_"noise"$ (e.g., standard Gaussian), $bold(x)_1 tilde p_"data"$ (scRNA-seq expression vectors), $bold(x)_t = (1-t) dot bold(x)_0 + t dot bold(x)_1$ (linear interpolation), and $u_t (bold(x)_t | bold(x)_0, bold(x)_1) = bold(x)_1 - bold(x)_0$ (target velocity for the linear path).

=== Full Loss Function

$ cal(L) = cal(L)_"CFM" + alpha dot ||bold(A)||_1 + beta dot cal(L)_"DAG" $

where $cal(L)_"DAG" = tr(e^(bold(A) circle.tiny bold(A))) - g$ is an optional acyclicity constraint (from NOTEARS) that penalizes cyclic structures, and $circle.tiny$ denotes the Hadamard product.

=== GRN Extraction

After training: (1) $|A_(i j)|$ gives regulatory strength of gene $j arrow.r$ gene $i$; (2) $"sign"(A_(i j))$ indicates activation ($+$) or repression ($-$); (3) rank all gene pairs by $|A_(i j)|$ for evaluation.


== Variant: Latent Flow Matching

For better noise handling, operate the flow in a compressed latent space:

+ *Pretrain an autoencoder* on scRNA-seq data: Encoder $cal(E): RR^g arrow.r RR^d$ ($d << g$), Decoder $cal(D): RR^d arrow.r RR^g$.
+ *Run flow matching in latent space:* $bold(z)_0 tilde cal(N)(0, bold(I)_d)$, $bold(z)_1 = cal(E)(bold(x))$, train velocity field on $bold(z)_t$ trajectories.
+ *Extract GRN from velocity field Jacobian:* Compute $frac(partial v_i, partial z_j)$ at representative points, then map back to gene space via the decoder Jacobian $frac(partial bold(x), partial bold(z))$.


== Extension: Transfer Learning via Shared-Private Decomposition

The latent-space formulation enables a natural decomposition into *shared* (transferable) and *private* (dataset-specific) components. Current GRN methods (DeepSEM, DAZZLE, GENIE3, GRNBoost2) all train from scratch on each dataset independently.

=== Biological Rationale

Gene regulation has two distinct layers:
+ *Conserved regulatory grammar*---the _mechanism_ by which TFs bind DNA, how signaling cascades propagate, how chromatin remodeling works. Broadly shared across cell types and partially conserved across species.
+ *Context-specific regulatory wiring*---_which_ TFs are active, _which_ enhancers are accessible, _which_ feedback loops are engaged. These differ between hESC, mDC, and hHEP.

FlowGRN's architecture naturally separates these:

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    inset: 6pt,
    [*Component*], [*What it captures*], [*Shared / Private*],
    [Encoder ($"MLP"_"enc"$)], [Gene expression $arrow.r$ latent mapping], [*Shared*],
    [Velocity MLPs ($"MLP"_"msg"$, $"MLP"_"self"$)], [How regulatory signals propagate], [*Shared*],
    [Attention parameters ($bold(W)_Q$, $bold(W)_K$, $bold(a)$)], [How to compute influence strength], [*Shared*],
    [Adjacency matrix $bold(A)_k$], [Which edges are active in cell type $k$], [*Private*],
    [Dataset embedding $bold(e)_k$ (optional)], [Cell-type-specific context], [*Private*],
  ),
  caption: [Shared-private decomposition of FlowGRN parameters.]
)

=== Transfer Variants

*Variant T1: Joint Multi-Task Training.* Train one FlowGRN on all $K$ datasets simultaneously:
$ cal(L)_"total" = sum_(k=1)^K w_k dot cal(L)_"CFM"^((k))(theta_"shared", bold(A)_k, bold(e)_k) + alpha dot sum_k ||bold(A)_k||_1 $
where $theta_"shared" = {theta_"enc", theta_"MLP_msg", theta_"MLP_self", bold(W)_Q, bold(W)_K, bold(a)}$ is trained on all datasets, and each $bold(A)_k$ is trained only on dataset $k$.

*Variant T2: Pretrain-then-Finetune.* Stage 1: train jointly on $K$ datasets $arrow.r$ get $theta_"shared"^*$. Stage 2: for a new target dataset, freeze $theta_"shared"^*$ and only train $bold(A)_"target"$:
$ bold(A)_"target"^* = arg min_(bold(A)_"target") cal(L)^("target")(theta_"shared"^*, bold(A)_"target") + alpha dot ||bold(A)_"target"||_1 $

*Variant T3: Cross-Species Transfer.* For organisms with ortholog mappings (e.g., mouse $arrow.l.r$ human), introduce an ortholog matrix $bold(O) in {0,1}^(g_h times g_m)$ where $O_(i j) = 1$ if human gene $i$ is orthologous to mouse gene $j$. Warm-start:
$ bold(A)_"human" arrow.l bold(O) dot bold(A)_"mouse" dot bold(O)^top $
then finetune $bold(A)_"human"$ on human data with $theta_"shared"$ frozen.

*Variant T4: Foundation Model Backbone.* Use a frozen pretrained encoder from scGPT (33M cells) or Geneformer (30M cells). Only train the velocity field (shared MLPs + per-dataset $bold(A)_k$).

=== Dataset Conditioning

Optionally condition the velocity field on dataset identity:
$ v_i (bold(z)_t, t; bold(A)_k, bold(e)_k) = sum_j alpha_(i j)^((k)) dot "MLP"_"msg"([bold(z)_t^j; gamma(t); bold(e)_k]) + "MLP"_"self"([bold(z)_t^i; gamma(t); bold(e)_k]) $
where $bold(e)_k in RR^(d_e)$ is a learnable embedding for dataset $k$, and $[; ]$ denotes concatenation.


== Architecture Comparison

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    inset: 5pt,
    [*Feature*], [*DeepSEM/DAZZLE*], [*GRNFormer*], [*FlowGRN*], [*FlowGRN + Transfer*],
    [Generative model], [VAE], [N/A (discrim.)], [Flow Matching], [Flow Matching],
    [GRN param.], [Adj. in enc/dec], [Graph transf.], [Adj-biased GAT], [Per-dataset $bold(A)_k$],
    [Training obj.], [Recon. + KL], [Supervised], [CFM (unsup.)], [Multi-task CFM],
    [Dynamics], [No (static)], [No], [Yes (vel. field)], [Yes],
    [Directionality], [Weak ($|bold(A)|$ avg)], [TF-subgraphs], [Strong], [Strong],
    [Cross-dataset], [None], [None], [None], [*Yes*],
    [Few-shot GRN], [No], [Needs labels], [No], [*Yes*],
    [Cross-species], [No], [No], [No], [*Yes*],
  ),
  caption: [Comparison of FlowGRN with existing GRN inference methods.]
)


= Mathematical Formulation

== Conditional Flow Matching: Foundations

A *continuous normalizing flow (CNF)* defines a time-dependent velocity field $v_theta: RR^d times [0,1] arrow.r RR^d$ generating a flow $psi_t$ via the ODE:
$ frac(d psi_t (bold(x)), d t) = v_theta (psi_t (bold(x)), t), quad quad psi_0(bold(x)) = bold(x) $

The flow transports a source distribution $p_0$ to a target distribution $p_1$: $p_t = (psi_t)_(\#) p_0$.

*Flow Matching Objective* (Lipman et al., 2022):
$ cal(L)_"FM" = EE_(t tilde cal(U)[0,1], med bold(x) tilde p_t) ||v_theta (bold(x), t) - u_t (bold(x))||^2 $

This is intractable since $p_t$ and $u_t$ are unknown. The key insight is to condition on endpoints.

*Conditional Flow Matching (CFM):* Define conditional probability paths $p_t (bold(x) | bold(z))$ with known conditional velocity fields $u_t (bold(x) | bold(z))$ such that $p_t (bold(x)) = integral p_t (bold(x) | bold(z)) q(bold(z)) d bold(z)$. Then:
$ cal(L)_"CFM" = EE_(t tilde cal(U)[0,1], med bold(z) tilde q(bold(z)), med bold(x) tilde p_t (dot | bold(z))) ||v_theta (bold(x), t) - u_t (bold(x) | bold(z))||^2 $

has the same gradients as $cal(L)_"FM"$ with respect to $theta$.


== OT-CFM with Minibatch Couplings

*Standard CFM* uses independent coupling: $bold(z) = (bold(x)_0, bold(x)_1)$ sampled independently. This leads to crossing paths and higher training variance.

*OT-CFM* uses an optimal transport coupling:
$ pi^* = arg min_(pi in Pi(p_0, p_1)) integral ||bold(x)_0 - bold(x)_1||^2 d pi(bold(x)_0, bold(x)_1) $

In practice, *minibatch OT* approximates this. Given a batch of $B$ noise samples ${bold(x)_0^i}_(i=1)^B$ and $B$ data samples ${bold(x)_1^j}_(j=1)^B$, solve the assignment:
$ sigma^* = arg min_(sigma in S_B) sum_(i=1)^B ||bold(x)_0^i - bold(x)_1^(sigma(i))||^2 $
via the Hungarian algorithm ($cal(O)(B^3)$) or Sinkhorn approximation.

The conditional paths for OT-CFM are:
$ bold(x)_t = (1-t) dot bold(x)_0 + t dot bold(x)_1 quad quad quad u_t (bold(x) | bold(x)_0, bold(x)_1) = bold(x)_1 - bold(x)_0 $

*Why this matters for GRN inference:* OT coupling produces straighter, non-crossing flows. Since gene expression distributions have complex multimodal structure (different cell types), OT coupling ensures the velocity field captures biologically meaningful transitions rather than arbitrary noise-to-data mappings.


== Applying CFM to GRN Inference

*Setup for single-cell data:*
- Source distribution $p_0$: Standard Gaussian $cal(N)(0, bold(I)_g)$
- Target distribution $p_1$: Empirical distribution of scRNA-seq expression vectors
- Each sample $bold(x) in RR^g$ is a cell's expression profile across $g$ genes

*GRN-structured velocity field:*
$ v_(theta, i) (bold(x)_t, t) = sum_(j in cal(N)(i)) alpha_(i j)(bold(x)_t, t) dot phi(x_t^j, t) + psi(x_t^i, t) $

where $cal(N)(i)$ is the neighborhood of gene $i$, $alpha_(i j)$ are attention weights with adjacency bias, and $phi, psi$ are shared MLPs.

*Attention mechanism with adjacency bias:*
$ e_(i j) = "LeakyReLU"(bold(a)^top dot [bold(W)_Q bold(x)_t^i || bold(W)_K bold(x)_t^j || gamma(t)]) $
$ alpha_(i j) = frac(exp(e_(i j) + lambda dot A_(i j)), sum_k exp(e_(i k) + lambda dot A_(i k))) $

*Full training loss (single-dataset):*
$ cal(L) = EE_(t tilde cal(U)[0,1], med (bold(x)_0, bold(x)_1) tilde pi_"OT") ||v_theta (bold(x)_t, t) - (bold(x)_1 - bold(x)_0)||^2 + alpha dot ||bold(A)||_1 + beta dot cal(R)(bold(A)) $

where $cal(R)(bold(A))$ is an optional structural regularizer:
- *Acyclicity (NOTEARS):* $cal(R)(bold(A)) = tr(e^(bold(A) circle.tiny bold(A))) - g$
- *Degree constraint:* $cal(R)(bold(A)) = sum_i max(0, sum_j |A_(i j)| - kappa)$
- *Prior incorporation:* $cal(R)(bold(A)) = ||bold(A) circle.tiny bold(M)||_F^2$ where $bold(M)$ is a mask from TF motif data


== Multi-Task Flow Matching for Transfer Learning

Given $K$ datasets ${cal(D)_1, dots, cal(D)_K}$, decompose parameters into:
- $theta_"shared" = {theta_"enc", theta_"MLP_msg", theta_"MLP_self", bold(W)_Q, bold(W)_K, bold(a), b_A, lambda}$
- $theta_"private" = {bold(A)_1, dots, bold(A)_K, bold(e)_1, dots, bold(e)_K}$

*Multi-task loss:*
$ cal(L)_"total" = sum_(k=1)^K w_k dot cal(L)^((k))(theta_"shared", bold(A)_k, bold(e)_k) + alpha dot sum_k ||bold(A)_k||_1 + beta dot sum_k cal(R)(bold(A)_k) $

where:
$ cal(L)^((k)) = EE_(t, med (bold(x)_0, bold(x)_1^((k))) tilde pi_"OT"^((k))) ||v_(theta_"shared") (bold(x)_t, t; bold(A)_k, bold(e)_k) - (bold(x)_1^((k)) - bold(x)_0)||^2 $

=== Gradient Flow Analysis

Gradients from $cal(L)^((k))$ flow into $theta_"shared"$ but are blocked from other datasets' adjacency matrices:
$ frac(partial cal(L)^((k)), partial theta_"shared") eq.not 0, quad quad frac(partial cal(L)^((k)), partial bold(A)_k) eq.not 0, quad quad frac(partial cal(L)^((k)), partial bold(A)_j) = 0 quad (j eq.not k) $

This ensures each $bold(A)_k$ reflects only dataset $k$'s GRN, while $theta_"shared"$ captures cross-dataset regulatory patterns.

=== Transfer to New Datasets

For a new dataset $cal(D)_"target"$:
$ bold(A)_"target"^* = arg min_(bold(A)_"target") cal(L)^("target")(theta_"shared"^*, bold(A)_"target", bold(e)_"target") + alpha dot ||bold(A)_"target"||_1 $
with $theta_"shared"^*$ frozen from pretraining.

=== Cross-Species Formulation

Introduce ortholog mapping $bold(O) in {0,1}^(g_h times g_m)$ ($O_(i j) = 1$ iff human gene $i$ is orthologous to mouse gene $j$). Warm-start the human adjacency:
$ bold(A)_"human"^("init") = bold(O) dot bold(A)_"mouse" dot bold(O)^top $
then finetune $bold(A)_"human"$ on human data with $theta_"shared"$ frozen.


== GRN Extraction and Scoring

*Method 1: Direct adjacency.* Edge score$(j arrow.r i) = |A_(i j)|$; edge sign$(j arrow.r i) = "sign"(A_(i j))$.

*Method 2: Jacobian analysis.* For representative cells $bold(x)^*$ (cluster centroids):
$ J_(i j)(bold(x)^*, t) = frac(partial v_i (bold(x)^*, t), partial x_j) $
$ "GRN"_(i j) = EE_(t, bold(x)^*) [|J_(i j)(bold(x)^*, t)|] $

*Method 3: Integrated Gradients.*
$ "IG"_(i j) = integral_0^1 frac(partial v_i (bold(x)_t, t), partial x_j) dot (x_1^j - x_0^j) med d t $

*Transfer-specific note:* In the multi-task setting, Methods 2 and 3 capture effects from both $bold(A)_k$ (dataset-specific) and $theta_"shared"$ (universal dynamics), potentially yielding more informative GRNs than the raw $bold(A)_k$ alone.


== Biological Interpretation of the Flow

- $v_i (bold(x), t) > 0$: gene $i$ is being upregulated at state $bold(x)$, flow time $t$
- $alpha_(i j)$ large: gene $j$ strongly influences gene $i$'s rate of change $arrow.r$ regulatory edge
- *Flow trajectories:* integrating $v_theta$ from noise to data traces cell state transitions
- *Cell-type-specific GRNs:* evaluate $J_(i j)(bold(x)^*, t)$ at different cell-type centroids
- *Shared vs specific regulation:* edges appearing in all $bold(A)_k$ are conserved; edges unique to one $bold(A)_k$ are context-specific


= Experimental Setup

== BEELINE Benchmark: Datasets and Ground Truths

*A) Synthetic Networks (6 datasets).* Generated via BoolODE: dyn-LI (linear), dyn-CY (cyclic), dyn-LL (long linear), dyn-BF (bifurcating), dyn-TF (trifurcating), dyn-BFC (bifurcating converging). Each with $\{100, 200, 500, 2000, 5000\}$ cells $times$ 10 samples. Ground truth: exact generating network.

*B) Curated Boolean Models (4 datasets).* HSC (hematopoietic stem cell), mCAD (cortical area development), VSC (ventral spinal cord), GSD (gonadal sex determination). 2000 cells $times$ 10 samples, plus 50% and 70% dropout variants. Ground truth: Boolean model structure.

*C) Experimental scRNA-seq (7 datasets).* hESC (human embryonic stem cells), hHEP (human hepatocytes), mDC (mouse dendritic cells), mESC (mouse embryonic stem cells), mHSC-E/GM/L (mouse hematopoietic stem cells, three lineages). Ground truths: non-specific ChIP-seq, STRING, cell-type-specific ChIP-seq, and LOF/GOF networks.


== Evaluation Metrics

+ *AUPRC ratio:* AUPRC divided by baseline (where baseline $=$ edge density). Primary metric.
+ *EPR* (Early Precision Ratio): precision among top $K$ predicted edges ($K =$ \# ground truth edges).
+ *AUROC*: secondary metric; less informative for sparse networks.


== Baselines

*Tier 1 (DL, unsupervised, on BEELINE):* DeepSEM, DAZZLE (current SOTA), GRN-VAE, HyperG-VAE.

*Tier 2 (Classical):* GENIE3, GRNBoost2, PIDC, PPCOR.

*Tier 3 (Recent SOTA):* GRNFormer, scKAN, scRegNet, GRANGER.

*Tier 4 (Transfer-aware):* GRNPT, Meta-TGLink, scMTNI, LINGER.


== Experimental Protocol

=== Experiment 1: BEELINE Benchmark---Single-Dataset

All three BEELINE categories. TFs + 500 and TFs + 1000 genes. 10 random seeds per dataset. Key comparisons: FlowGRN vs DAZZLE; OT coupling vs independent; with/without DAG constraint.

Hyperparameter search: $d in {32, 64, 128}$; $alpha in {0.001, 0.01, 0.1}$; $lambda in {0.1, 1.0, 10.0}$; batch OT size $B in {64, 128, 256}$; GAT heads $in {1, 4, 8}$; flow steps $in {10, 20, 50, 100}$.

=== Experiment 2: Ablation Studies

Velocity field architecture (GAT vs MLP vs Transformer); OT coupling type; GRN extraction method (direct $bold(A)$ vs Jacobian vs IG); latent vs gene space; dropout robustness (0%, 50%, 70%); with/without dropout augmentation from DAZZLE.

=== Experiment 3: External Biological Validation

*A) ChIP-seq validation:* Predict TF-target edges, validate against ENCODE ChIP-seq. Report AUC/AUPR per TF.

*B) Perturbation prediction:* Use inferred GRN to predict TF knockout effects against Perturb-seq and CRISPRi screens.

*C) Cross-species transfer:* Train on well-annotated species, evaluate on related species.

=== Experiment 4: Scalability Analysis

Runtime and memory vs number of genes (100--10,000) and cells (100--20,000). Compare against DeepSEM, DAZZLE, GENIE3, GRNBoost2.

=== Experiment 5: Multi-Task Joint Training

Test whether sharing regulatory dynamics across BEELINE datasets helps.

- *FlowGRN-Solo:* trained independently on each dataset (baseline)
- *FlowGRN-Joint:* trained on all 7 datasets (shared $theta$, per-dataset $bold(A)_k$)
- *FlowGRN-Joint-Cond:* same as Joint with dataset conditioning embeddings $bold(e)_k$

Report per-dataset AUPRC/EPR/AUROC. Analyze which datasets benefit most---hypothesis: smaller datasets and datasets from the same organism/lineage benefit most.

=== Experiment 6: Few-Shot GRN Inference

*Leave-one-out:* For each dataset $k$, pretrain on the other 6 $arrow.r$ get $theta_"shared"^*$, then finetune only $bold(A)_k$ (frozen $theta_"shared"$). Compare against Solo from scratch.

*Data titration:* Subsample to $\{100\%, 50\%, 20\%, 10\%, 5\%\}$ of cells. Compare Solo vs Transfer at each level. Plot AUPRC vs \# cells.

Expected: transfer helps most in low-data regime. If pretrained FlowGRN with 10% cells matches Solo with 100%, this is a strong result for rare cell types.

=== Experiment 7: Cross-Species Transfer (Mouse $arrow.r$ Human)

+ *Source:* Train jointly on 5 mouse datasets (mDC, mESC, mHSC-E/GM/L) $arrow.r$ $theta_"shared, mouse"^*$
+ *Ortholog mapping:* Ensembl BioMart one-to-one orthologs ($tilde$16,000 genes)
+ *Target:* Transfer to human datasets (hESC, hHEP):
  - Warm start: $bold(A)_"human" arrow.l bold(O) dot bold(A)_"mouse,avg" dot bold(O)^top$
  - Cold start: $bold(A)_"human" = bold(0)$
  - Finetune $bold(A)_"human"$ with $theta_"shared"$ frozen
+ *Controls:* Reverse direction (human $arrow.r$ mouse); same-species transfer

=== Experiment 8: Shared vs Specific Regulation Analysis

*A) Conserved edges:* Compare $bold(A)_k$ across datasets. Edges with high $|A_(i j)|$ in all datasets $arrow.r$ conserved regulation. Validate against known housekeeping interactions.

*B) Context-specific edges:* High in one $bold(A)_k$, low in others $arrow.r$ context-specific wiring. Validate against cell-type-specific ChIP-seq.

*C) Shared weight inspection:* Analyze what $"MLP"_"msg"$ learns universally. Visualize attention before/after adjacency bias.

*D) Latent space structure:* UMAP of shared latent space across datasets. Do cells undergoing similar processes cluster across datasets?


== Implementation Plan

*Libraries:* PyTorch, TorchCFM, PyTorch Geometric, Scanpy, POT, pybiomart.

*Hardware:* Single GPU (A100/H100); models trainable in minutes per dataset.

```
flowgrn/
├── models/
│   ├── encoder.py, velocity_field.py, flow_matching.py
│   ├── grn_extraction.py, transfer.py
├── data/
│   ├── beeline_loader.py, preprocessing.py
│   ├── multitask_loader.py, ortholog_mapper.py
├── evaluation/
│   ├── metrics.py, beeline_eval.py, transfer_analysis.py
└── experiments/
    ├── exp1–exp8 scripts
```


== Expected Outcomes and Risk Mitigation

*Core FlowGRN:* 5--15% AUPRC improvement over DAZZLE; lower variance across seeds; better on high-dropout data; stronger directionality; competitive runtime.

*Transfer extension:* Joint training $>=$ Solo on most datasets; pretrained FlowGRN at 20% data $approx$ Solo at 100%; cross-species transfer outperforms Solo when human data is limited; biologically interpretable conserved vs specific edges.

#figure(
  table(
    columns: 2,
    stroke: 0.5pt,
    inset: 6pt,
    [*Risk*], [*Mitigation*],
    [Flow matching $<$ VAE on static snapshots], [Add temporal datasets; use pseudotime for source$arrow.r$target],
    [OT computation too expensive], [Sinkhorn with $epsilon=0.1$; top-$k$ genes per batch],
    [Adjacency doesn't converge], [Initialize $bold(A)$ from GENIE3 (warm start); anneal $alpha$],
    [Marginal BEELINE gains], [Focus on bio validation + transfer novelty],
    [Negative transfer in joint training], [Report honestly; Solo baseline always included],
    [Cross-species transfer fails], [Same-species transfer as control to isolate species barrier],
    [Low gene overlap between datasets], [Focus on TF-centric subnetwork where overlap is higher],
    [Shared latent space collapses], [Monitor UMAP/silhouette; add auxiliary reconstruction loss],
  ),
  caption: [Risk mitigation strategy.]
)


= Novelty Claims and Related Work

+ *vs TrajectoryNet/TIGON/CellOT:* These use OT/flow for _trajectory inference_ with GRN as a downstream byproduct. FlowGRN embeds the GRN _directly into the velocity field_.

+ *vs CycleGRN:* Restricted to oscillatory/cell-cycle genes. FlowGRN is general-purpose.

+ *vs DeepSEM/DAZZLE:* Replaces the VAE with flow matching (no matrix inversion, OT-based noise handling, interpretable velocity field). The shared-private architecture enables transfer learning impossible in the SEM framework.

+ *vs GRNFormer:* Supervised, requires labels. FlowGRN is fully unsupervised.

+ *vs scGPT/Geneformer for GRN:* Foundation models extract embeddings for post-hoc GRN inference. FlowGRN learns the GRN as a structural component of the generative model---but can _use_ foundation encoders as its backbone (Variant T4).

+ *vs GRNPT:* Uses LLM text embeddings + supervised training. FlowGRN achieves transfer via shared generative dynamics, entirely unsupervised.

+ *vs Meta-TGLink:* Supervised meta-learning for few-shot GRN. FlowGRN achieves few-shot via pretrained generative dynamics without ground-truth labels.

+ *vs scMTNI:* Shares information via multi-task regularization on single lineages. FlowGRN shares through learned generative dynamics, applies to any dataset collection.

+ *vs LINGER:* Uses external bulk data via lifelong learning. FlowGRN transfers from other single-cell datasets---complementary approaches that could be combined.

*Target venues:* Nature Methods, Nature Computational Science, Genome Biology (full scope); Bioinformatics (core method); NeurIPS/ICML (ML angle with bio validation).
