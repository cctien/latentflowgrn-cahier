# LatentFlowGRN: Model Architecture

This document explains how the LatentFlowGRN model works, covering the
mathematical formulation, code implementation, and the relationship between
the two architectures (MLP and GAT).

## 1. Overview

LatentFlowGRN learns a gene regulatory network (GRN) by embedding a learnable
adjacency matrix A into a conditional flow matching model. The model learns to
transport noise to gene expression data, and the structure of A that helps this
transport reveals which genes regulate which.

The pipeline:

```
Expression data x ──┐
                     ├── Flow matching ──> Velocity field v(x_t, t; A) ──> Learned A
Gaussian noise x_0 ──┘
```

## 2. Conditional Flow Matching

### Mathematical formulation

Conditional flow matching (CFM) learns a velocity field that transports samples
from a source distribution (noise) to a target distribution (expression data).

Given:
- x_0 ~ N(0, I): Gaussian noise (source)
- x_1 ~ p_data: gene expression vector (target)
- t ~ Uniform(0, 1): time

The interpolation and target velocity are:

```
x_t = (1 - t) * x_0 + t * x_1
u_t = x_1 - x_0
```

The model learns v_theta(x_t, t) to approximate u_t by minimizing:

```
L = E_{t, x_0, x_1} [ || v_theta(x_t, t) - u_t ||^2 ]
```

This is the **flow loss** — a simple MSE between predicted and target velocity.

### OT-CFM variant

With optimal transport coupling (`ot_coupling="ot"`), the (x_0, x_1) pairs within
each batch are matched via OT before computing the flow, producing straighter
transport paths. In practice this had no measurable effect on GRN quality
(Finding 002).

### Code: training loop

From `train.py`:

```python
# Sample noise
x0 = torch.randn_like(x1)

# Compute interpolation and target velocity
t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)

# Model predicts velocity
vt = model(xt, t)

# Flow matching loss
flow_loss = F.mse_loss(vt, ut)
```

The dataloader samples one batch of real expression data per step (with
replacement). The `steps_per_epoch` parameter controls how many gradient
steps occur per epoch.


## 3. Data Normalization

Expression data is normalized before training. Two modes are available,
controlled by `data.normalization`:

### Default: two-stage z-score (`normalization: "zscore"`)

```python
# Stage 1: min-max per cell (row)
normalized = (exp_array - cell_min) / cell_range

# Stage 2: z-score per gene (column)
normalized = (normalized - gene_mean) / gene_std
```

After normalization, each gene has approximately zero mean and unit variance.
This means x_0 ~ N(0, I) and x_1 ~ normalized data have comparable scale,
and the target velocity u_t = x_1 - x_0 has expected squared norm ~2G
(where G is the number of genes).

### Alternative: arcsinh normalization (`normalization: "arcsinh"`)

Inspired by GRNFormer's preprocessing. Arcsinh is smooth at zero and handles
zero-inflated scRNA-seq count data better than log transforms:

```python
# Stage 1: arcsinh transform (handles zero-inflation gracefully)
normalized = arcsinh(exp_array)

# Stage 2: z-score per gene (column)
normalized = (normalized - gene_mean) / gene_std
```

Configuration: `{"data": {"normalization": "arcsinh"}}`


## 4. The Adjacency Matrix

### Parameterization

The adjacency matrix A is a learnable G x G parameter (`adj_A`) processed
through soft thresholding with diagonal zeroing.

From `adjacency.py`:

```python
class AdjacencyMatrix(nn.Module):
    def __init__(self, n_genes, adj_dropout=0.3, init_coef=5.0):
        self.gene_reg_norm = 1.0 / (n_genes - 1)
        self.tau = self.gene_reg_norm / 2

        # Initialize uniformly: all genes equally connected
        init_val = torch.ones(n_genes, n_genes) * self.gene_reg_norm * init_coef
        self.adj_A = nn.Parameter(init_val)
```

For G=1000: `gene_reg_norm = 1/999 ≈ 0.001`, `tau ≈ 0.0005`,
initial values ≈ 0.005.

### Correlation-based initialization (`adj_init: "corr"`)

Instead of uniform initialization, A can be warm-started from the gene
co-expression correlation matrix. Absolute correlation values are scaled to
match the default init magnitude:

```python
def init_from_corr(self, corr_matrix):
    scaled = corr_matrix.abs() * self.gene_reg_norm * 5.0
    scaled.fill_diagonal_(self.gene_reg_norm * 5.0)
    self.adj_A.copy_(scaled)
```

This gives the adjacency matrix a data-informed starting point rather than
learning structure entirely from scratch. Requires computing the Pearson
correlation matrix from expression data.

Configuration: `{"model": {"adj_init": "corr"}}`

### Soft thresholding

Soft thresholding drives small entries toward zero, encouraging sparsity:

```
S_tau(x) = sign(x) * max(|x| - tau, 0)
```

In code:

```python
def soft_threshold(self, x):
    return torch.sign(x) * F.relu(torch.abs(x) - self.tau)
```

Entries with |x| < tau become exactly zero. This creates natural sparsity
without explicit L1 regularization.

### Masked adjacency (for GAT)

`get_masked_adj()` returns the soft-thresholded A with diagonal zeroed
(no self-regulation) and no dropout. Used as attention bias in GAT mode:

```python
def get_masked_adj(self):
    mask = 1.0 - torch.eye(self.n_genes, device=self.adj_A.device)
    return self.soft_threshold(self.adj_A) * mask
```

### I - A (for MLP)

`i_minus_a()` computes the mixing matrix for the MLP architecture.
During training, inverted dropout randomly zeros 30% of off-diagonal entries
(scaled up by 1/(1-p) to preserve expected value):

```python
def i_minus_a(self):
    eye = torch.eye(self.n_genes, device=self.adj_A.device)
    if self.training:
        drop_mask = (torch.rand_like(self.adj_A) > self.adj_dropout).float()
        drop_mask /= (1.0 - self.adj_dropout)
        drop_mask.fill_diagonal_(0.0)
    else:
        mask = 1.0 - eye
    clean_a = self.soft_threshold(self.adj_A) * drop_mask
    return eye - clean_a
```

### TF-aware masking (`tf_mask: true`)

Real gene regulatory networks have a directional constraint: only
transcription factors (TFs) can be source nodes. When TF masking is enabled,
the model extracts the set of TF gene names from ground truth edges (all
source genes are TFs by definition), maps them to gene indices, and stores a
binary mask that zeros out non-TF rows in A.

**Important caveat:** The mask applies during both training and evaluation
(in `get_masked_adj()`, `i_minus_a()`, and `get_adj()`), which constrains
the model's learning capacity. Moreover, the GRNEvaluator already filters
predictions to TF→target edges at evaluation time regardless of this setting,
so the mask provides no evaluation benefit. Ablation experiments (Finding 009)
showed TF masking uniformly hurts performance.

Configuration: `{"model": {"tf_mask": true}}`

### GRN extraction

At evaluation time, the raw adjacency is extracted, normalized by
`gene_reg_norm`, optionally TF-masked, and compared against the ground-truth
regulatory network.

The magnitude |A_{gh}| represents the inferred regulatory strength from
gene g to gene h. Higher values indicate stronger predicted regulation.

### Optimizer setup

A has a separate optimizer parameter group with:
- Much lower learning rate: `lr_adj = gene_reg_norm / 50 ≈ 0.00002`
  (vs `lr = 0.001` for the network, ~50x slower)
- Zero weight decay (`wd_adj = 0.0`): weight decay was found to kill A
  (Finding 001)

```python
optimizer = torch.optim.Adam(
    [{"params": nn_params}, {"params": adj_params}],
    lr=cfg.train.lr, weight_decay=0.1, betas=(0.9, 0.99),
)
optimizer.param_groups[1]["lr"] = lr_adj
optimizer.param_groups[1]["weight_decay"] = cfg.train.wd_adj  # 0.0
```


## 5. Gene Embeddings

The gene embedding maps each gene's scalar expression value and identity
into a D-dimensional vector. Three styles are available, controlled by
`model.embed_style`:

### Concat embedding (`embed_style: "concat"`, default)

From RegDiffusion (Zhu & Slonim, 2024). Each gene g has a learnable
(D-1)-dimensional identity vector. The raw expression scalar x_t[:, g]
is concatenated as the first feature:

```python
h_i = [x_i ; emb_i]    # emb_i ∈ ℝ^{D-1}, output ∈ ℝ^D
```

This preserves the exact expression value but confines it to 1 out of D
dimensions (e.g., 1/128 = 0.8% of the representation).

A variational variant is available with `variational_embed: true` — the
gene embedding becomes a learned distribution (mu, logstd) sampled via
reparameterization during training, regularized by a KL divergence term.
Only applies to the concat style.

Configuration:
```json
{"model": {"embed_style": "concat"}}
{"model": {"embed_style": "concat", "variational_embed": true}, "train": {"alpha_kl": 0.001}}
```

### scDFM embedding (`embed_style: "scdfm"`)

From scDFM (ICLR 2026), `ContinuousValueEncoder`. Projects the expression
scalar through a two-layer MLP with LayerNorm, then adds a full D-dim
gene identity embedding:

```python
Ev(x_i) = Linear(1, D) → ReLU → Linear(D, D) → LayerNorm → Dropout
h_i = Ev(x_i) + emb_i    # both ∈ ℝ^D, additive combination
```

The learned projection gives expression magnitude access to all D
dimensions rather than a single scalar slot. LayerNorm normalizes the
projected expression to a consistent scale before adding the gene identity.

Configuration: `{"model": {"embed_style": "scdfm"}}`

### MLP embedding (`embed_style: "mlp"`)

A simpler projected variant without normalization or dropout:

```python
Ev(x_i) = Linear(1, D) → SiLU → Linear(D, D)
h_i = Ev(x_i) + emb_i    # both ∈ ℝ^D, additive combination
```

Uses SiLU (Swish) activation, which is standard in modern architectures.
Lighter than scDFM style — no LayerNorm or Dropout overhead.

Configuration: `{"model": {"embed_style": "mlp"}}`

### Comparison

| Style | Expression dims | Gene identity dims | Combination | Extras |
|-------|----------------|-------------------|-------------|--------|
| `concat` | 1 (raw scalar) | D-1 | Concatenation | Variational option |
| `scdfm` | D (projected) | D | Addition | LayerNorm + Dropout |
| `mlp` | D (projected) | D | Addition | None |

The projected styles (scdfm, mlp) give the model a richer representation
of expression magnitude. The tradeoff is that the raw expression value is
no longer directly accessible — the model must learn a useful projection.

Note: When using projected embeddings with the MLP velocity field, the
between-block skip connections (which re-concat the raw expression scalar)
are disabled, since expression information is already distributed across
all dimensions.


## 6. MLP Velocity Field (Phase 1)

### Architecture

The MLP velocity field follows RegDiffusion's noise-prediction network,
adapted for flow matching:

```
x_t (B, G) ──> GeneEmbedding ──> h (B, G, D)
t   (B,)   ──> SinusoidalTimeEmb ──> t_emb (B, 64)

h ──> VelocityBlock_1(h, t_emb) ──> h (B, G, D-1)
      ──> concat(h, x_t) ──> VelocityBlock_2(h, t_emb) ──> h (B, G, D-1)

h ──> einsum('bgd,gh->bhd', h, I-A) ──> h    # (I-A) mixing
h ──> Linear(D-1, 1) ──> squeeze ──> v (B, G)
```

### Time conditioning

Time is embedded via sinusoidal encoding (scaled by 1000 to match
RegDiffusion's frequency range), then injected into each VelocityBlock.

**Default: additive injection** — the time embedding is projected and added
to the hidden state:

```python
h = self.dropout(self.act(self.l1(x)))
h = h + self.act(self.time_mlp(t_emb)).unsqueeze(1)  # broadcast across genes
return self.act(self.l2(h))
```

**adaLN-Zero** (`adaln_zero: true`) — inspired by DiT and scDFM, replaces
additive injection with adaptive layer norm. The time embedding produces
(scale, shift, gate) parameters that modulate via LayerNorm:

```python
h = self.dropout(self.act(self.l1(x)))
scale, shift, gate = self.adaln_proj(t_emb).chunk(3)
h = self.norm(h) * (1 + scale) + shift
return gate * self.act(self.l2(h))
```

The gate is zero-initialized so the block starts as an identity function.
This allows the model to modulate feature distributions based on noise
level (t≈0 = mostly noise, t≈1 = mostly signal) rather than just shifting.

The activation function in VelocityBlock is controlled by `model.activation`
(default: Tanh).

### (I - A) mixing

After the velocity blocks, the hidden states are mixed across genes via
matrix multiplication with (I - A):

```python
ima = self.adj.i_minus_a()                      # (G, G)
h = torch.einsum("bgd,gh->bhd", h, ima)         # mix across genes
```

Expanding the einsum for a single gene h and feature d:

```
h_new[b, h, d] = sum_g h[b, g, d] * (I - A)[g, h]
               = h[b, h, d] - sum_g A[g, h] * h[b, g, d]
```

Each gene's representation becomes itself minus a weighted combination of
all other genes' representations. The weights A_{g,h} encode how much gene g's
state should influence gene h's velocity prediction. Genes that regulate h
have large A_{g,h} values.

### Why (I - A) and not just A

The identity term ensures that each gene retains its own information. Without
it (`h = h @ A`), the mixing would be purely cross-gene and lose each gene's
individual signal. The subtraction means regulators *modify* the target gene's
velocity prediction rather than replacing it.

### Gradient flow to A

The gradient path is: `flow_loss -> v_t -> final_linear -> h_mixed -> (I-A) -> A`.
This is indirect — A only affects the output through a single matrix multiply
after the network has already computed per-gene features. If the network learns
to predict velocity well without cross-gene information, the gradient on A
becomes very small (observed: A_grad_norm ~ 0.01 early, growing to ~0.1 later).


## 7. GAT Velocity Field (Phase 2)

### Motivation

The MLP's (I-A) mixing applies A as a post-hoc linear transformation, which
the network can learn to ignore. The GAT architecture integrates A directly
into the attention mechanism, creating a tighter gradient coupling.

### Architecture

```
x_t (B, G) ──> GeneEmbedding ──> activation ──> input_proj ──> h (B, G, D)
t   (B,)   ──> SinusoidalTimeEmb ──> t_emb (B, 64)

A_bias = a_scale * get_masked_adj()                    # (G, G)
A_bias = A_bias.unsqueeze(0).unsqueeze(0)              # (1, 1, G, G)

h ──> GATBlock_1(h, t_emb, A_bias [, edge_feat]) ──> h (B, G, D)
      ──> GATBlock_2(h, t_emb, A_bias [, edge_feat]) ──> h

h ──> LayerNorm ──> Linear(D, 1) ──> squeeze ──> v (B, G)
```

### GATBlock: attention with A bias

Each GATBlock is a pre-norm Transformer block where A biases the attention
logits. The block has two sub-blocks (attention + FFN) with several
configurable options:

**Standard attention** (default):

```python
def forward(self, h, t_emb, a_bias, edge_feat=None, knn_mask=None):
    # 1. Pre-norm multi-head attention
    h_norm = self.norm1(h)
    Q, K, V = self.qkv(h_norm).chunk(3)

    logits = Q @ K^T / sqrt(d_head)              # (B, H, G, G)
    logits = logits + a_bias                      # A enters here
    logits = logits + edge_proj(edge_feat)        # if edge_features enabled
    if knn_mask is not None:
        logits[~knn_mask] = -inf                  # sparse attention
    attn = dropout(softmax(logits))
    out = attn @ V
    h = h + out_proj(concat_heads(out))

    # 2. Pre-norm FFN + time injection
    h_norm = self.norm2(h)
    h = h + ffn(h_norm) + time_act(time_proj(t_emb))
    return h
```

**Differential attention** (`diff_attn: true`): Instead of standard softmax,
computes two attention distributions and takes their difference:

```
attn = softmax(Q1 K1^T / sqrt(d)) - λ · softmax(Q2 K2^T / sqrt(d))
```

This suppresses noisy/spurious correlations by subtracting a "background"
attention pattern. Inspired by scDFM's PAD-Transformer and the DIFF
Transformer paper. The Q/K projection is doubled (2Q + 2K + V = 5D),
and λ is a learnable per-head scalar (initialized to 0.8). Both attention
distributions receive the same A bias, edge features, and KNN mask.

Configuration: `{"model": {"diff_attn": true}}`

**adaLN-Zero time conditioning** (`adaln_zero: true`): Replaces additive
time injection with adaptive layer norm modulation for both the attention
and FFN sub-blocks. The time embedding produces 6D parameters
(scale₁, shift₁, gate₁, scale₂, shift₂, gate₂):

```python
# Attention sub-block
h_norm = norm1(h) * (1 + scale1) + shift1    # adaptive norm
out = attention(h_norm)
h = h + gate1 * out_proj(out)                 # gated residual

# FFN sub-block
h_norm = norm2(h) * (1 + scale2) + shift2
h = h + gate2 * ffn(h_norm)
```

Gates are zero-initialized so both sub-blocks start as identity.
Configuration: `{"model": {"adaln_zero": true}}`

### How A enters attention

The key line is:

```python
logits = torch.matmul(q, k.transpose(-2, -1)) / scale + a_bias
```

The attention logits for gene h attending to gene g are:

```
logit[h, g] = (q_h . k_g) / sqrt(d) + a_scale * A[g, h]
```

- `q_h . k_g`: content-based attention — how relevant is gene g's current
  state to gene h?
- `a_scale * A[g, h]`: structure-based bias — does gene g regulate gene h?

After softmax, genes with large A_{g,h} receive more attention weight,
meaning gene h's representation is more influenced by gene g's state.

### Edge features (`edge_features: true`)

Inspired by GRNFormer's TransformerConv which jointly uses node and edge
features. When enabled, the co-expression correlation matrix is passed
through a learned linear projection to produce per-head attention biases:

```python
# edge_feat: (G, G) correlation matrix
# edge_proj: Linear(1, n_heads)
ef_bias = edge_proj(edge_feat.unsqueeze(-1))    # (G, G, H)
ef_bias = ef_bias.permute(2, 0, 1).unsqueeze(0) # (1, H, G, G)
logits = logits + ef_bias
```

This differs from `corr_bias` (which adds a fixed scaled correlation to the
logits) in that the projection is learned per attention head, allowing the
model to learn how much and in what direction co-expression should influence
attention at each head. When `edge_features` is enabled and a correlation
matrix is provided, `corr_bias` mode is automatically disabled to avoid
double-counting.

Configuration: `{"model": {"edge_features": true}}`

Requires a correlation matrix, which is computed automatically when this
option is enabled.

### Sparse KNN correlation mask (`knn_mask_k: 30`)

Inspired by scDFM's gene-gene co-expression graph. When enabled, a KNN
graph is built from the gene-gene Pearson correlation matrix: for each gene,
the top-k most correlated neighbors (by absolute correlation) are selected.
The result is symmetrized and stored as a boolean buffer.

During attention, non-neighbor entries are masked to `-inf` before softmax,
creating sparse attention patterns constrained to biologically plausible
neighbors:

```python
# At init (once):
abs_corr = corr_matrix.abs()
_, topk_idx = abs_corr.topk(k, dim=1)    # top-k per gene
mask = scatter_to_bool(topk_idx)
mask = mask | mask.T                       # symmetrize
mask.fill_diagonal_(True)                  # always allow self-attention

# During forward (every step):
logits = logits.masked_fill(~knn_mask, -inf)
attn = softmax(logits)  # non-neighbors get zero attention weight
```

This complements the learnable A bias: A learns the GRN structure while
the KNN mask constrains the attention's search space to co-expressed gene
pairs, reducing noise from unrelated genes. The density scales as
~2k/G (e.g., k=30 with G=1000 gives ~6% density).

Configuration: `{"model": {"knn_mask_k": 30}}`

Requires a correlation matrix, computed automatically when `knn_mask_k > 0`.
GAT-only (MLP architecture does not use attention).

### The a_scale parameter

A values are small (~0.003) while QK^T logits are typically magnitude ~1-10.
Without scaling, A is invisible to softmax. `a_scale` is a learnable scalar
(initialized to 100.0) that amplifies A's contribution:

```python
self.a_scale = nn.Parameter(torch.tensor(a_scale_init))  # init 100.0

# In forward:
a_bias = self.a_scale * self.adj.get_masked_adj()
```

With `a_scale=100`, the bias becomes `100 * 0.003 = 0.3`, which is meaningful
relative to attention logits. The model can learn to adjust this balance
during training.

### Gradient flow to A in GAT

The gradient path is: `flow_loss -> v_t -> attention output -> softmax ->
logits -> a_scale * A -> A`.

This is more direct than the MLP path because A affects *which genes attend
to which* at every layer, not just a post-hoc mixing. The softmax provides
a differentiable routing mechanism that creates structured gradients on A.

Observed gradient magnitudes:
- GAT with `a_scale=100`: A_grad_norm ~ 0.02 (initial), growing with training
- GAT with `a_scale=1`: A_grad_norm ~ 0.0002 (vanishingly small)
- MLP: A_grad_norm ~ 0.01-0.17


## 8. Activations

Two config fields control activations, each defaulting to the original choice
in its respective context:

- **`activation`** (default: `"tanh"`) — controls the time embedding MLP,
  gene embedding output, MLP VelocityBlock layers, and the GATBlock time
  projection. These all originally used Tanh.

- **`ffn_activation`** (default: `"gelu"`) — controls the feed-forward
  network inside GATBlock. This originally used GELU (standard for
  Transformers). Only applies to the GAT architecture.

Available choices for both: `"tanh"`, `"leaky_relu"`, `"gelu"`.

GRNFormer uses LeakyReLU throughout, motivated by preserving gradient flow
for negative co-expression patterns (repressive regulation). To replicate
this:

```json
{"model": {"activation": "leaky_relu", "ffn_activation": "leaky_relu"}}
```


## 9. SEM-Style Residual Pathway

### Motivation

Both the MLP's (I-A) mixing and the GAT's attention bias couple A to the
velocity indirectly — through hidden representations. The SEM (structural
equation model) residual adds a **direct first-order pathway** where A
multiplies the raw expression:

```
v(x_t, t) = v_nn(x_t, t) + lambda_adj * (x_t @ A)
```

The `x_t @ A` term means: for gene h, the velocity contribution is
`sum_g x_t[g] * A[g,h]` — a weighted sum of all genes' expression values,
where A encodes the regulatory weights. This is the classic SEM formulation
from DeepSEM.

### Code

Both `MLPVelocityField` and `GATVelocityField` support this via the
`lambda_adj` parameter:

```python
v = self.final(h).squeeze(-1)  # (B, G) from the neural network

# SEM-style residual: v += lambda * (x_t @ A)
if self.lambda_adj > 0:
    a = self.adj.get_masked_adj()
    v = v + self.lambda_adj * torch.matmul(x_t, a)

return v
```

### Gradient flow

The gradient from the SEM residual to A is:

```
dL/dA[g,h] = dL/dv[h] * lambda_adj * x_t[g]
```

This is **much stronger** than the (I-A) mixing gradient because:
1. It's first-order in A (no intermediate hidden representations)
2. It scales with the expression value x_t[g] directly
3. It's not attenuated by softmax or post-hoc mixing

Observed: A_grad_norm increases from ~0.01 (without SEM) to ~0.15-0.6
(with `lambda_adj=1.0`).

### Configuration

```json
{"model": {"lambda_adj": 1.0}}  // 0.0 = disabled (default)
```


## 10. Auxiliary Losses on A

### Known-Edge Supervision (`alpha_sup`)

Semi-supervised loss using the ground-truth regulatory edges (the same edges
used for evaluation). Encourages A to have high values at known regulatory
positions and low values elsewhere.

The ground truth edges are converted to index pairs at the start of training:

```python
name_to_idx = {name: i for i, name in enumerate(gene_names)}
pos_src, pos_tgt = [], []
for edge in gt:
    if src_name in name_to_idx and tgt_name in name_to_idx:
        pos_src.append(name_to_idx[src_name])
        pos_tgt.append(name_to_idx[tgt_name])
```

The loss is binary cross-entropy on A (normalized by `gene_reg_norm` to
bring values to a reasonable logit range), with negative edges sampled
randomly:

```python
a_normed = adj.get_masked_adj() / adj.gene_reg_norm

# Positive: known edges should have high A
pos_logits = a_normed[pos_src, pos_tgt]

# Negative: random non-edges should have low A
neg_src = torch.randint(0, n_genes, (n_pos,))
neg_tgt = torch.randint(0, n_genes, (n_pos,))
neg_logits = a_normed[neg_src, neg_tgt]

sup_loss = F.binary_cross_entropy_with_logits(
    torch.cat([pos_logits, neg_logits]),
    torch.cat([ones, zeros]),
)
loss = loss + alpha_sup * sup_loss
```

**Important:** This changes the experimental protocol from **unsupervised** to
**semi-supervised**. Results must be reported accordingly since the model sees
the evaluation edges during training.

Configuration: `{"train": {"alpha_sup": 0.1}}`  (0.0 = disabled)

### Balanced negative sampling (`balanced_neg_sampling: true`)

By default, negative edges are sampled uniformly at random, which may
accidentally include known positive edges. When enabled, negative sampling
explicitly excludes known positives from the negative pool, ensuring a clean
1:1 positive/negative ratio. Inspired by GRNFormer's dynamic negative
sampling strategy.

Configuration: `{"train": {"balanced_neg_sampling": true}}`

Only has effect when `alpha_sup > 0`.


### NOTEARS Acyclicity Constraint (`alpha_dag`)

Real gene regulatory networks are directed acyclic graphs (DAGs) — no gene
can regulate itself through a cycle. The NOTEARS constraint penalizes
cyclic structure in A:

```
R(A) = tr(e^{A * A}) - G
```

where `*` is element-wise (Hadamard) product. This equals zero if and only
if A corresponds to a DAG. Positive values indicate cycles.

From `adjacency.py`:

```python
def dag_loss(self):
    a = self.get_masked_adj()
    a_sq = a * a  # element-wise square
    return torch.trace(torch.matrix_exp(a_sq)) - self.n_genes
```

The matrix exponential `e^{A*A}` has diagonal entries that count weighted
walks of all lengths. If there are no cycles, these walks don't return to
the starting node, and the trace equals G (from the identity term in the
Taylor expansion). Cycles create additional return paths that increase the
trace above G.

Configuration: `{"train": {"alpha_dag": 0.001}}`  (0.0 = disabled)


### KL Divergence (`alpha_kl`)

When variational gene embeddings are enabled, a KL divergence term
regularizes the learned distributions toward a standard normal prior:

```
L_kl = -0.5 * mean(1 + 2*logstd - mu^2 - exp(2*logstd))
```

Configuration: `{"train": {"alpha_kl": 0.001}}`  (0.0 = disabled)

Only has effect when `model.variational_embed` is true.


### MMD Distributional Alignment (`alpha_mmd`)

Inspired by scDFM's distributional flow matching objective. While the CFM
loss aligns individual velocity predictions, the MMD loss encourages the
model's one-step endpoint predictions to match the target distribution at
the population level.

At each training step, the model's predicted endpoint is computed:

```
x1_hat = x_t + (1 - t) * v_theta(x_t, t)
```

Then MMD^2 is computed between `x1_hat` and the true `x1` batch using a
mixture of Gaussian RBF kernels with dynamic bandwidth (median heuristic):

```python
# Dynamic bandwidth from median pairwise distance
median_dist = pairwise_distances.median()

# Multi-kernel RBF with 4 bandwidth scales
for s in [0.5, 1.0, 2.0, 4.0]:
    bw = 2 * s * median_dist
    k(x, y) += exp(-||x - y||^2 / bw)

# Unbiased MMD^2 estimate (excludes self-similarities)
mmd2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
```

Configuration: `{"train": {"alpha_mmd": 0.5}}`  (0.0 = disabled)

Works with both GAT and MLP architectures. Note: the MMD gradient flows
through `v_theta` to all model parameters, but does not directly affect A
beyond what the velocity field already provides.


## 11. Ground Truth Options

### Standard (single source)

By default, each experiment uses one ground truth type (`data.ground_truth`):
STRING, ChIP-seq, or Non-ChIP.

### Multi-source union (`ground_truth_union: true`)

Inspired by GRNFormer's integration of multiple evidence sources. When
enabled, edges from all three ground truth types are unioned into a single
edge set, providing denser supervision signal:

```python
if cfg.ground_truth_union:
    for other_gt in ["STRING", "ChIP-seq", "Non-ChIP"]:
        _, other_edges = load_beeline(..., other_gt)
        edge_set.update(other_edges)
```

The primary ground truth (from `data.ground_truth`) is still used for
evaluation metrics — the union only affects which edges are available for
supervision loss and TF mask extraction.

Configuration: `{"data": {"ground_truth_union": true}}`


## 12. Complete Loss Function

The total loss combines flow matching with optional auxiliary terms:

```
L_total = L_flow                               # always
         + alpha_l1 * L_sparse                  # L1 sparsity (after delay)
         + alpha_sup * L_supervision            # known-edge BCE
         + alpha_dag * L_dag                    # NOTEARS acyclicity
         + alpha_kl * L_kl                      # variational KL divergence
         + alpha_mmd * L_mmd                    # distributional alignment
```

where:
- `L_flow = MSE(v_theta(x_t, t), u_t)`: flow matching (~1.5)
- `L_sparse = mean(|S_tau(A)|)`: soft-thresholded L1 (~0.00003, negligible)
- `L_supervision`: BCE on A at known edge positions
- `L_dag = tr(e^{A*A}) - G`: acyclicity constraint
- `L_kl = KL(q(z) || N(0,I))`: variational embedding regularization
- `L_mmd = MMD^2(x1_hat, x1)`: distributional alignment of one-step predictions

The SEM residual (`lambda_adj`) is not a separate loss — it modifies the
velocity prediction itself, affecting L_flow through the changed v_theta.


## 13. Differences: MLP vs GAT

| Aspect | MLP | GAT |
|--------|-----|-----|
| How A enters | Post-hoc (I-A) einsum | Attention logit bias |
| A method used | `i_minus_a()` (with dropout) | `get_masked_adj()` (no dropout) |
| Gradient coupling | Weak (single matmul) | Stronger (through softmax) |
| Cross-gene mixing | Single linear transform | Multi-head attention per layer |
| Time conditioning | Additive (or adaLN-Zero) | Additive (or adaLN-Zero) |
| Residual connections | Skip-concat (re-add expression) | Standard Transformer residual |
| Normalization | None (LayerNorm with adaLN-Zero) | Pre-norm LayerNorm |
| `activation` controls | Time MLP, gene emb, VelocityBlock | Time MLP, gene emb, time proj |
| `ffn_activation` | N/A | FFN inside GATBlock |
| Edge features | Not supported | Supported via `edge_features` |
| Differential attention | Not supported | Supported via `diff_attn` |
| KNN correlation mask | Not supported | Supported via `knn_mask_k` |
| Memory | O(G * D) | O(G^2) for attention |
| Batch size | 256 | 32 (memory constrained) |
| SEM residual | Supported (`lambda_adj`) | Supported (`lambda_adj`) |

Both architectures support variational gene embeddings, SEM residual,
TF masking, correlation-based initialization, adaLN-Zero time conditioning,
and all auxiliary losses (including MMD).
Differential attention, edge features, and KNN mask are GAT-only
(they require attention logits).


## 14. Configuration Reference

All parameters with defaults. Parameters marked **(P5)** were added in
Phase 5 (GRNFormer-inspired) and **(P6)** in Phase 6 (scDFM-inspired).
All new parameters default to preserving the original behavior.

```json
{
    "data": {
        "dataset": "mESC",
        "n_genes": 1000,
        "ground_truth": "STRING",
        "normalization": "zscore",          // (P5) "zscore" or "arcsinh"
        "ground_truth_union": false         // (P5) union edges from all GT sources
    },
    "model": {
        "architecture": "gat",             // "mlp" or "gat"
        "hidden_dim": 256,
        "n_heads": 4,                      // GAT only
        "n_layers": 2,
        "adj_mixing": "post",              // MLP only: "pre", "post", "both"
        "lambda_adj": 0.0,                 // SEM residual strength
        "a_scale_init": 100.0,             // GAT only: initial A bias scale
        "init_coef": 5.0,                  // A initialization multiplier
        "adj_dropout": 0.3,                // dropout rate on A during training
        "adj_mode": "full",                // "full" or "lowrank"
        "corr_bias": false,                // fixed correlation bias in attention
        "adj_init": "default",             // (P5) "default" or "corr"
        "edge_features": false,            // (P5) GAT only: learned edge feature projection
        "variational_embed": false,        // (P5) variational gene embeddings
        "tf_mask": false,                  // (P5) TF-only masking (hurts — see note)
        "activation": "tanh",              // (P5) time/emb/block activation
        "ffn_activation": "gelu",          // (P5) GAT FFN activation
        "diff_attn": false,                // (P6) GAT only: differential attention
        "knn_mask_k": 0,                   // (P6) GAT only: KNN sparse attention mask (0=off)
        "adaln_zero": false,               // (P6) adaptive layer norm time conditioning
        "embed_style": "concat"            // (P6) "concat" (RegDiffusion), "scdfm", or "mlp"
    },
    "train": {
        "lr": 1e-3,
        "batch_size": 256,
        "epochs": 500,
        "wd_adj": 0.0,                    // weight decay on A (must be 0)
        "alpha_l1": 0.001,                // L1 sparsity
        "l1_delay": 100,                  // epochs before L1 starts
        "l1_ramp": 100,                   // epochs to ramp L1 to full strength
        "alpha_sup": 0.0,                 // known-edge supervision
        "alpha_dag": 0.0,                 // NOTEARS acyclicity
        "alpha_kl": 0.0,                  // (P5) variational KL weight
        "balanced_neg_sampling": false,    // (P5) exclude positives from negatives
        "alpha_mmd": 0.0                   // (P6) MMD distributional alignment weight
    }
}
```


## 15. Transfer Learning (Multi-Dataset Training)

### Motivation

The single-dataset model trains independently per dataset. Transfer learning
shares the velocity field's learned "regulatory grammar" across datasets
while keeping per-dataset gene embeddings and adjacency matrices.

### What is shared vs per-dataset

The key insight is that velocity blocks operate on hidden dimension D,
independent of the number of genes G:

```
Shared (D-dependent, G-independent):
  - time_mlp: time embedding
  - blocks: GATBlock or VelocityBlock layers
  - input_proj, final_norm, final: projections
  - a_scale: attention bias scale

Per-dataset (G-dependent):
  - gene_emb: GeneEmbedding(G_k, D-1)  -- different genes per dataset
  - adj: AdjacencyMatrix(G_k, G_k)      -- different regulatory network
```

### Implementation: weight sharing by reference

Multiple model instances are created (one per dataset), then shared modules
are assigned by Python reference:

```python
model_0 = GATVelocityField(n_genes_0, adj_0, ...)  # first dataset
model_k = GATVelocityField(n_genes_k, adj_k, ...)  # other dataset

# Share blocks -- same tensor objects, not copies
model_k.time_mlp = model_0.time_mlp
model_k.blocks = model_0.blocks
model_k.input_proj = model_0.input_proj
model_k.final_norm = model_0.final_norm
model_k.final = model_0.final
model_k.a_scale = model_0.a_scale
```

PyTorch deduplicates shared parameters in the optimizer automatically.
Each dataset's A_k only receives gradients from its own flow loss.

### Training loop

```python
for epoch in range(epochs):
    for ds_name in dataset_names:           # round-robin
        model = models[ds_name]
        x1 = next(dataloaders[ds_name])     # dataset-specific batch
        x0 = torch.randn_like(x1)
        t, xt, ut = fm.sample(x0, x1)
        vt = model(xt, t)                   # uses dataset's gene_emb + adj
        loss = F.mse_loss(vt, ut)           # shared blocks get gradients
        loss.backward()
        optimizer.step()
```

### Modes

- **joint**: all datasets train simultaneously
- **pretrain**: train on K-1 datasets, save shared checkpoint
- **finetune**: load shared checkpoint, freeze shared params, train only A
  on target dataset (optionally with reduced cell count for few-shot)

### Findings (Finding 008)

Joint training was roughly neutral (mean AUROC -0.004 vs solo). Few-shot
finetuning showed remarkable stability (10% cells = 100% cells) but
absolute performance fell below solo training. The root cause is near-zero
gene overlap across datasets when using HVG selection (see the gene panels
exposition for details).

### Configuration

```python
class TransferConfig(BaseModel):
    datasets: list[DatasetEntry]           # list of datasets to train on
    mode: Literal["joint", "pretrain", "finetune"] = "joint"
    finetune_freeze_shared: bool = True    # freeze shared params in finetune
    data_fraction: float = 1.0             # cell subsampling for titration
```


## 16. What the Model Does NOT Do

- **No latent space**: the flow operates directly on gene expression vectors,
  not a learned latent representation. The `latent_dim` config field is unused.
- **No encoder/decoder**: expression data enters the velocity field directly.
- **No effective L1**: while L1 regularization is implemented, at current
  scales it contributes ~0.00003 to the loss vs ~1.5 for flow loss. Sparsity
  emerges naturally from soft thresholding and training dynamics.
- **No trajectory reconstruction**: the model only predicts velocity at
  sampled time points, not full cell trajectories.
- **No gene-level transfer**: gene embeddings are per-dataset. Genes present
  in multiple datasets do not share embeddings. See the gene panels exposition
  for how this could be addressed.
