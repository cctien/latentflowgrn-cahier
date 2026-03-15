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

Expression data undergoes two-stage normalization before training:

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

### GRN extraction

At evaluation time, the raw adjacency is extracted, normalized by
`gene_reg_norm`, and compared against the ground-truth regulatory network:

```python
def get_adj(self):
    adj = self.get_masked_adj().detach().cpu().numpy() / self.gene_reg_norm
    return adj.astype(np.float16)
```

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


## 5. MLP Velocity Field (Phase 1)

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

### Gene embedding

Each gene g has a learnable (D-1)-dimensional vector. The expression scalar
x_t[:, g] is concatenated as the first feature:

```python
class GeneEmbedding(nn.Module):
    def __init__(self, n_genes, hidden_dim):
        self.gene_emb = nn.Parameter(torch.randn(n_genes, hidden_dim - 1))

    def forward(self, x):
        batch_emb = self.gene_emb.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([x.unsqueeze(-1), batch_emb], dim=-1)
```

This gives each gene its own identity while preserving the expression value.

### Time conditioning

Time is embedded via sinusoidal encoding (scaled by 1000 to match
RegDiffusion's frequency range), then added to the hidden state inside
each VelocityBlock:

```python
class VelocityBlock(nn.Module):
    def forward(self, x, t_emb):
        h = self.dropout(self.act(self.l1(x)))
        h = h + self.act(self.time_mlp(t_emb)).unsqueeze(1)  # broadcast across genes
        return self.act(self.l2(h))
```

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


## 6. GAT Velocity Field (Phase 2)

### Motivation

The MLP's (I-A) mixing applies A as a post-hoc linear transformation, which
the network can learn to ignore. The GAT architecture integrates A directly
into the attention mechanism, creating a tighter gradient coupling.

### Architecture

```
x_t (B, G) ──> GeneEmbedding ──> Tanh ──> input_proj ──> h (B, G, D)
t   (B,)   ──> SinusoidalTimeEmb ──> t_emb (B, 64)

A_bias = a_scale * get_masked_adj()                    # (G, G)
A_bias = A_bias.unsqueeze(0).unsqueeze(0)              # (1, 1, G, G)

h ──> GATBlock_1(h, t_emb, A_bias) ──> h (B, G, D)
      ──> GATBlock_2(h, t_emb, A_bias) ──> h

h ──> LayerNorm ──> Linear(D, 1) ──> squeeze ──> v (B, G)
```

### GATBlock: attention with A bias

Each GATBlock is a pre-norm Transformer block where A biases the attention
logits. The full computation:

```python
class GATBlock(nn.Module):
    def forward(self, h, t_emb, a_bias):
        # 1. Pre-norm multi-head attention
        h_norm = self.norm1(h)
        Q, K, V = self.qkv(h_norm).chunk(3)        # each (B, G, D)
        # reshape to (B, H, G, d_head) where H=n_heads, d_head=D/H

        logits = Q @ K^T / sqrt(d_head)              # (B, H, G, G)
        logits = logits + a_bias                      # A enters here
        attn = dropout(softmax(logits, dim=-1))       # (B, H, G, G)
        out = attn @ V                                # (B, H, G, d_head)
        h = h + out_proj(concat_heads(out))           # residual

        # 2. Pre-norm FFN + time injection
        h_norm = self.norm2(h)
        h = h + ffn(h_norm) + tanh(time_proj(t_emb)).unsqueeze(1)

        return h
```

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


## 7. SEM-Style Residual Pathway

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


## 8. Auxiliary Losses on A

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


## 9. Complete Loss Function

The total loss combines flow matching with optional auxiliary terms:

```
L_total = L_flow                               # always
         + alpha_l1 * L_sparse                  # L1 sparsity (after delay)
         + alpha_sup * L_supervision            # known-edge BCE
         + alpha_dag * L_dag                    # NOTEARS acyclicity
```

where:
- `L_flow = MSE(v_theta(x_t, t), u_t)`: flow matching (~1.5)
- `L_sparse = mean(|S_tau(A)|)`: soft-thresholded L1 (~0.00003, negligible)
- `L_supervision`: BCE on A at known edge positions
- `L_dag = tr(e^{A*A}) - G`: acyclicity constraint

The SEM residual (`lambda_adj`) is not a separate loss — it modifies the
velocity prediction itself, affecting L_flow through the changed v_theta.


## 10. Differences: MLP vs GAT

| Aspect | MLP | GAT |
|--------|-----|-----|
| How A enters | Post-hoc (I-A) einsum | Attention logit bias |
| A method used | `i_minus_a()` (with dropout) | `get_masked_adj()` (no dropout) |
| Gradient coupling | Weak (single matmul) | Stronger (through softmax) |
| Cross-gene mixing | Single linear transform | Multi-head attention per layer |
| Time conditioning | Additive in VelocityBlock | Additive in FFN sublayer |
| Residual connections | Skip-concat (re-add expression) | Standard Transformer residual |
| Normalization | None | Pre-norm LayerNorm |
| Activation | Tanh throughout | Tanh (embeddings), GELU (FFN) |
| Memory | O(G * D) | O(G^2) for attention |
| Batch size | 256 | 32 (memory constrained) |
| SEM residual | Supported (`lambda_adj`) | Supported (`lambda_adj`) |

Both architectures support the SEM residual pathway and all auxiliary losses
(supervision, NOTEARS). These are orthogonal to the architecture choice.


## 11. Configuration Reference

All coupling and regularization parameters with defaults:

```json
{
    "model": {
        "architecture": "gat",      // "mlp" or "gat"
        "adj_mixing": "post",       // MLP only: "pre", "post", "both"
        "lambda_adj": 0.0,          // SEM residual strength (0 = disabled)
        "a_scale_init": 100.0,      // GAT only: initial A bias scale
        "init_coef": 5.0,           // A initialization multiplier
        "adj_dropout": 0.3          // dropout rate on A during training
    },
    "train": {
        "wd_adj": 0.0,              // weight decay on A (must be 0)
        "alpha_l1": 0.001,          // L1 sparsity (negligible at this scale)
        "l1_delay": 100,            // epochs before L1 starts
        "l1_ramp": 100,             // epochs to ramp L1 to full strength
        "alpha_sup": 0.0,           // known-edge supervision (0 = disabled)
        "alpha_dag": 0.0            // NOTEARS acyclicity (0 = disabled)
    }
}
```


## 12. Transfer Learning (Multi-Dataset Training)

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


## 13. What the Model Does NOT Do

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
