# Joint Training Architecture: Shared Gene Vocabulary

This document describes how multi-dataset joint training works in
LatentFlowGRN, covering the shared gene vocabulary, mixed dataloader,
overlap-adaptive consistency regularization, and checkpoint flow.

For the problem statement (why earlier transfer learning failed), see
`gene_panels_and_transfer.md`.

## 1. Architecture Overview

```
Dataset 1 (mESC, 1620 genes)          Dataset 2 (hESC, 1410 genes)
─────────────────────────              ─────────────────────────
gene_indices: [0, 1, ..., 1619]        gene_indices: [0, 5, ..., 2740]
       │                                      │
       ▼                                      ▼
┌──────────────────────────────────────────────────────┐
│            SHARED nn.Embedding(5805, D-1)            │  ◀── Shared
│    Union vocabulary: all unique genes across datasets │
└──────────────────────────────────────────────────────┘
       │                                      │
       ▼                                      ▼
   [x_i ; emb_i]                          [x_i ; emb_i]
       │                                      │
       ▼                                      ▼
┌──────────────────────────────────────────────────────┐
│                  SHARED GAT Blocks                   │  ◀── Shared
│  time_mlp, input_proj, blocks, final_norm, final     │
└──────────────────────────────────────────────────────┘
       │                                      │
       ▼                                      ▼
  v(x_t, t)                              v(x_t, t)
  + λ·(x_t @ A₁)                        + λ·(x_t @ A₂)
       │                                      │
  ┌────┴────┐                           ┌────┴────┐
  │  A₁     │  ◀── Per-dataset          │  A₂     │  ◀── Per-dataset
  │ a_scale₁│                           │ a_scale₂│
  └────┬────┘                           └────┬────┘
       │                                      │
       └──────────┬───────────────────────────┘
                  ▼
  ┌──────────────────────────────────────────┐
  │  Overlap-adaptive consistency loss       │
  │  w_ij · ||A₁[shared] - A₂[shared]||²   │
  │  w_ij = 1 - overlap_fraction            │
  └──────────────────────────────────────────┘
```

## 2. What is shared vs per-dataset

| Component | Shared? | Why |
|-----------|---------|-----|
| `shared_emb` (nn.Embedding) | **Shared** | Gene knowledge transfers via shared embedding vectors |
| `time_mlp` | **Shared** | Time conditioning is task-generic |
| `input_proj` | **Shared** | Projection to hidden dim is task-generic |
| `blocks` (GAT layers) | **Shared** | Attention + FFN learn general velocity field dynamics |
| `final_norm`, `final` | **Shared** | Output mapping is task-generic |
| `gene_indices` (buffer) | Per-dataset | Maps local gene positions → vocab indices |
| `adj_A` | Per-dataset | Each dataset has its own regulatory network |
| `a_scale` | Per-dataset | Each dataset's A needs its own attention bias scale |

## 3. Shared Gene Vocabulary

### The problem it solves

With per-dataset `nn.Parameter(n_genes, D-1)`, a gene like SOX2 appearing
in both mESC and hESC has two independent embedding vectors. No knowledge
transfers between them.

### How it works

1. **Build union vocabulary** — collect all unique gene names across all
   datasets. BEELINE genes are already uppercase, so no normalization needed.

2. **Create `nn.Embedding(vocab_size, D-1)`** — a single embedding table
   shared across all models via Python reference assignment.

3. **Per-dataset `gene_indices`** — a buffer (not trainable) mapping each
   dataset's local gene positions to indices in the shared embedding.

4. **Forward pass** — each dataset looks up its genes:
   ```python
   emb = shared_emb(gene_indices)  # (G_local, D-1)
   h = [x_i ; emb_i]              # concat expression + identity
   ```

### Gene overlap in practice

Pairwise overlaps between BEELINE datasets (all genes already UPPERCASE):

| Pair | Shared genes | % of smaller set |
|------|-------------|------------------|
| mHSC-E & mHSC-GM | 618 | 54.6% |
| hESC & hHep | 401 | 28.4% |
| mESC & hHep | 333 | 20.6% |
| mESC & mDC | 299 | 18.5% |
| mESC & hESC | 289 | 17.8% |

These shared genes receive gradient signal from all datasets they appear in.

## 4. Mixed Dataloader Training

Each epoch randomly interleaves mini-batches from all datasets:

```
step_schedule = shuffle([ds1, ds2, ..., ds7] × steps_per_epoch)
for ds_name in step_schedule:
    x1 = sample_batch(ds_name)
    loss = flow_matching_loss(model[ds_name], x1)
    loss += l1_sparsity(adj[ds_name])
    loss += consistency_loss(adjs, pairs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Why mixed dataloader over gradient accumulation

Gradient accumulation (1 optimizer step per epoch) gives Adam only 500
updates over 500 epochs. Mixed dataloader gives 28 steps/epoch = 14,000
total updates, which provides:

- **Better Adam adaptation** — momentum and variance estimates update
  28x more frequently
- **No dataset ordering bias** — random shuffling eliminates systematic
  effects from always processing datasets in the same order
- **Empirically superior** — accumulation gave mDC -0.023 AUROC while
  mixed dataloader reduced this to -0.010 (and -0.001 without consistency)

Gradient accumulation is still available via `--grad_accumulate` flag for
cases where consensus gradients are preferred (e.g., very few datasets).

## 5. Overlap-Adaptive Consistency Regularization

### The problem

Per-dataset adjacency matrices A_k are fully independent. Even for the 618
genes shared between mHSC-E and mHSC-GM, their regulatory edges are learned
independently. A uniform consistency penalty helps low-overlap datasets
(mDC: +0.009) but hurts high-overlap ones (mHSC-GM: -0.010) — a
fundamental tension.

### The solution: pair-weighted consistency

Each dataset pair gets a consistency weight inversely proportional to its
gene overlap fraction:

```
w_ij = 1 - (n_shared / min(n_genes_i, n_genes_j))

loss_consistency = alpha * sum_ij [ w_ij * ||A_i[shared] - A_j[shared]||² ] / sum_ij(w_ij)
```

### Intuition

- **High-overlap pairs** (mHSC-E/GM: 55% overlap, w=0.45) → weak
  consistency. These cell types already benefit from shared embeddings and
  have substantial cell-type-specific regulation to preserve.

- **Low-overlap pairs** (mDC/mHSC-L: 15% overlap, w=0.85) → strong
  consistency. These distantly related cell types share few genes, so the
  consistency loss provides a stronger signal to borrow regulatory structure
  where overlap exists.

### Example weights (7 BEELINE datasets)

```
  hESC & hHep:    401 shared, weight=0.715    (28% overlap → moderate)
  hESC & mDC:     139 shared, weight=0.895    (10% overlap → strong)
  mHSC-E & mHSC-GM: 618 shared, weight=0.454 (55% overlap → weak)
  mDC & mHSC-L:   103 shared, weight=0.851    (15% overlap → strong)
```

### Configuration

```bash
python -m latentflowgrn.experiments.run_transfer \
  --mode joint \
  --alpha_consistency 0.01    # optimal from sweep
```

Set `--alpha_consistency 0` to disable consistency (shared vocab only).

## 6. Checkpoint Flow (Pretrain → Finetune)

### Pretrain phase

Train on N source datasets jointly. Save:
- `shared_state_dict` — all shared module weights + shared_emb
- `shared_vocab` — `dict[str, int]` mapping gene names to indices

### Finetune phase

1. Load checkpoint and its vocabulary
2. Extend vocab with any new genes from the finetune dataset
3. Resize `shared_emb` if vocab grew (old weights preserved, new genes
   get random init)
4. Create model with extended vocab, load shared weights
5. Freeze shared params (group 0)
6. Train only `adj_A` and `a_scale` on the target dataset

### What freezing means

- **Frozen:** shared_emb, time_mlp, blocks, final → gene identities and
  velocity field dynamics from pretrain are locked in
- **Trainable:** adj_A (regulatory network), a_scale (attention bias
  magnitude) → the finetune dataset learns its own GRN

## 7. What Didn't Work

### Shared low-rank adjacency (Finding 014, Exp 2)

Decomposing A_k = B[indices] + R_k with shared low-rank base B caused
catastrophic failure (mean AUROC 0.542, -0.130 vs solo). This confirms
Finding 006: low-rank A factorization is fundamentally incompatible with
GRN inference. Sparse, high-dimensional regulatory networks cannot be
captured by low-rank structure.

### Gradient accumulation

One optimizer step per epoch (500 total) gave Adam too few updates to
adapt properly. The mixed dataloader (14,000 steps) is strictly better.

### Uniform consistency

A single alpha for all dataset pairs creates a tension: it helps
low-overlap datasets but over-constrains high-overlap ones. The
overlap-adaptive weighting resolves this.

## 8. Implementation Details

### Key files

- `model.py:SharedGeneEmbedding` — nn.Embedding + gene_indices buffer
- `train_transfer.py:_build_shared_vocab()` — union vocabulary construction
- `train_transfer.py:_build_consistency_pairs()` — overlap-adaptive pair weights
- `train_transfer.py:_consistency_loss()` — weighted L2 on shared gene edges
- `train_transfer.py:_share_weights()` — copies shared modules + shared_emb by reference
- `train_transfer.py:_collect_params()` — deduplicates shared params for optimizer

### Weight sharing mechanism

Python reference assignment (`setattr(target, attr, getattr(source, attr))`)
makes all models point to the same PyTorch modules. Changes to shared
weights in one model are immediately visible to all others. The optimizer
deduplicates parameters by `id()` so each shared parameter gets one update
per step.

### Optimizer parameter groups

| Group | Contents | Learning rate | Weight decay |
|-------|----------|--------------|--------------|
| 0 | Shared NN params (blocks, time_mlp, final, shared_emb) | lr | wd_nn |
| 1 | Per-dataset params (a_scale) | lr | wd_nn |
| 2 | Per-dataset adj params (adj_A) | lr_adj | wd_adj |
| 3 | Shared adj base (z_src, z_tgt) — if enabled | lr_adj | wd_adj |

In finetune mode, groups 0 and 3 are frozen.

### Backward compatibility

Solo training (`train.py`) never passes `shared_vocab_size` to
`build_velocity_field`, so `_make_gene_embedding` falls through to the
original `GeneEmbedding` with a per-dataset `nn.Parameter`. No changes to
the solo training path.

### Configurable options

| Flag | Default | Description |
|------|---------|-------------|
| `--alpha_consistency` | 0.0 | Overlap-adaptive consistency strength (0 = disabled) |
| `--grad_accumulate` | false | Use accumulation instead of mixed dataloader |
| `--shared_adj_rank` | 0 | Shared low-rank adj base rank (0 = disabled, not recommended) |
| `--alpha_residual` | 0.0 | L2 penalty on adj residual (only with shared_adj_rank > 0) |
