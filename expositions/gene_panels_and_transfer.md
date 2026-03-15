# Gene Panels, Ortholog Mapping, and the Transfer Learning Problem

This document explains why transfer learning in LatentFlowGRN currently
underperforms solo training, and outlines strategies to fix it.

## 1. The Problem: Gene Sets Don't Overlap

### How genes are currently selected

Each BEELINE dataset independently selects its top highly variable genes
(HVGs) — genes with the highest variance across cells. The `n_genes=1000`
setting means each dataset picks its own 1000 most variable genes.

Different cell types express different genes variably. A hematopoietic stem
cell dataset will have high variance in hematopoietic TFs (GATA1, TAL1),
while an embryonic stem cell dataset will have high variance in pluripotency
factors (POU5F1, NANOG). The resulting gene lists barely overlap:

```
mESC:    1620 genes (after BEELINE loading with 1000 HVG setting)
mHSC-E:  1204 genes
mHSC-GM: 1132 genes
mHSC-L:   692 genes
mDC:     1321 genes

Pairwise overlaps:
  mESC    & mHSC-E:  188 shared genes  (out of 2636 union)
  mESC    & mHSC-GM: 171 shared genes  (out of 2581 union)
  mHSC-E  & mHSC-GM: 618 shared genes  (out of 1718 union)
  All 5 mouse:        13 shared genes  (out of 4223 union)
```

### Why this kills transfer learning

In our transfer learning setup, each dataset has its own gene embedding
`GeneEmbedding(G_k, D)` and adjacency `AdjacencyMatrix(G_k, G_k)`. The
shared components (velocity blocks, time embedding) operate on the hidden
dimension D, blind to gene identity.

When gene sets don't overlap, the shared blocks learn generic hidden-state
processing (how to transform D-dimensional vectors conditioned on time) but
cannot transfer **gene-specific regulatory knowledge** — which TFs regulate
which targets, how strongly, in what direction.

The regulatory grammar hypothesis — that the rules of gene regulation are
universal — may be true, but it can only be tested if the model sees the
**same genes** across datasets. With different gene sets, even identical
regulatory rules produce different A matrices.


## 2. Strategy 1: Shared Gene Panels

### Concept

Instead of each dataset picking its own HVGs, use a single fixed list of
genes across all datasets. This ensures gene embeddings and A structure are
directly comparable and transferable.

### Option A: Consensus HVG

1. For each dataset, compute per-gene variability scores
2. Average scores across datasets
3. Select top K genes by average variability

**Pros:** Data-driven, captures broadly important genes.
**Cons:** Loses dataset-specific variable genes that may be regulatory.

### Option B: Transcription Factor Panel

Use a curated list of known transcription factors (TFs). For mouse,
databases like AnimalTFDB list ~1,500 TFs. For human, similar resources
exist.

BEELINE already supports a `TF+500` and `TF+1000` setting that includes
all significantly varying TFs plus top HVGs. This partially addresses
the overlap problem since TFs are more conserved across cell types.

**Pros:** Biologically motivated — TFs are the regulators we want to infer.
**Cons:** Misses non-TF regulatory genes (lncRNAs, chromatin remodelers).

### Option C: Union with Masking

1. Compute the union of all datasets' gene sets
2. Create a single large embedding for all union genes
3. Each dataset uses a binary mask to indicate which genes are present
4. Missing genes get zero expression and their A entries are masked

**Pros:** No information lost; all genes accessible.
**Cons:** Large model (4223 genes for 5 mouse datasets); many masked entries.

### Implementation

For our codebase, the simplest approach is Option B (TF panel):

```python
# In DataConfig, change gene_selection
class DataConfig(BaseModel):
    gene_selection: Literal["hvg", "all", "tf"] = "hvg"
```

The data loader would then select a fixed TF list rather than per-dataset
HVGs. All datasets would have the same gene count and gene identities,
enabling direct weight sharing of gene embeddings and A structure.


## 3. Strategy 2: Ortholog Mapping (Cross-Species Transfer)

### Concept

Transfer regulatory knowledge from one species to another by mapping genes
through ortholog relationships. If mouse gene Gata1 regulates Tal1, and
human GATA1 is orthologous to mouse Gata1, then human GATA1 likely
regulates human TAL1.

### The ortholog mapping matrix

Given mouse genes {m_1, ..., m_M} and human genes {h_1, ..., h_H}, the
ortholog mapping O is a binary matrix:

```
O ∈ {0, 1}^{H x M}

O[i, j] = 1  if human gene h_i is orthologous to mouse gene m_j
```

This is typically many-to-many (gene duplications) but mostly one-to-one
for well-conserved TFs.

### Warm-starting A

Given a learned mouse adjacency A_mouse:

```
A_human_init = O @ A_mouse @ O^T
```

This transfers the regulatory structure: if A_mouse[m_j, m_k] is large
(gene j regulates gene k in mouse), then A_human_init[h_i, h_l] will be
large for all orthologs h_i of m_j and h_l of m_k.

After warm-starting, A_human is fine-tuned on human expression data to
adjust for species-specific differences.

### Data sources

Ortholog mappings can be obtained from:
- **Ensembl BioMart** (pybiomart Python API): most comprehensive
- **NCBI HomoloGene**: curated ortholog groups
- **OMA (Orthologous Matrix)**: algorithmic ortholog detection

For mouse-human, most TFs have clear one-to-one orthologs (~85% of TFs
are conserved between mouse and human with identifiable orthologs).

### Implementation sketch

```python
import pybiomart

def get_ortholog_mapping(source_species, target_species, gene_names):
    """Fetch ortholog mapping from Ensembl BioMart."""
    server = pybiomart.Server(host='http://www.ensembl.org')
    dataset = server.marts['ENSEMBL_MART_ENSEMBL'] \
                    .datasets[f'{source_species}_gene_ensembl']

    # Query for orthologs
    result = dataset.query(
        attributes=['external_gene_name', f'{target_species}_homolog_associated_gene_name'],
        filters={'link_external_gene_name': gene_names}
    )

    # Build mapping matrix O
    # ...
    return O  # (n_target_genes, n_source_genes)
```

### Applicability to BEELINE

BEELINE has 5 mouse datasets and 2 human datasets:

| Mouse | Human |
|-------|-------|
| mDC | hESC |
| mESC | hHep |
| mHSC-E | |
| mHSC-GM | |
| mHSC-L | |

Cross-species transfer would pretrain on mouse datasets and warm-start
A_human for hESC/hHep evaluation. This is the Phase 4d experiment from
the proposal.


## 4. Strategy 3: Gene-Name-Aware Embeddings

### Concept

Instead of positional gene embeddings (gene 0 gets embedding 0), use a
shared namespace where the same gene in different datasets shares the
same embedding vector.

### Implementation

1. Build a **union vocabulary**: sorted union of all gene names across
   all datasets
2. Create a single `GeneEmbedding(n_union, D)` shared across datasets
3. Each dataset has a mapping `gene_indices[k]: (G_k,)` that indexes
   into the union embedding

```python
# Build union vocabulary
all_genes = sorted(set().union(*[set(gn) for gn in gene_names_per_dataset]))
gene_to_union_idx = {g: i for i, g in enumerate(all_genes)}

# Per-dataset index mapping
gene_indices = {
    ds: torch.tensor([gene_to_union_idx[g] for g in gene_names[ds]])
    for ds in datasets
}

# In forward pass:
h = shared_gene_emb.weight[gene_indices[ds_name]]  # (G_k, D)
```

### Trade-offs

**Pros:**
- Genes appearing in multiple datasets share embeddings and gradients
- A entries for shared genes become comparable across datasets
- No gene information lost

**Cons:**
- Union vocabulary can be large (4223 for 5 mouse datasets)
- Most genes appear in only 1-2 datasets — limited sharing
- Requires modifying the model's forward pass to accept gene indices

### When this helps

Gene-name-aware embeddings help most when there IS significant gene overlap
but the overlap isn't 100%. For the BEELINE HVG setting (13-gene overlap),
it wouldn't help much. For a TF panel (where most TFs appear in most
datasets), it would be very effective.


## 5. Relationship to Our Findings

### What we tried (Finding 008)

We implemented weight sharing at the **architecture level** — shared velocity
blocks, per-dataset gene embeddings and A. This transfers the "regulatory
grammar" (how hidden states are processed) but not the "regulatory vocabulary"
(which genes regulate which).

Results: joint training was neutral (-0.004 AUROC on average). Few-shot
finetuning showed stable performance across cell fractions but 0.015 AUROC
below solo training.

### Why it didn't work well

The near-zero gene overlap means:
1. Gene embeddings are entirely independent across datasets
2. A_k matrices represent different gene sets — no direct transfer
3. The only transferable knowledge is in the velocity blocks, which are
   already generic enough to converge similarly with or without sharing

### What would help (ordered by expected impact)

1. **Shared TF panel** (Strategy 1B): Highest impact, lowest implementation
   cost. Use a fixed TF list across all datasets. Gene embeddings, A, and
   evaluation all become directly comparable.

2. **Gene-name-aware embeddings** (Strategy 3): Medium impact. Useful when
   gene overlap exists but isn't 100%. Requires model forward pass changes.

3. **Cross-species ortholog transfer** (Strategy 2): High novelty but
   requires external data (ortholog mapping) and careful validation. Best
   suited as a paper contribution.

4. **Union with masking** (Strategy 1C): Comprehensive but complex. Large
   model with many masked entries. Best for maximizing information retention.

### Implementation priority

For the next iteration, we recommend:
1. Check if BEELINE's existing `TF+1000` setting provides better overlap
2. If so, re-run Phase 4 transfer experiments with TF panel
3. If overlap improves significantly, implement gene-name-aware embeddings
4. Cross-species transfer as a separate experiment track
