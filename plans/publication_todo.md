# Publication TODO: Cross-Cell-Type GRN Inference via Flow Matching

**Date:** 2026-03-17
**Status:** In progress
**Target:** Computational biology venue (e.g., Genome Research, Bioinformatics, RECOMB)

---

## Paper Framing

**Title idea:** "Cross-cell-type gene regulatory network inference via
conditional flow matching with shared gene representations"

**Core claims:**
1. Conditional flow matching with a learnable adjacency matrix is a novel
   unsupervised approach for GRN inference (architecture contribution)
2. Shared gene vocabulary + mixed dataloader enables multi-dataset joint
   training without performance degradation (method contribution)
3. The architecture supports cross-cell-type generalization: train on N
   cell types, infer GRNs on held-out cell types (evaluation contribution)
4. GenePT-initialized shared embeddings provide biologically meaningful
   gene representations for transfer (practical contribution)

**Key differentiator vs existing work:**
- GRNFormer achieves 0.90+ AUROC but is **supervised** (trains on known edges)
- We are the first **unsupervised** method to demonstrate cross-cell-type
  GRN generalization via flow matching
- scDFM uses flow matching but for perturbation prediction, not GRN inference
- DigNet uses discrete diffusion for GRN but doesn't do multi-dataset training

---

## Experiments TODO

### Priority 1: Core results (MUST HAVE)

- [ ] **Run benchmark v2** (`scripts/run_transfer_benchmark.sh`)
  - Stage 1: Solo baselines (7 datasets) — may reuse existing results
  - Stage 2: Joint all 7, random init
  - Stage 3: Joint all 7, GenePT init
  - Stage 4: Cross-cell-type (train 5, eval 7), GenePT init
  - Stage 5: Cross-cell-type + consistency (α=0.01)
  - **Key result needed:** held-out mESC and mHSC-L performance vs solo

- [ ] **Multi-seed validation** (3 seeds: 42, 123, 456)
  - Run the best joint config with 3 seeds
  - Run solo baselines with 3 seeds
  - Compute mean ± std for all metrics
  - **Critical:** determines if any improvement is statistically significant

- [ ] **Few-shot with fixed checkpoint**
  - Pretrain on 5 datasets → finetune on mESC with {100%, 50%, 20%, 10%} cells
  - The checkpoint bug is now fixed; Finding 008 results were invalid
  - **Key result:** does pretrained→finetune outperform solo at low cell counts?

### Priority 2: Strengthen the paper

- [ ] **Evaluate on all 3 ground truth types** (STRING, ChIP-seq, Non-ChIP)
  - Current results only report STRING
  - Joint training may show different patterns on experimental GTs

- [ ] **Add geneRNIB evaluation**
  - Newer benchmark with causal inference metrics
  - Addresses BEELINE's known limitations
  - Shows we're not over-optimizing for one benchmark

- [ ] **Cross-species transfer** (mouse → human)
  - Train on 5 mouse datasets → evaluate on hESC and hHep
  - Ortholog mapping already implemented (`load_ortholog_map`)
  - Shared gene vocabulary handles cross-species naturally (genes already uppercase)
  - **High novelty:** no unsupervised method has shown this

- [ ] **Ablation table**
  - Shared vocab ON/OFF
  - Mixed dataloader vs accumulation vs sequential
  - GenePT init ON/OFF
  - Consistency ON/OFF
  - Show each component's marginal contribution

### Priority 3: Nice to have

- [ ] **Adaptive consistency alpha sweep** (0.015, 0.02, 0.03)
  - Currently running; results will refine the optimal alpha
  - May not change the paper's conclusions

- [ ] **Foundation model comparison**
  - Try scGPT embeddings alongside GenePT
  - Try GeneFormer if accessible
  - Compare init strategies: frozen vs learnable projection

- [ ] **Scalability analysis**
  - Training time vs number of datasets
  - Memory usage with shared vocab vs per-dataset
  - Show linear scaling with dataset count

- [ ] **Visualization**
  - UMAP of shared gene embeddings colored by dataset
  - Heatmap of learned A for shared genes across datasets
  - Overlap-adaptive consistency weights matrix

---

## Code TODO

- [x] Shared gene vocabulary (`SharedGeneEmbedding`)
- [x] Mixed dataloader training loop
- [x] Overlap-adaptive consistency regularization
- [x] Checkpoint save/load with vocab
- [x] Pretrained embedding init for shared vocab
- [x] Held-out evaluation mode (`--held_out`)
- [x] Per-dataset `a_scale`
- [x] Tests (65 passing)
- [ ] Save per-dataset metrics to a structured JSON for easy plotting
- [ ] Add ChIP-seq and Non-ChIP evaluation to transfer training
      (currently only evaluates primary GT)
- [ ] Multi-seed support in `run_transfer.py` (loop over seeds)

---

## Writing TODO

- [ ] Introduction: unsupervised GRN inference gap, flow matching novelty
- [ ] Methods: CFM + learnable A, shared vocab, mixed dataloader, consistency
- [ ] Results: solo vs joint vs cross-cell-type vs few-shot
- [ ] Discussion: what transfers (gene identity) vs what doesn't (regulatory
  structure), implications for atlas-scale GRN inference
- [ ] Figures:
  - Fig 1: Architecture diagram (solo vs joint training)
  - Fig 2: BEELINE results table (solo vs joint, all 7 datasets × 3 GTs)
  - Fig 3: Cross-cell-type generalization (held-out performance)
  - Fig 4: Few-shot titration curves
  - Fig 5: Ablation heatmap
  - Supp: Gene overlap statistics, consistency weight matrix, training curves

---

## Key Risks

1. **Cross-cell-type may not show clear improvement** — held-out datasets
   get no training signal for their unique genes. If the held-out
   performance is much worse than solo, the story weakens.

2. **Single-seed noise** — current results are all seed=42. The +0.009 mDC
   improvement and -0.010 mHSC-GM regression could both be noise.

3. **BEELINE limitations** — small gene sets, incomplete ground truth,
   few cell types. Reviewers may want larger-scale experiments.

4. **Competition from supervised methods** — GRNFormer (0.90+ AUROC) makes
   our 0.67 look weak. Must frame as "unsupervised" clearly.

---

## Decision Points

- **If cross-cell-type works (held-out AUROC > 0.55):** strong paper,
  focus on generalization story
- **If cross-cell-type fails:** pivot to "joint training as regularization"
  story (matches solo, doesn't hurt, enables few-shot)
- **If few-shot works (pretrained@20% ≈ solo@100%):** strongest result,
  lead with data efficiency
- **If GenePT init helps significantly:** lead with "foundation model
  embeddings enable GRN transfer"
