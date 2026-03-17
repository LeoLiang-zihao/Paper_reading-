# scGPT Learning Notes (By Paper Chapter)

> Organized by paper structure (Abstract → Figure 1 → 2.1 → 2.2 → ... → 4.8)

---

## Abstract / Introduction

### My Understanding
- scGPT is a foundation model for single-cell, inspired by GPT's success in NLP
- Core analogy: gene ↔ word, cell ↔ sentence
- Pre-training data: over 10 million normal human cells
- Downstream tasks: multi-batch integration, multi-omic integration, cell type annotation, genetic perturbation prediction, gene network inference, pseudo-cell generation
- Engineering: in-memory data structures, generative workflow for non-sequential omics, reusable fine-tuning pipeline

### My Questions / Later Clarifications
- **What is NLG?**
  - *Clarification*: NLG = Natural Language Generation, i.e., the model's ability to generate text
- **What exactly does multi-omic integration do?**
  - *Clarification*: Aligning different modalities (RNA/ATAC/Protein) from the same cell into a unified representation space, handling batch/modality differences while preserving biological signals
- **What is the specific task of gene network inference?**
  - *Clarification*: Building gene relationship networks from gene embedding similarity, validating whether known functional modules (e.g., HLA, CD networks) are recovered

---

## Figure 1: Model Overview

### My Understanding
- Figure 1A: Two-stage workflow → pre-training (large atlas data) → fine-tuning (task-specific small data) → application to various downstream tasks
- Figure 1B: Input composition → gene token + expression value + condition token (batch/modality/perturbation)
- Figure 1C: Core mechanism → generative attention mask designed for non-sequential omics data

### My Questions / Later Clarifications
- **What is the specific mechanism of GEP (Gene Expression Prediction)?**
  - *Clarification*: Iteratively predicting gene expression values for unknown tokens from known tokens in an autoregressive manner
- **What is the specific mechanism of GEPC (Gene Expression Prediction for Cell Modelling)?**
  - *Clarification*: Making the model predict gene expression based on cell embedding, establishing the link between cell representation and gene expression
- **How does attention mask work on non-sequential data?**
  - *Clarification*: Unknown genes can only attend to known genes + themselves, not to other unknown genes; generates iteratively

---

## 2.1 Single-cell transformer foundation model overview

### My Understanding
- Input design: three-way fusion of gene token + expression value + condition token
- Expression value processing: per-cell binning (each cell binned independently), making "high/medium/low" semantics comparable across batches
- Model structure: stacked Transformer layers, learning both gene embedding and cell embedding simultaneously
- Pre-training: 10.3 million blood and bone marrow cells, using specially designed attention mask and generative training workflow
- Fine-tuning: flexible pipeline adaptable to various downstream tasks

### My Questions / Later Clarifications
- **Data flow: Is each cell three-dimensional data?**
  - *Clarification*: Not 3D spatial coordinates, but 3 feature channels at each gene position (gene id, expr bin, condition)
- **Feed single cell matrix or batch?**
  - *Clarification*: Feed mini-batches. Raw matrix (N, G) → take batch (B, M) → construct 3-way input (B, M, 3) → sum embeddings to get H0 (B, M, D)
- **Where does cell embedding come from?**
  - *Clarification*: Aggregated from Transformer output HL (B, M, D): cls style takes `<cls>` position, or avg-pool all positions

---

## 2.2 Integration of multiple scRNA-seq data with batch correction

### My Understanding
- Task goal: Remove batch effects while preserving cell type biological differences
- Evaluation metrics: AvgBIO (biological conservation) + AvgBATCH (batch correction) + Overall (weighted combination)
- AvgBIO: Average of NMIcell + ARIcell + ASWcell, measuring whether cell type structure is preserved
- AvgBATCH: Average of ASWbatch + GraphConn, measuring whether batches are mixed
- Results: scGPT achieves SOTA on Immune Human and PBMC 10K, correcting batch effects without erasing biological structure

### My Questions / Later Clarifications
- **Is AvgBIO classification accuracy?**
  - *Clarification*: No, it's the preservation of biological structure in embedding space (clustering quality), not classification accuracy
- **Specific meaning of biological conservation?**
  - *Clarification*: After batch correction, whether real cell type structure is still preserved (not mixing all cells together)

---

## 2.3 Cell type annotation

### My Understanding
- Task: Train on reference dataset, classify query cells
- scGPT advantage: Directly receives complete highly variable gene set, no pre-dimensionality reduction (avoids information loss)
- Results: 96.7% accuracy on hPancreas dataset, surpassing TOSICA
- Pre-training benefit: Fine-tuned model outperforms training from scratch

### My Questions / Later Clarifications
- This section is relatively straightforward, mainly understanding the advantage of "direct gene input vs pre-dimensionality reduction"

---

## 2.4 Perturbation Prediction

### My Understanding
- Task: Input control cell + perturbation condition, predict perturbed cell expression
- Train-test split (key):
  - Single-gene: test perturbations not seen in training
  - Two-gene (Norman): 3 difficulty levels
    - 0/2 unseen: both genes seen in training (combination not seen)
    - 1/2 unseen: one gene not seen in training
    - 2/2 unseen: neither gene seen in training (hardest)
- Evaluation metrics: corr (predicted vs true correlation), corr(Δ) (change amount correlation), DE/ALL (differentially expressed/all genes)
- Results: scGPT best on 7 of 8 metrics, especially significant improvement on DE corr(Δ)

### My Questions / Later Clarifications
- **Concrete example understanding of 3 difficulty levels**
  - *Clarification*: Assume training saw A,B,C,D; didn't see E,F
    - `A+B` → 0/2 unseen
    - `A+E` → 1/2 unseen
    - `E+F` → 2/2 unseen
- **How exactly are training and testing data fed?**
  - *Clarification*: Not random masking, but specific pairing
    - Input: control cell expression + perturbation condition
    - Target: perturbed cell true expression
    - Mask positions are target perturbation genes, input values are unperturbed expression values

---

## 2.5 Multi-omic integration and multi-modal representation learning

### My Understanding
- Task: Integrate multiple sequencing modalities from the same cell (RNA + ATAC + Protein)
- Two settings:
  - Paired integration: each cell measured for all modalities simultaneously
  - Mosaic integration: different batches share partial modalities (e.g., batches 1-2 have RNA+Protein, batches 3-4 have ATAC+Protein)
- Implementation: Extend vocabulary, treat different modalities as different "languages", joint optimization
- Results: Surpasses scGLUE, Seurat v4, scMoMat on 10X Multiome and ASAP PBMC
- Pre-training benefit: Fine-tuned model converges faster, biological fidelity scores >10% higher than training from scratch

### My Questions / Later Clarifications
- **Specific challenge of mosaic data integration?**
  - *Clarification*: Different batches have different modality combinations, need to handle missing modalities during cross-batch alignment

---

## 2.6 Gene embeddings for Gene Regulatory Network inference

### My Understanding
- Gene embedding: vector representation of genes learned by the model
- Gene embedding network: similarity graph built from vector similarity (cosine)
- Difference from GRN:
  - Embedding network is similarity graph (functional/pattern similarity)
  - GRN is biological regulatory graph (TF→target gene, with directionality and mechanism)
- Validation methods:
  - Zero-shot: Directly extract gene embeddings from pre-trained model, check if known networks are recovered (e.g., HLA I/II classification, CD network)
  - Post-fine-tuning: Extract more dataset-specific networks
  - Pathway validation: Compare against Reactome database, embedding similarity positively correlates with number of shared pathways (Pearson 0.316)
- Gene program: Co-expressed gene modules obtained by clustering gene embeddings

### My Questions / Later Clarifications
- **Why does the model have a gene embedding network?**
  - *Clarification*: The model itself is not a network, but trained vectors can be post-processed (similarity calculation) into a network
- **How exactly are gene programs discovered?**
  - *Clarification*: Embedding clustering → filter clusters with ≥5 genes → validate differential expression and pathway enrichment

---

## 3 Discussion

### My Understanding
- Core contributions:
  1. First foundation model pre-trained on 10-million-level single-cell data
  2. Generative attention mask designed for non-sequential omics
  3. Unified framework covering multiple downstream tasks
  4. Zero-shot recovers known structures, fine-tuning improves 8-12% over training from scratch
- Pre-trained model is a "universal feature extractor", not memorizing data but extracting patterns
- Future directions: Larger scale/more modalities/spatial omics/perturbation+time series/causal discovery/in-context learning

### My Questions / Later Clarifications
- This section is mainly summary and future outlook, relatively straightforward

---

## 4 Methods

### 4.1 Input embeddings

**My Understanding**
- Three-way input: gene token id + expression value/bin + condition token
- Key design: per-cell binning (each cell binned independently), making expression level semantics comparable across batches
- Sum of three embeddings gives final input representation

**My Questions / Later Clarifications**
- **Specific data flow?**
  - *Clarification*: Raw (N, G) → batch (B, M) → gene_ids (B, M) + expr_vals (B, M) + cond_ids (B, M) → respective embeddings → sum to get H0 (B, M, D)

---

### 4.2 Cell and gene expression modeling by transformers

**My Understanding**
- Stacked Transformer layers, self-attention mechanism learns gene-gene relationships
- Supports simultaneously:
  - Gene-level: predict gene expression from any position (GEP)
  - Cell-level: get cell embedding from `<cls>` position or pooling
- Can use efficient implementations like FlashAttention

**My Questions / Later Clarifications**
- **How does cell embedding aggregate from transformer output?**
  - *Clarification*: HL (B, M, D) → take cls position or average pool → cell_emb (B, D)

---

### 4.2.3 Condition tokens for batch and modality

**My Understanding**
- Batch/modality information participates in downstream objectives as additional embeddings
- Helps model do batch correction and cross-modality alignment
- DSBN (Domain-Specific Batch Normalization) handles batch differences

---

### 4.3 Generative pre-training

**My Understanding**
- Core innovation: Generative attention mask designed for non-sequential omics data
- Rule: Unknown genes can only attend to known genes + themselves, cannot attend to other unknown genes
- Iterative generation: Predict one batch of unknowns, add high-confidence to known set, predict next batch
- Essence: Adapting GPT's autoregressive idea to "gene dimension" (no natural order)

**My Questions / Later Clarifications**
- **Difference from NLP masking?**
  - *Clarification*: NLP has natural order (left→right), omics genes have no order, need specially designed visibility rules

---

### 4.4 Finetuning objectives

**My Understanding**
- GEP: masked gene expression prediction (gene-level)
- GEPC: predict gene expr from cell embedding (establish cell-gene link)
- ECS: similarity regularization on cells (pull similar cells closer)
- DAR: domain adaptation via gradient reversal (batch correction)
- CLS: cell type classification (supervised)
- Actual usage: Combine these objectives by task

---

### 4.5 Finetuning on downstream tasks

**My Understanding**
- Same pre-trained backbone, different task configurations:
  - scRNA integration: GEP+GEPC+ECS+DAR+DSBN
  - Cell annotation: CLS
  - Perturbation: control → perturbed pairing
  - Multiomic: extend vocabulary, RNA/ATAC/Protein tokens
  - GRN inference: gene embeddings → similarity graph
- Perturbation task data pairing (most critical):
  - Input: control cell expression + perturbation condition
  - Target: perturbed cell expression

---

### 4.6 Datasets

**My Understanding**
- Pre-training: CELLxGENE PBMC (~10.3 million cells)
- Evaluation datasets:
  - Integration: PBMC 10K, Immune Human
  - Annotation: hPancreas
  - Perturbation: Adamson, Norman
  - Multiomic: 10X Multiome, ASAP PBMC
  - GRN validation: Reactome, immune gene sets

---

### 4.7 Experiment setup

**My Understanding**
- Unified feature count: 1200 HVG
- Preprocessing: per-cell normalization + log transform
- Evaluation metric structure:
  - AvgBIO = (NMIcell + ARIcell + ASWcell) / 3
  - AvgBATCH = (ASWbatch + GraphConn) / 2
  - Overall = 0.6 × AvgBIO + 0.4 × AvgBATCH
- Perturbation evaluation: corr and corr(Δ), reported on ALL/DE gene sets

---

### 4.8 Implementation details

**My Understanding**
- Model scale example: d_model=512, n_layers=12, n_heads=8
- Training: train/val split ~97/3, iterative generation ratio sampled from {0.25, 0.5, 0.75}
- Fine-tuning: Load pre-trained weights, task-specific heads/loss combination, fixed epochs + best-val checkpoint

---

## Appendix: My Question List (in order of appearance)

1. What is NLG?
2. What exactly does multi-omic integration do?
3. What is the specific task of gene network inference?
4. Specific mechanisms of GEP/GEPC/attention mask?
5. Is each cell three-dimensional data?
6. Feed single cell matrix or batch?
7. Is embedding "generated" or network output?
8. How does cell embedding aggregate from transformer output?
9. Is AvgBIO classification accuracy?
10. Specific meaning of biological conservation?
11. Training and testing process for perturbation experiments?
12. How are 3 difficulty levels defined for two-gene perturbation? (0/2, 1/2, 2/2 unseen)
13. Specific challenge of mosaic data integration?
14. Why does the model have a gene embedding network?
15. How exactly are gene programs discovered?
16. Difference from NLP masking?