# Geneformer (Theodoris et al., Nature 2023) - Reading Notes

## Abstract & Introduction

### Background
Mapping gene networks requires large amounts of transcriptomic data to learn connections between genes. This impedes discoveries in settings with limited data, including rare diseases and diseases affecting clinically inaccessible tissues.

### Solution: Transfer Learning with Foundation Model
Geneformer is a context-aware, attention-based deep learning model pretrained on ~30 million single cell transcriptomes (Genecorpus-30M) to enable context-specific predictions in settings with limited data.

**Key Innovation**: Unlike traditional approaches that train new models from scratch for each task, transfer learning leverages:
- Large-scale pretraining on general data
- Fine-tuning on limited task-specific data
- Knowledge transfer to diverse downstream applications

### Technical Approach
- **Self-supervised learning**: Uses masked learning objective (15% masking) during pretraining
- **No labels required**: Can utilize massive unlabeled training data
- **Context-aware**: Attention mechanism captures gene importance in specific cellular contexts

---

## Geneformer Architecture and Pretraining

### Model Architecture
- **Type**: 6-layer transformer encoder (BERT-style, not autoregressive)
- **Parameters**:
  - Input size: 2048 tokens (covers 93% of rank value encodings)
  - Embedding dimensions: 256
  - Attention heads: 4 per layer
  - Feed forward size: 512
  - Activation: ReLU
  - Dropout: 0.02

### Input: Rank-Value Encoding
Each single cell's transcriptome is encoded as a novel rank value encoding:

**Process**:
1. Calculate non-zero median expression of each gene across entire Genecorpus-30M
2. Normalize each cell's transcript counts by cell total, then divide by gene's global median
3. Rank genes within each cell by this normalized value

**Rationale**:
- Deprioritizes housekeeping genes (ubiquitously highly expressed)
- Prioritizes transcription factors and other state-distinguishing genes that may be lowly expressed
- More robust against technical artifacts that bias absolute counts

### Pretraining Data (Genecorpus-30M)
- **Scale**: 29.9 million human single cell transcriptomes (27.4M after QC)
- **Sources**: 561 datasets from publicly available repositories (GEO, Human Cell Atlas, etc.)
- **Tissues**: Broad range including adipose, brain, heart, immune, liver, lung, etc.
- **Filtering**:
  - Excluded cells with high mutational burdens (malignant, immortalized lines)
  - Excluded possible doublets and damaged cells
  - Quality control: total reads and mitochondrial reads within 3SD of dataset mean

### Pretraining Objective
- **Method**: Masked Language Modeling (MLM)
- **Masking rate**: 15% of genes per transcriptome
- **Task**: Predict which gene should be in each masked position using context of unmasked genes
- **Advantage**: Entirely self-supervised, no labels needed

### Training Details
- **Learning rate**: 1e-3 with linear warmup (10,000 warmup steps)
- **Optimizer**: Adam with weight decay fix (decoupled)
- **Weight decay**: 0.001
- **Batch size**: 12
- **Epochs**: 3
- **Efficiency optimizations**:
  - Dynamic length-grouped padding (29.4x speedup)
  - DeepSpeed for distributed training
  - Training time: ~3 days on 12x Nvidia V100 32GB GPUs

---

## Context-Awareness and Batch Integration

### Context-Specific Gene Embeddings
For each single cell, the model embeds each gene into a 256-dimensional space encoding gene characteristics specific to that cell's context.

**Validation of Context-Awareness**:
- **In silico reprogramming**: Adding OCT4/SOX2/KLF4/MYC (OSKM) to fibroblasts shifts embeddings toward iPSC state
- **In silico differentiation**: Adding MYOD shifts myogenic cells toward differentiated state
- **Context-dependent genes**: NOTCH receptors show higher variance across cell types than housekeeping genes (e.g., GAPDH)

### Batch Integration Capabilities
Gene embeddings and cell embeddings demonstrate robustness to:
- Sequencing platform effects
- Preservation method differences
- Individual patient variability

**Cross-platform validation**:
- Model fine-tuned on Drop-seq data generalizes to DroNc-seq data
- Out-of-sample accuracy: 84%
- Performance exceeds traditional batch correction methods (ComBat, Harmony)

---

## Gene Dosage Sensitivity Predictions

### Background
Interpreting copy number variants (CNVs) in genetic diagnosis requires determining which genes are sensitive to dosage changes. Traditional features (conservation, allele frequency) don't vary across cell states.

### Fine-tuning Setup
- **Task**: Distinguish dosage-sensitive vs. dosage-insensitive transcription factors
- **Training data**: 10,000 random single cell transcriptomes
- **Results**: AUC 0.91 (significantly outperforms alternative methods)

### Scale Effects
Pretraining with larger and more diverse corpora consistently improves predictive power in downstream tasks, even when using the same limited fine-tuning data.

### Context-Specific Predictions
**Collins et al. neurodevelopmental disease genes** (recently reported CNV analysis of 753,994 individuals):
- High confidence genes (>0.85 score): 96% concordance in fetal cerebral cells
- Moderate confidence genes (0.15-0.85): 84% concordance
- **Context-specificity**: Moderate confidence genes predicted dosage-sensitive at higher rates in fetal cerebral cells vs. adult neurons vs. random cells

### In Silico Deletion Strategy
**Method**:
1. Remove gene from cell's rank value encoding
2. Quantify cosine similarity between original and perturbed embeddings
3. Impact measured at:
   - Cell level: Predicted deleterious effect on cell state
   - Gene level: Which downstream genes are most sensitive to deletion

**Fetal cardiomyocyte application**:
- In silico deletion of known cardiomyopathy genes has larger effect than control hyperlipidemia genes
- GWAS-linked cardiac MRI trait genes also show larger effects
- Top 25 most deleterious deletions include known regulators (FOXM1) and novel candidates (TEAD4)

**Experimental validation**:
- CRISPR-mediated knockout of TEAD4 in iPSC-derived cardiac microtissues
- Result: Significant reduction in contractile stress generation

---

## Chromatin Dynamics Predictions

### Bivalent Chromatin Structure
Bivalent domains (H3K27me3 + H3K4me3) mark key developmental genes in embryonic stem cells, maintaining promoters poised for activation.

### Prediction Task
- **Task**: Distinguish bivalently marked genes vs. unmethylated vs. H3K4me3-only
- **Training**: ~15,000 ESCs, only 56 highly conserved loci as labels
- **Results**:
  - Bivalent vs. unmethylated: AUC 0.93
  - Bivalent vs. H3K4me3-only: AUC 0.88
- **Generalization**: Model trained on 56 loci predicts genome-wide bivalent domains

### Transcription Factor Regulatory Range
- **Background**: TFs vary in genomic distance of regulatory influence (long-range vs. short-range)
- **Training**: ~34,000 cells from iPSC to cardiomyocyte differentiation
- **No ChIP-seq or distance data provided**
- **Result**: Significantly outperforms alternative methods (others near random)

---

## Network Dynamics Predictions

### Central vs. Peripheral Factors
**Problem**: Identifying hierarchy in gene networks enables therapies targeting core regulatory elements rather than peripheral effectors.

**NOTCH1 (N1)-dependent cardiac valve disease network**:
- **Training**: ~30,000 normal endothelial cells from Heart Atlas
- **No perturbation data provided**
- **Task**: Distinguish central vs. peripheral factors
- **Result**: AUC 0.81
- **Additional**: Can also distinguish N1-activated targets from non-targets

### Data Efficiency Tests
**Minimal data requirements**:
- Reducing from 30K to 5K cells: Maintains nearly equivalent performance
- Using only 884 cells from healthy vs. dilated aortas (more relevant to task):
  - Outperforms alternative methods trained on 30K general cardiac ECs
  - **Conclusion**: Data relevance > quantity

---

## Pretraining Encoded Network Hierarchy

### Attention Head Analysis
Examining pretrained Geneformer attention weights reveals:

**Layer-wise patterns**:
- **Early layers**: Diverse attention across gene ranks - initial broad survey of cell state
- **Middle layers**: Broadest coverage
- **Final layers**: Dominated by centrality-driven attention focusing on highest ranked genes

**Biological learning** (unsupervised):
- ~20% of attention heads significantly attend to transcription factors more than other genes
- Specific heads attend more to central regulatory nodes than peripheral genes
- Centrality-driven heads consistently attend to highest ranked genes across different cell types

### TF Importance Learning
The model learned, in a completely self-supervised manner, the relative importance of transcription factors in distinguishing cell states.

---

## In Silico Gene Network Analysis

### Network Connection Inference
**Pretrained model encodes TF-target relationships prior to fine-tuning**:

**GATA4 validation** (fetal cardiomyocytes):
- In silico deletion has higher effect on:
  - Direct targets (ChIP-seq defined) vs. indirect targets
  - Direct targets vs. housekeeping genes
  - Previously reported disease-dysregulated genes

**TBX5 replication**:
- Same pattern: direct targets > indirect targets > housekeeping genes

### Cooperative Action Detection
**GATA4-TBX5 interaction**:
- Background: GATA4 variant disrupts interaction with partner TF TBX5
- **Test**: Single deletion vs. combined deletion
- **Results**:
  - Individual deletions significantly affect co-bound targets vs. housekeeping
  - Combined deletion has even greater impact than sum of individual effects
  - **Conclusion**: Model recognizes cooperative TF action, not just additive effects

---

## In Silico Treatment Analysis

### Disease Modeling Workflow

**Step 1: Model fine-tuning**
- Distinguish: non-failing cardiomyocytes vs. hypertrophic cardiomyopathy (HCM) vs. dilated cardiomyopathy (DCM)
- Training: non-failing n=9, HCM n=11, DCM n=9 (93,589 cells total)
- Out-of-sample accuracy: 90%

**Step 2: Disease modeling**
- In non-failing cardiomyocytes: delete/activate random genes to establish background distribution
- Identify genes whose perturbation shifts embeddings toward HCM or DCM states

**Step 3: Treatment modeling**
- In HCM/DCM patient cells: identify pathways whose inhibition/activation shifts embeddings back toward non-failing state

### Candidate Therapeutic Targets
**Hypertrophic cardiomyopathy**:
- Top pathways: ADCY5 (associated with longevity and cardioprotection in mice), SRPK3 (druggable, downstream of MEF2)
- Enrichment: Titin binding, sarcomere organization pathways

**Dilated cardiomyopathy**:
- Enrichment: Muscle contraction, mitochondrial function pathways
- 478 genes whose loss shifts toward DCM state

### Experimental Validation
**TTN+/- iPSC model** (truncating mutation, leading cause of DCM):
- Control: TTN+/- cells show reduced contractile stress vs. wild-type
- CRISPR knockout of Geneformer-predicted targets (GSN and PLN) in TTN+/- cells
- **Result**: Significant improvement in contractile stress
- **Conclusion**: In silico treatment predictions validated by functional rescue

---

## Discussion

### Core Contributions
1. **Foundation model for network biology**: Pretraining on large-scale single-cell data enables predictions with limited task-specific data
2. **In silico perturbation capabilities**: Context-aware deletion/activation strategies for dosage sensitivity analysis and therapeutic target discovery
3. **Experimental validation**: Predicted candidates (TEAD4, GSN, PLN) validated by CRISPR experiments

### Scalability and Future Directions
- Pretraining with larger, more diverse corpora consistently improves downstream performance
- As public single-cell data expands, future models pretrained on even larger scales may achieve meaningful predictions with increasingly minimal task-specific data
- Model represents democratized knowledge that can be fine-tuned toward broad range of downstream applications

### Limitations and Considerations
- Rank value encoding does not fully exploit precise transcript count measurements
- Hyperparameter tuning generally enhances performance (results shown use intentionally uniform parameters)
- Performance is task-dependent and related to relevance of pretraining corpus to specific application

---

## Methods Summary

### Rank-Value Encoding Calculation
```
1. Global statistics:
   For each gene g, calculate non-zero median m_g across Genecorpus-30M

2. Per-cell normalization:
   For cell c with total reads T_c:
   normalized_score_{g,c} = (count_{g,c} / T_c) / m_g

3. Ranking:
   Within each cell, sort genes by normalized_score descending
   Take top 2048 as input sequence

4. Tokenization:
   Map gene Ensembl IDs to vocabulary tokens
   Add <pad> and <mask> special tokens
```

### Fine-tuning Protocol
- **Initialization**: Pretrained Geneformer weights
- **Architecture addition**: Task-specific transformer layer
- **Hyperparameters** (uniform across tasks):
  - Learning rate: 5e-5
  - Warmup: 500 steps
  - Weight decay: 0.001
  - Batch size: 12
  - Epochs: 1 (to avoid overfitting)
- **Layer freezing strategy**:
  - Tasks similar to pretraining: freeze more layers
  - Tasks distant from pretraining: freeze fewer layers

### In Silico Perturbation Implementation
**Deletion**:
- Remove target gene token(s) from rank value encoding
- Forward pass through model
- Compare embeddings before/after via cosine similarity

**Activation**:
- Move target gene(s) to front of rank value encoding
- Same comparison process

**Metrics**:
- Cell embedding impact: 1 - cos(E_cell_original, E_cell_deleted)
- Gene embedding impact: Per-gene embedding changes for downstream targets

### Evaluation
- Gene classification: 5-fold cross-validation (80/20 split), report AUC and F1
- Cell classification: Train/validation/test splits, out-of-sample accuracy

---

## Key Terms Glossary

| Term | Definition |
|------|------------|
| **scRNA-seq** | Single-cell RNA sequencing |
| **TF** | Transcription Factor - protein regulating gene expression |
| **CNV** | Copy Number Variation - genetic alteration affecting gene dosage |
| **DCM** | Dilated Cardiomyopathy - heart condition with enlarged, weakened ventricles |
| **HCM** | Hypertrophic Cardiomyopathy - heart condition with thickened heart muscle |
| **iPSC** | induced Pluripotent Stem Cell - reprogrammed cells with pluripotency |
| **MLM** | Masked Language Modeling - pretraining objective masking and predicting tokens |
| **Embedding** | Dense vector representation capturing semantic information |
| **Attention** | Mechanism allowing model to focus on relevant parts of input |
| **In silico** | Computer simulation or virtual experiment |
| **Dosage-sensitive** | Genes where copy number changes cause phenotypic effects |
| **Central node** | Core regulatory element in network with broad influence |
| **Peripheral node** | Downstream effector in network with limited influence |
| **Rank-value encoding** | Non-parametric representation ranking genes by normalized expression |
| **Transfer learning** | Leveraging knowledge from pretraining for new tasks |
| **Foundation model** | Large-scale pretrained model adaptable to diverse downstream tasks |