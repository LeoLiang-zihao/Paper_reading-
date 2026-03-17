# Cell2Sentence (C2S) 技术文档

> **Cell2Sentence: 使用大语言模型进行单细胞分析**

---

## 1. 概述

### 1.1 核心目标和功能

Cell2Sentence (C2S) 是一个创新性框架，旨在将**单细胞RNA测序（scRNA-seq）数据转换为自然语言形式**，使大语言模型（LLMs）能够原生处理和理解单细胞转录组学数据。

**核心功能包括：**
- 将基因表达矩阵转换为"细胞句子"（Cell Sentences）
- 使用LLM进行细胞类型预测
- 基于细胞类型条件生成新细胞
- 提取细胞嵌入向量用于下游分析
- 组织类型预测和自然语言生物学解释

### 1.2 核心创新点

1. **秩序变换（Rank Transformation）**：将基因表达值转换为基因名的排序序列，保留表达模式的相对信息
2. **自然语言桥接**：利用LLM的预训练知识和自然语言能力处理生物学数据
3. **双向转换**：支持从细胞句子逆向重建表达向量，保留约87%的原始方差
4. **多任务统一框架**：单一模型支持预测、生成、嵌入等多种任务
5. **规模化扩展**：C2S-Scale模型扩展到27B参数，支持更复杂的单细胞任务

### 1.3 与现有方法的对比

| 特性 | Cell2Sentence | 传统方法 | 其他基础模型 |
|------|---------------|----------|--------------|
| 数据表示 | 自然语言（基因序列） | 数值向量 | 专用嵌入 |
| 模型基础 | 预训练LLM | 从头训练 | 专用架构 |
| 可解释性 | 高（自然语言输出） | 低 | 中等 |
| 迁移能力 | 强（利用LLM知识） | 弱 | 中等 |
| 任务灵活性 | 高（提示驱动） | 低（任务特定） | 中等 |

### 1.4 解决的实际问题

- **细胞类型注释**：自动化识别未知细胞群体的类型
- **数据增强**：生成虚拟细胞数据用于训练和分析
- **跨数据集泛化**：利用LLM的预训练知识实现零样本/少样本迁移
- **生物学解释**：自动生成细胞群体的自然语言摘要

---

## 2. 核心架构

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Cell2Sentence 系统架构                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │   AnnData       │    │   CSData        │    │     CSModel             │ │
│  │   (h5ad)        │───▶│   (Arrow)       │───▶│     (LLM Wrapper)       │ │
│  │                 │    │                 │    │                         │ │
│  │ • Expression    │    │ • cell_sentence │    │ • Tokenizer             │ │
│  │ • obs metadata  │    │ • metadata      │    │ • CausalLM              │ │
│  │ • var (genes)   │    │ • vocabulary    │    │ • Training/Inference    │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘ │
│          │                      │                         │                │
│          ▼                      ▼                         ▼                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     数据转换流程 (Transformation Pipeline)            │   │
│  │                                                                      │   │
│  │  Expression Matrix     Rank Ordering       Cell Sentence            │   │
│  │  ┌─────────────┐      ┌─────────────┐     ┌─────────────────────┐   │   │
│  │  │ Gene  Expr  │      │ Rank  Gene  │     │ "MALAT1 TMSB4X B2M  │   │   │
│  │  │ MALAT1 3.2  │ ───▶ │  1   MALAT1 │ ──▶│  MT-CO1 RPLP1 ACTB  │   │   │
│  │  │ TMSB4X 2.8  │      │  2   TMSB4X │     │  MT-CO2 RPS27..."   │   │   │
│  │  │ B2M    2.5  │      │  3   B2M    │     └─────────────────────┘   │   │
│  │  │ ...    ...  │      │ ...  ...    │                               │   │
│  │  └─────────────┘      └─────────────┘                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      提示格式化 (Prompt Formatting)                   │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │ Model Input:                                                  │   │   │
│  │  │ "The following is a list of 200 gene names ordered by        │   │   │
│  │  │  descending expression level in a Homo sapiens cell.         │   │   │
│  │  │  Cell sentence: MALAT1 TMSB4X B2M MT-CO1...                  │   │   │
│  │  │  The cell type corresponding to these genes is:"              │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │ Model Output: "macrophage."                                   │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         任务类型 (Task Types)                        │   │
│  │                                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │ Cell Type   │  │ Cell        │  │ Cell        │  │ Tissue      │ │   │
│  │  │ Prediction  │  │ Generation  │  │ Embedding   │  │ Prediction  │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心类代码结构

#### CSData 类

```python
class CSData:
    """数据管理包装类"""
    
    def __init__(self, vocab, data_path, dataset_backend='arrow'):
        self.vocab = vocab           # OrderedDict: {gene_name: num_expressed_cells}
        self.data_path = data_path   # Arrow数据集路径
        self.dataset_backend = dataset_backend
    
    @classmethod
    def adata_to_arrow(cls, adata, random_state, sentence_delimiter, label_col_names):
        """AnnData → Arrow数据集 + 词汇表"""
        # 返回: (arrow_dataset, vocabulary)
        pass
    
    @classmethod
    def csdata_from_arrow(cls, arrow_dataset, vocabulary, save_dir, save_name):
        """从Arrow数据集创建CSData对象"""
        pass
    
    def get_sentence_strings(self):
        """获取所有细胞句子"""
        pass
```

#### CSModel 类

```python
class CSModel:
    """模型包装类"""
    
    def __init__(self, model_name_or_path, save_dir, save_name):
        self.model_name_or_path = model_name_or_path
        self.save_dir = save_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # 加载/保存模型...
    
    def fine_tune(self, csdata, task, train_args, loss_on_response_only, 
                  top_k_genes, max_eval_samples, ...):
        """微调模型"""
        pass
    
    def generate_from_prompt(self, model, prompt, max_num_tokens, **kwargs):
        """单个prompt生成"""
        pass
    
    def generate_from_prompt_batched(self, model, prompt_list, max_num_tokens, **kwargs):
        """批量生成"""
        pass
    
    def embed_cells_batched(self, model, prompt_list, max_num_tokens):
        """批量获取细胞嵌入"""
        pass
```

### 2.3 数据流动方式

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   AnnData     │     │  Vocabulary   │     │ Cell Sentence │     │   Prompt      │
│   Input       │────▶│  Generation   │────▶│  Generation   │────▶│   Formatting  │
│               │     │               │     │               │     │               │
│ Shape:        │     │ OrderedDict:  │     │ List[str]:    │     │ Dict:         │
│ (N_cells,     │     │ {gene: count} │     │ "GENE1 GENE2  │     │ model_input   │
│  N_genes)     │     │               │     │  GENE3..."    │     │ response      │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
                                                                          │
                                                                          ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   Output      │     │   Decoding    │     │   Model       │     │  Tokenization │
│   (Text/      │◀────│   + Post-     │◀────│   Forward     │◀────│               │
│   Embedding)  │     │   processing  │     │   Pass        │     │               │
│               │     │               │     │               │     │ input_ids     │
│ Cell type /   │     │ Remove non-   │     │ CausalLM      │     │ attention_mask│
│ Generated cell│     │ genes, avg    │     │ output        │     │ labels        │
└───────────────┘     │ duplicates    │     └───────────────┘     └───────────────┘
                      └───────────────┘
```

### 2.4 关键维度变换

| 阶段 | 输入形状 | 输出形状 | 说明 |
|------|----------|----------|------|
| 表达矩阵 → 细胞句子 | `(N_cells, N_genes)` | `List[str]` 长度 N_cells | 每个细胞一个基因名序列 |
| 细胞句子 → Token IDs | `str` | `(seq_len,)` | 根据tokenizer词表映射 |
| Token IDs → Embeddings | `(batch, seq_len)` | `(batch, seq_len, hidden_dim)` | hidden_dim通常为1024-4096 |
| 细胞嵌入提取 | `(batch, seq_len, hidden_dim)` | `(batch, hidden_dim)` | 最后一层mean pooling |
| 表达重建 | `List[str]` (genes) | `(N_genes,)` | 使用线性模型逆向映射 |

---

## 3. 模型组件详解

### 3.1 数据处理模块 (CSData)

**功能说明**：管理单细胞数据的加载、转换和存储，将AnnData格式转换为Arrow数据集格式。

**架构参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| vocab | OrderedDict | - | 基因词汇表及其表达细胞数 |
| data_path | str | - | Arrow数据集存储路径 |
| dataset_backend | str | 'arrow' | 数据后端类型 |
| sentence_delimiter | str | ' ' | 细胞句子中基因名分隔符 |

**关键代码片段 - 细胞句子生成**：

```python
def generate_sentences(adata, vocab, delimiter=' ', random_state=42):
    """
    将表达矩阵转换为细胞句子
    基因按表达量降序排列
    """
    np.random.seed(random_state)
    mat = sparse.csr_matrix(adata.X)
    enc_map = list(vocab.keys())
    
    sentences = []
    for i in tqdm(range(mat.shape[0])):
        cols = mat.indices[mat.indptr[i] : mat.indptr[i + 1]]
        vals = mat.data[mat.indptr[i] : mat.indptr[i + 1]]
        cols, vals = shuffle(cols, vals)  # 随机打乱以处理ties
        # 按表达值降序排列基因
        sentences.append(delimiter.join([enc_map[x] for x in cols[np.argsort(-vals, kind="stable")]]))
    
    return sentences
```

**设计动机**：
- 使用稀疏矩阵操作提高大规模数据处理效率
- 随机打乱后稳定排序处理表达值相同的基因
- Arrow格式提供高效的磁盘存储和内存映射

### 3.2 模型包装模块 (CSModel)

**功能说明**：封装Hugging Face的CausalLM模型，提供训练、推理和嵌入提取接口。

**架构参数表**：

| 参数 | 类型 | 说明 |
|------|------|------|
| model_name_or_path | str | HuggingFace模型名或本地路径 |
| save_dir | str | 模型保存目录 |
| save_name | str | 模型保存名称 |

**支持的基础模型**：

| 模型 | 参数量 | 用途 |
|------|--------|------|
| Pythia-160M | 160M | 快速实验 |
| Pythia-410M | 410M | 标准任务 |
| Pythia-1B | 1B | 复杂任务 |
| Gemma-2 2B | 2B | 高性能任务 |
| Gemma-2 27B | 27B | 最高性能 |

**关键代码片段 - 生成推理**：

```python
def generate_from_prompt_batched(self, model, prompt_list, max_num_tokens=1024, **kwargs):
    """批量文本生成"""
    tokens = self.tokenizer(prompt_list, padding=True, return_tensors='pt')
    input_ids = tokens['input_ids'].to(self.device)
    attention_mask = tokens['attention_mask'].to(self.device)
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_num_tokens,
        pad_token_id=self.tokenizer.pad_token_id,
        **kwargs
    )
    pred_list = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # 移除输入提示，只保留生成内容
    predictions_without_input_prompt = []
    for pred, prompt in zip(pred_list, prompt_list):
        pred_cleaned = pred.replace(prompt, "").replace("<|endoftext|>", "").lstrip()
        predictions_without_input_prompt.append(pred_cleaned)
    
    return predictions_without_input_prompt
```

**关键代码片段 - 嵌入提取**：

```python
def embed_cells_batched(self, model, prompt_list, max_num_tokens=1024):
    """批量获取细胞嵌入向量"""
    tokens = self.tokenizer(prompt_list, padding=True, return_tensors='pt')
    input_ids = tokens['input_ids'].to(self.device)
    attention_mask = tokens['attention_mask'].to(self.device)
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    
    # 取最后一层，沿序列维度平均
    all_embeddings = []
    for idx in range(len(prompt_list)):
        embedding = outputs.hidden_states[-1][idx].mean(0).detach().cpu().numpy()
        all_embeddings.append(embedding)
    return all_embeddings
```

### 3.3 提示格式化模块 (PromptFormatter)

**功能说明**：将细胞句子格式化为LLM能理解的提示模板。

**支持的任务类型**：

| 任务 | 输入键 | 输出键 | 说明 |
|------|--------|--------|------|
| cell_type_prediction | num_genes, organism, cell_sentence | cell_type | 预测细胞类型 |
| cell_type_generation | num_genes, organism, cell_type | cell_sentence | 生成细胞 |
| tissue_prediction | num_genes, num_cells, organism, multi_cell_sentences | tissue | 组织预测 |
| natural_language_interpretation | num_genes, num_cells, organism, multi_cell_sentences | abstract | 生物学解释 |

**提示模板示例 - 细胞类型预测**：

```json
{
    "model_input": [
        "The following is a list of {num_genes} gene names ordered by descending expression level in a {organism} cell. Your task is to give the cell type which this cell belongs to based on its gene expression.\nCell sentence: {cell_sentence}.\nThe cell type corresponding to these genes is:"
    ],
    "response": ["{cell_type}."]
}
```

**关键代码片段**：

```python
def format_hf_ds(self, hf_ds):
    """格式化HuggingFace数据集"""
    model_inputs_list = []
    responses_list = []
    model_input_keys, response_keys = self.get_keys_for_task()
    
    for cell_idx in range(hf_ds.num_rows):
        sample = hf_ds[cell_idx]
        
        # 获取细胞句子（限制基因数量）
        single_cell_sentence_str, num_genes_str = get_cell_sentence_str(
            sample, num_genes=self.top_k_genes
        )
        sample["cell_sentence"] = single_cell_sentence_str
        sample["num_genes"] = num_genes_str
        
        # 随机选择一个输入模板并填充
        model_input_str = random.choice(self.prompts_dict["model_input"])
        model_input_str = model_input_str.format(**{key: sample[key] for key in model_input_keys})
        
        # 格式化响应
        response_str = self.prompts_dict["response"][0]
        response_str = response_str.format(**{key: sample[key] for key in response_keys})
        
        model_inputs_list.append(model_input_str)
        responses_list.append(response_str)
    
    return Dataset.from_dict({
        "sample_type": [self.task] * hf_ds.num_rows,
        "model_input": model_inputs_list,
        "response": responses_list,
    })
```

---

## 4. 数据预处理与输入格式

### 4.1 输入数据格式

**原始输入**：AnnData对象 (`.h5ad`文件)

```
adata.X: (N_cells, N_genes) - 表达矩阵（原始或归一化计数）
adata.obs: DataFrame - 细胞元数据（cell_type, tissue, organism等）
adata.var: DataFrame - 基因信息（gene_names）
```

**处理后格式**：Arrow Dataset

```python
{
    "cell_name": str,           # 细胞唯一标识符
    "cell_sentence": str,       # 空格分隔的基因名序列
    "cell_type": str,           # 细胞类型标签（可选）
    "tissue": str,              # 组织类型（可选）
    "organism": str,            # 物种（Homo sapiens / Mus musculus）
    # ... 其他元数据
}
```

### 4.2 预处理流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        数据预处理流程                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. 质量控制过滤                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │ sc.pp.filter_cells(adata, min_genes=200)  # 每个细胞至少200个基因  │    │
│  │ sc.pp.filter_genes(adata, min_cells=3)    # 每个基因至少3个细胞表达 │   │
│  └──────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  2. 计数归一化                                                            │
│  ┌────────────────────────────────────────────────────────────────── ┐   │
│  │ sc.pp.normalize_total(adata)  # 总计数归一化到中位数               │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  3. Log变换 (⚠️ 必须使用base=10)                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │ sc.pp.log1p(adata, base=10)   # 重要：C2S使用log10而非自然对数     │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  4. 细胞句子转换                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │ arrow_ds, vocab = CSData.adata_to_arrow(adata, ...)              │    │
│  │ 基因按log10表达值降序排列，生成空格分隔的基因名序列                   │   │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| min_genes | 200 | 最少基因数过滤阈值 |
| min_cells | 3 | 最少细胞数过滤阈值 |
| log_base | **10** | 对数变换底数（关键参数！） |
| top_k_genes | 100-200 | 细胞句子中使用的基因数 |
| sentence_delimiter | ' ' | 基因名分隔符 |

**⚠️ 重要提示**：C2S使用**base=10的log1p变换**而非默认的自然对数。这对于逆向重建表达向量至关重要。

---

## 5. 损失函数与优化目标

### 5.1 损失函数

C2S使用标准的**因果语言模型损失（Causal Language Modeling Loss）**：

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$$

其中：
- $x_t$：第t个token
- $x_{<t}$：前t-1个tokens
- $\theta$：模型参数
- $T$：序列长度

### 5.2 损失计算策略

提供两种策略，通过`loss_on_response_only`参数控制：

**策略1：仅在响应上计算损失** (`loss_on_response_only=True`)

```python
def tokenize_loss_on_response(examples, tokenizer, ignore_token_id=-100):
    """损失仅在模型响应部分计算"""
    prompt_inputs = examples["model_input"]
    responses = examples["response"]
    model_inputs = tokenizer(prompt_inputs)
    labels = tokenizer(responses)
    
    for i in range(len(prompt_inputs)):
        prompt_input_ids = model_inputs["input_ids"][i]
        response_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        
        # 输入 = 提示 + 响应
        model_inputs["input_ids"][i] = prompt_input_ids + response_input_ids
        # 标签：提示部分忽略(-100)，仅在响应部分计算损失
        labels["input_ids"][i] = [-100] * len(prompt_input_ids) + response_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

**策略2：在全部token上计算损失** (`loss_on_response_only=False`)

```python
def tokenize_all(examples, tokenizer):
    """损失在所有token上计算"""
    full_inputs = []
    for sample_idx in range(len(examples["model_input"])):
        full_inputs.append(
            examples["model_input"][sample_idx] + " " + examples["response"][sample_idx]
        )
    
    model_inputs = tokenizer(full_inputs)
    for i in range(len(full_inputs)):
        model_inputs["input_ids"][i] += [tokenizer.eos_token_id]
        model_inputs["attention_mask"][i] += [tokenizer.eos_token_id]
    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs
```

### 5.3 两种损失策略的设计动机 ⚠️重要澄清

**为什么提供两种策略？** 这**不是**为了多任务训练，而是针对**不同任务特性**的优化策略：

| 策略 | 适用任务 | 原因 |
|------|----------|------|
| `loss_on_response_only=True` | **细胞类型预测** | 提示中包含细胞句子（长序列），我们只关心模型能否正确预测细胞类型（短回答）。在长提示上计算loss会稀释学习信号 |
| `loss_on_response_only=False` | **细胞生成**、**组织预测** | 响应本身就是长序列（生成的细胞句子），或者我们希望模型也学习理解提示中的细胞句子模式 |

**关键点**：
1. **不是多任务联合训练**：C2S的不同任务（预测、生成、嵌入）是**分别独立训练**的
2. **没有前后依赖关系**：可以直接加载预训练的C2S模型进行任意任务的微调
3. **策略选择基于任务性质**：
   - 预测任务（短输出）→ 推荐 `loss_on_response_only=True`
   - 生成任务（长输出）→ 可用 `loss_on_response_only=False`

```python
# 不同任务的训练是独立的，每次fine_tune只训练一个任务
csmodel.fine_tune(csdata, task="cell_type_prediction", loss_on_response_only=True, ...)
# 或者
csmodel.fine_tune(csdata, task="cell_type_generation", loss_on_response_only=False, ...)
```

### 5.5 训练样本构建

| 任务类型 | 输入（不计算损失） | 输出（计算损失） |
|----------|-------------------|------------------|
| 细胞类型预测 | 提示 + 细胞句子 | 细胞类型 |
| 细胞生成 | 提示 + 细胞类型 | 细胞句子 |
| 组织预测 | 提示 + 多细胞句子 | 组织类型 |

### 5.6 数据来源澄清 ⚠️重要

**数据不是用GPT/LLM生成的！** 所有训练数据均来自真实的单细胞数据集：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          数据来源流程                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  真实scRNA-seq数据集                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ • CellxGene (57M+ 细胞)                                              │   │
│  │ • Human Cell Atlas                                                    │   │
│  │ • Domínguez Conde Immune Tissue 等                                   │   │
│  │                                                                       │   │
│  │ 每个数据集本身就包含：                                                │   │
│  │   - 表达矩阵 (adata.X)                                               │   │
│  │   - 细胞类型标签 (adata.obs["cell_type"]) ← 人工注释/聚类分析得到    │   │
│  │   - 组织信息 (adata.obs["tissue"])                                   │   │
│  │   - 物种 (adata.obs["organism"])                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  C2S数据转换（自动，无需LLM）                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 表达矩阵 → Rank排序 → 细胞句子 (自动生成)                            │   │
│  │ 标签直接使用数据集原有的注释                                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  提示模板（人工设计的固定模板，不是LLM生成）                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ "The following is a list of {num_genes} gene names ordered by        │   │
│  │  descending expression level in a {organism} cell..."                │   │
│  │                                                                       │   │
│  │ 共有几种固定的提示模板变体，在训练时随机选择                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**总结**：
- **细胞句子**：从真实表达数据自动转换，不是生成的
- **标签**：使用数据集原有的人工注释（cell_type, tissue等）
- **提示模板**：人工设计的固定模板（JSON文件中定义），随机选择增加多样性
- **没有使用GPT或其他LLM生成任何训练数据**

---

## 6. 训练策略

### 6.1 训练配置

**推荐训练参数**：

```python
train_args = TrainingArguments(
    bf16=True,                          # 使用bfloat16混合精度
    fp16=False,
    per_device_train_batch_size=8,      # 每设备批大小
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,       # 梯度累积（有效批大小=32）
    gradient_checkpointing=False,
    learning_rate=1e-5,                  # 学习率
    load_best_model_at_end=True,
    logging_steps=50,
    lr_scheduler_type="cosine",          # 余弦退火
    num_train_epochs=5,
    eval_steps=50,
    evaluation_strategy="steps",
    save_steps=100,
    save_total_limit=3,                  # 保存最近3个检查点
    warmup_ratio=0.05,                   # 5%预热
    output_dir=output_dir
)
```

### 6.2 超参数设置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| learning_rate | 1e-5 ~ 5e-5 | 微调学习率 |
| batch_size | 8-16 | 根据GPU内存调整 |
| gradient_accumulation | 2-8 | 增大有效批大小 |
| warmup_ratio | 0.05-0.1 | 学习率预热比例 |
| num_epochs | 3-10 | 训练轮数 |
| top_k_genes | 100-200 | 细胞句子长度 |
| max_eval_samples | 500 | 验证集采样数 |

### 6.3 数据划分

```python
def train_test_split_arrow_ds(arrow_ds):
    """80/10/10 训练/验证/测试划分"""
    cell_indices_list = list(range(arrow_ds.num_rows))
    train_and_val_indices, test_indices = train_test_split(
        cell_indices_list, test_size=0.1
    )
    train_indices, val_indices = train_test_split(
        train_and_val_indices, test_size=0.11  # 0.11 * 0.9 ≈ 0.1
    )
    # 返回DatasetDict和索引字典
```

### 6.4 多阶段训练策略

C2S-Scale模型采用多阶段训练：

1. **预训练阶段**：在大规模单细胞数据上进行下一基因预测
2. **任务特定微调**：针对具体任务（预测/生成）微调
3. **数据集适配**：在目标数据集上进一步微调

---

## 7. 推理流程与结果生成 ⭐

### 7.1 推理流程详解

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          完整推理流程                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入: AnnData对象                                                          │
│    Shape: (N_cells, N_genes)                                                │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Step 1: 数据转换                                                      │   │
│  │   adata → arrow_ds (细胞句子)                                         │   │
│  │   Shape: List[str], length=N_cells                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Step 2: 提示格式化                                                    │   │
│  │   cell_sentence → formatted_prompt                                    │   │
│  │   "MALAT1 TMSB4X..." → "The following is a list of 200 genes..."     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Step 3: Tokenization                                                  │   │
│  │   text → input_ids                                                    │   │
│  │   Shape: (batch_size, seq_len) e.g., (8, 512)                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Step 4: 模型前向传播                                                  │   │
│  │   CausalLM forward pass                                               │   │
│  │   input_ids → logits / hidden_states                                  │   │
│  │   Shape: logits (batch, seq_len, vocab_size)                         │   │
│  │          hidden_states (batch, seq_len, hidden_dim)                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│              ┌───────────────┼───────────────┐                              │
│              ▼               ▼               ▼                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐               │
│  │ 预测任务        │ │ 生成任务        │ │ 嵌入任务        │               │
│  │                 │ │                 │ │                 │               │
│  │ model.generate  │ │ model.generate  │ │ mean(hidden     │               │
│  │ max_new=64      │ │ max_new=1024    │ │ _states[-1])    │               │
│  │                 │ │ do_sample=True  │ │                 │               │
│  │ output: text    │ │ top_k=50        │ │ output: vector  │               │
│  │ e.g."macrophage"│ │                 │ │ Shape: (hidden  │               │
│  │                 │ │ output: genes   │ │ _dim,)          │               │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 结果生成方法分析

#### a) 输出是如何生成的？

**细胞类型预测**：
- 最终输出层：**CausalLM的softmax层**
- 输出维度：`(vocab_size,)` ≈ 50,000+ tokens
- 生成方式：自回归生成，直到遇到EOS token或达到max_new_tokens

```python
# 细胞类型预测生成
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=64,           # 细胞类型名通常较短
    pad_token_id=tokenizer.pad_token_id,
)
# 输出: "macrophage." 或 "CD8-positive, alpha-beta memory T cell."
```

**细胞生成**：
- 输出：基因名序列（细胞句子）
- 使用采样策略增加多样性

```python
# 细胞生成
outputs = model.generate(
    input_ids=input_ids,
    max_new_tokens=1024,         # 生成较长的细胞句子
    do_sample=True,              # 启用采样
    top_k=50,                    # Top-K采样
    top_p=0.95,                  # Nucleus采样
)
# 输出: "MALAT1 TMSB4X B2M MT-CO1 RPLP1 ACTB..."
```

**细胞嵌入** ⚠️重要澄清：
- 输出：最后隐藏层**所有位置**的**平均池化**
- 维度：`(hidden_dim,)` 通常为1024-4096

```python
outputs = model(
    input_ids=input_ids,
    output_hidden_states=True
)
# 取最后一层，沿序列维度平均（不是最后一个token！）
embedding = outputs.hidden_states[-1][idx].mean(0)  # Shape: (hidden_dim,)
```

**为什么用平均池化而不是取最后一个token？**

```
输入序列:  [提示词...] [MALAT1] [TMSB4X] [B2M] [MT-CO1] ...
位置:        1...k      k+1      k+2     k+3    k+4    ...
隐藏状态:    h_1...h_k   h_{k+1}  h_{k+2} h_{k+3} h_{k+4} ...

方法1 (NOT USED): 取最后一个token → h_T 
  问题：只包含序列末尾的信息，丢失了前面基因的贡献

方法2 (C2S使用): 平均池化 → mean(h_1, h_2, ..., h_T)
  优势：
  - 整合了所有token（基因）的语义信息
  - 每个基因对嵌入都有贡献
  - 更稳健，不依赖于单个位置
```

**细胞嵌入的用途**：
- **不是**用来做生成任务的中间表示
- **是**用于下游分析：UMAP可视化、聚类、分类、批次效应校正等
- 可以将嵌入存入 `adata.obsm["c2s_embeddings"]` 与Scanpy集成

#### b) 生成过程的数学推理

**自回归生成数学表示**：

给定输入序列 $x_{1:t}$，预测下一个token的概率分布：

$$P(x_{t+1} | x_{1:t}) = \text{softmax}(W_o \cdot h_t + b_o)$$

其中：
- $h_t$：最后一层第t个位置的隐藏状态
- $W_o$：输出投影矩阵 `(hidden_dim, vocab_size)`
- $b_o$：偏置向量

**Top-K采样**：

$$P'(x) = \begin{cases} \frac{P(x)}{\sum_{x' \in V_K} P(x')} & \text{if } x \in V_K \\ 0 & \text{otherwise} \end{cases}$$

其中 $V_K$ 是概率最高的K个tokens。

**细胞嵌入计算**：

$$e_{\text{cell}} = \frac{1}{T} \sum_{t=1}^{T} h_t^{(L)}$$

其中 $h_t^{(L)}$ 是最后一层（第L层）第t个位置的隐藏状态。

#### c) 后处理步骤

**细胞类型预测后处理**：

```python
# 移除输入提示
pred_cleaned = pred.replace(prompt, "")
pred_cleaned = pred_cleaned.replace("<|endoftext|>", "")
pred_cleaned = pred_cleaned.lstrip()

# 移除末尾句号
if model_pred[-1] == ".":
    model_pred = model_pred[:-1]
```

**生成细胞后处理**：

```python
def post_process_generated_cell_sentences(cell_sentence, vocab_list, replace_nonsense_string="NOT_A_GENE"):
    """
    1. 将非基因词替换为标记
    2. 处理重复基因：计算平均位置
    """
    words = cell_sentence.upper().split(" ")
    
    # 替换非基因词
    generated_gene_names = [
        word if word in vocab_list else replace_nonsense_string 
        for word in words
    ]
    
    # 处理重复基因
    for gene_name in gene_name_to_occurrences:
        if gene_name_to_occurrences[gene_name] > 1 and gene_name != replace_nonsense_string:
            occurrence_positions = [idx for idx, elem in enumerate(post_processed_sentence) if elem == gene_name]
            average_position = int(sum(occurrence_positions) / len(occurrence_positions))
            
            # 移除所有出现，在平均位置重新插入
            post_processed_sentence = [elem for elem in post_processed_sentence if elem != gene_name]
            post_processed_sentence.insert(average_position, gene_name)
    
    return post_processed_sentence, num_genes_replaced
```

**表达向量重建**：

```python
def reconstruct_expression_from_cell_sentence(cell_sentence_str, delimiter, vocab_list, slope, intercept):
    """
    使用线性模型从细胞句子重建表达向量
    predicted_expression = intercept + slope * log10(rank + 1)
    """
    cell_sentence = cell_sentence_str.split(delimiter)
    expression_vector = np.zeros(len(vocab_list), dtype=np.float32)
    
    log_ranks = np.log10(1 + np.arange(len(cell_sentence)))
    
    for pos, gene_name in enumerate(cell_sentence):
        gene_idx = gene_to_index.get(gene_name)
        if gene_idx is not None:
            gene_expr_val = intercept + (slope * log_ranks[pos])
            expression_vector[gene_idx] = gene_expr_val
    
    return expression_vector
```

#### d) 不同任务模式下的结果生成

| 任务 | 生成策略 | 输出格式 | 典型长度 |
|------|----------|----------|----------|
| 细胞类型预测 | 贪婪解码 | 文本（细胞类型名） | 1-20 tokens |
| 细胞生成 | Top-K/Top-P采样 | 文本（基因名序列） | 200-1000 tokens |
| 细胞嵌入 | 无（直接取隐藏状态） | 向量 | hidden_dim |
| 组织预测 | 贪婪解码 | 文本（组织名） | 1-10 tokens |
| 自然语言解释 | 采样 | 文本（摘要） | 50-200 tokens |

### 7.3 结果解读指南

**细胞类型预测**：
- 输出：细胞类型名称字符串
- 评估：与真实标签比较计算准确率
- 典型准确率：微调后可达80%+

**生成细胞质量评估**：
- 非基因词数量：平均应<2个
- 可通过UMAP可视化验证生成细胞是否与真实细胞在同一分布

**嵌入向量应用**：
- 可用于：降维可视化、聚类、分类
- 使用Scanpy的neighbors/UMAP流程处理

---

## 8. 评估指标与实验设计

### 8.1 评估指标

| 任务 | 主要指标 | 次要指标 |
|------|----------|----------|
| 细胞类型预测 | Accuracy | F1-score, Confusion Matrix |
| 细胞生成 | 非基因词率 | UMAP分布一致性, 基因表达热图相似度 |
| 嵌入质量 | 聚类一致性 | Silhouette Score, ARI |
| 逆向重建 | R² Score | Pearson/Spearman相关系数 |

### 8.2 基准数据集

| 数据集 | 细胞数 | 物种 | 用途 |
|--------|--------|------|------|
| CellxGene | 57M+ | Human/Mouse | 预训练 |
| Human Cell Atlas | 多种组织 | Human | 预训练/评估 |
| Domínguez Conde Immune | ~30K | Human | 教程示例 |

### 8.3 逆向重建评估

```python
# 基准测试结果示例
R² Score: 0.867           # 解释了86.7%的方差
Pearson r: ~0.93
Spearman r: ~0.93

# 线性模型参数（log10 rank → normalized expression）
slope: -0.676
intercept: 2.238
```

---

## 9. 扩展功能与应用

### 9.1 迁移学习

```python
# 从预训练模型开始微调
csmodel = cs.CSModel(
    model_name_or_path="vandijklab/C2S-Pythia-410m-cell-type-prediction",
    save_dir="./my_model",
    save_name="finetuned_model"
)

# 在新数据集上微调
csmodel.fine_tune(csdata, task="cell_type_prediction", ...)
```

### 9.2 下游任务应用

1. **细胞注释**：自动化细胞类型注释工作流
2. **数据增强**：生成虚拟细胞扩充训练数据
3. **批次效应分析**：使用嵌入进行数据整合
4. **差异分析**：比较不同条件下的细胞句子差异

### 9.3 与Scanpy集成

```python
# 将C2S嵌入集成到Scanpy工作流
adata.obsm["c2s_embeddings"] = embedded_cells
sc.pp.neighbors(adata, use_rep="c2s_embeddings")
sc.tl.umap(adata)
sc.pl.umap(adata, color="cell_type")
```



## 12. 技术栈总结

| 类别 | 具体技术 |
|------|----------|
| **深度学习框架** | PyTorch |
| **预训练模型** | Pythia (160M-1B), Gemma-2 (2B-27B) |
| **模型库** | Hugging Face Transformers |
| **数据处理** | AnnData, Scanpy, Arrow/Datasets |
| **数值计算** | NumPy, SciPy |
| **数据分析** | Pandas, scikit-learn |
| **可视化** | Matplotlib, Plotnine |
| **训练加速** | Mixed Precision (bf16), Flash Attention (可选) |
| **分布式训练** | Hugging Face Accelerate |

### 依赖版本

```
torch
transformers
datasets
anndata
scanpy
numpy
pandas
scipy
tqdm
scikit-learn
accelerate
plotnine
flash-attn (optional)
```

---

## 13. 常见问题解答（FAQ）

### Q1: Cell2Sentence的核心思想是什么？

**A**: 将单细胞表达数据转换为基因名序列（按表达量降序排列），使LLM能够像处理自然语言一样处理单细胞数据。这种转换保留了基因表达的相对排序信息，并利用LLM对基因名的预训练知识。

### Q2: 为什么采用这种架构设计？

**A**: 
1. **信息保留**：秩序变换保留了约87%的表达方差
2. **LLM优势**：利用LLM对生物学术语的预训练理解
3. **灵活性**：同一架构支持预测、生成、嵌入等多种任务
4. **可解释性**：输入输出都是人类可读的文本

### Q3: 如何在自己的数据上微调？

**A**:
```python
# 1. 准备数据
adata = anndata.read_h5ad("your_data.h5ad")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata, base=10)  # 重要：使用base=10

# 2. 转换为C2S格式
arrow_ds, vocab = cs.CSData.adata_to_arrow(adata, label_col_names=["cell_type"])
csdata = cs.CSData.csdata_from_arrow(arrow_ds, vocab, save_dir, save_name)

# 3. 加载预训练模型并微调
csmodel = cs.CSModel("vandijklab/C2S-Pythia-410m-cell-type-prediction", save_dir, save_name)
csmodel.fine_tune(csdata, task="cell_type_prediction", train_args=train_args)
```

### Q4: 常见的使用误区有哪些？

**A**:
1. **使用自然对数变换**：必须使用`base=10`的log1p变换
2. **忽略数据预处理**：必须先进行标准的scanpy预处理流程
3. **期望完美重建**：逆向重建会损失约13%的方差，这是预期内的
4. **top_k_genes设置过小**：建议使用100-200个基因以获得足够的信息

### Q5: 如何选择合适的基础模型？

**A**:

| 使用场景 | 推荐模型 | 原因 |
|----------|----------|------|
| 快速实验/调试 | Pythia-160M | 轻量快速 |
| 标准任务 | Pythia-410M | 性能/效率平衡 |
| 复杂任务/大数据 | Pythia-1B / Gemma-2 2B | 更强的表示能力 |
| 最高性能 | Gemma-2 27B | 最佳效果 |

### Q6: 细胞生成的质量如何评估？

**A**:
1. **定量指标**：非基因词率（应<1%）
2. **分布一致性**：生成细胞与真实细胞在UMAP上的分布
3. **基因表达热图**：与真实数据的表达模式相似度
4. **下游任务性能**：用生成数据训练的模型在真实数据上的表现

### Q7: 推理速度如何优化？

**A**:
1. 使用**Flash Attention 2**加速长序列生成
2. 使用**批量推理**而非逐个处理
3. 使用**bfloat16/float16**精度
4. 适当减少`top_k_genes`数量

```python
# 启用Flash Attention
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

### Q8: 支持哪些物种？

**A**: 当前支持：
- **Homo sapiens**（人类）
- **Mus musculus**（小鼠）

模型在提示中明确指定物种，以便利用物种特异性的基因知识。

### Q9: 如何处理未见过的细胞类型？

**A**: C2S利用LLM的预训练知识，对未见过的细胞类型有一定的零样本泛化能力。但建议：
1. 在包含目标细胞类型的数据上微调
2. 使用C2S-Scale基础模型，它在更多数据上预训练
3. 检查生成的细胞类型是否合理

### Q10: 多细胞任务（组织预测）如何工作？它和细胞生成是什么关系？ ⚠️重要澄清

**A**: 组织预测**不是**细胞生成的多次进行！它们是**完全不同的任务**：

**组织预测 vs 细胞生成 对比**：

| 特性 | 组织预测 (tissue_prediction) | 细胞生成 (cell_type_generation) |
|------|------------------------------|--------------------------------|
| **输入** | 多个细胞句子拼接 | 细胞类型名称 |
| **输出** | 组织类型（如"lung", "spleen"） | 单个细胞句子（基因序列） |
| **目标** | 分类任务（预测组织来源） | 生成任务（创造新细胞） |
| **数据流** | 多细胞 → 组织标签 | 标签 → 单细胞 |

**组织预测的实际流程**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          组织预测任务流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入: 同一组织的5个细胞句子                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ "Consider these 100 highly expressed genes from 5 Homo sapiens       │   │
│  │  cells. Predict the associated tissue type:                          │   │
│  │                                                                       │   │
│  │  Cell 1: MALAT1 B2M MT-CO1 TMSB4X HSP90AA1 RPLP1...                  │   │
│  │  Cell 2: FTH1 DNAJB1 EEF1A1 HSPA1A RPLP1 MALAT1...                   │   │
│  │  Cell 3: FTL TMSB4X RPLP1 TMSB10 ACTB FTH1...                        │   │
│  │  Cell 4: ...                                                          │   │
│  │  Cell 5: ...                                                          │   │
│  │                                                                       │   │
│  │  The tissue type is:"                                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  输出: 组织类型名称                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ "ileum."  (或 "lung.", "spleen.", "bone marrow." 等)                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**关键区别**：
1. **组织预测**是**分类/预测**任务：给定多个细胞，预测它们来自什么组织
2. **细胞生成**是**生成**任务：给定细胞类型，生成一个新的细胞句子
3. 组织预测的输入是**拼接的多细胞句子**，输出是**短文本标签**
4. 这两个任务没有先后依赖关系，可以独立训练和使用

### Q11: scRNA-seq原始数据是什么样子？如何转换成细胞句子？

**A**: 让我用具体例子说明整个转换流程：

**Step 1: 原始scRNA-seq数据（Raw Counts）**

```
                基因A    基因B    基因C    基因D    基因E    ...    基因N (共~20000个基因)
细胞1            0        5        0        120      0      ...    3
细胞2            80       0        0        45       200    ...    0
细胞3            0        0        15       0        8      ...    0
```

这是一个**稀疏矩阵**：每行=细胞，每列=基因，数值=该基因被检测到的次数（UMI counts）

**Step 2: 预处理（Normalize + Log1p）**

```python
sc.pp.normalize_total(adata)      # 总计数归一化
sc.pp.log1p(adata, base=10)       # log10(1 + x) 变换
```

**Step 3: Rank排序生成细胞句子**

对每个细胞，把非零表达的基因按数值从大到小排序，只保留基因名：

```
细胞1的表达值: 基因D=2.08, 基因B=0.70, 基因N=0.48, ...
         ↓ 排序
细胞1的句子: "基因D 基因B 基因N ..."  ← 这就是"Cell Sentence"！
```

**完整流程图**：

```
┌──────────────────────────────────────────────────────────────────────────┐
│  原始数据 (稀疏矩阵)                                                       │
│  ┌─────────────────────────────────────────────┐                         │
│  │       GeneA  GeneB  GeneC  GeneD  GeneE ... │  ~20000列（基因）        │
│  │ Cell1   0      5      0     120     0   ... │                         │
│  │ Cell2   80     0      0      45    200  ... │  很多0！这就是"稀疏"      │
│  └─────────────────────────────────────────────┘                         │
│                          ↓                                                │
│                  normalize + log10(1+x)                                   │
│                          ↓                                                │
│  ┌─────────────────────────────────────────────┐                         │
│  │       GeneA  GeneB  GeneC  GeneD  GeneE ... │                         │
│  │ Cell1  0.00   0.70   0.00   2.08   0.00 ... │  数值被压缩到0-4之间     │
│  └─────────────────────────────────────────────┘                         │
│                          ↓                                                │
│              按表达值降序排列 (只保留非零基因名)                            │
│                          ↓                                                │
│  ┌─────────────────────────────────────────────┐                         │
│  │ Cell1: "GeneD GeneB GeneN GeneX ..."        │  ← 细胞句子！            │
│  │ Cell2: "GeneE GeneA GeneD GeneM ..."        │  位置=排名=相对表达量    │
│  └─────────────────────────────────────────────┘                         │
└──────────────────────────────────────────────────────────────────────────┘
```

### Q12: 什么是scRNA-seq数据的稀疏性？为什么Rank能规避这个问题？

**A**: 

**稀疏性的直观理解**：scRNA-seq数据**95%以上都是0**！原因是：
1. 每个细胞只表达约2000-5000个基因（共有~20000个）
2. 测序深度有限，低表达基因检测不到
3. 技术噪声导致"dropout"（本来有表达但测不到）

**直观图示**：

```
一个典型的scRNA-seq矩阵（假设100个基因）：

         G1  G2  G3  G4  G5  G6  G7  G8  G9  G10 ... G100
Cell1:   0   0   0   45  0   0   120 0   0   3   ... 0
Cell2:   0   80  0   0   0   0   0   200 0   0   ... 15
Cell3:   0   0   0   0   8   0   0   0   0   0   ... 0

稀疏！大部分都是0。而且非零值范围很大：从1到几千。
```

**为什么稀疏性是问题？**
1. **数值分布不均匀**：表达值从0到几万，跨度太大
2. **0太多**：神经网络难以从大量0中学习有意义的模式
3. **噪声影响**：低表达基因的数值波动很大（1变成2就是翻倍了）

**Rank如何解决这个问题？**

```
原始数值（Cell1）:   
    GeneD=120, GeneA=45, GeneB=3, 其他都是0

转换为Rank后:
    GeneD=1（第1名）, GeneA=2（第2名）, GeneB=3（第3名）

好处：
1. ✅ 消除了绝对数值的差异（120 vs 3 → 都变成排名）
2. ✅ 0被自然忽略（不参与排名）
3. ✅ 所有细胞的"句子长度"相似（都是表达的基因数）
4. ✅ 保留了相对关系（最高表达的还是第一）
```

**关键洞见**：在生物学上，**基因的相对表达排序**比绝对数值更重要！巨噬细胞的标志是 CD68 > CD14 > ...（排序模式），而不是具体数值。

### Q13: Log-Normalization到底取了什么的对数？

**A**: Log变换发生在**Rank之前**，是对**表达数值**取对数，不是对排名取对数。

**完整的数据处理顺序**：

```
原始counts → normalize → log1p(base=10) → Rank排序 → 细胞句子
                              ↑
                         这一步取对数！
```

**Log1p具体做了什么？**

```python
log1p(x, base=10) = log10(1 + x)
```

**例子**：

```
原始表达值:    0,   1,   10,   100,   1000
log10(1+x):   0,  0.30, 1.04,  2.00,  3.00

效果：
- 压缩了数值范围（从0-1000 → 0-3）
- 0还是0（因为log10(1+0)=0）
- 保留了相对顺序
```

**为什么需要Log？** 即使最终我们只用Rank，Log变换也很重要：
1. **处理ties（并列）**：如果两个基因表达值都是100，排谁前面？Log变换后能更好地区分细微差异
2. **标准做法**：这是单细胞分析的标准预处理流程（scanpy推荐）
3. **逆向重建需要**：可逆性依赖于 log表达值 和 log排名 的线性关系

### Q14: 可逆性证明是什么意思？为什么能从排名恢复表达值？

**A**: 这是论文的关键发现。

**核心发现**：**log(表达值)** 和 **log(排名)** 之间存在**线性关系**！

$$\log_{10}(\text{expression}) \approx a \times \log_{10}(\text{rank} + 1) + b$$

其中 $a$（斜率）≈ -0.676，$b$（截距）≈ 2.238

**直观理解**：这其实就是**Zipf定律**在基因表达中的体现：

```
排名第1的基因（表达最高）→ 表达值很高
排名第2的基因           → 表达值明显下降
排名第3的基因           → 继续下降
...
排名第100的基因         → 表达值很低

这个下降遵循规律：log(表达值) = -0.676 × log(排名) + 2.238
```

**图示说明**：

```
            表达值 (log scale)
               ^
           4  |  *
               |   *
           3  |    *
               |     **
           2  |       ***
               |          ****
           1  |              ******
               |                   **********
           0  +-------------------------> 排名 (log scale)
               1   10   100   1000

在log-log坐标下是一条直线！
```

**可逆性的意义**：因为有这个线性关系，我们可以：
1. **正向**：表达值 → 排名 → 细胞句子（"GeneD GeneB GeneN..."）
2. **逆向**：细胞句子 → 排名 → 用线性模型预测表达值

```python
# 逆向重建代码（来自utils.py）
def reconstruct_expression_from_cell_sentence(...):
    for pos, gene_name in enumerate(cell_sentence):
        # pos就是排名（0, 1, 2, ...）
        log_rank = log10(1 + pos)
        # 用线性模型预测表达值
        expression = intercept + slope * log_rank
```

论文验证这个重建能恢复 **R² > 0.81** 的原始方差，即损失不到19%的信息！

### Q15: 生成的细胞是Rank格式的吗？

**A**: **是的，生成的结果是细胞句子（基因名序列），本质上就是Rank！**

```
模型输入: "Generate a macrophage cell: "
模型输出: "MALAT1 TMSB4X B2M MT-CO1 RPLP1 ACTB MT-CO2 RPS27..."
           ↑      ↑      ↑
          Rank1  Rank2  Rank3  ...

这个输出就是：
- 按表达量从高到低排列的基因名列表
- 第一个基因表达最高，第二个次之，以此类推
```

如果需要得到表达向量，可以通过Q14中的逆向重建方法转换回去。

### Q16: 细胞生成的三个评估指标（k-NN、GW距离、UMAP）分别是什么意思？

**A**:

**1. k-NN（k-Nearest Neighbors）**

测量"生成的细胞和真实细胞有多相似"：

```
步骤：
1. 把生成的细胞句子转换回表达向量（用逆向重建方法）
2. 计算生成细胞与真实细胞的距离
3. 看生成细胞的k个最近邻居是不是同类型的真实细胞

好的生成器：生成的巨噬细胞应该靠近真实的巨噬细胞
```

**2. Gromov-Wasserstein (GW) 距离**

测量"生成细胞群体的分布和真实细胞群体的分布有多像"：

```
不是比较单个细胞，而是比较整体分布：
- 真实巨噬细胞们形成一个"团"
- 生成巨噬细胞们也形成一个"团"
- GW距离衡量这两个"团"的形状和位置有多接近

GW越小越好！
```

**3. UMAP可视化**

最直观的评估方法：

```
把生成细胞和真实细胞一起做UMAP降维：
- 如果生成的巨噬细胞落在真实巨噬细胞的区域 → 生成质量好✅
- 如果生成的细胞乱飞或者形成单独的cluster → 生成质量差❌

┌─────────────────────────────────────┐
│      UMAP图                          │
│                                      │
│    ● ●●   T细胞(真实)                │
│   ●  ●●                             │
│         ○ ○   T细胞(生成)            │
│          ○○                         │
│                                      │
│           ★★★  巨噬细胞(真实)        │
│          ★ ★                        │
│           ☆ ☆  巨噬细胞(生成)        │
│                                      │
│   好的生成：○靠近●，☆靠近★           │
└─────────────────────────────────────┘
```

---

## 14. 参考资料

### 论文

1. **C2S-Scale论文**：[Scaling Large Language Models for Next-Generation Single-Cell Analysis](https://www.biorxiv.org/content/10.1101/2025.04.14.648850v2) (2025)
2. **原始C2S论文**：[Cell2Sentence: Teaching Large Language Models the Language of Biology](https://www.biorxiv.org/content/10.1101/2023.09.11.557287v4) (2023, ICML 2024)

### 官方资源

- **GitHub仓库**：[https://github.com/vandijklab/cell2sentence](https://github.com/vandijklab/cell2sentence)
- **文档**：[https://vandijklab-cell2sentence.readthedocs.io/](https://vandijklab-cell2sentence.readthedocs.io/)
- **HuggingFace模型**：[https://huggingface.co/collections/vandijklab/cell2sentence-models](https://huggingface.co/collections/vandijklab/cell2sentence-models)

### 相关博客

- [Google Research Blog: Teaching Machines the Language of Biology](https://research.google/blog/teaching-machines-the-language-of-biology-scaling-large-language-models-for-next-generation-single-cell-analysis/)
- [Google Blog: Gemma AI for Cancer Therapy Discovery](https://blog.google/technology/ai/google-gemma-ai-cancer-therapy-discovery/)

### 依赖项目

- [Scanpy](https://scanpy.readthedocs.io/)：单细胞分析工具包
- [AnnData](https://anndata.readthedocs.io/)：单细胞数据结构
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)：LLM框架
- [Pythia](https://github.com/EleutherAI/pythia)：基础语言模型
- [Gemma](https://ai.google.dev/gemma)：Google基础语言模型

---

*文档版本：1.0 | 基于Cell2Sentence v1.2.0 | 更新日期：2026-01-26*
