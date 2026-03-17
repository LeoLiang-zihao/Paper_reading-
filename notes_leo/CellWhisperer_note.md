# CellWhisperer 模型架构详解

> 本文档是原始 `模型架构详解.md` 的整理版本，消除了重复内容，保持核心信息完整。

## 目录
1. [概述](#概述)
2. [核心架构：双编码器对比学习](#核心架构双编码器对比学习)
3. [转录组编码器](#转录组编码器)
4. [文本编码器](#文本编码器)
5. [投影层与对比学习](#投影层与对比学习)
6. [训练策略](#训练策略)
7. [LLaVA聊天模型](#llava聊天模型)
8. [Agent技术分析](#agent技术分析)
9. [Single Cell知识要求](#single-cell知识要求)
10. [技术栈总结](#技术栈总结)
11. [常见问题解答](#常见问题解答)

---

## 概述

CellWhisperer是一个**多模态AI模型**，将单细胞RNA测序（scRNA-seq）数据与自然语言描述对齐，实现用自然语言查询和探索单细胞数据。借鉴CLIP对比学习范式，将图像替换为转录组数据。

### 核心创新点
- **跨模态对齐**：将高维基因表达数据映射到与自然语言相同的语义空间
- **零样本检索**：无需重新训练即可用自然语言查询新的细胞类型或状态
- **交互式探索**：通过LLaVA聊天模型实现对话式数据探索

---

## 核心架构：双编码器对比学习

### 整体架构图

```
┌─────────────────────┐         ┌─────────────────────┐
│  转录组编码器        │         │   文本编码器         │
│  (Geneformer/       │         │   (BioBERT)         │
│   scGPT/UCE)        │         │                     │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           │ 特征提取                      │ 特征提取
           ▼                               ▼
    ┌──────────────┐              ┌──────────────┐
    │ 转录组特征    │              │  文本特征     │
    │ (hidden_size)│              │ (hidden_size)│
    └──────┬───────┘              └──────┬───────┘
           │                              │
           │  投影层 (Discriminator)      │
           └─▶│  MILinearBlock       │◀──┘
              │  (投影到共同空间)      │
              └──────────┬─────────────┘
                        ▼
              ┌──────────────────┐
              │  联合嵌入空间     │
              │  (projection_dim)│
              └────────┬─────────┘
                       │ 对比学习损失
                       ▼
              ┌──────────────────┐
              │   CLIP Loss      │
              └──────────────────┘
```

### 核心模型代码

```python
class TranscriptomeTextDualEncoderModel(PreTrainedModel):
    def __init__(self, config, transcriptome_model, text_model):
        self.transcriptome_model = transcriptome_model  # Geneformer/scGPT/UCE
        self.text_model = text_model  # BioBERT
        self.discriminator = GlobalDiscriminatorDot(
            image_sz=self.transcriptome_embed_dim,
            text_sz=self.text_embed_dim,
            units=self.projection_dim,
        )
    
    def forward(self, ...):
        transcriptome_features = self.transcriptome_model(...)
        text_features = self.text_model(...)
        logits_per_transcriptome, transcriptome_embeds, text_embeds = \
            self.discriminator(transcriptome_features, text_features)
        return CLIPOutput(...)
```

---

## 转录组编码器

CellWhisperer支持三种预训练的单细胞转录组模型：

### 1. Geneformer
| 属性 | 值 |
|------|-----|
| 架构 | BERT-based Transformer |
| 层数 | 12层 |
| 隐藏维度 | 512 |
| 注意力头数 | 4 |
| 词汇表大小 | 25,426 |
| 预训练任务 | 掩码语言建模（MLM） |

### 2. scGPT
| 属性 | 值 |
|------|-----|
| 架构 | Transformer Encoder |
| 特点 | 支持Flash Attention、批次效应校正 |

### 3. UCE (Universal Cell Embedding)
| 属性 | 值 |
|------|-----|
| 架构 | 33层Transformer |
| Token维度 | 5120 |
| 隐藏维度 | 1280 |
| 注意力头数 | 20 |
| 特点 | 结合蛋白质编码信息，支持跨物种 |

### 数据预处理差异
- **Geneformer**：归一化/排序后生成token序列
- **scGPT**：可选归一化与log1p，可选HVG筛选
- **UCE**：生成序列与位置编码相关输入

---

## 文本编码器

### BioBERT
| 属性 | 值 |
|------|-----|
| 模型 | `dmis-lab/biobert-v1.1` |
| 隐藏维度 | 768 |
| 层数 | 12层 |
| 预训练数据 | PubMed摘要和全文 |

### 文本池化策略
1. **CLS Pooling**：使用[CLS] token
2. **Mean Pooling**：对token embeddings加权平均

---

## 投影层与对比学习

### 投影层（Discriminator）

**核心问题**：转录组编码器（如512维）和文本编码器（768维）输出维度和语义空间不匹配。

**解决方案**：投影层将两个特征映射到**同一语义空间**（默认1024维，可配置）。

```python
class GlobalDiscriminatorDot(nn.Module):
    def __init__(self, image_sz, text_sz, units=projection_dim, bln=True):
        self.img_block = MILinearBlock(image_sz, units=units, bln=bln)   # 转录组投影块
        self.text_block = MILinearBlock(text_sz, units=units, bln=bln)  # 文本投影块
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, features1, features2):
        feat1 = self.img_block(features1)   # [batch, projection_dim]
        feat2 = self.text_block(features2)  # [batch, projection_dim]
        feat1, feat2 = F.normalize(feat1, p=2, dim=-1), F.normalize(feat2, p=2, dim=-1)
        logits = torch.einsum("nd,md->nm", [feat1, feat2]) * self.temperature.exp()
        return logits, feat1, feat2
```

### 投影块（MILinearBlock）

```python
class MILinearBlock(nn.Module):
    def __init__(self, feature_sz, units=projection_dim, bln=True):
        self.feature_nonlinear = nn.Sequential(
            nn.Linear(feature_sz, units, bias=False),
            nn.BatchNorm1d(units),
            nn.ReLU(),
            nn.Linear(units, units),
        )
        self.feature_shortcut = nn.Linear(feature_sz, units)  # 残差连接
        self.feature_block_ln = nn.LayerNorm(units)
    
    def forward(self, feat):
        f = self.feature_nonlinear(feat) + self.feature_shortcut(feat)
        return self.feature_block_ln(f) if self.bln else f
```

### 对比学习：正负样本机制

**关键理解**：正负样本**自动从batch生成**，不需要手动标注。

```python
# 假设 batch_size = 4
batch = [
    (cell_1, "activated CD4+ T cells"),   # 样本0
    (cell_2, "naive B cells"),            # 样本1
    (cell_3, "macrophage"),               # 样本2
    (cell_4, "neutrophil"),               # 样本3
]

# 相似度矩阵 [batch_size, batch_size]
# 正样本对（对角线）：cell_i ↔ text_i ✅
# 负样本对（非对角线）：cell_i ↔ text_j (i≠j) ❌
```

### CLIP Loss

```python
class ClipLoss(nn.Module):
    def forward(self, logits_per_text, ...):
        text_loss = contrastive_loss(logits_per_text)
        transcriptome_loss = contrastive_loss(logits_per_text.t())
        return (text_loss + transcriptome_loss) / 2.0

def contrastive_loss(logits):
    labels = torch.arange(len(logits), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)
```

### 温度参数

- **logit_scale高（temperature低）**：相似度差异被放大，分布更尖锐
- **logit_scale低（temperature高）**：相似度差异更平滑
- **可学习**：模型自动调整

---

## 训练策略

### 锁定模式（Locking Mode）

| 模式 | 转录组编码器 | 文本编码器 | 适用场景 |
|------|------------|-----------|---------|
| **LL** | 锁定 | 锁定 | 快速训练，保护预训练权重 |
| **UL** | 解锁 | 锁定 | 转录组数据需要适应 |
| **LU** | 锁定 | 解锁 | 文本描述需要适应 |
| **UU** | 解锁 | 解锁 | 端到端优化 |

### 渐进式解冻

**阶段1：Warmup（冻结编码器，只训练投影层）**
- 让投影层"适应"编码器输出
- 保护预训练知识
- 降低计算成本

**阶段2：Fine-tuning（根据锁定模式解冻）**
- 达到warmup步数后，解冻U标记的编码器
- 微调以进一步优化

### 冻结模型的工作方式

**关键**：冻结的模型**参与前向传播**，但**不计算梯度**，**不更新参数**。

```python
class FrozenCachedModel(nn.Module):
    def forward(self, *args, **kwargs):
        with torch.no_grad():  # 不计算梯度
            model_output = self.model(**kwargs)
            return model_output
```

| 特性 | 冻结的模型 | 未冻结的模型 |
|------|-----------|-------------|
| 前向传播 | ✅ 参与 | ✅ 参与 |
| 梯度计算 | ❌ 不计算 | ✅ 计算 |
| 参数更新 | ❌ 不更新 | ✅ 更新 |
| 缓存 | ✅ 使用 | ❌ 不使用 |

### 学习率调度

- **Warmup**：线性warmup（默认3%训练步数）
- **主训练**：余弦退火调度
- **权重衰减**：仅对解锁的编码器（0.01）

---

## LLaVA聊天模型

### 整体架构

```
输入层
  ├─ 用户查询 → Tokenizer → [token_1, ..., token_n]
  └─ 细胞嵌入（来自CellWhisperer）[batch, projection_dim]
       ↓
  多模态投影器 (mm_projector)
  - Linear(projection_dim → 4096) → GELU → Linear(4096 → 4096)
  - 输出: [batch, 4, 4096]（4个tokens）
       ↓
Token拼接 → [cell_tokens, text_tokens]
       ↓
基础LLM (Mistral-7B / Llama-3.3-70B)
  - 32层Transformer
       ↓
输出文本（自回归生成）
```

### 两个阶段训练

| 阶段 | Stage 1: Pretrain | Stage 2: Fine-tune |
|------|-------------------|-------------------|
| **目标** | 训练投影器 | 微调LLM |
| **训练组件** | mm_projector ✅ / LLM ❌ | mm_projector ✅ / LLM ✅ |
| **学习率** | 1e-4 | 2e-5 |
| **训练数据** | 简单问答对 | 复杂对话 |

### 训练数据结构

**关键理解**：**N个训练样本 = N个细胞嵌入 + N组对话**（一一对应）

```
训练样本 1:
├─ 细胞嵌入_1 [projection_dim]  ← 来自CellWhisperer
└─ 对话_1:
    - Q: "这些细胞是什么类型？"
    - A: "这些细胞似乎是激活的CD4+ T细胞..."

训练样本 2:
├─ 细胞嵌入_2 [projection_dim]  ← 来自CellWhisperer
└─ 对话_2:
    - Q: "描述这些细胞的生物学状态"
    - A: "根据基因表达谱，这些细胞是巨噬细胞..."

训练样本 N:
├─ 细胞嵌入_N [projection_dim]
└─ 对话_N: ...
```

**实际训练数据格式（JSON）**：

```json
{
  "id": "cell_001",
  "image": [0.23, 0.45, ..., 0.67],  // 细胞嵌入向量 [projection_dim]
  "conversations": [
    {"from": "human", "value": "<image>\n这些细胞是什么类型？"},
    {"from": "gpt", "value": "根据基因表达谱，这些细胞..."}
  ]
}
```

| 概念 | 说明 |
|------|------|
| **1个样本** | = 1个细胞嵌入 + 1组对话 |
| **N个样本** | = N个细胞嵌入 + N组对话 |
| **`<image>`标记** | 告诉模型"这里插入细胞嵌入" |

### 多模态投影器 vs 对比学习投影层

| 特性 | 对比学习投影层 | LLaVA多模态投影器 |
|------|---------------|------------------|
| **功能** | 对齐两个特征空间 | 格式转换（embedding→tokens） |
| **架构** | 两个MILinearBlock | 单个MLP |
| **输入** | 两个输入 | 一个输入 |
| **输出** | embedding [batch, projection_dim] | tokens [batch, 4, 4096] |
| **训练阶段** | 对比学习（阶段1） | LLaVA训练（阶段3-4） |
| **归一化** | ✅ L2归一化 | ❌ 不归一化 |

### 为什么是4个tokens？

> "Our transcriptome model returns a single embedding, so to provide more 'information power' to the language model, I convert it to 4 tokens."

单个embedding可能信息量不足，转换为4个tokens提供更多信息容量。

---

## 完整流程

```
┌─────────────────────────────────────────────────────────┐
│ 阶段1：预训练模型（已有）                                 │
├─────────────────────────────────────────────────────────┤
│ - BioBERT（文本编码器）                                   │
│ - Geneformer/scGPT/UCE（转录组编码器）                   │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 阶段2：对比学习训练（训练CellWhisperer）                  │
├─────────────────────────────────────────────────────────┤
│ 输入：转录组数据 + 文本描述                              │
│ 训练：投影层（和可能的编码器）                           │
│ 输出：训练好的CellWhisperer模型                         │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 阶段3：生成LLaVA训练数据                                 │
├─────────────────────────────────────────────────────────┤
│ 1. CellWhisperer生成细胞嵌入                            │
│ 2. GPT-4生成问答对                                      │
│ 3. 配对：embedding + 问答对                             │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 阶段4：LLaVA预训练（Stage 1）                            │
├─────────────────────────────────────────────────────────┤
│ 训练：mm_projector ✅ / LLM ❌                           │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 阶段5：LLaVA微调（Stage 2）                              │
├─────────────────────────────────────────────────────────┤
│ 训练：mm_projector ✅ / LLM ✅                           │
└─────────────────────────────────────────────────────────┘
```

---

## Agent技术分析

**结论**：CellWhisperer**没有**传统Agent技术。

| 特性 | 传统Agent | CellWhisperer |
|------|----------|---------------|
| 工具调用 | ✅ | ❌ |
| 多步推理 | ✅ | ❌ |
| 自主规划 | ✅ | ❌ |
| 环境交互 | ✅ | ⚠️ 有限 |

**类似Agent的特性**：
- 检索增强生成（RAG-like）
- 多模态理解
- 上下文感知

---

## Single Cell知识要求

### 需要掌握的程度

| 概念 | 需要程度 |
|------|---------|
| **基础概念** | ⭐⭐⭐⭐⭐ |
| 基因表达、细胞类型、转录组 | 必须深入理解 |
| **中级概念** | ⭐⭐⭐⭐ |
| 通路分析、差异表达、批次效应 | 需要理解 |
| **高级概念** | ⭐⭐ |
| 单细胞技术细节、序列比对 | 了解即可 |

### 核心工具
- **Scanpy**：数据预处理、降维、聚类
- **AnnData**：数据格式（`.h5ad`）

```python
# 数据结构
AnnData(
    X: np.ndarray,       # 表达矩阵
    obs: pd.DataFrame,   # 细胞注释
    var: pd.DataFrame,   # 基因注释
    obsm: dict,          # embeddings
)
```

---

## 技术栈总结

| 类别 | 技术 |
|------|------|
| **对比学习** | CLIP范式、对称对比损失、温度参数学习 |
| **Transformer** | BERT (Geneformer, BioBERT)、GPT-style (scGPT) |
| **深度学习框架** | PyTorch、PyTorch Lightning、Transformers |
| **生物信息学** | Scanpy、AnnData、NumPy/SciPy |
| **其他** | WandB、Snakemake、Docker、LLaVA |

---

## 常见问题解答

### Q1: 对比学习就是微调吗？

**是的**，有两个层面：
1. **对比学习模型的微调**：微调预训练编码器和投影层
2. **LLaVA的微调**：微调基础LLM

### Q2: 为什么要先训练投影层，再解冻编码器？

1. **稳定训练**：避免梯度冲突
2. **保护知识**：避免破坏预训练权重
3. **降低成本**：先训练小模型
4. **渐进学习**：先学对齐，再学微调

### Q3: CellWhisperer如何应用于LLaVA？

通过**生成训练数据**：
1. CellWhisperer生成细胞嵌入
2. GPT-4生成问答对
3. 配对形成LLaVA训练数据
4. 训练LLaVA模型

### Q4: 两个投影器有什么区别？

- **对比学习投影层**：对齐两个特征空间，用于相似度计算
- **LLaVA多模态投影器**：格式转换，将embedding转为LLM tokens

---

## 参考资料

1. CLIP论文：Radford et al., "Learning Transferable Visual Models From Natural Language Supervision"
2. Geneformer论文：Theodoris et al., "Transfer learning enables predictions in network biology"
3. scGPT论文：Cui et al., "scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI"
4. CellWhisperer论文：Schaefer et al., "Multimodal learning enables chat-based exploration of single-cell data" (Nature Biotechnology)

---

*整理版文档 - 基于原始模型架构详解.md*
*整理日期：2026年1月19日*
