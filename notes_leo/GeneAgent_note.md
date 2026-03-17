# GeneAgent 技术文档

> **Self-verification Language Agent for Gene Set Analysis using Domain Databases**

---

## 目录

1. [概述](#1-概述)
2. [核心架构](#2-核心架构)
3. [模型组件详解](#3-模型组件详解)
4. [数据预处理与输入格式](#4-数据预处理与输入格式)
5. [系统配置与设计理念](#5-系统配置与设计理念)
6. [推理流程与结果生成](#6-推理流程与结果生成)
7. [评估指标与实验设计](#7-评估指标与实验设计)
8. [扩展功能与应用](#8-扩展功能与应用)
9. [相关技术分析](#9-相关技术分析)
10. [技术栈总结](#10-技术栈总结)
11. [常见问题解答](#11-常见问题解答faq)
12. [参考资料](#12-参考资料)

---

## 1. 概述

### 1.1 核心目标和功能

**GeneAgent** 是首个基于 GPT-4 构建的语言智能体（Language Agent），专门用于自动与领域特定数据库交互，为基因集（Gene Set）提供功能注释。其核心功能包括：

- **生物过程命名**：为用户提供的基因集生成可解释且上下文准确的生物过程名称
- **富集分析对齐**：与显著富集分析结果对齐或引入新术语
- **自我验证**：通过与专家策划的生物数据库交互，自主进行事实验证
- **证据支持**：提供客观证据来支持或反驳原始 LLM 输出，减少幻觉

### 1.2 核心创新点

1. **自我验证机制（Self-verification）**：首次在基因集分析中引入自动化事实核查机制，通过调用外部数据库验证 LLM 生成的声明（claims）

2. **级联验证架构（Cascade Verification）**：采用两阶段验证流程，先验证主题名称（Topic），再验证分析叙述（Analysis）

3. **多数据库工具调用**：集成 8 种专业生物数据库 API，包括 PubTator、Enrichr、g:Profiler、NCBI Gene 等

4. **声明解构与验证**：将复杂的生物学声明分解为可验证的原子声明，逐一进行事实核查

5. **基于证据的修正**：根据验证报告自动修正原始输出，确保结果的可靠性和可追溯性

### 1.3 与现有方法的独特之处

| 方法 | 特点 | GeneAgent 优势 |
|------|------|----------------|
| **传统富集分析** | 依赖预定义术语库 | 可生成新颖术语 |
| **纯 GPT-4** | 可能产生幻觉 | 自我验证减少幻觉 |
| **Chain-of-Thought** | 单次推理无验证 | 迭代验证修正 |
| **RAG 方法** | 被动检索 | 主动工具调用验证 |

### 1.4 解决的实际问题

- **LLM 幻觉问题**：生物学领域对准确性要求极高，GeneAgent 通过多源验证减少虚假信息
- **可解释性**：提供完整的验证报告，研究人员可追溯每个结论的证据来源
- **专业术语标准化**：将自由文本生成与标准生物学术语库（如 GO、KEGG）对齐

---

## 2. 核心架构

### 2.1 ASCII 架构图

```
                          ┌─────────────────────────────────────────────────────────────┐
                          │                    GeneAgent 系统架构                        │
                          └─────────────────────────────────────────────────────────────┘
                                                      │
                          ┌───────────────────────────▼───────────────────────────┐
                          │                     输入: 基因集                        │
                          │              例如: "ERBB2,ERBB4,FGFR2,FGFR4"          │
                          └───────────────────────────┬───────────────────────────┘
                                                      │
                          ┌───────────────────────────▼───────────────────────────┐
                          │              STEP 1: 初始分析生成 (GPT-4o)             │
                          │    ┌─────────────────────────────────────────────┐    │
                          │    │  输入: 基因集 + System Prompt                │    │
                          │    │  输出: Process Name + Analysis Summary      │    │
                          │    └─────────────────────────────────────────────┘    │
                          └───────────────────────────┬───────────────────────────┘
                                                      │
          ┌───────────────────────────────────────────┴───────────────────────────────────────────┐
          │                                                                                       │
          ▼                                                                                       ▼
┌─────────────────────────────┐                                           ┌─────────────────────────────┐
│   STEP 2: 主题声明生成       │                                           │   STEP 4: 分析声明生成       │
│  (Topic Claim Generation)   │                                           │ (Analysis Claim Generation) │
│  ┌───────────────────────┐  │                                           │  ┌───────────────────────┐  │
│  │ 解构 Process Name     │  │                                           │  │ 解构 Analysis Summary │  │
│  │ 为可验证的声明列表     │  │                                           │  │ 为基因级别声明        │  │
│  └───────────────────────┘  │                                           │  └───────────────────────┘  │
└──────────────┬──────────────┘                                           └──────────────┬──────────────┘
               │                                                                          │
               ▼                                                                          ▼
┌─────────────────────────────┐                                           ┌─────────────────────────────┐
│   STEP 3: AgentPhD 验证     │                                           │   STEP 5: AgentPhD 验证     │
│  ┌───────────────────────┐  │                                           │  ┌───────────────────────┐  │
│  │   循环调用 8 种 API   │  │                                           │  │   循环调用 8 种 API   │  │
│  │  ├─ Complex API       │  │                                           │  │  ├─ Gene Summary API  │  │
│  │  ├─ Pathway API       │  │                                           │  │  ├─ PubMed API        │  │
│  │  ├─ Enrichment API    │  │                                           │  │  ├─ Domain API        │  │
│  │  └─ ...               │  │                                           │  │  └─ ...               │  │
│  └───────────────────────┘  │                                           │  └───────────────────────┘  │
│        验证报告输出          │                                           │        验证报告输出          │
└──────────────┬──────────────┘                                           └──────────────┬──────────────┘
               │                                                                          │
               ▼                                                                          │
┌─────────────────────────────┐                                                           │
│  STEP 3.5: 主题修正          │                                                           │
│  ┌───────────────────────┐  │                                                           │
│  │ 根据验证报告修正       │  │                                                           │
│  │ Process Name 和分析   │  │────────────────────────────────────────────────────────────│
│  └───────────────────────┘  │                                                           │
└──────────────┬──────────────┘                                                           │
               │                                                                          │
               │◄─────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              STEP 6: 最终摘要生成                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │  整合所有验证报告，生成最终的 Process Name 和 Analysis Summary                    │   │
│  │  • 支持的声明：保留并补充证据                                                     │   │
│  │  • 部分支持：修剪不支持部分                                                       │   │
│  │  • 反驳的声明：替换为验证报告中最显著的术语                                        │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────┬─────────────────────────────────────┘
                                                    │
                                                    ▼
                          ┌───────────────────────────────────────────────────┐
                          │                    最终输出                        │
                          │  ┌─────────────────────────────────────────────┐  │
                          │  │  Process: MAPK Signaling Pathway            │  │
                          │  │  [详细的基因功能分析叙述...]                  │  │
                          │  └─────────────────────────────────────────────┘  │
                          └───────────────────────────────────────────────────┘
```

### 2.2 核心模型类的代码结构

```python
# worker.py - 核心验证 Agent 类
class AgentPhD:
    """
    事实核查验证智能体
    负责调用多种生物数据库 API 验证声明的正确性
    """
    
    def __init__(self, function_names: List[str]):
        """
        初始化验证智能体
        
        Args:
            function_names: 要使用的 API 函数名称列表
                可选: get_complex_for_gene_set, get_disease_for_single_gene,
                      get_domain_for_single_gene, get_enrichment_for_gene_set,
                      get_pathway_for_gene_set, get_interactions_for_gene_set,
                      get_gene_summary_for_single_gene, get_pubmed_articles
        """
        self.name2function = {...}  # 函数名 → 函数映射
        self.function_docs = [...]   # OpenAI Function Calling 文档

    def inference(self, claim: str) -> str:
        """
        验证单个声明
        
        Args:
            claim: 待验证的生物学声明
            
        Returns:
            验证报告字符串，包含证据和结论
        
        流程:
            1. 构建验证 prompt
            2. 循环调用 GPT-4o（最多 20 次迭代）
            3. GPT-4o 自主选择调用哪些 API
            4. 收集 API 返回的证据
            5. 生成最终验证报告
        """
        pass
```

```python
# main_cascade.py - 主工作流函数
def GeneAgent(ID: str, genes: str) -> None:
    """
    GeneAgent 主入口函数
    
    Args:
        ID: 基因集唯一标识符
        genes: 逗号分隔的基因符号字符串
        
    工作流程:
        1. 基因集预处理
        2. 调用 GPT-4o 生成初始摘要
        3. 提取 Process Name
        4. 生成主题声明并验证
        5. 根据验证结果修正主题
        6. 生成分析声明并验证
        7. 生成最终修正摘要
    """
    pass
```

### 2.3 数据流动方式

```
输入基因集 (str)
    │
    ▼ [预处理: 标准化分隔符]
标准化基因集 (str: "GENE1,GENE2,GENE3")
    │
    ▼ [GPT-4o: baseline prompt]
初始摘要 (dict: {"process": str, "analysis": str})
    │
    ├──▶ Process Name (str)
    │        │
    │        ▼ [GPT-4o: 声明生成]
    │    主题声明列表 (List[str])
    │        │
    │        ▼ [AgentPhD.inference × N]
    │    验证报告 (str)
    │        │
    │        ▼ [GPT-4o: modification prompt]
    │    修正后的摘要 (str)
    │
    ├──▶ Updated Summary (str)
    │        │
    │        ▼ [GPT-4o: 声明生成]
    │    分析声明列表 (List[str])
    │        │
    │        ▼ [AgentPhD.inference × M]
    │    验证报告 (str)
    │
    ▼ [GPT-4o: summarization prompt]
最终输出 (str: "Process: XXX\n[Analysis...]")
```

### 2.4 关键维度变换

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| 预处理 | `"A B/C"` | `"A,B,C"` | 标准化分隔符 |
| 初始生成 | 基因集字符串 | 多行文本 | GPT-4o 生成 |
| 声明解构 | 摘要文本 | `List[str]` (3-10个声明) | JSON 解析 |
| 单声明验证 | 1 个声明 | 1 个验证报告 | 最多 20 轮 API 调用 |
| 最终修正 | 所有验证报告拼接 | 修正后的完整摘要 | 单次 GPT-4o 调用 |

---

## 3. 模型组件详解

### 3.1 AgentPhD - 事实验证智能体

#### 功能说明

AgentPhD 是 GeneAgent 的核心验证组件，负责调用外部生物数据库 API 来验证 LLM 生成的声明。它采用 **ReAct（Reasoning + Acting）** 模式，让 GPT-4o 自主决定调用哪些工具。

#### 架构参数表

| 参数 | 值 | 说明 |
|------|-----|------|
| 基础模型 | GPT-4o | Azure OpenAI API |
| 温度参数 | 0 | 确定性输出 |
| 最大迭代轮数 | 20 | 防止无限循环 |
| 可用工具数量 | 8 | 8 种生物数据库 API |
| Token 限制 | 127,900 | GPT-4 上下文窗口 |

#### 关键代码片段

```python
def inference(self, claim):
    system = f"""
    You are a helpful fact-checker. 
    Your task is to verify the claim using the provided tools. 
    If there are evidences in your contents, please start a message 
    with "Report:" and return your findings along with evidences.
    """
    content = f"""
    Here is the claim needed to be verified:\n{claim} 
    Try to use multiple tools to verify a claim and the verification 
    process should be factual and objective.
    """
    
    message_verification = [
        {"role": "system", "content": system},
        {"role": "user", "content": content} 
    ]

    loop = 0
    while loop < 20:
        loop += 1
        completion = openai.ChatCompletion.create(
            engine="gpt-4o",
            messages=message_verification,
            functions=self.function_docs,  # OpenAI Function Calling
            temperature=0,
        )
        
        message = completion.choices[0]["message"]
        
        if "function_call" in message:
            # 执行工具调用
            function_name = message["function_call"]["name"]
            function_params = json.loads(message["function_call"]["arguments"])
            function_to_call = self.name2function[function_name]
            function_response = function_to_call(**function_params)
            
            message_verification.append({
                "role": "function",
                "name": function_name,
                "content": function_response
            })
        else:
            # 检查是否生成了报告
            if "Report: " in message["content"]:
                return message["content"].split("Report: ")[-1]
    
    return "Failed."
```

#### 设计动机

1. **循环迭代设计**：单次 API 调用可能无法获取足够证据，循环允许 Agent 多次探索
2. **自主工具选择**：GPT-4o 根据声明内容自主决定调用哪些 API，而非固定流程
3. **Function Calling**：利用 OpenAI 原生的 Function Calling 能力，确保参数格式正确
4. **20 轮限制**：防止 Agent 陷入无限循环，平衡效率与完整性

### 3.2 外部数据库 API 工具集

#### 3.2.1 get_complex_for_gene_set

| 属性 | 描述 |
|------|------|
| **功能** | 查询基因集对应的蛋白质复合体信息 |
| **数据源** | NCBI PubTator3 |
| **输入** | 逗号分隔的基因集 `"x,y,z"` |
| **输出** | JSON 格式的复合体 ID 和名称列表 |
| **用途** | 验证基因间的物理相互作用 |

```python
def get_complex_for_gene_set(gene_set):
    url = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/agentapi/complex/?"
    params = {"name": gene_set, "retmode": "json", "limit": 10}
    response = requests.get(url, params=params)
    return json.dumps(response.json().get("results", {}))
```

#### 3.2.2 get_enrichment_for_gene_set

| 属性 | 描述 |
|------|------|
| **功能** | 执行基因集富集分析 |
| **数据源** | g:Profiler (biit.cs.ut.ee) |
| **输入** | 逗号分隔的基因集 |
| **输出** | Top-5 显著富集的功能术语 |
| **用途** | 验证基因集的整体功能主题 |

```python
def get_enrichment_for_gene_set(gene_set):
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    payload = {
        "organism": "hsapiens",
        "query": gene_set.split(","),
        "all_results": False,
        "user_threshold": 0.05  # p-value 阈值
    }
    response = requests.post(url, json=payload)
    return json.dumps(response.json()["result"][:5])
```

#### 3.2.3 get_pathway_for_gene_set

| 属性 | 描述 |
|------|------|
| **功能** | 查询基因集涉及的生物通路 |
| **数据源** | Enrichr (KEGG, Reactome, BioPlanet, MSigDB) |
| **输入** | 逗号分隔的基因集 |
| **输出** | Top-5 相关通路及重叠基因 |
| **用途** | 验证基因集的通路关联 |

```python
def get_pathway_for_gene_set(gene_set):
    # Step 1: 上传基因列表到 Enrichr
    ENRICHR_URL_ADD = 'http://maayanlab.cloud/Enrichr/addList'
    response_add = requests.post(ENRICHR_URL_ADD, 
        files={'list': (None, '\n'.join(gene_list))})
    list_id = json.loads(response_add.text)['userListId']
    
    # Step 2: 查询多个数据库
    for backgroundType in ["KEGG_2021_Human", "Reactome_2022", 
                           "BioPlanet_2019", "MSigDB_Hallmark_2020"]:
        ENRICHR_URL_RESULTS = f'.../enrich?userListId={list_id}&backgroundType={backgroundType}'
        # ... 收集结果并排序
    
    return json.dumps(pathway_analysis[:5])
```

#### 3.2.4 get_gene_summary_for_single_gene

| 属性 | 描述 |
|------|------|
| **功能** | 获取单个基因的功能摘要 |
| **数据源** | NCBI Gene (E-utilities) |
| **输入** | 单个基因名称 + 物种 (Homo/Mus) |
| **输出** | 基因功能描述、染色体位置等 |
| **用途** | 验证单基因的具体功能 |

#### 3.2.5 get_pubmed_articles

| 属性 | 描述 |
|------|------|
| **功能** | 搜索相关 PubMed 文献 |
| **数据源** | NCBI PubMed (E-utilities) |
| **输入** | 查询词 |
| **输出** | Top-5 相关文章的标题和摘要 |
| **用途** | 文献级别的证据支持 |

#### 3.2.6 其他工具

| 工具名 | 数据源 | 功能 |
|--------|--------|------|
| `get_disease_for_single_gene` | PubTator | 基因关联疾病 |
| `get_domain_for_single_gene` | PubTator (CDD) | 蛋白质结构域 |
| `get_interactions_for_gene_set` | PubTator (PPI) | 蛋白-蛋白相互作用 |

### 3.3 Prompt 模板系统

#### 3.3.1 初始分析生成 Prompt

```python
system = "You are an efficient and insightful assistant to a molecular biologist."

baseline = lambda genes: f"""
Write a critical analysis of the biological processes performed by this system 
of interacting proteins.
Propose a brief name for the most prominent biological process performed by 
the system. 
Put the name at the top of the analysis as "Process: <name>".
Be concise, do not use unnecessary words.
Be textual, do not use any format symbols such as "*", "-" or other tokens.
Be specific, avoid overly general statements such as "the proteins are involved 
in various cellular processes".
Be factual, do not editorialize.
For each important point, describe your reasoning and supporting information.
For each biological function name, show the corresponding gene names.
Here is the gene set: {genes}
"""
```

#### 3.3.2 声明生成 Prompt

```python
topic = lambda genes, process: f"""
Here is the original process name for the gene set {genes}:\n{process}
However, the process name might be false. Please generate decontextualized 
claims for the process name that need to be verified.
Only Return a list type that contain all generated claim strings, 
for example, ["claim_1", "claim_2"]
"""

topic_instruction = """
Only generate claims with affirmative sentence for the entire gene set.
The gene set should only be separated by comma, e.g., "a,b,c".
Don't generate claims for the single gene or incomplete gene set.
Don't generate hypotheis claims over the previous analysis.
Please replace the statement like 'these genes', 'this system' with the core 
genes in the given gene set.
"""
```

#### 3.3.3 修正 Prompt

```python
modification = lambda verification_topic: f"""
I have finished the verification for process name. Here is the verification 
report:\n{verification_topic}
You should only consider the successfully verified claims.
If claims are supported, you should retain the original process name and 
only can make a minor grammar revision. 
If claims are partially supported, you should discard the unsupported part.
If claims are refuted, you must replace the original process name with the 
most significant (i.e., top-1) biological function term summarized from 
the verification report.
Meanwhile, revise the original summaries using the verified (or updated) 
process name.
"""
```

---

## 4. 数据预处理与输入格式

### 4.1 输入数据格式

#### CSV 格式 (MsigDB, Gene Ontology)

```csv
"ID","Name","Count","Genes"
"MsigDB:69","Peroxisome","8","PEX1 PEX2 PEX3 PEX4 PEX5 PEX6 PEX7 PEX8"
```

| 字段 | 类型 | 说明 |
|------|------|------|
| ID | string | 唯一标识符（如 `MsigDB:69`, `GO:0035845`） |
| Name | string | 标准术语名称（Ground Truth） |
| Count | integer | 基因数量 |
| Genes | string | 空格或逗号分隔的基因符号 |

#### TSV 格式 (NeST)

```tsv
NEST ID	name_new	Genes
NEST:256	MAPK signaling	"ERBB2,ERBB4,FGFR2,FGFR4,HRAS,KRAS"
```

### 4.2 预处理流程

```python
def preprocess_genes(genes: str) -> str:
    """
    标准化基因集字符串
    
    转换规则:
        - "/" → ","
        - " " → ","
        - 移除多余空格
    
    输入: "ERBB2 ERBB4/FGFR2 FGFR4"
    输出: "ERBB2,ERBB4,FGFR2,FGFR4"
    """
    genes = genes.replace("/", ",").replace(" ", ",")
    return genes
```

### 4.3 支持的数据集

| 数据集 | 样本数 | 来源 | 特点 |
|--------|--------|------|------|
| **Gene Ontology** | 1,000 | GO:BP 分支 | 标准化生物过程术语 |
| **MsigDB** | 56 | Hallmark gene sets | 经典信号通路 |
| **NeST** | 50 | 人类癌症蛋白组学 | 癌症相关基因集 |

### 4.4 字符过滤

系统使用正则表达式过滤非法字符：

```python
pattern = re.compile(r'^[a-zA-Z0-9,.;?!*()_-]+$')

if not re.match(pattern, claim):
    claim = re.sub(r'[^a-zA-Z0-9,.;?!*()_-]+$', "_", claim)
```

---

## 5. 系统配置与设计理念

### 5.1 零样本推理系统

GeneAgent 是一个基于预训练大语言模型的 **零样本（Zero-shot）推理系统**，无需额外训练即可直接使用：

| 配置项 | 设置 |
|--------|------|
| **基础模型** | GPT-4o（Azure OpenAI 服务） |
| **适配方式** | Prompt Engineering + 工具调用 |
| **温度参数** | 0（确保输出确定性，便于复现） |

### 5.2 核心设计理念：生成-验证-修正

GeneAgent 的设计目标是通过 **验证-修正循环** 减少 LLM 幻觉：

```
目标: 最小化幻觉 + 最大化证据支持

实现机制:
  1. LLM 生成初步答案
  2. 将答案分解为可验证的原子声明
  3. 调用权威数据库 API 逐一验证
  4. 根据验证结果修正最终输出
```

### 5.3 Prompt 设计策略

| 策略 | 实现方式 | 目的 |
|------|----------|------|
| **角色定义** | `"You are an efficient and insightful assistant to a molecular biologist."` | 激活领域知识 |
| **任务分解** | 分步骤 Prompt（生成 → 声明 → 验证 → 修正） | 降低任务复杂度 |
| **格式约束** | 明确输出格式（`"Process: <name>"`） | 便于后处理 |
| **质量约束** | `Be concise, Be specific, Be factual` | 提高输出质量 |

### 5.4 级联验证策略

```
┌─────────────────────────────────────────────────────┐
│            级联验证策略 (Cascade Verification)       │
├─────────────────────────────────────────────────────┤
│  Phase 1: Topic Verification                        │
│  ├─ 生成 3-5 个主题相关声明                          │
│  ├─ 逐一调用 AgentPhD 验证                          │
│  └─ 汇总验证报告，修正 Process Name                  │
├─────────────────────────────────────────────────────┤
│  Phase 2: Analysis Verification                     │
│  ├─ 基于修正后的摘要生成分析声明                      │
│  ├─ 逐一调用 AgentPhD 验证                          │
│  └─ 生成最终修正摘要                                 │
└─────────────────────────────────────────────────────┘
```

**级联设计的优势**：
- **粒度分离**：主题（Topic）和分析叙述（Analysis）的验证需求不同
- **迭代修正**：先修正主题名称，再基于修正后的主题验证分析
- **错误传播控制**：如果主题错误，不会影响后续分析的验证

---

## 6. 推理流程与结果生成 ⭐重点

### 6.1 推理流程详解

#### 完整流程图

```
┌──────────────────────────────────────────────────────────────────────┐
│                        GeneAgent 推理流程                            │
└──────────────────────────────────────────────────────────────────────┘

输入: genes = "ERBB2,ERBB4,FGFR2,FGFR4,HRAS,KRAS"

Step 1: 初始分析生成
────────────────────────────────────────────────────────────────────────
    输入: genes (str)
           ↓
    [GPT-4o + baseline prompt]
           ↓
    输出: summary (str)
          "Process: MAPK Signaling Pathway
           The proteins encoded by the genes ERBB2, ERBB4..."

Step 2: 主题声明生成
────────────────────────────────────────────────────────────────────────
    输入: genes (str), process_name (str)
           ↓
    [GPT-4o + topic prompt + topic_instruction]
           ↓
    输出: claims_topic (List[str])
          ["ERBB2,ERBB4,FGFR2,FGFR4,HRAS,KRAS are involved in MAPK signaling",
           "These genes regulate cell growth and differentiation through MAPK"]

Step 3: 主题声明验证
────────────────────────────────────────────────────────────────────────
    对每个 claim in claims_topic:
        输入: claim (str)
               ↓
        [AgentPhD.inference]
               ↓
        内部循环 (最多 20 轮):
            GPT-4o 决定调用哪个 API
                ↓
            调用 API 获取证据
                ↓
            将证据加入对话历史
                ↓
            GPT-4o 判断是否有足够证据
               ↓
        输出: verification_result (str)
              "The claim is SUPPORTED. Evidence: KEGG pathway analysis shows..."

    汇总: verification_topic (str)
          "Original_claim: ...
           Verified_claim: The claim is SUPPORTED..."

Step 4: 主题修正
────────────────────────────────────────────────────────────────────────
    输入: original_messages + modification prompt + verification_topic
           ↓
    [GPT-4o]
           ↓
    输出: updated_topic (str)
          "Process: MAPK/RTK Signaling Pathway
           [Modified analysis text...]"

Step 5: 分析声明生成与验证
────────────────────────────────────────────────────────────────────────
    输入: updated_topic (str)
           ↓
    [GPT-4o + analysis prompt + analysis_instruction]
           ↓
    输出: claims_analysis (List[str])
          ["ERBB2 is a receptor tyrosine kinase",
           "HRAS activates downstream MAPK cascade"]
           ↓
    对每个声明重复 Step 3 的验证流程
           ↓
    汇总: verification_analysis (str)

Step 6: 最终摘要生成
────────────────────────────────────────────────────────────────────────
    输入: all_messages + summarization prompt + verification_analysis
           ↓
    [GPT-4o]
           ↓
    输出: final_output (str)
          "Process: MAPK Signaling Pathway
           The proteins encoded by the genes ERBB2, ERBB4, FGFR2, FGFR4,
           HRAS, and KRAS are all integral components of the MAPK signaling
           pathway, which is crucial for cell growth, differentiation, and
           survival.
           
           ERBB2 and ERBB4 are members of the epidermal growth factor receptor
           (EGFR) family of receptor tyrosine kinases (RTKs)..."
```

#### 每个步骤的数据形状变化

| 步骤 | 输入形状 | 输出形状 | 说明 |
|------|----------|----------|------|
| 输入预处理 | `str` (任意格式) | `str` (标准化) | 分隔符统一为逗号 |
| 初始生成 | `str` | `str` (多行) | 约 200-500 词 |
| 声明提取 | `str` | `List[str]` (3-10 项) | JSON 解析 |
| 单声明验证 | `str` | `str` (报告) | 约 100-300 词 |
| 验证汇总 | `List[str]` | `str` (拼接) | 所有报告合并 |
| 最终修正 | `str` | `str` (多行) | 约 200-500 词 |

### 6.2 结果生成方法分析

#### a) 输出是如何生成的？

**最终输出层**: GPT-4o 的文本生成能力

```
GPT-4o 内部架构 (简化):
    Input Tokens → Transformer Encoder-Decoder → Softmax → Next Token

    具体流程:
    1. 输入所有对话历史（包括验证报告）
    2. Transformer 计算注意力权重
    3. 最后一层生成词汇表上的 logits
    4. Softmax 转换为概率分布
    5. 贪婪解码（temperature=0）选择最高概率 token
    6. 重复直到生成 <EOS> 或达到长度限制
```

**输出维度和含义**:

```python
output = {
    "process_name": str,      # 生物过程名称（如 "MAPK Signaling Pathway"）
    "analysis": str,          # 详细功能分析（多段落文本）
}

# 实际以纯文本形式输出:
"""
Process: <process_name>
<analysis paragraph 1>
<analysis paragraph 2>
...
"""
```

#### b) 生成过程的数学推理

**核心计算流程**:

```
设:
    C = 声明集合 {c₁, c₂, ..., cₙ}
    V(cᵢ) = 验证函数，返回验证报告
    S = 支持的声明集合
    R = 反驳的声明集合

验证决策逻辑:
    对于每个 cᵢ ∈ C:
        evidence_i = V(cᵢ)  # 调用多个 API 收集证据
        
        if evidence_i 支持 cᵢ:
            S = S ∪ {cᵢ}
        else:
            R = R ∪ {cᵢ}

最终输出生成:
    if |S| > |R|:
        output = refine(original_summary, S)  # 保留并补充
    else:
        output = replace(original_summary, top_term(evidence))  # 替换

其中 refine 和 replace 由 GPT-4o 通过自然语言指令实现
```

**语义相似度计算（评估时）**:

```
使用 MedCPT 编码器:
    embed(text) : str → ℝᵈ (d = 768)
    
余弦相似度:
    cos_sim(a, b) = (a · b) / (||a|| × ||b||)

评估指标:
    score = cos_sim(embed(generated_term), embed(ground_truth_term))
```

#### c) 后处理步骤

```python
# 1. 字符过滤
pattern = re.compile(r'^[a-zA-Z0-9,.;?!*()_-]+$')
if not re.match(pattern, output):
    output = re.sub(r'[^a-zA-Z0-9-_]+', "_", output)

# 2. 编码统一
output = output.encode('utf-8').decode('utf-8')

# 3. 格式解析
lines = output.split("\n")
process_name = lines[0].split("Process: ")[1]
analysis = "\n".join(lines[1:])

# 4. 输出保存
with open("output.txt", "a") as f:
    f.write(output + "\n")
    f.write("//\n")  # 分隔符
```

**阈值和筛选策略**:

| 参数 | 值 | 用途 |
|------|-----|------|
| API 返回限制 | 5-10 条 | 控制证据数量 |
| p-value 阈值 | 0.05 | 富集分析显著性 |
| 最大迭代轮数 | 20 | AgentPhD 循环限制 |
| PubMed 结果数 | 5 | 文献证据数量 |

#### d) 不同任务模式下的结果生成

GeneAgent 主要支持一种核心任务，但有多种变体：

| 模式 | 脚本 | 输出特点 |
|------|------|----------|
| **Cascade (主模式)** | `main_cascade.py` | 两阶段验证，最完整 |
| **Chain-of-Thought** | `main_CoT.py` | 无验证，纯推理 |
| **Summary** | `main_summary.py` | 基于验证报告的富集分析 |

```python
# Chain-of-Thought 模式（无验证）
chain = f"""
Let do the task step-by-step:
Step1, write a critical analysis for gene functions.
Step2, analyze the functional associations among different genes.
Step3, summarize a brief name for the most significant biological process.
"""
# 直接输出，无 AgentPhD 验证

# Summary 模式（后处理富集）
base = lambda genes, functions: f"""
I will give you a list of genes together with their functional descriptions.  
Perform a term enrichment test on these genes.
...
"""
# 输入包含预先获取的验证结果
```

### 6.3 结果解读指南

#### 输出数值的含义和范围

| 组件 | 范围/格式 | 含义 |
|------|-----------|------|
| Process Name | 自由文本 | 基因集的核心生物功能 |
| Analysis | 多段落文本 | 每个基因的功能解释 |
| 验证状态 | SUPPORTED / PARTIALLY / REFUTED | 声明的验证结果 |

#### 判断结果质量的方法

1. **证据完整性**：检查验证报告中是否包含多源证据
2. **术语标准化**：生成的术语是否与 GO/KEGG 等标准库匹配
3. **基因覆盖率**：分析是否涵盖了输入的所有基因
4. **一致性**：Process Name 与 Analysis 内容是否一致

#### 常见结果形式示例

**良好输出示例**:

```
Process: MAPK Signaling Pathway

The proteins encoded by the genes ERBB2, ERBB4, FGFR2, FGFR4, HRAS, and KRAS 
are all integral components of the MAPK signaling pathway, which is crucial 
for cell growth, differentiation, and survival.

ERBB2 and ERBB4 are members of the epidermal growth factor receptor (EGFR) 
family of receptor tyrosine kinases (RTKs). ERBB2 is unique in that it has 
no known ligands, and it prefers to form heterodimers with other EGFR family 
members, enhancing their kinase activity.

FGFR2 and FGFR4 are part of the fibroblast growth factor receptor (FGFR) 
family of RTKs. They are activated by fibroblast growth factors, leading to 
receptor dimerization and autophosphorylation.

HRAS and KRAS are GTPases that act as molecular switches in RTK signaling. 
They are activated by guanine nucleotide exchange factors (GEFs) that catalyze 
the exchange of GDP for GTP.
```

**需要注意的输出示例**:

```
Process: Cellular Processes  # 过于笼统
The genes are involved in various cellular processes...  # 缺乏具体信息
```

---

## 7. 评估指标与实验设计

### 7.1 评估指标

#### 7.1.1 ROUGE 分数

```python
from rouge_score import rouge_scorer

metrics = ["rouge1", "rouge2", "rougeL"]
scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

for ref, hyp in zip(reference_terms, generated_terms):
    scores = scorer.score(ref, hyp)
    # scores["rouge1"].fmeasure, scores["rouge2"].fmeasure, scores["rougeL"].fmeasure
```

| 指标 | 含义 | 用途 |
|------|------|------|
| ROUGE-1 | Unigram 重叠 | 词汇级别匹配 |
| ROUGE-2 | Bigram 重叠 | 短语级别匹配 |
| ROUGE-L | 最长公共子序列 | 结构相似性 |

#### 7.1.2 语义相似度

**MedCPT 模型**（主要评估指标）:

```python
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

def get_similarity(text1, text2):
    encoded = tokenizer([text1, text2], truncation=True, padding=True, 
                        return_tensors='pt', max_length=64)
    embeds = model(**encoded).last_hidden_state[:, 0, :]  # [CLS] token
    similarity = cos_sim(embeds[0], embeds[1])
    return similarity
```

**其他模型**:
- SentenceBERT (text2vec)
- SapBERT (生物医学领域专用)

#### 8.1.3 相对排名评估

```python
# 计算生成术语在背景分布中的排名
# 背景: ~1100 个标准 GO/MSigDB/NeST 术语

for generated, ground_truth_index in zip(generated_terms, indices):
    similarities = [cos_sim(generated, bg_term) for bg_term in background_terms]
    rank = sum(1 for s in similarities if s > similarities[ground_truth_index]) + 1
    # rank 越小越好，1 表示最相关
```

#### 7.1.4 富集术语精确匹配

```python
# 检查生成的术语是否与标准富集分析结果匹配
with open("GSEATerms/MsigDB.EnrichTerms.top5.json") as f:
    enriched_terms = json.load(f)

match_rate = sum(
    any(term in enriched for term in generated_terms[i])
    for i, enriched in enumerate(enriched_terms)
) / len(enriched_terms)
```

### 8.2 基准数据集

| 数据集 | 样本数 | Ground Truth | 评估用途 |
|--------|--------|--------------|----------|
| Gene Ontology | 1,000 | GO:BP 术语 | 标准生物过程匹配 |
| MsigDB | 56 | Hallmark 名称 | 经典通路匹配 |
| NeST | 50 | 癌症相关术语 | 疾病上下文匹配 |

### 7.3 对比实验设计

| 方法 | 描述 | 脚本 |
|------|------|------|
| **GPT-4 Baseline** | 无验证的直接生成 | `main_cascade.py` (Step 1 输出) |
| **Chain-of-Thought** | 分步推理无验证 | `main_CoT.py` |
| **GeneAgent (Cascade)** | 完整级联验证 | `main_cascade.py` |
| **GeneAgent + Summary** | 多术语富集汇总 | `main_summary.py` |

---

## 8. 扩展功能与应用

### 8.1 扩展能力

#### 零样本泛化

GeneAgent 无需针对特定基因集进行训练，可直接应用于：
- 任意人类基因集
- 不同来源的基因表达数据
- 新发现的基因组合

#### 工具扩展

添加新的验证 API 只需：

```python
# 1. 在 apis/ 目录下创建新的 API 文件
# apis/get_new_database.py

def get_new_database_info(gene):
    # API 调用逻辑
    return result

get_new_database_info_doc = {
    "name": "get_new_database_info",
    "description": "...",
    "parameters": {...}
}

# 2. 在 worker.py 中注册
from apis.get_new_database import get_new_database_info, get_new_database_info_doc

func2info["get_new_database_info"] = [get_new_database_info, get_new_database_info_doc]

# 3. 在 main_cascade.py 中启用
reposits = [..., "get_new_database_info"]
```

### 9.2 下游任务应用

| 任务 | 应用方式 |
|------|----------|
| **基因功能注释** | 直接使用 GeneAgent 输出 |
| **通路富集分析** | 结合 `main_summary.py` 获取富集术语 |
| **文献挖掘** | 使用 PubMed API 结果作为起点 |
| **疾病关联分析** | 利用 disease API 输出 |

### 8.3 集成方案

#### Web API 集成

NCBI 提供了演示网站：
```
https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/GeneAgent/
```

#### 本地部署

```python
# 作为 Python 模块使用
from main_cascade import GeneAgent

# 处理单个基因集
GeneAgent(ID="custom_001", genes="BRCA1,BRCA2,TP53,ATM")
```

---

## 9. 相关技术分析

### 9.1 Agent 技术

✅ **是的，GeneAgent 是一个典型的 LLM Agent**

| Agent 特征 | GeneAgent 实现 |
|------------|----------------|
| **工具调用** | 8 种生物数据库 API |
| **多步推理** | 生成 → 声明 → 验证 → 修正 |
| **自主规划** | GPT-4o 自主选择调用哪些 API |
| **环境交互** | 与外部数据库实时交互 |

**Agent 架构模式**: ReAct (Reasoning + Acting)

```
┌─────────────────────────────────────────────────────┐
│                  ReAct 循环                          │
│                                                     │
│   Thought: 我需要验证这个声明关于 MAPK 通路的陈述     │
│   Action: get_pathway_for_gene_set("ERBB2,KRAS")   │
│   Observation: {"term": "MAPK signaling", ...}      │
│   Thought: 证据支持这个声明                          │
│   Action: 生成验证报告                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 9.2 RAG（检索增强生成）

✅ **GeneAgent 采用了 RAG 的变体形式**

| RAG 组件 | GeneAgent 对应 |
|----------|----------------|
| **检索器** | 8 种 API 作为"动态检索器" |
| **知识库** | 外部生物数据库（NCBI, Enrichr 等） |
| **生成器** | GPT-4o |
| **增强方式** | 将 API 返回结果注入对话历史 |

**与传统 RAG 的区别**:
- 传统 RAG：基于向量相似度的被动检索
- GeneAgent：基于 LLM 决策的主动工具调用

### 9.3 数据模态

GeneAgent 是一个 **纯文本系统**：

| 阶段 | 数据形式 |
|------|----------|
| 输入 | 文本（基因符号列表） |
| 中间表示 | 文本（验证报告、声明） |
| 输出 | 文本（Process Name + Analysis） |

---

## 10. 技术栈总结

| 类别 | 具体技术 |
|------|----------|
| **深度学习框架** | PyTorch 1.13.0 |
| **预训练模型** | GPT-4o (Azure OpenAI), MedCPT, SapBERT |
| **数据处理工具** | Pandas 2.1.4, NumPy 1.26.3 |
| **NLP 工具** | tiktoken 0.7.0, NLTK, Gensim |
| **评估工具** | rouge-score, text2vec, transformers |
| **HTTP 客户端** | requests 2.31.0, requests-oauthlib 1.3.1 |
| **可视化** | Seaborn 0.13.2 |
| **开发环境** | Python 3.11.0, Conda |
| **LLM API** | OpenAI API 0.28.0 (Azure) |

### 外部 API 依赖

| API | 用途 | 端点 |
|-----|------|------|
| NCBI PubTator3 | 复合体、疾病、PPI | `ncbi.nlm.nih.gov/research/pubtator3-api/` |
| g:Profiler | 富集分析 | `biit.cs.ut.ee/gprofiler/api/` |
| Enrichr | 通路分析 | `maayanlab.cloud/Enrichr/` |
| NCBI E-utilities | 基因摘要、PubMed | `eutils.ncbi.nlm.nih.gov/` |

---

## 13. 常见问题解答（FAQ）

### Q1: GeneAgent 的核心思想是什么？

**A**: GeneAgent 的核心思想是 **"生成-验证-修正"** 循环。与直接使用 LLM 生成结果不同，GeneAgent 会：
1. 先让 LLM 生成初步答案
2. 将答案分解为可验证的声明
3. 通过调用权威数据库 API 验证每个声明
4. 根据验证结果修正最终输出

这种机制显著减少了 LLM 的幻觉问题，提高了结果的可靠性。

### Q2: 为什么采用级联验证架构（Cascade）而不是一次性验证？

**A**: 级联验证的设计考虑：
1. **粒度分离**：主题（Topic）和分析叙述（Analysis）的验证需求不同
2. **迭代修正**：先修正主题名称，再基于修正后的主题验证分析
3. **错误传播控制**：如果主题错误，不会影响后续分析的验证
4. **计算效率**：分阶段验证可以更早终止明显错误的输出

### Q3: 如何在自己的数据上使用 GeneAgent？

**A**: 

```python
# 1. 准备数据文件 (CSV 格式)
"""
ID,Name,Count,Genes
my_gene_set_001,Unknown,5,GENE1 GENE2 GENE3 GENE4 GENE5
"""

# 2. 修改 main_cascade.py 中的数据路径
data = pd.read_csv("your_data.csv", header=0, index_col=None)

# 3. 配置 OpenAI API 密钥
openai.api_key = "your_api_key"

# 4. 运行
python main_cascade.py

# 5. 结果在 Outputs/GeneAgent/Cascade/ 目录下
```

### Q4: API 调用失败时如何处理？

**A**: GeneAgent 有内置的错误处理机制：

```python
try:
    function_response = function_to_call(**function_params)
except Exception as E:
    message_verification.append({
        "role": "function",
        "name": function_name,
        "content": f"Function returned error: {E}. Please try again.",
    })
```

Agent 会尝试重新调用或选择其他 API。最大迭代次数（20次）防止无限循环。

### Q5: 为什么使用 Azure OpenAI 而不是官方 OpenAI API？

**A**: 原论文使用 Azure OpenAI 主要是因为：
- NIH 内部的合规要求
- 更稳定的企业级服务
- 数据隐私考虑

你可以轻松切换到官方 API：

```python
# 修改 main_cascade.py
import openai
openai.api_key = "sk-..."  # 官方 API 密钥
# 移除 api_type, api_base, api_version 设置
```

### Q6: 如何评估自己的输出质量？

**A**: 使用 `evaluate.ipynb` 中的评估流程：

1. **ROUGE 分数**：如果有 Ground Truth 术语
2. **语义相似度**：使用 MedCPT 计算嵌入相似度
3. **富集匹配**：检查是否与 g:Profiler 富集结果一致
4. **人工评估**：领域专家审核

### Q7: GeneAgent 的主要限制是什么？

**A**:
1. **依赖外部 API**：数据库 API 的可用性和响应速度影响性能
2. **成本较高**：大量 GPT-4o 调用产生费用
3. **仅支持人类基因**：部分 API 仅支持 Homo sapiens
4. **术语标准化有限**：生成的术语不总是与标准术语完全匹配

### Q8: 如何减少 API 调用成本？

**A**:
1. 使用 `GO_toy.csv` 等小数据集测试
2. 减少 `reposits` 中的 API 数量
3. 降低 AgentPhD 的最大迭代次数
4. 考虑使用 GPT-3.5-turbo 进行初步测试

### Q9: 为什么验证报告中有时显示 "Failed"？

**A**: "Failed" 表示 AgentPhD 在 20 轮迭代内未能生成有效的验证报告。可能原因：
- API 返回空结果（基因集过于罕见）
- 声明过于复杂无法验证
- API 调用多次失败

### Q10: 如何扩展支持鼠标或其他物种？

**A**: 目前部分 API 支持多物种：

```python
# get_gene_summary_for_single_gene 支持 Homo 和 Mus
get_gene_summary_for_single_gene(gene_name="Tp53", specie="Mus")
```

扩展其他物种需要修改相应 API 的物种参数。

---

## 12. 参考资料

### 相关论文

1. **GeneAgent 原论文**
   - 标题: *GeneAgent: Self-verification Language Agent for Gene Set Analysis using Domain Databases*
   - 作者: NCBI/NLM 计算生物学分支
   - DOI: [10.5281/zenodo.15008591](https://zenodo.org/records/15008591)

2. **LLM 基因集解释评估**
   - GitHub: [idekerlab/llm_evaluation_for_gene_set_interpretation](https://github.com/idekerlab/llm_evaluation_for_gene_set_interpretation)

3. **Talisman 基准**
   - GitHub: [monarch-initiative/talisman-paper](https://github.com/monarch-initiative/talisman-paper)

### 官方资源

- **演示网站**: https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/GeneAgent/
- **GitHub 仓库**: https://github.com/ncbi-nlp/GeneAgent
- **Azure OpenAI 文档**: https://learn.microsoft.com/en-us/azure/ai-services/

### 相关数据库

| 数据库 | 网址 | 用途 |
|--------|------|------|
| Gene Ontology | https://geneontology.org/ | 标准生物过程术语 |
| MSigDB | https://www.gsea-msigdb.org/ | 基因集数据库 |
| KEGG | https://www.kegg.jp/ | 通路数据库 |
| Reactome | https://reactome.org/ | 通路数据库 |
| g:Profiler | https://biit.cs.ut.ee/gprofiler/ | 富集分析工具 |
| Enrichr | https://maayanlab.cloud/Enrichr/ | 富集分析工具 |
| PubTator | https://www.ncbi.nlm.nih.gov/research/pubtator3/ | 生物医学文本挖掘 |

### 技术教程

- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- MedCPT 模型: https://huggingface.co/ncbi/MedCPT-Query-Encoder
- ReAct Agent 模式: https://arxiv.org/abs/2210.03629

---

## 附录 A: 完整示例运行

### 输入

```python
genes = "ERBB2,ERBB4,FGFR2,FGFR4,HRAS,KRAS"
```

### Step 1: 初始生成

```
Process: MAPK Signaling Pathway

The proteins encoded by the genes ERBB2, ERBB4, FGFR2, FGFR4, HRAS, and KRAS 
are all integral components of the MAPK signaling pathway...
```

### Step 2: 主题声明

```json
[
  "ERBB2,ERBB4,FGFR2,FGFR4,HRAS,KRAS are involved in MAPK signaling pathway",
  "These genes encode proteins that regulate cell growth and differentiation"
]
```

### Step 3: 验证报告

```
Original_claim: ERBB2,ERBB4,FGFR2,FGFR4,HRAS,KRAS are involved in MAPK signaling
Verified_claim: SUPPORTED. Evidence from KEGG pathway analysis shows these genes 
overlap with "MAPK signaling pathway" (KEGG:04010) with p-value < 0.001.
```

### Step 4: 最终输出

```
Process: MAPK Signaling Pathway

The proteins encoded by the genes ERBB2, ERBB4, FGFR2, FGFR4, HRAS, and KRAS 
are all integral components of the MAPK signaling pathway, which is crucial 
for cell growth, differentiation, and survival.

ERBB2 and ERBB4 are members of the epidermal growth factor receptor (EGFR) 
family of receptor tyrosine kinases (RTKs). ERBB2 is unique in that it has 
no known ligands, and it prefers to form heterodimers with other EGFR family 
members, enhancing their kinase activity. ERBB4 is activated by neuregulins 
and other factors and induces a variety of cellular responses including 
mitogenesis and differentiation.

FGFR2 and FGFR4 are part of the fibroblast growth factor receptor (FGFR) 
family of RTKs. They are activated by fibroblast growth factors, leading to 
receptor dimerization and autophosphorylation.

HRAS and KRAS are GTPases that act as molecular switches in RTK signaling. 
They are activated by guanine nucleotide exchange factors (GEFs) that catalyze 
the exchange of GDP for GTP. Once activated, RAS proteins can interact with 
a variety of effector proteins to propagate the signal downstream.

The interaction between these proteins forms a complex network of signaling 
events that regulate key cellular processes. Dysregulation of this system can 
result in uncontrolled cell growth and cancer.
```

---

*文档版本: 1.0 | 生成日期: 2026-01-23 | 基于 GeneAgent 仓库分析*
