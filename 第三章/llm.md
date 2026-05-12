# RadioLLM 项目 — LLM 理解指南

## 项目概述

本项目实现了一个基于 **GPT2/LLaMA/BERT** 等大语言模型的无线电信号处理框架，包含两个主要模型：

- **RadioLLM**：基础版本，使用 LLM 作为骨干网络处理 I/Q 信号
- **RadioLLM_RAG**：增强版本，引入 HNSW (FAISS) 检索增强生成

支持**预训练**（多数据集多任务）和**微调**（分类任务）两种模式。

本框架针对LLM处理无线电信号面临的**信号-语言模态割裂**与**高低频表征解耦**两大核心挑战，提出了三项关键创新技术：**混合提示与令牌重编程（HPTR）**、**频率调谐融合机制（FAF）** 以及**检索增强生成与自适应门控融合（RAG-AGF）**。

---

## 目录结构

```
第三章/
├── main.py                    # 统一入口：python main.py --cfg config/xxx.yaml
├── config/                    # YAML 配置文件
│   ├── pretrain_radiollm.yaml
│   ├── pretrain_radiollm_rag.yaml
│   ├── finetune_radiollm.yaml
│   ├── finetune_radiollm_rag.yaml
│   ├── evaluate_radiollm.yaml
│   ├── evaluate_radiollm_rag.yaml
│   └── datasets.yaml          # 数据集参数（按数据集名称索引）
├── scheme/                    # 算法策略实现
│   ├── pretrain.py            # 预训练流程
│   ├── finetune.py            # 微调流程
│   └── evaluate.py            # 评估流程
├── model/                     # 神经网络模型定义
│   ├── radiollm.py            # RadioLLM 模型（含 HPTR、FAF 核心实现）
│   ├── RadioLLM_RAG.py       # RadioLLM_RAG 模型（含 HNSW、AGF）
│   ├── embed.py               # 嵌入层
│   ├── norm.py                # 归一化层
│   └── CVCNN.py               # CNN 基线模型
├── tool/                      # 工具函数
│   ├── parser.py              # 配置解析器（Config 类）
│   ├── utils.py               # 通用工具（EarlyStopping, CSVStats, MergedLoader 等）
│   ├── rml2016a.py            # 数据集加载（.mat/.npy/.pkl）
│   ├── model_component.py     # 模型组件（HFE高频提取、SWT_RAG对比学习等）
│   ├── sig_tools.py           # 信号处理工具（滤波、归一化、频域变换）
│   └── signal_aug.py          # 信号增强（噪声、掩码、时间扭曲等）
├── data/                      # 实验数据
│   ├── pretrain_data/         # 预训练数据
│   └── fintune_data/          # 微调数据
├── pretrain_model/gpt2/       # 预训练 LLM 权重
├── checkpoint_pretrtain/      # 预训练检查点
├── checkpoint_fintune/        # 微调检查点
├── result/                    # 评估结果 (.mat)
└── runs/                      # 训练日志 (.csv)
```

---

## 核心数据流

### 配置加载

```
YAML 文件 → load_config() → Config 对象 → resolve_path_templates() → 最终 Config
```

[`Config`](tool/parser.py:13) 类将字典转换为属性访问方式，同时兼容 dict 风格操作（`len()`、`items()`、`keys()`、`config[key]`）。

### 预训练流程

```
main.py → run_pretrain(config)
  ├── load_config(dataset_config) → yaml_config (数据集参数)
  ├── prepare_data(yaml_config, ...) → 加载多数据集
  ├── 为每个数据集创建 DataLoader
  ├── build_model(config, yaml_config) → RadioLLM / RadioLLM_RAG_HNSW
  ├── 训练循环（train_one_epoch / validate_one_epoch）
  │   ├── MergedLoader 按概率采样多数据集
  │   ├── 三种损失策略：masking / noise / mixup（按阈值控制）
  │   └── 损失：MSE + SmoothL1 + CrossEntropy
  └── EarlyStopping 保存最佳模型
```

### 微调流程

```
main.py → run_finetune(config)
  ├── 加载单数据集（RMLA/RMLB/RMLC）
  ├── 构建模型 + 加载预训练权重
  ├── 训练循环（分类任务）
  └── 保存最佳模型
```

### 评估流程

```
main.py → run_evaluate(config)
  ├── 加载模型权重
  ├── 遍历测试集计算 OA / AA / 每类准确率 / 每 SNR 准确率
  └── 保存 .mat 结果文件
```

---

## 关键组件详解

### 1. [`Config`](tool/parser.py:13) 配置类

```python
config = load_config('config/pretrain_radiollm.yaml')
config.lr          # → 0.00001 (自动从 '1e-5' 转换)
config.model_name  # → 'RadioLLM'
config.Names       # → ['RML2016', ...]
len(config)        # → 属性数量
config['lr']       # → 0.00001 (dict 风格访问)
```

自动数值转换：YAML 中 `'1e-5'`、`'42'`、`'None'` 等字符串会被自动转为 `float`/`int`/`None`。

### 2. [`RadioLLM`](model/radiollm.py:348) 模型

```
输入: (batch, 2, seq_len) I/Q 信号
  ↓
High_freq_conv_3layer (信号特征提取)  ← FAF 高频提取
  ↓
PatchEmbedding (分片嵌入)
  ↓
ReprogrammingLayer (跨注意力 → LLM 空间)  ← HPTR 信号重编程
  ↓
AttentionFusion (高低频融合)  ← FAF 融合
  ↓
Soft/Hard Prompt (Top-K 语义锚点检索)  ← HPTR 混合提示
  ↓
LLM Backbone (GPT2/LLaMA/BERT)
  ↓
输出: 预测信号 / 分类结果
```

关键参数（来自 `yaml_config` 数据集配置）：
- `seq_len`、`patch_len`、`stride`：信号分片参数
- `min_mask_ratio`、`max_mask_ratio`：掩码比例范围
- `numclass`：分类类别数

### 3. [`RadioLLM_RAG_HNSW`](model/RadioLLM_RAG.py:1202) 模型

在 RadioLLM 基础上增加 HNSW 检索增强：

```
信号特征 → HNSW 检索 (FAISS) → 检索结果融合 → LLM 处理
```

- 使用 FAISS HNSW 索引进行相似性检索
- `NeuralWeightedFusion` 自适应融合原始特征和检索特征
- 仅支持 RML2016 数据集

### 4. [`prepare_data`](tool/rml2016a.py:122) 数据准备

```python
Names, batch_sizes, num_workers, balence_list, thresholds, data_list, label_list
  = prepare_data(yaml_config, split_ratios, Names, data_paths)
```

- 从 `yaml_config` 中按数据集名称读取 `threshold`、`balence`、`batchsize`、`numworks`
- 支持 `.mat`、`.npy`、`.pkl` 三种格式
- 支持 `Config` 对象和原始 dict 两种输入

### 5. [`MergedLoader`](tool/utils.py:864) 多数据集加载器

```python
loader = MergedLoader([loader1, loader2, ...], probabilities=[p1, p2, ...])
for batch, dataset_idx in loader:
    # batch 来自按概率采样的某个数据集
```

- 按概率从多个 DataLoader 中采样
- 某个 loader 遍历完后自动移除
- 所有 loader 用尽后抛出 `StopIteration`

### 6. 早停机制

四种早停策略（[`tool/utils.py`](tool/utils.py)）：
- [`EarlyStopping`](tool/utils.py:175)：基于验证 loss，含 LoRA 权重保存
- [`EarlyStoppingNoLora`](tool/utils.py:287)：不含 LoRA 的版本
- [`EarlyStopping_acc`](tool/utils.py:399)：基于验证准确率
- [`EarlyStoppingTra`](tool/utils.py:453)：带训练 loss 监控

---

## 核心创新点详解

### 创新点一：混合提示与令牌重编程（HPTR）

**HPTR（Hybrid Prompt and Token Reprogramming）** 旨在消除信号-语言模态鸿沟，实现LLM对无线电信号的原生理解。

#### 背景问题

LLM预训练于离散文本语料，而无线电信号是**高维、连续、复值的时序序列**。现有方法依赖自然语言中介（将信号转为文本描述），导致：
- **特征丢失**：连续波形被压缩为有限词元，滤除高频瞬态细节
- **语义失真**：依赖专家经验，易产生"幻觉"误判

#### 技术方案

##### 1. 提示模板构成（[`radiollm.py`](model/radiollm.py:829-838)）

提示模板由三个语义组件构成：
- **数据集描述**：概括信号来源、调制类型集合、信噪比范围及采集环境
- **任务描述**：明确当前下游任务目标（去噪或掩码重建）
- **输入统计特征**：动态计算当前样本的时域与频域统计量（最小值、最大值、中位数、整体趋势、前五大滞后特征）

```python
prompt_ = (
    f"<|start_prompt|>Dataset description: {self.description[dataset_name]}"
    f"{task_prompt}"
    "Input statistics: "
    f"min value {min_values_str}, "
    f"max value {max_values_str}, "
    f"median value {median_values_str}, "
    f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
    f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
)
```

##### 2. 语义锚点检索与混合提示生成（[`radiollm.py`](model/radiollm.py:874-876)）

针对传统固定模板提示的效率瓶颈，提出语义锚点检索策略：

1. 从预训练词嵌入矩阵 `word_embeddings` 中，通过映射层 `mapping_layer` 构建高信息密度的**语义锚点库**
2. 通过**余弦相似度**度量模板嵌入与锚点库的语义亲和性
3. 通过 `torch.topk` 选取相似度最高的K个锚点作为混合提示（[`soft_hard_prompt`](model/radiollm.py:223-238)）

```python
# 语义锚点检索
prompt_word_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
prompt_embeddings = soft_hard_prompt(prompt_embeddings, prompt_word_embeddings, self.K)

# soft_hard_prompt 核心：余弦相似度 + Top-K 选择
sim = torch.einsum('bld,vd->bv', prompt_embeddings, prompt_word_embeddings)
topk_values, topk_indices = torch.topk(sim, k=K, dim=-1)
prompt_embeddings = torch.gather(prompt_word_embeddings.expand(B, -1, -1), 1, topk_indices_expanded)
```

##### 3. 信号重编程（[`radiollm.py`](model/radiollm.py:861-866)）

摒弃自然语言中介，采用**令牌重编程技术**实现端到端原生建模：

1. 原始I/Q序列经 `patch_embedding` 分块生成信号嵌入
2. 通过 `ReprogrammingLayer`（多头交叉注意力）将信号嵌入映射至LLM令牌空间
3. 该过程保留信号完整的时频结构与物理可解释性

```python
enc_out_sgn, n_vars = self.patch_embedding[dataset_name](x_enc)
enc_out = self.reprogramming_layer(enc_out_sgn, source_embeddings, source_embeddings)
```

> **效果**：消除了模态鸿沟，实现了高效知识注入，使LLM能够直接处理原始波形并保持物理一致性。

---

### 创新点二：频率调谐融合机制（FAF）

**FAF（Frequency Attuned Fusion）** 旨在解决高低频表征解耦问题，通过跨模态融合策略协同整合局部高频特征与全局语义表征。

#### 背景问题

标准Transformer的自注意力机制存在固有的**低通滤波效应**，倾向于捕获低频全局结构而抑制高频成分。然而无线电信号的关键判别信息（如BPSK/QPSK的码元跳变点、线性调频信号的频率斜率突变、雷达脉冲的前沿/后沿瞬态）恰恰集中于高频瞬态区域。

#### 技术方案

##### 1. 高频提取层（HFE）（[`High_freq_conv_3layer`](tool/model_component.py:96-126)）

FAF模块由三个HFE层组成，每层包含：
- **复数卷积**：利用卷积检测局部变化，保留相位信息
- **ReLU激活**：增强非线性特征
- **残差特征级联**：保留原始信息流
- **MaxPool1d池化**：压缩冗余信息

```python
class High_freq_conv_3layer(nn.Module):
    def __init__(self, d_model, out_channel, patch_len, stride, dropout):
        self.high_freq_exter  = high_freq_extract(in_channels=d_model,      out_channels=d_model*2)
        self.high_freq_exter2 = high_freq_extract(in_channels=d_model,      out_channels=out_channel*2)
        self.high_freq_exter3 = high_freq_extract(in_channels=out_channel,  out_channels=out_channel*2)
        self.pool = nn.MaxPool1d(kernel_size=2)
    
    def forward(self, x):
        x = self.high_freq_exter(x)   # 第一层高频提取
        x = self.pool(x)              # 池化降维
        x = self.high_freq_exter2(x)  # 第二层高频提取
        x = self.pool(x)
        x = self.high_freq_exter3(x)  # 第三层高频提取
        x = self.pool(x)
        return x
```

##### 2. 高低频融合（[`radiollm.py`](model/radiollm.py:861-866) + [`AttentionFusion`](model/radiollm.py:240-258)）

```python
enc_out_sgn, n_vars = self.patch_embedding[dataset_name](x_enc)     # 低频全局表征
enc_out_sgn_1 = self.high_freq_extract[dataset_name](x_enc)         # 高频局部特征
enc_out_sgn_1 = einops.rearrange(enc_out_sgn_1, 'b (n d) l -> (b n) l d', n=n_vars)
enc_out = self.reprogramming_layer(enc_out_sgn, source_embeddings, source_embeddings)
enc_out = self.attn_fusion(enc_out, enc_out_sgn_1)                  # 注意力融合
```

`AttentionFusion` 通过可学习的注意力机制实现门控融合：
- 低频全局表征作为 **Query**
- 高频局部特征作为 **Key** 和 **Value**
- 输出为残差连接后的融合特征

> **效果**：弥补了标准Transformer因低通滤波效应导致的细节丢失，使模型兼顾全局低频信息与细粒度高频细节。

---

### 创新点三：检索增强生成与自适应门控融合（RAG-AGF）

引入**检索增强生成（RAG）** 机制，构建面向无线电信号的动态知识注入通路。

#### 1. 检索向量数据库构建（[`SWT_RAG`](tool/model_component.py:128-164)）

基于**MoCo框架**的对比学习策略：
- 以原始I/Q序列为输入，经FAF编码器生成高维特征向量
- 通过动量更新的队列机制拉近同一信号的增强视图距离、推远异类样本
- 学习对信道失真与设备漂移鲁棒的通用表征

#### 2. 分层导航特征检索（HNSW）（[`RadioLLM_RAG.py`](model/RadioLLM_RAG.py:861-946)）

采用**分层可导航小世界图（HNSW）** 算法：
- **多层跳表结构**：高层稀疏（粗粒度导航），低层稠密（细粒度精搜）
- **自顶向下检索**：从顶层入口贪婪搜索，到底层扩展搜索半径
- 复杂度从线性降至对数级别

```python
def _faiss_search_topk(self, q_flat_np, dataset_name, k):
    index = self._get_faiss_index(dataset_name)
    dist, idx = index.search(q_flat_np, k)  # (B, k)
    return index, dist, idx

def HNSW_RAG(self, enc_out_sgn_high, dataset_name, k=5):
    # 逐样本重构k个邻居，加权聚合
    for i in range(B):
        ids = nn_indices[i].tolist()
        neigh = np.stack([index.reconstruct(int(j)) for j in ids], axis=0)
        V = torch.from_numpy(neigh).to(device=device, dtype=dtype)
        qi = q_flat[i]
        di = (V - qi.unsqueeze(0)).pow(2).sum(-1)
        w = torch.softmax(-di, dim=-1)
        out[i] = (w.unsqueeze(1) * V).sum(0)
```

#### 3. 自适应门控融合（AGF）（[`NeuralWeightedFusion`](model/RadioLLM_RAG.py:345-372)）

轻量级门控网络学习原始特征与检索特征之间的最优融合策略：

```python
def fuse_with_hnsw(self, enc_out_sgn_high, dataset_name, k=5):
    hnsw_enh = self.HNSW_RAG(enc_out_sgn_high, dataset_name, k=k)
    fused = self.hnsw_fusion(A, B)  # NeuralWeightedFusion
    return fused
```

> **效果**：有效缓解LLM在面对未知调制样式或新型设备时的"幻觉"风险，提升开放电磁环境中的决策鲁棒性。

---

## 配置文件详解

### 主配置（如 [`pretrain_radiollm.yaml`](config/pretrain_radiollm.yaml)）

| 字段 | 说明 | 示例值 |
|------|------|--------|
| `task_name` | 任务类型 | `soft_hard_prompt3`(预训练) / `classification`(微调) |
| `model_name` | 模型选择 | `RadioLLM` / `RadioLLM_RAG` |
| `mode` | 运行模式 | 留空(训练) / `evaluate` |
| `lr` | 学习率 | `1e-5` |
| `epochs` | 训练轮数 | `100` |
| `batchsize` | 批次大小 | `32` |
| `is_LORA` | 是否使用 LoRA | `True` |
| `llm_model` | 骨干 LLM | `GPT2` / `LLAMA3.2` / `BERT` |
| `dataset_config` | 数据集配置路径 | `./config/datasets.yaml` |
| `Names` | 数据集名称列表 | `[RML2016, RML2018, ADSB, WIFI]` |
| `data_paths` | 数据路径列表 | `[./data/pretrain_data/xxx.mat, ...]` |
| `model_path` | 预训练权重路径 | `./checkpoint_pretrtain/RadioLLM/best_network.pth` |

### 数据集配置（[`datasets.yaml`](config/datasets.yaml)）

按数据集名称索引，每个数据集包含：

| 字段 | 说明 |
|------|------|
| `dataset_name` | 数据集内部名称 |
| `min_mask_ratio` / `max_mask_ratio` | 掩码比例范围 |
| `threshold` | 损失策略阈值 `[mask, noise, mixup]` |
| `balence` | 损失平衡系数 |
| `batchsize` / `numworks` | 批次大小 / 工作进程数 |
| `numclass` | 类别数 |
| `seq_len` / `patch_len` / `stride` | 信号分片参数 |
| `content` | 数据集描述文本 |

---

## 数据集

| 数据集 | 用途 | 类别数 | 说明 |
|--------|------|--------|------|
| RML2016 | 预训练+微调 | 11 | 合成调制信号，含 SNR 变化 |
| RML2018 | 预训练 | 29 | 扩展版本，更多调制类型 |
| ADSB | 预训练 | 198 | 真实飞机应答器信号 |
| WIFI | 预训练 | 16 | USRP 采集的 WiFi 信号 |
| RMLA | 微调 | 11 | 微调专用数据集 |
| RMLB | 微调 | 10 | 微调专用数据集 |
| RMLC | 微调 | 11 | 微调专用数据集 |
| RML22 | 评估泛化性 | 10 | 未参与预训练，用于泛化性测试 |

---

## 运行命令

```bash
# 预训练 (RadioLLM)
python main.py --cfg config/pretrain_radiollm.yaml

# 预训练 (RadioLLM_RAG)
python main.py --cfg config/pretrain_radiollm_rag.yaml

# 微调 (RadioLLM)
python main.py --cfg config/finetune_radiollm.yaml

# 微调 (RadioLLM_RAG)
python main.py --cfg config/finetune_radiollm_rag.yaml

# 评估 (RadioLLM)
python main.py --cfg config/evaluate_radiollm.yaml

# 评估 (RadioLLM_RAG)
python main.py --cfg config/evaluate_radiollm_rag.yaml
```

---

## 注意事项

1. **RadioLLM_RAG 仅支持 RML2016 数据集**，因为 HNSW 索引基于 RML2016 特征构建
2. **预训练**使用 `MergedLoader` 多数据集联合训练，`task_name=soft_hard_prompt3`
3. **微调**使用单数据集分类，`task_name=classification`
4. 配置中的 `{dataset}` 占位符会被自动替换为实际数据集名称
5. YAML 中的科学计数法（如 `1e-5`）会被自动转换为数值类型
