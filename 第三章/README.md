# RadioLLM - 无线电信号大模型

基于大语言模型（LLM）的无线电信号处理框架，支持预训练、微调和评估。本框架针对LLM处理无线电信号面临的**信号-语言模态割裂**与**高低频表征解耦**两大核心挑战，提出了三项关键创新技术：**混合提示与令牌重编程（HPTR）**、**频率调谐融合机制（FAF）** 以及**检索增强生成与自适应门控融合（RAG-AGF）**。

## 项目结构

```
├── main.py                      # 统一入口文件
├── config/                      # YAML 配置文件
│   ├── datasets.yaml            # 数据集参数配置
│   ├── pretrain_radiollm.yaml   # RadioLLM 预训练配置
│   ├── pretrain_radiollm_rag.yaml # RadioLLM_RAG 预训练配置
│   ├── finetune_radiollm.yaml   # RadioLLM 微调配置
│   ├── finetune_radiollm_rag.yaml # RadioLLM_RAG 微调配置
│   ├── evaluate_radiollm.yaml   # RadioLLM 评估配置
│   └── evaluate_radiollm_rag.yaml # RadioLLM_RAG 评估配置
├── model/                       # 模型定义
│   ├── radiollm.py              # RadioLLM 模型（含 HPTR、FAF 核心实现）
│   ├── RadioLLM_RAG.py          # RadioLLM_RAG 模型（含 HNSW、AGF）
│   ├── embed.py                 # 嵌入模块
│   ├── norm.py                  # 归一化模块
│   └── CVCNN.py                 # CNN 模型
├── scheme/                      # 算法策略
│   ├── pretrain.py              # 预训练策略
│   ├── finetune.py              # 微调策略
│   └── evaluate.py              # 评估策略
├── tool/                        # 工具函数
│   ├── __init__.py              # 工具模块入口
│   ├── parser.py                # 配置解析器
│   ├── rml2016a.py              # 数据集加载工具
│   ├── utils.py                 # 核心工具函数
│   ├── model_component.py       # 模型组件（HFE高频提取、SWT_RAG对比学习等）
│   ├── sig_tools.py             # 信号处理工具
│   └── signal_aug.py            # 信号增强工具
├── data/                        # 实验数据
│   ├── pretrain_data/           # 预训练数据
│   └── fintune_data/            # 微调数据
├── pretrain_model/              # 预训练 LLM 权重
├── checkpoint_pretrtain/        # 预训练 checkpoint
├── checkpoint_fintune/          # 微调 checkpoint
├── result/                      # 实验结果
└── runs/                        # 训练日志
```

## 快速开始

### 环境要求

- Python 3.10.9
- PyTorch 2.3.0+cu121
- CUDA (推荐)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 预训练

```bash
# RadioLLM 预训练
python main.py --cfg config/pretrain_radiollm.yaml

# RadioLLM_RAG 预训练（仅支持 RML2016 数据集）
python main.py --cfg config/pretrain_radiollm_rag.yaml
```

### 微调

```bash
# RadioLLM 微调（RMLA 数据集，11 类）
python main.py --cfg config/finetune_radiollm.yaml

# RadioLLM_RAG 微调
python main.py --cfg config/finetune_radiollm_rag.yaml
```

### 评估

```bash
# RadioLLM 评估（默认评估 RMLA 数据集，11 类）
python main.py --cfg config/evaluate_radiollm.yaml

# RadioLLM_RAG 评估（默认评估 RMLA 数据集，11 类）
python main.py --cfg config/evaluate_radiollm_rag.yaml
```

通过 `--override` 参数可以评估其他微调数据集：

```bash
# RadioLLM 评估 RMLB 数据集（10 类）
python main.py --cfg config/evaluate_radiollm.yaml --override dataset=RMLB numclass=10

# RadioLLM 评估 RMLC 数据集（11 类）
python main.py --cfg config/evaluate_radiollm.yaml --override dataset=RMLC numclass=11

# RadioLLM_RAG 评估 RMLB 数据集（10 类）
python main.py --cfg config/evaluate_radiollm_rag.yaml --override dataset=RMLB numclass=10

# RadioLLM_RAG 评估 RMLC 数据集（11 类）
python main.py --cfg config/evaluate_radiollm_rag.yaml --override dataset=RMLC numclass=11
```


### 命令行覆盖配置

可以在命令行中覆盖配置文件中的任意参数：

```bash
# 覆盖微调数据集
python main.py --cfg config/finetune_radiollm.yaml --override dataset=RMLB numclass=10

# 覆盖模型和数据集
python main.py --cfg config/finetune_radiollm.yaml --override model_name=RadioLLM_RAG dataset=RMLC numclass=11
```

## 配置说明

所有配置通过 YAML 文件管理，主要配置项：

| 配置项 | 说明 | 可选值 |
|--------|------|--------|
| `task_name` | 任务类型 | `soft_hard_prompt3`(预训练), `classification`(分类) |
| `model_name` | 模型名称 | `RadioLLM`, `RadioLLM_RAG` |
| `mode` | 运行模式 | `train`(训练), `evaluate`(评估) |
| `dataset` | 微调数据集 | `RMLA`(11类), `RMLB`(10类), `RMLC`(11类) |
| `llm_model` | 大语言模型 | `GPT2`, `LLAMA`, `BERT` |
| `is_LORA` | 是否使用 LoRA | `True`, `False` |

### 数据集约束

- **RadioLLM**: 支持 RML2016, RML2018, ADSB数据集
- **RadioLLM_RAG**: 仅支持 RML2016 数据集
- **微调数据集**: RMLA(11类), RMLB(10类), RMLC(11类)

## 模型说明

### RadioLLM

基于 LLM 的无线电信号处理模型，核心组件：
- **PatchEmbedding**: 将 I/Q 信号切分为 patches
- **ReprogrammingLayer**: 多层交叉注意力，将信号嵌入映射到 LLM 空间
- **AttentionFusion**: 注意力融合机制
- **Soft/Hard Prompt**: 基于 top-k 词嵌入选择的提示工程

### RadioLLM_RAG

检索增强生成版本，在 RadioLLM 基础上增加：
- **HNSW 索引**: 使用 FAISS HNSW 进行高效相似性检索
- **NeuralWeightedFusion**: 可学习的门控融合机制
- **RAG 增强**: 从数据库中检索相似信号特征辅助生成

---

## 核心创新点

本框架针对LLM应用于认知无线电任务（CRT）时面临的两大核心挑战，提出了三项关键技术。

### 研究动机

#### 挑战一：信号-语言模态割裂

LLM预训练于海量离散文本语料，其输入空间天然被限定于文本令牌序列。然而无线电信号本质上是**高维、连续、复值的时序序列**，其关键判别信息（如调制跳变点、脉冲前沿相位突变、载频微偏等）均以非符号化的物理量形式存在。现有方法依赖自然语言中介（如将信号转为文本描述），导致：

- **特征丢失**：文本化过程将连续波形压缩为有限词元组合，滤除高频瞬态细节
- **语义失真**：人工描述准确性依赖领域专家经验，易产生"幻觉"误判

#### 挑战二：高低频表征解耦

标准Transformer的自注意力机制存在固有的**低通滤波效应**，倾向于捕获低频全局结构而抑制高频成分。然而无线电信号的关键判别信息（如BPSK/QPSK的码元跳变点、线性调频信号的频率斜率突变、雷达脉冲的前沿/后沿瞬态）恰恰集中于高频瞬态区域。若仅依赖LLM基座建模，这些关键特征将被平滑或丢失，导致相似类别混淆。

---

### 创新点一：混合提示与令牌重编程（HPTR）

**HPTR（Hybrid Prompt and Token Reprogramming）** 旨在消除信号-语言模态鸿沟，实现LLM对无线电信号的原生理解与可控推理。

#### 1. 提示模板构成

提示模板由三个语义组件构成（[`radiollm.py`](model/radiollm.py:829-838)）：

- **数据集描述**：概括信号来源、调制类型集合、信噪比范围及采集环境等先验信息
- **任务描述**：明确当前下游任务目标（如"对输入信号执行调制分类"或"复原被噪声污染的原始波形"）
- **输入统计特征**：动态计算当前样本的时域与频域统计量，包括最小值、最大值、中位数、整体趋势及前五大滞后特征

#### 2. 语义锚点检索与混合提示生成

针对传统固定模板提示存在的效率瓶颈（语法冗余词元挤占信号表征空间、过长提示序列稀释梯度更新强度），提出语义锚点检索策略（[`radiollm.py`](model/radiollm.py:874-876)）：

1. 从预训练词嵌入矩阵 `word_embeddings` 中，通过映射层 `mapping_layer` 构建高信息密度的**语义锚点库**（由通信领域术语、调制类标签、设备指纹描述等构成）
2. 通过**余弦相似度**度量模板嵌入与锚点库的语义亲和性，选取相似度最高的K个锚点（[`soft_hard_prompt`](model/radiollm.py:223-238)）
3. 仅保留高信息密度的关键词元，替代冗长的文本模板，显著降低计算开销

```python
# 语义锚点检索核心代码
prompt_word_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
prompt_embeddings = soft_hard_prompt(prompt_embeddings, prompt_word_embeddings, self.K)
```

其中 `soft_hard_prompt` 通过 `torch.topk` 选取与当前提示最相似的K个词嵌入向量作为混合提示。

#### 3. 信号重编程

摒弃依赖自然语言中介的间接适配范式，采用**令牌重编程技术**（[`radiollm.py`](model/radiollm.py:861-866)）：

1. 原始I/Q序列经分块处理生成信号嵌入 `enc_out_sgn`
2. 通过**多头交叉注意力机制**（`ReprogrammingLayer`）将信号嵌入重编程至LLM令牌空间
3. 该过程不依赖自然语言描述，保留了信号完整的时频结构与物理可解释性

```python
# 信号重编程核心流程
enc_out_sgn, n_vars = self.patch_embedding[dataset_name](x_enc)          # 信号分块嵌入
enc_out = self.reprogramming_layer(enc_out_sgn, source_embeddings, ...)  # 跨注意力重编程
enc_out = self.attn_fusion(enc_out, enc_out_sgn_1)                       # 高频特征融合
```

> **效果**：消除了模态鸿沟，实现了高效知识注入，使LLM能够直接处理原始波形并保持物理一致性。

---

### 创新点二：频率调谐融合机制（FAF）

**FAF（Frequency Attuned Fusion）** 旨在解决高低频表征解耦问题，通过跨模态融合策略协同整合局部高频特征与全局语义表征。

#### 核心架构

FAF模块由三个**高频提取层（HFE, High-Frequency Extraction）** 组成（[`High_freq_conv_3layer`](tool/model_component.py:96-126)），每个HFE层：

1. **复数卷积**：利用卷积检测局部变化，支持复数域运算（实部/虚部分别卷积），保留相位信息
2. **ReLU激活**：增强非线性特征
3. **残差特征级联**：每个HFE内部进行残差连接，保留原始信息流
4. **池化降维**：通过 `MaxPool1d` 压缩冗余信息

```python
# HFE三层级联结构
self.high_freq_exter  = high_freq_extract(in_channels=d_model,  out_channels=d_model*2)
self.high_freq_exter2 = high_freq_extract(in_channels=d_model,  out_channels=out_channel*2)
self.high_freq_exter3 = high_freq_extract(in_channels=out_channel, out_channels=out_channel*2)
self.pool = nn.MaxPool1d(kernel_size=2)
```

#### 融合流程

FAF的完整融合流程在 [`radiollm.py`](model/radiollm.py:861-866) 中实现：

1. 原始信号经 `patch_embedding` 生成信号嵌入 `enc_out_sgn`
2. 同时经 `high_freq_extract`（即 `High_freq_conv_3layer`）提取高频特征 `enc_out_sgn_1`
3. 通过 `ReprogrammingLayer` 将信号嵌入重编程至LLM空间
4. 通过 `AttentionFusion` 将高频特征与重编程后的信号表征进行注意力融合

```python
enc_out_sgn, n_vars = self.patch_embedding[dataset_name](x_enc)          # 低频全局表征
enc_out_sgn_1 = self.high_freq_extract[dataset_name](x_enc)              # 高频局部特征
enc_out_sgn_1 = einops.rearrange(enc_out_sgn_1, 'b (n d) l -> (b n) l d', n=n_vars)
enc_out = self.reprogramming_layer(enc_out_sgn, source_embeddings, source_embeddings)
enc_out = self.attn_fusion(enc_out, enc_out_sgn_1)                       # 高低频融合
```

`AttentionFusion`（[`radiollm.py`](model/radiollm.py:240-258)）通过可学习的注意力机制计算高频特征与低频表征的相似度，生成融合后的增强表征：

```python
class AttentionFusion(nn.Module):
    def forward(self, enc_out, enc_out_sgn):
        query = self.query(enc_out)           # 低频全局表征作为查询
        key = self.key(enc_out_sgn)           # 高频局部特征作为键
        value = self.value(enc_out_sgn)       # 高频局部特征作为值
        attention_scores = matmul(query, key.transpose(-2, -1))
        attention_weights = softmax(attention_scores / sqrt(d))
        fused = enc_out + matmul(attention_weights, value)  # 残差融合
        return fused
```

> **效果**：弥补了标准Transformer因低通滤波效应导致的细节丢失，使模型能够对输入无线电信号实现更全面的表征，兼顾全局低频信息与细粒度高频细节。

---

### 创新点三：检索增强生成与自适应门控融合（RAG-AGF）

引入**检索增强生成（RAG）** 机制，构建面向无线电信号的动态知识注入通路，增强大模型在开放电磁环境中的即时知识获取能力。

#### 1. 检索向量数据库构建

基于**MoCo框架**的对比学习策略（[`SWT_RAG`](tool/model_component.py:128-164)）对FAF网络进行预训练：

- 以原始I/Q序列为输入，经FAF编码器生成高维特征向量
- 通过动量更新的队列机制拉近同一信号的增强视图距离、推远异类样本
- 学习对信道失真与设备漂移鲁棒的通用表征
- 对数据库中所有样本提取L2归一化后的特征向量作为键值

#### 2. 分层导航特征检索（HNSW）

采用**分层可导航小世界图（HNSW）** 算法构建索引结构（[`RadioLLM_RAG.py`](model/RadioLLM_RAG.py:861-946)）：

- **多层跳表结构**：高层图节点稀疏，用于粗粒度全局导航；低层图节点稠密，用于细粒度局部精搜
- **自顶向下检索**：从顶层入口节点开始贪婪搜索，快速缩小候选区域；到底层后扩展搜索半径，返回K最近邻
- 将最近邻搜索复杂度从线性降至对数级别，同时保持高召回率

```python
# HNSW检索核心
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

#### 3. 自适应门控融合（AGF）

设计**自适应门控融合机制**（[`NeuralWeightedFusion`](model/RadioLLM_RAG.py:345-372)）实现检索先验与当前信号观测的高效融合：

1. **相似度加权聚合**：计算当前特征与检索特征的余弦相似度，生成聚合检索特征
2. **门控神经网络**：轻量级门控网络学习原始特征与检索特征之间的最优融合策略
3. **动态权衡**：在保留原始信号信息的同时，注入检索得到的增强特征

```python
# AGF融合核心
def fuse_with_hnsw(self, enc_out_sgn_high, dataset_name, k=5):
    hnsw_enh = self.HNSW_RAG(enc_out_sgn_high, dataset_name, k=k)  # HNSW检索增强
    fused = self.hnsw_fusion(A, B)  # NeuralWeightedFusion门控融合
    return fused
```

> **效果**：有效缓解LLM在面对未知调制样式或新型设备时的"幻觉"风险，提升开放电磁环境中的决策鲁棒性。

---

## 多任务端到端认知

RadioLLM支持统一的预训练与下游任务适配框架（[`scheme/pretrain.py`](scheme/pretrain.py)）：

### 预训练阶段

- 使用**多数据集联合训练**（RMLA、RMLB、RMLC、ADS-B），通过 `MergedLoader` 按概率采样
- 采用**损失平衡因子** `balence_list` 归一化各数据集的梯度贡献，缓解数据规模差异引起的偏差
- 通过**LoRA微调**更新LLM参数，冻结大部分预训练权重
- 三种损失策略：`masking`（掩码重建）、`noise`（去噪）、`mixup`（混合增强），按阈值控制
- 预训练目标：最小化与真实信号之间的MSE损失

### 下游任务适配

- **分类任务**：从LLM输出中提取特征表示，经池化后输入线性分类头
- **降噪/信号重建任务**：解码器输出直接用作最终预测

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

### 数据划分

- RMLA、RMLB、RMLC和ADS-B按 **8:1:1** 比例划分为训练集、验证集和测试集
- 预训练阶段：RML系列仅选取 **SNR ≥ 14dB** 的高质量样本；ADS-B使用完整训练集
- 对比实验采用 **"多数据集预训练/训练，单数据集微调测试"** 的统一范式
