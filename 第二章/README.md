# 电磁信号识别训练框架

基于 **ACSF-TMAE**（Attentional Channel-Spectral Fusion Transformer-based Masked AutoEncoder）模型的电磁信号自监督认知方法框架，支持生成式预训练和有监督微调，覆盖信号分类、信号降噪与信号缺失复原三大下游任务。

---

## 核心创新点

### 创新点一：通道-光谱双路径注意力机制（AC-SF）

**问题**：PWVD 时频图中频谱能量分布不均（能量集中于低频与高频区域，中频区域稀疏），导致模型难以从所有 patch 中提取有效信息。

**方案**：在编码器输出的潜在特征上，设计双路径注意力模块 [`AC_SF`](model/pwvdswin_ViT.py:332-370)，并行执行：

- **谱注意力路径**：在 patch 序列维度（`B, H*W, C`）上通过 MLP（Tanh + Sigmoid）生成加权系数，自适应聚焦能量集中的频谱区域。
- **通道注意力路径**：在特征通道维度（`B, C, H*W`）上通过 MLP（Tanh + Sigmoid）生成加权系数，聚焦信息量丰富的特征通道。

两条路径输出逐元素相加融合，有效缓解时频能量分布不均导致的表征偏差。该模块在解码器入口处调用（[`model/ACSF_TMAE.py:220`](model/ACSF_TMAE.py:220)）。

---

### 创新点二：基于互信息最大化的噪声鲁棒特征学习（MIM）

**问题**：低信噪比或高比例信号缺失等极端场景下，强增强样本与原始信号互信息过低，MAE 重建困难，编码器难以提取高层频谱语义。

**方案**：引入 MIM 框架，通过 **InfoNCE 损失**（[`tool/loss.py:362-476`](tool/loss.py:362)）最大化强增强与弱增强潜在特征之间的互信息。以弱增强作为隐式目标引导强增强的特征学习，迫使编码器提取深层结构化特征而非依赖局部像素统计。

**损失函数构成**（[`scheme/pretrain_scheme.py:124-129`](scheme/pretrain_scheme.py:124)）：

```python
loss1 = MAELoss(pred_noise, target)       # 强增强重建损失（SmoothL1）
loss2 = MAELoss(pred_clean, target)       # 弱增强重建损失（SmoothL1）
loss3 = InfoNCE(latent_clean, latent_noise)  # 互信息最大化

loss = lamba * (loss1 + loss2) + (1 - lamba) * loss3
```

---

### 创新点三：生成式多任务统一框架

**问题**：传统自监督方法（MoCo、SimCLR 等对比学习）局限于单一 AMC 任务，无法支持信号降噪、缺失复原等像素级重建任务。

**方案**：构建统一的生成式预训练框架，天然支持三大下游任务：

| 下游任务 | 实现方式 | 代码入口 |
|---------|---------|---------|
| **信号分类** | 预训练编码器 + AC-SF + MLP 分类头微调 | [`ACSF_TMAE.forward_classification()`](model/ACSF_TMAE.py:246-252) |
| **信号降噪** | 编码器-解码器重建 + 逆时频变换 | [`ACSF_TMAE.forward_pretrain()`](model/ACSF_TMAE.py:233-245) |
| **信号缺失复原** | 编码器-解码器重建 + 逆时频变换 | 同上 |

实现从"判别性特征学习"向"结构性表征建模"的范式转变，一次预训练即可迁移至多种认知无线电任务。

---

### 创新点四：PWVD 时频变换与 Swin Transformer 的深度融合

**方案**：

- **PWVD 时频变换**（[`tool/signeltoimage.py:77-110`](tool/signeltoimage.py:77)）：将 I/Q 信号转换为伪魏格纳-维尔分布时频图，提供优异的时频聚集性，相比 STFT 具有更高的时频分辨率。
- **Swin Transformer 编码器**（[`model/pwvdswin_ViT.py:478-519`](model/pwvdswin_ViT.py:478)）：采用移动窗口多头自注意力（SW-MSHA），实现线性计算复杂度下的全局感受野，逐步提取从局部到全局的频谱特征。
- **图像块合并（Patch Merging）**：类似 CNN 的层级下采样结构，逐步扩大感受野并降低计算量。
- **图像块扩展（Patch Expanding）**：解码器中的对称上采样结构，将下采样的特征逐步恢复至原始分辨率。

---

## 项目结构

```
├── main.py                          # 主程序入口
├── config/                          # 配置文件目录
│   ├── pretrain/                    # 预训练配置文件
│   │   ├── pretrain_RMLA.yaml       # RMLA 预训练配置
│   │   ├── pretrain_RMLB.yaml       # RMLB 预训练配置
│   │   └── pretrain_RMLC.yaml       # RMLC 预训练配置
│   └── fintune/                     # 微调配置文件
│       ├── fintune_RMLA.yaml        # RMLA 微调配置
│       ├── fintune_RMLB.yaml        # RMLB 微调配置
│       └── fintune_RMLC.yaml        # RMLC 微调配置
├── data/                            # 数据目录
│   ├── pretrain_data/               # 预训练数据
│   │   ├── RMLA_gaodbdata.npy
│   │   ├── RMLB_gaodbdata.npy
│   │   └── RMLC_gaodbdata.npy
│   └── fintune_data/                # 微调数据
│       ├── RMLA_train_data.npy
│       ├── RMLA_test_data.npy
│       ├── RMLB_train_data.npy
│       ├── RMLB_test_data.npy
│       ├── RMLC_train_data.npy
│       └── RMLC_test_data.npy
├── model/                           # 模型定义
│   ├── ACSF_TMAE.py                 # ACSF-TMAE 主模型
│   └── pwvdswin_ViT.py             # Swin Transformer 基础组件 + AC-SF 模块
├── scheme/                          # 训练方案
│   ├── base_scheme.py               # 基础方案类
│   ├── pretrain_scheme.py           # 预训练方案（含 MIM 损失）
│   ├── fintune_scheme.py            # 微调方案
│   └── evaluate_scheme.py           # 评估方案
├── tool/                            # 工具函数
│   ├── parser.py                    # 命令行参数解析
│   ├── rml2016a.py                  # 数据集加载和预处理
│   ├── utils.py                     # 通用工具函数
│   ├── loss.py                      # 损失函数（MAELoss, InfoNCE 等）
│   ├── signal_aug.py                # 信号增强（加噪、掩码、时间扭曲等）
│   ├── signeltoimage.py             # 信号转图像（PWVD, STFT, CWT 等）
│   └── pos_embed.py                 # 位置编码
├── checkpoint_pretrain/             # 预训练检查点
│   ├── RMLA/
│   ├── RMLB/
│   └── RMLC/
├── checkpoint_fintune/              # 微调检查点
│   ├── RMLA/
│   ├── RMLB/
│   └── RMLC/
└── runs/                            # 训练日志
```

---

## 环境要求

- Python 3.8+
- PyTorch 2.3.0+
- CUDA 12.1（推荐）

### 安装依赖

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 1. 预训练

```bash
python main.py --cfg config/pretrain/pretrain_RMLA.yaml
python main.py --cfg config/pretrain/pretrain_RMLB.yaml
python main.py --cfg config/pretrain/pretrain_RMLC.yaml
```

### 2. 微调

```bash
python main.py --cfg config/fintune/fintune_RMLA.yaml
python main.py --cfg config/fintune/fintune_RMLB.yaml
python main.py --cfg config/fintune/fintune_RMLC.yaml
```

### 3. 评估

```bash
python main.py --cfg config/fintune/fintune_RMLA.yaml --evaluate
```

### 4. 命令行覆盖参数

```bash
python main.py --cfg config/pretrain/pretrain_RMLA.yaml --batchsize 64 --lr 0.0005 --epochs 500
```

---

## 配置文件说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `task` | 任务类型 | `pretraining` / `classification` |
| `dataset` | 数据集名称 | `RMLA` / `RMLB` / `RMLC` |
| `trans_choose` | 时频变换方式 | `pwvd` / `stft` / `wave` |
| `batchsize` | 批大小 | `128` / `256` |
| `epochs` | 训练轮数 | `1000` / `50` |
| `lr` | 学习率 | `0.001` / `0.0005` |
| `patience` | 早停耐心值 | `20` / `200` |
| `wait` | 学习率调整等待轮数 | `5` |
| `samplenum` | 信号下采样倍数 | `6` (pwvd) / `4` (stft) |
| `lamba` | 重建损失与 MIM 损失平衡系数 | `0.99` |

---

## 数据集

支持 RML2016.10a 数据集的三个变体：

- **RMLA**: 原始 RML2016.10a 数据集（11 类调制信号）
- **RMLB**: RML2016.10a 的变体 B
- **RMLC**: RML2016.10a 的变体 C

### 数据格式

- 预训练：`{dataset}_gaodbdata.npy`（含 `data` 和 `label` 字段的字典）
- 微调：`{dataset}_train_data.npy` / `{dataset}_test_data.npy`

---

## 训练流程

1. **预训练阶段**：在无标签数据上使用 MAE + MIM 进行生成式自监督学习
2. **微调阶段**：加载预训练权重，在有标签数据上进行分类微调
3. **评估阶段**：在测试集上评估 Overall Accuracy (OA) 和各 SNR 准确率

---

## 结果输出

训练过程中自动保存：
- 最佳模型检查点（`.pth` 文件）
- 训练日志 CSV 文件（`runs/` 目录）
- 评估结果（OA 和各 SNR 准确率）
