# 基于深度学习的无线电调制识别

本项目包含三个章节的代码，围绕**无线电调制识别**任务，从传统深度学习模型到自监督预训练，再到大语言模型（LLM）的应用，逐步深入探索。

---

## 目录结构

```
├── 第一章/          # 传统深度学习模型与对比实验
├── 第二章/          # 自监督预训练 + 微调范式
├── 第三章/          # 基于大语言模型（LLM）的无线电信号识别
├── .gitignore
└── README.md
```

---

## 第一章：传统深度学习模型

### 概述
实现了多种经典的深度学习模型用于无线电调制识别，包括 CNN、ResNet、Transformer 等架构，并提供完整的训练、评估流程。

### 支持的模型
| 模型 | 说明 |
|------|------|
| CNN2 | 基础卷积神经网络 |
| ResNet1D / ResNet2D | 残差网络（一维/二维） |
| LSTM | 长短期记忆网络 |
| FLAN | 频域注意力网络 |
| ICAMCNet | 改进的通道注意力网络 |
| PETCGDNN | 并行卷积网络 |
| TSFFN | 时频融合网络 |
| Swin Transformer | Swin Transformer |
| ViT | Vision Transformer |
| SKNet1D / SKNet2D | 选择性核网络 |
| MCLDNN | 多尺度卷积网络 |
| DAE | 深度自编码器 |
| HCGDNN | 混合卷积网络 |
| CMTSF | 跨模态时频融合网络 |

### 数据集
- **RML2016.10a**：11 种调制类型，不同 SNR
- **RMLB / RMLC**：扩展调制集
- **ADSB**：航空广播式自动相关监视信号

### 使用方式
```bash
cd 第一章
pip install -r requirements.txt
python main.py --config config/replay/cnn2/cnn2_rml2016a.yaml
```

---

## 第二章：自监督预训练 + 微调

### 概述
采用 **自监督预训练（Masked Autoencoder）** 策略，先在大规模无标签数据上预训练，再在下游任务上微调，提升少样本场景下的识别性能。

### 模型架构
- **ACSF_TMAE**：自适应跨尺度融合时序掩码自编码器
- **PWVD-Swin ViT**：基于 PWVD 时频图的 Swin Transformer

### 流程
1. **预训练阶段**：在无标签数据上进行掩码重建任务
2. **微调阶段**：在少量标签数据上微调分类头

### 使用方式
```bash
cd 第二章
pip install -r requirements.txt

# 预训练
python main.py --mode pretrain --config config/pretrain/pretrain_RMLA.yaml

# 微调
python main.py --mode finetune --config config/fintune/fintune_RMLA.yaml
```

---

## 第三章：基于大语言模型（LLM）的无线电信号识别

### 概述
将 **GPT-2** 大语言模型引入无线电信号识别领域，探索 LLM 在信号处理中的潜力。包含两个变体：
- **RadioLLM**：基础版本，将信号特征映射到 LLM 的嵌入空间
- **RadioLLM-RAG**：引入检索增强生成（RAG），通过检索相似信号辅助识别

### 模型架构
- **RadioLLM**：信号编码器 + GPT-2 解码器
- **RadioLLM-RAG**：信号编码器 + 检索模块 + GPT-2 解码器

### 流程
1. **预训练**：在大规模信号数据上预训练 LLM
2. **微调**：在特定调制数据集上微调
3. **评估**：少样本（few-shot）评估

### 使用方式
```bash
cd 第三章
pip install -r requirements.txt

# 预训练
python main.py --mode pretrain --config config/pretrain_radiollm.yaml

# 微调
python main.py --mode finetune --config config/finetune_radiollm.yaml

# 评估
python main.py --mode evaluate --config config/evaluate_radiollm.yaml
```

---

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+（推荐 GPU 训练）
- 各章节详细依赖见对应目录下的 `requirements.txt`

## 引用

如果您使用了本项目的代码，请考虑引用相关论文（待补充）。

## 许可证

本项目仅供学术研究使用。
