# 信号分类训练项目

基于深度学习的电磁信号分类项目，支持多种数据集和多种深度学习模型。

## 项目架构

项目采用模块化设计，分为配置管理、数据处理、模型定义、训练方案和工具函数五个主要部分。

```
├── config/                    # 配置管理模块
│   ├── parser.py              # 命令行参数解析器
│   ├── replay/                # 各模型配置文件
│   │   ├── tsffn/             # TSFFN 模型配置
│   │   ├── cnn2/              # CNN2 模型配置
│   │   ├── dae/               # DAE 模型配置
│   │   ├── lstm/              # LSTM 模型配置
│   │   ├── flan/              # FLAN 模型配置
│   │   ├── icamcnet/          # ICAMCNET 模型配置
│   │   └── petcgdnn/          # PETCGDNN 模型配置
│   └── __init__.py
│
├── data/                      # 数据模块
│   ├── rml2016a.py            # RML2016a数据集加载和预处理
│   ├── data1/                 # 实验数据目录
│   │   ├── train_X.npy
│   │   ├── train_Y.npy
│   │   ├── val_X.npy
│   │   ├── val_Y.npy
│   │   ├── test_X.npy
│   │   └── test_Y.npy
│   └── __init__.py
│
├── model/                     # 模型定义模块
│   ├── HydraAttention_cutmix_dropout_RMLdrop.py  # TSFFN模型
│   ├── CNN2.py                # CNN2 模型
│   ├── DAE.py                 # DAE 模型（深度自编码器）
│   ├── lstm.py                # LSTM 模型
│   ├── FLAN.py                # FLAN 模型
│   ├── ICAMCNET.py            # ICAMCNET 模型
│   ├── PETCGDNN.py            # PETCGDNN 模型
│   ├── HCGDNN.py              # HCGDNN 模型
│   ├── resnet1d.py
│   ├── resnet2d.py
│   ├── sknet1d.py
│   ├── sknet2d.py
│   ├── vit.py
│   └── ...                    # 其他模型定义
│
├── scheme/                    # 训练方案模块
│   ├── base.py                # 基础训练方案类（train/evaluate）
│   └── __init__.py
│
├── tool/                      # 工具函数模块
│   ├── parser.py              # 参数加载入口
│   ├── rml2016a.py            # 数据集加载入口
│   ├── utils.py               # 通用工具函数（日志、模型保存、早停等）
│   ├── signeltoimage.py       # 信号转时频图像
│   ├── tezhengtiqu.py         # 特征提取
│   ├── chazhi.py              # 插值函数
│   ├── loss.py                # 损失函数
│   ├── opter.py               # 优化器
│   └── ...                    # 其他工具函数
│
├── main.py                    # 主程序入口（训练）
├── evaluate.py                # 评估脚本
├── README.md                  # 项目说明文档
├── checkpoint_TSFFN/          # TSFFN 模型检查点目录
├── checkpoint_CNN2/           # CNN2 模型检查点目录
├── checkpoint_DAE/            # DAE 模型检查点目录
├── checkpoint_LSTM/           # LSTM 模型检查点目录
├── checkpoint_FLAN/           # FLAN 模型检查点目录
├── checkpoint_ICAMCNET/       # ICAMCNET 模型检查点目录
├── checkpoint_PETCGDNN/       # PETCGDNN 模型检查点目录
└── checkpoint_HCGDNN/         # HCGDNN 模型检查点目录
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- tqdm
- PyYAML
- tftb（时频分析工具包）
- opencv-python

### 安装依赖

```bash
pip install torch numpy scipy tqdm pyyaml tftb opencv-python matplotlib
```

## 数据集说明

### 支持的数据集

| 数据集 | 类别数 | 说明 |
|--------|--------|------|
| RMLA (RML2016.10a) | 11 | 公开调制识别数据集 |
| RMLB (RML2016.10b) | 10 | 公开调制识别数据集 |
| RMLC (RML2016.10c) | 11 | 公开调制识别数据集 |
| ADSB | 198 | 广播式自动相关监视信号 |

### 数据格式

数据以 `.npy` 格式存储，每个数据集包含以下文件：

- `{dataset}_train_data.npy` - 训练集特征
- `{dataset}_train_label.npy` - 训练集标签
- `{dataset}_train_snr.npy` - 训练集信噪比（RML系列）
- `{dataset}_val_data.npy` - 验证集特征
- `{dataset}_val_label.npy` - 验证集标签
- `{dataset}_val_snr.npy` - 验证集信噪比（RML系列）
- `{dataset}_test_data.npy` - 测试集特征
- `{dataset}_test_label.npy` - 测试集标签
- `{dataset}_test_snr.npy` - 测试集信噪比（RML系列）

## 支持的模型

| 模型 | 文件 | 说明 |
|------|------|------|
| TSFFN | `model/HydraAttention_cutmix_dropout_RMLdrop.py` | Transformer信号特征融合网络（默认） |
| CNN2 | `model/CNN2.py` | 卷积神经网络 |
| DAE | `model/DAE.py` | 深度自编码器 |
| LSTM | `model/lstm.py` | 长短期记忆网络 |
| FLAN | `model/FLAN.py` | 频域注意力网络 |
| ICAMCNET | `model/ICAMCNET.py` | 改进卷积注意力网络 |
| PETCGDNN | `model/PETCGDNN.py` | 并行卷积GRU网络 |
| HCGDNN | `model/HCGDNN.py` | 混合卷积GRU网络 |

所有模型均通过 `--model` 参数选择，通过 `main.py` 中的 `build_model()` 函数动态构建。

## 训练方案

训练方案位于 `scheme/` 目录，当前实现了基础训练方案 `BaseScheme`，支持早停、学习率衰减等功能。

训练方案兼容所有模型的输出格式：
- **单输出模型**（CNN2, LSTM, FLAN, ICAMCNET, PETCGDNN）：直接使用分类输出
- **DAE**：自动取 `(reconstructed, classification_output)` 中的分类输出
- **HCGDNN**：自动取三个分类输出的平均值

## 训练

### 使用配置文件（推荐）

每个模型在 `config/replay/` 下有独立的配置目录，包含4个数据集的配置。

#### TSFFN

```bash
# RML2016.10a
python main.py --cfg config/replay/tsffn/tsffn_rml2016a.yaml

# RML2016.10b
python main.py --cfg config/replay/tsffn/tsffn_rmlb.yaml

# RML2016.10c
python main.py --cfg config/replay/tsffn/tsffn_rmlc.yaml

# ADSB
python main.py --cfg config/replay/tsffn/tsffn_adsb.yaml
```

#### CNN2

```bash
python main.py --cfg config/replay/cnn2/cnn2_rml2016a.yaml
python main.py --cfg config/replay/cnn2/cnn2_rmlb.yaml
python main.py --cfg config/replay/cnn2/cnn2_rmlc.yaml
python main.py --cfg config/replay/cnn2/cnn2_adsb.yaml
```

#### DAE

```bash
python main.py --cfg config/replay/dae/dae_rml2016a.yaml
python main.py --cfg config/replay/dae/dae_rmlb.yaml
python main.py --cfg config/replay/dae/dae_rmlc.yaml
python main.py --cfg config/replay/dae/dae_adsb.yaml
```

#### LSTM

```bash
python main.py --cfg config/replay/lstm/lstm_rml2016a.yaml
python main.py --cfg config/replay/lstm/lstm_rmlb.yaml
python main.py --cfg config/replay/lstm/lstm_rmlc.yaml
python main.py --cfg config/replay/lstm/lstm_adsb.yaml
```

#### FLAN

```bash
python main.py --cfg config/replay/flan/flan_rml2016a.yaml
python main.py --cfg config/replay/flan/flan_rmlb.yaml
python main.py --cfg config/replay/flan/flan_rmlc.yaml
python main.py --cfg config/replay/flan/flan_adsb.yaml
```

#### ICAMCNET

```bash
python main.py --cfg config/replay/icamcnet/icamcnet_rml2016a.yaml
python main.py --cfg config/replay/icamcnet/icamcnet_rmlb.yaml
python main.py --cfg config/replay/icamcnet/icamcnet_rmlc.yaml
python main.py --cfg config/replay/icamcnet/icamcnet_adsb.yaml
```

#### PETCGDNN

```bash
python main.py --cfg config/replay/petcgdnn/petcgdnn_rml2016a.yaml
python main.py --cfg config/replay/petcgdnn/petcgdnn_rmlb.yaml
python main.py --cfg config/replay/petcgdnn/petcgdnn_rmlc.yaml
python main.py --cfg config/replay/petcgdnn/petcgdnn_adsb.yaml
```

### 使用命令行参数（不依赖配置文件）

```bash
# CNN2
python main.py --model CNN2 --dataset RMLA --classesnum 11 --batchsize 64 --adsbis false --resample false

# DAE
python main.py --model DAE --dataset RMLA --classesnum 11 --batchsize 64 --adsbis false --resample false

# LSTM
python main.py --model LSTM --dataset RMLA --classesnum 11 --batchsize 64 --adsbis false --resample false

# FLAN
python main.py --model FLAN --dataset RMLA --classesnum 11 --batchsize 64 --adsbis false --resample false

# ICAMCNET
python main.py --model ICAMCNET --dataset RMLA --classesnum 11 --batchsize 64 --adsbis false --resample false

# PETCGDNN
python main.py --model PETCGDNN --dataset RMLA --classesnum 11 --batchsize 64 --adsbis false --resample false

# ADSB 数据集示例
python main.py --model CNN2 --dataset ADSB --classesnum 198 --batchsize 32 --adsbis true --resample true --samplenum 10
```

## 评估

### 一键批量评估

提供一键脚本，自动顺序评估所有方法在全部数据集上的表现。

#### Windows

```batch
batch_eval_all.bat
```

#### Linux / macOS

```bash
chmod +x batch_eval_all.sh
./batch_eval_all.sh
```

### TSFFN

```bash
# RML2016.10a
python evaluate.py --cfg config/replay/tsffn/tsffn_rml2016a.yaml
python evaluate.py --checkpoint ./checkpoint_TSFFN/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64

# RML2016.10b
python evaluate.py --cfg config/replay/tsffn/tsffn_rmlb.yaml
python evaluate.py --checkpoint ./checkpoint_TSFFN/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64

# RML2016.10c
python evaluate.py --cfg config/replay/tsffn/tsffn_rmlc.yaml
python evaluate.py --checkpoint ./checkpoint_TSFFN/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64

# ADSB
python evaluate.py --cfg config/replay/tsffn/tsffn_adsb.yaml
python evaluate.py --checkpoint ./checkpoint_TSFFN/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10
```

### MCLDNN

```bash
python evaluate.py --checkpoint ./checkpoint_MCLDNN/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_MCLDNN/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_MCLDNN/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_MCLDNN/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10
```

### CNN2

```bash
python evaluate.py --checkpoint ./checkpoint_CNN2/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_CNN2/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_CNN2/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_CNN2/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10
```

### DAE

```bash
python evaluate.py --checkpoint ./checkpoint_DAE/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_DAE/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_DAE/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_DAE/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10
```

### ResNet2d

```bash
python evaluate.py --checkpoint ./checkpoint_ResNet2d/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_ResNet2d/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_ResNet2d/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_ResNet2d/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10
```

### ResNet1d

```bash
python evaluate.py --checkpoint ./checkpoint_ResNet1d/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_ResNet1d/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_ResNet1d/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_ResNet1d/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10
```

### PETCGDNN

```bash
python evaluate.py --checkpoint ./checkpoint_PETCGDNN/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_PETCGDNN/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_PETCGDNN/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_PETCGDNN/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10
```

### ICAMCNET

```bash
python evaluate.py --checkpoint ./checkpoint_ICAMCNET/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_ICAMCNET/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_ICAMCNET/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_ICAMCNET/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10
```

### HCGDNN

```bash
python evaluate.py --checkpoint ./checkpoint_HCGDNN/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_HCGDNN/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_HCGDNN/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_HCGDNN/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10
```

### FLAN

```bash
python evaluate.py --checkpoint ./checkpoint_FLAN/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_FLAN/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_FLAN/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_FLAN/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10
```

### MMNet

```bash
python evaluate.py --checkpoint ./checkpoint_MMNet/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_MMNet/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_MMNet/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_MMNet/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10
```

### AWN

```bash
python evaluate.py --checkpoint ./checkpoint_AWN/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_AWN/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_AWN/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64
python evaluate.py --checkpoint ./checkpoint_AWN/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10
```



## 配置文件说明

所有超参数和数据集路径都在配置文件中管理，无需修改代码。

### 配置文件列表

| 配置文件 | 模型 | 数据集 | 类别数 | batchsize |
|----------|------|--------|--------|-----------|
| `config/replay/tsffn/tsffn_rml2016a.yaml` | TSFFN | RML2016.10a | 11 | 64 |
| `config/replay/tsffn/tsffn_rmlb.yaml` | TSFFN | RML2016.10b | 10 | 64 |
| `config/replay/tsffn/tsffn_rmlc.yaml` | TSFFN | RML2016.10c | 11 | 64 |
| `config/replay/tsffn/tsffn_adsb.yaml` | TSFFN | ADSB | 198 | 32 |
| `config/replay/cnn2/cnn2_rml2016a.yaml` | CNN2 | RML2016.10a | 11 | 64 |
| `config/replay/cnn2/cnn2_rmlb.yaml` | CNN2 | RML2016.10b | 10 | 64 |
| `config/replay/cnn2/cnn2_rmlc.yaml` | CNN2 | RML2016.10c | 11 | 64 |
| `config/replay/cnn2/cnn2_adsb.yaml` | CNN2 | ADSB | 198 | 32 |
| `config/replay/dae/dae_rml2016a.yaml` | DAE | RML2016.10a | 11 | 64 |
| `config/replay/dae/dae_rmlb.yaml` | DAE | RML2016.10b | 10 | 64 |
| `config/replay/dae/dae_rmlc.yaml` | DAE | RML2016.10c | 11 | 64 |
| `config/replay/dae/dae_adsb.yaml` | DAE | ADSB | 198 | 32 |
| `config/replay/lstm/lstm_rml2016a.yaml` | LSTM | RML2016.10a | 11 | 64 |
| `config/replay/lstm/lstm_rmlb.yaml` | LSTM | RML2016.10b | 10 | 64 |
| `config/replay/lstm/lstm_rmlc.yaml` | LSTM | RML2016.10c | 11 | 64 |
| `config/replay/lstm/lstm_adsb.yaml` | LSTM | ADSB | 198 | 32 |
| `config/replay/flan/flan_rml2016a.yaml` | FLAN | RML2016.10a | 11 | 64 |
| `config/replay/flan/flan_rmlb.yaml` | FLAN | RML2016.10b | 10 | 64 |
| `config/replay/flan/flan_rmlc.yaml` | FLAN | RML2016.10c | 11 | 64 |
| `config/replay/flan/flan_adsb.yaml` | FLAN | ADSB | 198 | 32 |
| `config/replay/icamcnet/icamcnet_rml2016a.yaml` | ICAMCNET | RML2016.10a | 11 | 64 |
| `config/replay/icamcnet/icamcnet_rmlb.yaml` | ICAMCNET | RML2016.10b | 10 | 64 |
| `config/replay/icamcnet/icamcnet_rmlc.yaml` | ICAMCNET | RML2016.10c | 11 | 64 |
| `config/replay/icamcnet/icamcnet_adsb.yaml` | ICAMCNET | ADSB | 198 | 32 |
| `config/replay/petcgdnn/petcgdnn_rml2016a.yaml` | PETCGDNN | RML2016.10a | 11 | 64 |
| `config/replay/petcgdnn/petcgdnn_rmlb.yaml` | PETCGDNN | RML2016.10b | 10 | 64 |
| `config/replay/petcgdnn/petcgdnn_rmlc.yaml` | PETCGDNN | RML2016.10c | 11 | 64 |
| `config/replay/petcgdnn/petcgdnn_adsb.yaml` | PETCGDNN | ADSB | 198 | 32 |

### 修改参数

打开对应数据集的 YAML 配置文件，修改参数值即可：

```yaml
# config/replay/cnn2/cnn2_rml2016a.yaml
model: CNN2              # 模型名称
batchsize: 64            # 批次大小 (ADSB: 32, 其他: 64)
lr: 0.0005               # 学习率
epochs: 200              # 训练轮数
dataset: RMLA            # 数据集名称
adsbis: false            # 是否为ADSB数据集
```

## 命令行参数说明

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型名称 (TSFFN, CNN2, DAE, LSTM, FLAN, ICAMCNET, PETCGDNN, HCGDNN) | TSFFN |
| `--cfg` | YAML配置文件路径 | None |
| `--batchsize` | 批次大小 | 32 |
| `--lr` | 学习率 | 0.0005 |
| `--epochs` | 训练轮数 | 200 |
| `--classesnum` | 类别数 | 198 |
| `--dataset` | 数据集名称 | ADSB |
| `--adsbis` | 是否ADSB数据集 | True |
| `--resample` | 是否重采样 | True |
| `--samplenum` | 重采样点数 | 10 |
| `--data_dir` | 数据目录 | ./data |
| `--numworks` | 工作进程数 | 4 |

### 命令行覆盖参数

命令行参数优先级高于配置文件：

```bash
# 使用配置文件，但覆盖模型、batchsize和lr
python main.py --cfg config/replay/tsffn/tsffn_rml2016a.yaml --model CNN2 --batchsize 32 --lr 0.001

# 评估时覆盖参数
python evaluate.py --cfg config/replay/tsffn/tsffn_rml2016a.yaml --batchsize 128
```

### 评估参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cfg` | YAML配置文件路径 | None |
| `--checkpoint` | 模型权重路径 | `./checkpoint_TSFFN/RMLA/PWVD/pwvd_best_network_acc_best.pth` |
| `--dataset` | 数据集名称 | RMLA |
| `--classesnum` | 类别数 | 11 |
| `--batchsize` | 批次大小 | 64 |
| `--adsbis` | 是否ADSB数据集 | False |
| `--resample` | 是否重采样 | False |
| `--samplenum` | 重采样点数 | 10 |
| `--netdepth` | 网络特征维度 | 64 |
| `--cutmixsize` | CutMix大小 | 4 |
| `--data_dir` | 数据目录 | ./data |
| `--numworks` | 工作进程数 | 4 |
