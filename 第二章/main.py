#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
电磁信号识别训练框架 - 主程序入口

使用方式:
    # 预训练
    python main.py --cfg config/pretrain/pretrain_RMLA.yaml

    # 微调 (ACSF_TMAE)
    python main.py --cfg config/fintune/fintune_RMLA.yaml

    # 微调 (对比方法)
    python main.py --model mcldnn --cfg config/fintune/fintune_RMLA.yaml

    # 评估
    python main.py --cfg config/fintune/fintune_RMLA.yaml --evaluate

    # 命令行覆盖参数
    python main.py --cfg config/pretrain/pretrain_RMLA.yaml --dataset RMLB --batchsize 64
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tool.parser import parse_args
from scheme.pretrain_scheme import PretrainScheme
from scheme.fintune_scheme import FintuneScheme
from scheme.evaluate_scheme import EvaluateScheme
from scheme.fintune_comparison_scheme import FintuneComparisonScheme


def fintune(args):
    """
    对比方法微调入口
    支持多种对比方法的分类微调

    支持的模型:
        1D 信号模型: mcldnn, hcgdnn, pet, AWN, ICAMCNET, moco1d, TcssAMR
        自监督模型:  Sgnc2freq, Sgnc2freq_moco, Sgnc2freq_swav, SWT
        2D 图像模型: MAE_ViT_gacf

    使用方式:
        python main.py --model mcldnn --cfg config/fintune/fintune_RMLA.yaml
    """
    print(f"\n{'='*60}")
    print(f"对比方法微调")
    print(f"{'='*60}")
    print(f"模型名称:  {args.model_name}")
    print(f"数据集:    {args.dataset}")
    print(f"配置文件:  {args.cfg}")
    print(f"{'='*60}\n")

    scheme = FintuneComparisonScheme(args)
    scheme.run()


def main():
    """主函数：解析参数并执行对应任务"""

    # 解析完整配置（包括 --cfg, --evaluate 和 YAML 中的参数）
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"电磁信号识别训练框架")
    print(f"{'='*60}")
    print(f"配置文件: {args.cfg}")
    print(f"任务类型: {args.task}")
    print(f"数据集:   {args.dataset}")
    print(f"{'='*60}\n")

    # 如果指定了 --model 参数，则执行对比方法微调
    if hasattr(args, 'model_name') and args.model_name is not None:
        fintune(args)
        return

    # 根据任务类型选择执行方案
    if args.evaluate:
        # 评估模式
        print("运行模式: 评估\n")
        scheme = EvaluateScheme(args)
        scheme.run()
    elif args.task == 'pretraining':
        # 预训练模式
        print("运行模式: 预训练\n")
        scheme = PretrainScheme(args)
        scheme.run()
    elif args.task == 'classification':
        # 微调模式
        print("运行模式: 微调\n")
        scheme = FintuneScheme(args)
        scheme.run()
    else:
        print(f"错误: 未知的任务类型 '{args.task}'")
        print("支持的任务类型: pretraining, classification")
        sys.exit(1)


if __name__ == '__main__':
    main()
