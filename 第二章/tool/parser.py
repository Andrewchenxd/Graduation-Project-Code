"""
命令行参数解析模块
从 YAML 配置文件加载参数，支持命令行覆盖
"""

import yaml
import argparse
import os
from typing import Dict, Any


def load_config(cfg_path: str) -> Dict[str, Any]:
    """
    从 YAML 文件加载配置

    参数:
        cfg_path: YAML 配置文件路径

    返回:
        配置字典
    """
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数，加载 YAML 配置文件

    使用方式:
        python main.py --cfg config/pretrain/pretrain_RMLA.yaml
        python main.py --cfg config/fintune/fintune_RMLA.yaml --evaluate

    返回:
        包含所有配置参数的 Namespace 对象
    """
    parser = argparse.ArgumentParser(description='电磁信号识别训练框架')

    # 配置文件路径（必需参数）
    parser.add_argument('--cfg', type=str, required=True,
                        help='配置文件路径 (YAML)')

    # 评估模式标志
    parser.add_argument('--evaluate', action='store_true',
                        help='评估模式，在测试集上评估模型')

    # 对比方法模型选择
    parser.add_argument('--model', dest='model_name', type=str, default=None,
                        help='对比方法模型名称: mcldnn, hcgdnn, pet, AWN, ICAMCNET, '
                             'moco1d, TcssAMR, Sgnc2freq, Sgnc2freq_moco, '
                             'Sgnc2freq_swav, SWT, MAE_ViT_gacf')

    # 允许命令行覆盖的关键参数
    parser.add_argument('--task', type=str, default=None,
                        help='任务类型: pretraining / classification')
    parser.add_argument('--dataset', type=str, default=None,
                        help='数据集名称: RMLA / RMLB / RMLC')
    parser.add_argument('--batchsize', type=int, default=None,
                        help='批大小')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型检查点路径')

    # 先解析 --cfg 参数
    known_args, _ = parser.parse_known_args()

    if known_args.cfg:
        # 从 YAML 加载配置
        cfg_dict = load_config(known_args.cfg)
    else:
        cfg_dict = {}

    # 将 YAML 中的值设为 argparse 的默认值（仅当命令行未提供时生效）
    for key, value in cfg_dict.items():
        for action in parser._actions:
            if action.dest == key:
                # 参数已注册，更新其默认值
                # 但仅当命令行未显式提供该参数时才使用 YAML 值
                action.default = value
                break
        else:
            # 参数未注册，动态添加
            if isinstance(value, bool):
                parser.add_argument(f'--{key}', type=str2bool, default=value)
            elif isinstance(value, int):
                parser.add_argument(f'--{key}', type=int, default=value)
            elif isinstance(value, float):
                parser.add_argument(f'--{key}', type=float, default=value)
            elif isinstance(value, str):
                parser.add_argument(f'--{key}', type=str, default=value)
            elif isinstance(value, list):
                parser.add_argument(f'--{key}', type=type(value[0]) if value else str,
                                    nargs='+', default=value)

    # 重新解析所有参数（命令行参数会覆盖 YAML 中的值）
    args = parser.parse_args()

    return args


def str2bool(v):
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('布尔值期望: True/False')
