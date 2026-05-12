"""
配置管理模块 - 命令行参数解析器

从YAML配置文件加载参数，支持命令行覆盖。
使用方式: python main.py --cfg config/replay/icarl/icarl_rml2016a.yaml
"""

import argparse
import yaml
import os


def str2bool(v):
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    解析命令行参数，支持从YAML配置文件加载

    优先级: 命令行参数 > YAML配置文件 > 默认值

    使用方式:
        python main.py --cfg config/replay/icarl/icarl_rml2016a.yaml
    """
    parser = argparse.ArgumentParser(description='信号分类训练参数配置')

    # ========== 模型选择 ==========
    parser.add_argument("--model", type=str, default='TSFFN',
                        help="模型名称 (TSFFN, CNN2, DAE, LSTM, FLAN, ICAMCNET, PETCGDNN, HCGDNN 等)")

    # ========== 配置文件路径 ==========
    parser.add_argument("--cfg", type=str, default=None,
                        help="YAML配置文件路径，配置文件中可覆盖以下所有参数")

    # ========== 训练超参数 ==========
    parser.add_argument("--batchsize", type=int, default=32,
                        help="批次大小 (ADSB: 32, 其他: 64)")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="初始学习率")
    parser.add_argument("--epochs", type=int, default=200,
                        help="最大训练轮数")
    parser.add_argument("--classesnum", type=int, default=198,
                        help="分类类别数 (ADSB: 198, RMLA: 11, RMLB: 10, RMLC: 11)")
    parser.add_argument("--netdepth", type=int, default=64,
                        help="网络深度/特征维度")
    parser.add_argument("--cutmixsize", type=int, default=4,
                        help="CutMix大小")
    parser.add_argument("--patience", type=int, default=200,
                        help="早停耐心值")
    parser.add_argument("--wait", type=int, default=5,
                        help="学习率调整等待轮数")
    parser.add_argument("--declay", type=float, default=0.5,
                        help="学习率衰减因子")
    parser.add_argument("--yuzhi", type=int, default=10,
                        help="学习率衰减阈值")
    parser.add_argument("--numworks", type=int, default=4,
                        help="数据加载工作进程数")
    parser.add_argument("--pref", type=int, default=20,
                        help="数据预取因子")

    # ========== 数据相关参数 ==========
    parser.add_argument("--trans_choose", type=str, default='pwvd',
                        help="时频变换方式 (pwvd, stft等)")
    parser.add_argument("--dataset", type=str, default='ADSB',
                        help="数据集名称 (RMLA, RMLB, RMLC, ADSB)")
    parser.add_argument("--withoutis", type=str, default='no')
    parser.add_argument("--adsbis", type=str2bool, default=True,
                        help="是否为ADSB数据集 (ADSB: True, 其他: False)")
    parser.add_argument("--resample", type=str2bool, default=True,
                        help="是否重采样 (ADSB: True, 其他: False)")
    parser.add_argument("--chazhi", type=str2bool, default=False,
                        help="是否进行插值")
    parser.add_argument("--cnum", type=int, default=2,
                        help="插值倍数")
    parser.add_argument("--samplenum", type=int, default=10,
                        help="重采样点数")

    # ========== 路径参数 ==========
    parser.add_argument("--data_dir", type=str, default='./data',
                        help="数据文件目录")
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint_TSFFN',
                        help="模型保存目录")

    # 先解析基本参数（获取cfg路径）
    args, _ = parser.parse_known_args()

    # 如果指定了配置文件，从配置文件加载
    if args.cfg is not None:
        cfg_path = args.cfg
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg_data = yaml.safe_load(f)
            # 将配置文件中的参数设置为默认值
            parser.set_defaults(**cfg_data)
        else:
            raise FileNotFoundError(f"配置文件未找到: {cfg_path}")

    # 重新解析，此时命令行参数会覆盖配置文件中的值
    final_args = parser.parse_args()

    return final_args
