"""
评估模型在测试集上的准确率

使用方式:
    # 使用配置文件（推荐）
    python evaluate.py --cfg config/replay/icarl/icarl_rml2016a.yaml

    # 使用命令行参数
    python evaluate.py --checkpoint ./checkpoint_TSFFN/RMLA/PWVD/pwvd_best_network_acc_best.pth
                       --dataset RMLA
                       --classesnum 11
                       --batchsize 64

    # 评估ADSB数据集
    python evaluate.py --checkpoint ./checkpoint_TSFFN/ADSB/PWVD/pwvd_best_network_acc_best.pth
                       --dataset ADSB
                       --classesnum 198
                       --batchsize 32
                       --adsbis True
                       --resample True
                       --samplenum 10
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse

from config.parser import get_args, str2bool
from data.rml2016a import SigDataSet_pwvd
from model.HydraAttention_cutmix_dropout_RMLdrop import TSFFN


def evaluate_model(model, data_loader, device, classesnum=11):
    """
    评估模型在数据集上的准确率
    返回: 总体准确率, 每个类别的准确率, 每个信噪比的准确率
    """
    model.eval()
    correct = 0
    total = 0

    # 每个类别的统计
    class_correct = np.zeros(classesnum)
    class_total = np.zeros(classesnum)

    # 每个信噪比的统计
    snr_correct = {}
    snr_total = {}

    with torch.no_grad():
        for input1, input2, input3, target, snr in tqdm(data_loader, desc='Evaluating'):
            images, sgn, te, labels = input1, input2, input3, target
            output = model(images.to(device), sgn.to(device), te.to(device))
            preds = torch.max(output, dim=1)[1].cpu().numpy()
            labels_np = labels.numpy()
            snr_np = snr.numpy()

            for i in range(len(labels)):
                label = labels_np[i]
                pred = preds[i]
                s = snr_np[i]

                # 总体统计
                if pred == label:
                    correct += 1
                total += 1

                # 类别统计
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

                # 信噪比统计
                if s not in snr_correct:
                    snr_correct[s] = 0
                    snr_total[s] = 0
                snr_total[s] += 1
                if pred == label:
                    snr_correct[s] += 1

    # 计算准确率
    overall_acc = correct / total

    # 类别准确率
    class_acc = {}
    for c in range(classesnum):
        if class_total[c] > 0:
            class_acc[c] = class_correct[c] / class_total[c]

    # 信噪比准确率
    snr_acc = {}
    for s in sorted(snr_correct.keys()):
        snr_acc[s] = snr_correct[s] / snr_total[s]

    return overall_acc, class_acc, snr_acc


def build_eval_parser():
    """构建评估专用的命令行参数解析器"""
    parser = argparse.ArgumentParser(description='评估模型准确率')

    # 支持从配置文件加载
    parser.add_argument("--cfg", type=str, default=None,
                        help="YAML配置文件路径")

    # 模型参数
    parser.add_argument("--checkpoint", type=str,
                        default='./checkpoint_TSFFN/RMLA/PWVD/pwvd_best_network_acc_best.pth',
                        help="模型权重路径")
    parser.add_argument("--classesnum", type=int, default=11,
                        help="分类类别数 (ADSB: 198, RMLA: 11, RMLB: 10, RMLC: 11)")
    parser.add_argument("--netdepth", type=int, default=64,
                        help="网络特征维度")
    parser.add_argument("--cutmixsize", type=int, default=4,
                        help="CutMix大小")

    # 数据参数
    parser.add_argument("--dataset", type=str, default='RMLA',
                        help="数据集名称 (RMLA, RMLB, RMLC, ADSB)")
    parser.add_argument("--data_dir", type=str, default='./data',
                        help="数据文件目录")
    parser.add_argument("--batchsize", type=int, default=64,
                        help="批次大小 (ADSB: 32, 其他: 64)")
    parser.add_argument("--adsbis", type=str2bool, default=False,
                        help="是否为ADSB数据集")
    parser.add_argument("--resample", type=str2bool, default=False,
                        help="是否重采样")
    parser.add_argument("--samplenum", type=int, default=10,
                        help="重采样点数")
    parser.add_argument("--numworks", type=int, default=4,
                        help="数据加载工作进程数")

    return parser


def main():
    """评估主函数"""
    # 1. 解析参数
    parser = build_eval_parser()
    args, _ = parser.parse_known_args()

    # 如果指定了配置文件，从配置文件加载
    if args.cfg is not None:
        import os
        import yaml
        if os.path.exists(args.cfg):
            with open(args.cfg, 'r', encoding='utf-8') as f:
                cfg_data = yaml.safe_load(f)
            parser.set_defaults(**cfg_data)
        else:
            raise FileNotFoundError(f"配置文件未找到: {args.cfg}")

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"评估数据集: {args.dataset}")
    print(f"类别数: {args.classesnum}")

    # 2. 加载模型
    print("\n加载模型...")
    model = TSFFN(args.classesnum, args.netdepth, args.cutmixsize,
                  in_features=2368, num_block_lists=[1, 1, 2, 2])
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU")
        model = nn.DataParallel(model)
    model = model.to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"模型加载完成: {args.checkpoint}")

    # 3. 加载数据集
    data_path = f'{args.data_dir}/{args.dataset}_test_data'
    print(f"\n加载测试集: {data_path}")
    test_set = SigDataSet_pwvd(
        data_path,
        adsbis=args.adsbis,
        resample_is=args.resample,
        samplenum=args.samplenum
    )
    print(f"测试集大小: {len(test_set)}")

    # 4. 创建数据加载器
    test_loader = DataLoader(
        test_set, batch_size=args.batchsize,
        shuffle=False, num_workers=args.numworks
    )

    # 5. 评估
    print("\n=== 开始评估 ===")
    overall_acc, class_acc, snr_acc = evaluate_model(
        model, test_loader, device, args.classesnum
    )

    # 6. 输出结果
    print("\n" + "=" * 50)
    print(f"总体准确率: {overall_acc * 100:.2f}%")
    print("=" * 50)

    if class_acc:
        print("\n各类别准确率:")
        for c in sorted(class_acc.keys()):
            print(f"  类别 {c}: {class_acc[c] * 100:.2f}%")


    return overall_acc, class_acc, snr_acc


if __name__ == '__main__':
    main()
