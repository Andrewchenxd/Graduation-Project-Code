"""
主程序入口

协调整个训练流程，包括数据加载、模型构建、训练和评估。

使用方式:
    python main.py --cfg config/replay/tsffn/tsffn_rml2016a.yaml
    python main.py --model CNN2 --dataset RMLA --classesnum 11 --batchsize 64
    python main.py --model DAE --dataset RMLA --classesnum 11 --batchsize 64
    python main.py --model LSTM --dataset RMLA --classesnum 11 --batchsize 64
    python main.py --model FLAN --dataset RMLA --classesnum 11 --batchsize 64
    python main.py --model ICAMCNET --dataset RMLA --classesnum 11 --batchsize 64
    python main.py --model PETCGDNN --dataset RMLA --classesnum 11 --batchsize 64

参数修改:
    直接在config文件夹中修改相应的YAML配置文件即可，
    无需修改代码中的参数。
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from config.parser import get_args
from data.rml2016a import SigDataSet_pwvd
from scheme.base import BaseScheme


def build_model(args):
    """
    构建神经网络模型

    Args:
        args: 配置参数

    Returns:
        model: 构建好的模型
    """
    model_name = args.model.upper()

    if model_name == 'TSFFN':
        from model.HydraAttention_cutmix_dropout_RMLdrop import TSFFN
        model = TSFFN(
            args.classesnum,
            args.netdepth,
            args.cutmixsize,
            in_features=2368,
            num_block_lists=[1, 1, 2, 2]
        )
    elif model_name == 'CNN2':
        from model.CNN2 import CNN2
        model = CNN2(args.classesnum)
    elif model_name == 'DAE':
        from model.DAE import DAE
        model = DAE(args.classesnum)
    elif model_name == 'LSTM':
        from model.lstm import lstm2
        model = lstm2(output_size=128, numclass=args.classesnum)
    elif model_name == 'FLAN':
        from model.FLAN import FLAN
        model = FLAN(args.classesnum, num_block_lists=[3, 4, 6, 3])
    elif model_name == 'ICAMCNET':
        from model.ICAMCNET import ICAMCNET
        model = ICAMCNET(args.classesnum)
    elif model_name == 'PETCGDNN':
        from model.PETCGDNN import PETCGDNN
        model = PETCGDNN(args.classesnum)
    elif model_name == 'HCGDNN':
        from model.HCGDNN import HCGDNN
        model = HCGDNN(args.classesnum)
    else:
        raise ValueError(f"未知模型: {args.model}，支持的模型: TSFFN, CNN2, DAE, LSTM, FLAN, ICAMCNET, PETCGDNN, HCGDNN")

    print("use dataset: {}".format(args.dataset))
    print("use model: {}".format(model_name))

    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()

    return model


def load_pretrained(model, checkpoint_path, device):
    """
    加载预训练权重

    Args:
        model: 模型实例
        checkpoint_path: 权重文件路径
        device: 设备

    Returns:
        model: 加载权重后的模型
    """
    model_dict = model.state_dict()
    pretrained_dict = torch.load(checkpoint_path, map_location=device)
    load_key, no_load_key, temp_dict = [], [], {}

    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)

    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print('load success')
    return model


def build_dataloaders(args):
    """
    构建数据加载器

    Args:
        args: 配置参数

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    data_dir = args.data_dir

    if not args.adsbis:
        # RML系列数据集：使用预划分的训练/验证/测试集
        train_set = SigDataSet_pwvd(
            '{}/{}_train_data'.format(data_dir, args.dataset),
            adsbis=args.adsbis,
            resample_is=args.resample,
            samplenum=args.samplenum,
            aug_is=True
        )
        test_set = SigDataSet_pwvd(
            '{}/{}_test_data'.format(data_dir, args.dataset),
            adsbis=args.adsbis,
            resample_is=args.resample,
            samplenum=args.samplenum
        )
        val_set = SigDataSet_pwvd(
            '{}/{}_val_data'.format(data_dir, args.dataset),
            adsbis=args.adsbis,
            resample_is=args.resample,
            samplenum=args.samplenum
        )
    else:
        # ADSB数据集：从训练集中划分验证集
        train_set = SigDataSet_pwvd(
            '{}/{}_train_data'.format(data_dir, args.dataset),
            adsbis=args.adsbis,
            resample_is=args.resample,
            samplenum=args.samplenum,
            aug_is=True
        )
        test_set = SigDataSet_pwvd(
            '{}/{}_test_data'.format(data_dir, args.dataset),
            adsbis=args.adsbis,
            resample_is=args.resample,
            samplenum=args.samplenum
        )

        split_ratio = 0.2
        train_set, val_set = torch.utils.data.random_split(
            train_set,
            [int(len(train_set) * (1 - split_ratio)),
             int(len(train_set) * split_ratio)]
        )

    train_loader = DataLoader(
        train_set, batch_size=args.batchsize, shuffle=True,
        num_workers=args.numworks, prefetch_factor=args.pref, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batchsize,
        num_workers=args.numworks, prefetch_factor=args.pref,
        shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batchsize,
        num_workers=args.numworks, prefetch_factor=args.pref,
        shuffle=True, drop_last=True
    )

    return train_loader, val_loader, test_loader


def main():
    """主函数：协调训练流程"""
    # 1. 加载配置
    args = get_args()
    print("=" * 50)
    print("配置参数:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 50)

    # 2. 构建模型
    model = build_model(args)

    # 3. 加载预训练权重（可选）
    # checkpoint_path = './checkpoint_TSFFN/ADSB/PWVD/pwvd_best_network_acc_best.pth'
    # model = load_pretrained(model, checkpoint_path, torch.device('cuda:0'))

    # 4. 构建数据加载器
    train_loader, val_loader, test_loader = build_dataloaders(args)

    # 5. 创建训练方案并执行训练
    scheme = BaseScheme(model, args)
    scheme.train(train_loader, val_loader)

    # 6. 评估模型
    print("\n" + "=" * 50)
    print("开始评估模型...")
    overall_acc, class_acc, snr_acc = scheme.evaluate(test_loader)

    print("\n" + "=" * 50)
    print(f"测试集总体准确率: {overall_acc * 100:.2f}%")
    print("=" * 50)

    print("\n各类别准确率:")
    for c in sorted(class_acc.keys()):
        print(f"  类别 {c}: {class_acc[c] * 100:.2f}%")

    print("\n各信噪比准确率:")
    for s in sorted(snr_acc.keys()):
        print(f"  SNR {s}: {snr_acc[s] * 100:.2f}%")

    print("\n训练完成!")


if __name__ == '__main__':
    main()
