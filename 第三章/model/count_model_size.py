"""
统计 RadioLLM 模型大小（MB）
只统计 RML2016a+b_total_snr 数据集相关的参数
"""
import sys
import os

# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import argparse
import yaml
import importlib.util


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False


def main():
    parser = argparse.ArgumentParser(description='Time-LLM')
    parser.add_argument('--task_name', type=str, required=False, default='soft_hard_prompt3')
    parser.add_argument("--decoder_is", type=str2bool, default=False)
    parser.add_argument('--prompt_domain', type=str2bool, default=True)
    parser.add_argument('--is_LORA', type=str2bool, default=True)
    parser.add_argument('--numclass', type=int, default=11)
    parser.add_argument('--enc_in', type=int, default=2)
    parser.add_argument('--dec_in', type=int, default=2)
    parser.add_argument('--c_out', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nmb_prototypes', type=int, default=300)
    parser.add_argument('--T', type=float, default=0.07)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--sinkhorn_iterations', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--patch_len', type=int, default=4)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--llm_model', type=str, default='GPT2')
    parser.add_argument('--llm_path', type=str, default=r'D:\planedemo\SignalGPT\pretrain_model\gpt2')
    parser.add_argument('--llm_dim', type=int, default='768')
    parser.add_argument('--ReprogrammingLayer', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--right_prob', type=float, default=0.5)
    parser.add_argument('--K', type=int, default=7)
    parser.add_argument("--decode_mask", type=str2bool, default=True)
    parser.add_argument("--mix", type=str2bool, default=True)
    parser.add_argument("--attn", type=str, default='prob')
    parser.add_argument('--d_ff2', type=int, default=1024)

    args = parser.parse_args()

    # 读取YAML文件
    yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'script', 'multi_task_pretrain.yaml')
    with open(yaml_path, 'r') as file:
        yaml_config = yaml.safe_load(file)

    # 使用 importlib 直接加载模块，避免 model/__init__.py 中 RadioLLM_RAG 的 faiss 依赖
    radiollm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'model', 'radiollm.py')
    spec = importlib.util.spec_from_file_location("radiollm_module", radiollm_path)
    radiollm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(radiollm_module)
    RadioLLM = radiollm_module.RadioLLM

    model = RadioLLM(args, yaml_config=yaml_config)

    dataset_name = 'RML2016a+b_total_snr'

    # ========== 1. 模型总参数量（全部参数） ==========
    total_params = sum(p.numel() for p in model.parameters())
    total_params_MB = total_params * 4 / (1024 ** 2)

    # ========== 2. 可训练参数量 ==========
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params_MB = trainable_params * 4 / (1024 ** 2)

    # ========== 3. 冻结参数量 ==========
    frozen_params = total_params - trainable_params
    frozen_params_MB = frozen_params * 4 / (1024 ** 2)

    ratio = trainable_params / total_params if total_params > 0 else 0

    # ========== 4. 统计 high_freq_extract 中 RML2016a+b_total_snr 的参数量 ==========
    hfe_rml2016_params = 0
    for name, param in model.high_freq_extract[dataset_name].named_parameters():
        hfe_rml2016_params += param.numel()
    hfe_rml2016_MB = hfe_rml2016_params * 4 / (1024 ** 2)

    # ========== 5. 统计各 dataset 相关模块的参数量 ==========
    dataset_params = {}
    for ds_name in model.high_freq_extract.keys():
        ds_params = 0
        # high_freq_extract
        for name, param in model.high_freq_extract[ds_name].named_parameters():
            ds_params += param.numel()
        # patch_embedding
        for name, param in model.patch_embedding[ds_name].named_parameters():
            ds_params += param.numel()
        # output_projection
        for name, param in model.output_projection[ds_name].named_parameters():
            ds_params += param.numel()
        # predictor_ins
        for name, param in model.predictor_ins[ds_name].named_parameters():
            ds_params += param.numel()
        dataset_params[ds_name] = ds_params

    # ========== 打印结果 ==========
    print('=' * 80)
    print(f'  RadioLLM 模型大小统计 (数据集: {dataset_name})')
    print('=' * 80)
    print()
    print(f'  模型总参数量:          {total_params:>15,} 个参数  ({total_params_MB:.2f} MB)')
    print(f'  可训练参数量:          {trainable_params:>15,} 个参数  ({trainable_params_MB:.2f} MB)')
    print(f'  冻结参数量:            {frozen_params:>15,} 个参数  ({frozen_params_MB:.2f} MB)')
    print(f'  可训练参数占比:        {ratio:.4f} ({ratio*100:.2f}%)')
    print()
    print('-' * 80)
    print(f'  high_freq_extract[{dataset_name}] 参数量: {hfe_rml2016_params:>10,}  ({hfe_rml2016_MB:.4f} MB)')
    print('-' * 80)
    print()
    print('  各数据集模块参数量对比:')
    print(f'  {"数据集名":30s} {"参数量":>12s} {"内存(MB)":>10s}')
    print('  ' + '-' * 52)
    for ds_name, pcount in sorted(dataset_params.items(), key=lambda x: x[1], reverse=True):
        mb = pcount * 4 / (1024 ** 2)
        marker = ' <--' if ds_name == dataset_name else ''
        print(f'  {ds_name:30s} {pcount:>12,} {mb:>9.2f} MB{marker}')
    print()
    print('=' * 80)


if __name__ == '__main__':
    main()
