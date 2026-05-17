"""
统一入口文件

使用方法:
    # 预训练
    python main.py --cfg config/pretrain_radiollm.yaml
    python main.py --cfg config/pretrain_radiollm_rag.yaml

    # 微调
    python main.py --cfg config/finetune_radiollm.yaml
    python main.py --cfg config/finetune_radiollm_rag.yaml

    # 评估
    python main.py --cfg config/evaluate_radiollm.yaml
    python main.py --cfg config/evaluate_radiollm_rag.yaml
"""

import argparse
import warnings
import sys
import os

# 将项目根目录添加到 sys.path，确保模块导入正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='RadioLLM Unified Entry Point')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--override', type=str, nargs='+', default=None,
                        help='Override config parameters, e.g. --override model_name=RadioLLM_RAG dataset=RMLB')

    args, _ = parser.parse_known_args()

    # 加载配置
    from tool.parser import parse_args, load_config, resolve_path_templates
    config = parse_args(args.cfg)

    # 命令行覆盖配置参数
    if args.override:
        for override_item in args.override:
            if '=' in override_item:
                key, value = override_item.split('=', 1)
                # 尝试转换为合适的类型
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                setattr(config, key, value)
        # 重新从 YAML 加载原始配置并重新解析路径模板
        raw_config = load_config(args.cfg)
        for k, v in config.__dict__.items():
            if k not in ('test_data_path', 'train_data_path', 'model_path', 'result_dir'):
                setattr(raw_config, k, v)
        config = resolve_path_templates(raw_config)

    # 打印配置信息
    print(f"{'=' * 60}")
    print(f"RadioLLM - Configuration: {args.cfg}")
    print(f"{'=' * 60}")
    print(f"Task: {config.task_name}")
    print(f"Model: {config.model_name}")
    print(f"Mode: {getattr(config, 'mode', 'train')}")
    print(f"{'=' * 60}\n")

    # 根据 task_name 和 mode 决定执行策略
    task_name = config.task_name
    mode = getattr(config, 'mode', 'train')

    # task_name=soft_hard_prompt3 表示预训练模式
    if task_name == 'soft_hard_prompt3' and mode != 'evaluate':
        from scheme.pretrain import run_pretrain
        run_pretrain(config)

    elif task_name == 'classification' and mode == 'evaluate':
        from scheme.evaluate import run_evaluate
        run_evaluate(config)

    elif task_name == 'classification':
        from scheme.finetune import run_finetune
        run_finetune(config)

    else:
        print(f"Unknown task/mode combination: task_name={task_name}, mode={mode}")
        print("Available options:")
        print("  Pretrain:  task_name=soft_hard_prompt3, mode=train (default)")
        print("  Finetune:  task_name=classification, mode=train (default)")
        print("  Evaluate:  task_name=classification, mode=evaluate")
        sys.exit(1)


if __name__ == '__main__':
    main()
