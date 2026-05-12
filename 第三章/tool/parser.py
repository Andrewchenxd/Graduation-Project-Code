"""
配置解析器模块

将 YAML 配置文件解析为类似 argparse.Namespace 的对象，
支持路径模板替换（如 {dataset} 会被替换为实际数据集名称）。
"""

import yaml
import argparse
from typing import Any, Dict, List, Optional


class Config:
    """
    配置类，将字典转换为属性访问方式。
    支持嵌套属性访问，如 config.model_name, config.dataset 等。
    同时兼容 dict 风格访问（__len__, __getitem__, keys, values, items），
    以便与期望 dict 的旧代码无缝协作。
    """

    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """如果属性不存在，返回 None 而不是抛出异常"""
        return None

    def __repr__(self) -> str:
        items = sorted(self.__dict__.items())
        return '\n'.join([f"  {k}: {v}" for k, v in items if not k.startswith('_')])

    # ---- dict 兼容接口 ----

    def __len__(self) -> int:
        """支持 len(config)"""
        return len(self.__dict__)

    def __getitem__(self, key: str) -> Any:
        """支持 config[key] 访问"""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """支持 config[key] = value 赋值"""
        if isinstance(value, dict):
            setattr(self, key, Config(value))
        else:
            setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """支持 key in config"""
        return hasattr(self, key)

    def __iter__(self):
        """支持 for key in config / list(config)"""
        return iter(self.__dict__)

    def keys(self):
        """支持 config.keys()"""
        return self.__dict__.keys()

    def values(self):
        """支持 config.values()"""
        return self.__dict__.values()

    def items(self):
        """支持 config.items()"""
        return self.__dict__.items()

    # ---- 原始接口 ----

    def get(self, key: str, default: Any = None) -> Any:
        """安全获取属性"""
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """递归转换为普通字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def update(self, other: 'Config') -> None:
        """用另一个 Config 对象更新当前配置"""
        for key, value in other.__dict__.items():
            if isinstance(value, Config) and hasattr(self, key) and isinstance(getattr(self, key), Config):
                getattr(self, key).update(value)
            else:
                setattr(self, key, value)


def _convert_numeric_strings(obj: Any) -> Any:
    """
    递归地将字符串形式的数值（如 '1e-5', '42', '3.14'）转换为实际数值类型。
    同时将字符串 'None' 转换为 Python None。
    """
    if isinstance(obj, str):
        # 处理 'None' 字符串
        if obj.lower() == 'none':
            return None
        # 尝试转换为数值
        try:
            # 先尝试 int（如 '42'）
            return int(obj)
        except (ValueError, TypeError):
            try:
                # 再尝试 float（如 '1e-5', '3.14'）
                return float(obj)
            except (ValueError, TypeError):
                return obj
    elif isinstance(obj, dict):
        return {k: _convert_numeric_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numeric_strings(item) for item in obj]
    return obj


def load_config(cfg_path: str) -> Config:
    """
    加载 YAML 配置文件并返回 Config 对象。

    Args:
        cfg_path: YAML 配置文件路径

    Returns:
        Config 对象
    """
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    # 自动转换字符串形式的数值（如 '1e-5' → 0.00001）
    config_dict = _convert_numeric_strings(config_dict)
    return Config(config_dict)


def resolve_path_templates(config: Config) -> Config:
    """
    解析路径模板，将 {dataset} 等占位符替换为实际值。

    例如:
        train_data_path: ./data/fintune_data/{dataset}_train_data.npy
        如果 config.dataset = 'RMLA'，则解析为 ./data/fintune_data/RMLA_train_data.npy

    Args:
        config: 原始 Config 对象

    Returns:
        路径模板解析后的 Config 对象
    """
    config_dict = config.to_dict()

    # 收集所有模板变量
    template_vars = {}
    for key in ['dataset', 'model_name', 'llm_model']:
        val = config_dict.get(key)
        if val is not None:
            template_vars[key] = str(val)

    # 递归替换路径中的模板
    def _resolve(obj):
        if isinstance(obj, str) and '{' in obj:
            try:
                return obj.format(**template_vars)
            except KeyError:
                return obj
        elif isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_resolve(item) for item in obj]
        return obj

    resolved_dict = _resolve(config_dict)
    return Config(resolved_dict)


def str2bool(v) -> bool:
    """字符串转布尔值"""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ('true', 'yes', '1', 'y')
    return bool(v)


def parse_args(cfg_path: str) -> Config:
    """
    主入口：加载配置文件并解析路径模板。

    Args:
        cfg_path: YAML 配置文件路径

    Returns:
        解析后的 Config 对象
    """
    config = load_config(cfg_path)
    config = resolve_path_templates(config)
    return config


def get_dataset_config(dataset_config_path: str, dataset_name: str) -> Optional[Config]:
    """
    从数据集配置文件中获取指定数据集的配置。

    Args:
        dataset_config_path: 数据集 YAML 配置文件路径
        dataset_name: 数据集名称 (如 'RML2016', 'RML2018')

    Returns:
        数据集配置 Config 对象，如果未找到则返回 None
    """
    try:
        with open(dataset_config_path, 'r', encoding='utf-8') as f:
            all_datasets = yaml.safe_load(f)
        if dataset_name in all_datasets:
            return Config(all_datasets[dataset_name])
        return None
    except FileNotFoundError:
        print(f"Warning: Dataset config file not found: {dataset_config_path}")
        return None
