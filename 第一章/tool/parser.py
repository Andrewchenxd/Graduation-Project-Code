"""
工具模块 - 命令行参数解析

提供参数加载功能，从config/parser.py导入。
此文件作为参数加载的入口点。
"""

import sys
import os

# 将项目根目录添加到系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.parser import get_args, str2bool

__all__ = ['get_args', 'str2bool']
