"""数据归一化函数 - GPU加速版本"""
import sys
sys.path.append('..')
from gpu_utils import xp, to_gpu, to_cpu
import numpy as np


def normalize_data(X):
    """
    对数据进行归一化处理（按列归一化）（GPU加速）
    
    参数:
        X: 数组，形状为 (nFea, nSmp)（numpy或cupy数组）
    
    返回:
        ProcessData: 归一化后的数据（与输入相同类型）
    """
    # 检测输入类型
    is_numpy = isinstance(X, np.ndarray)
    
    # 如果是numpy数组，保持在CPU上处理
    if is_numpy:
        X_proc = np.asarray(X)
        norms = np.linalg.norm(X_proc, axis=0)
        norms = np.maximum(norms, 1e-12)  # 避免除以零
        ProcessData = X_proc / norms[np.newaxis, :]
    else:
        # CuPy数组，在GPU上处理
        norms = xp.linalg.norm(X, axis=0)
        norms = xp.maximum(norms, 1e-12)  # 避免除以零
        ProcessData = X / norms[xp.newaxis, :]
    
    return ProcessData

