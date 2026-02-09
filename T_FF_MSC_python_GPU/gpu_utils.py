"""GPU工具模块 - 自动检测GPU并提供CPU回退"""
import sys
import warnings

# 尝试导入CuPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ 检测到GPU，使用CuPy进行GPU加速")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    warnings.warn("⚠ 未检测到CuPy，将使用NumPy（CPU模式）运行")

import numpy as np


def get_array_module(x=None):
    """
    获取数组模块（cupy或numpy）
    
    参数:
        x: 可选的数组对象，用于自动检测
    
    返回:
        适当的数组模块（cupy或numpy）
    """
    if GPU_AVAILABLE:
        if x is None:
            return cp
        return cp.get_array_module(x)
    return np


def to_gpu(x):
    """
    将数据转移到GPU
    
    参数:
        x: numpy数组或列表
    
    返回:
        GPU数组（如果可用）或原始numpy数组
    """
    if not GPU_AVAILABLE:
        return x
    
    if isinstance(x, list):
        return [cp.asarray(item) for item in x]
    else:
        return cp.asarray(x)


def to_cpu(x):
    """
    将数据转移到CPU
    
    参数:
        x: cupy数组、numpy数组或列表
    
    返回:
        numpy数组
    """
    if isinstance(x, list):
        result = []
        for item in x:
            if GPU_AVAILABLE and isinstance(item, cp.ndarray):
                result.append(cp.asnumpy(item))
            else:
                result.append(np.asarray(item))
        return result
    else:
        if GPU_AVAILABLE and isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        else:
            return np.asarray(x)


def get_gpu_info():
    """获取GPU信息"""
    if not GPU_AVAILABLE:
        return "GPU不可用 - 使用CPU模式"
    
    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        memory_info = device.mem_info
        free_mem = memory_info[0] / 1024**3  # GB
        total_mem = memory_info[1] / 1024**3  # GB
        
        info = f"""GPU信息:
        设备: {props['name'].decode()}
        计算能力: {props['major']}.{props['minor']}
        总内存: {total_mem:.2f} GB
        可用内存: {free_mem:.2f} GB
        多处理器数量: {props['multiProcessorCount']}
        """
        return info
    except Exception as e:
        return f"无法获取GPU信息: {str(e)}"


# 导出常用的数组创建函数（GPU优先）
xp = cp  # 主数组模块

__all__ = [
    'xp', 'cp', 'np', 
    'GPU_AVAILABLE',
    'get_array_module',
    'to_gpu',
    'to_cpu',
    'get_gpu_info'
]

