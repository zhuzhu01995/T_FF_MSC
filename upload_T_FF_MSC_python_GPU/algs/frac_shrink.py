"""Fractional收缩算子 - GPU加速版本"""
import sys
import os

# 处理相对导入问题
try:
    # 尝试相对导入（当作为包的一部分导入时）
    from .frac_update_sigma import frac_update_sigma
except ImportError:
    # 如果相对导入失败（直接运行时），使用绝对导入
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from algs.frac_update_sigma import frac_update_sigma

# 导入gpu_utils
sys.path.append('..')
from gpu_utils import xp


def frac_shrink(X, lambda_val, mode, a):
    """
    Fractional收缩算子（GPU加速）
    
    参数:
        X: 输入张量（GPU数组）
        lambda_val: lambda参数
        mode: 模式（3表示张量模式）
        a: alpha参数
    
    返回:
        X: 处理后的张量
        objV: 目标值
    """
    sX = X.shape
    
    # 辅助函数：兼容的复制操作
    def safe_copy(x):
        """兼容NumPy/CuPy的复制函数"""
        if hasattr(x, 'copy'):  # NumPy/CuPy
            return x.copy()
        else:
            return x  # 如果都没有，返回原对象
    
    if mode == 3:
        Y = xp.moveaxis(X, 0, -1)  # shiftdim(X, 1)的Python等价
        Y = xp.moveaxis(Y, 0, 1)
    else:
        Y = safe_copy(X)
    
    # FFT变换（在GPU上执行）
    Yhat = xp.fft.fft(Y, axis=2)
    
    objV = 0
    if mode == 3:
        n3 = sX[0]
        m = min(sX[1], sX[2])
    else:
        n3 = sX[2]
        m = min(sX[0], sX[1])
    
    # 对每个前切片进行SVD（在GPU上执行）
    for i in range(n3):
        uhat, shat, vhat = xp.linalg.svd(Yhat[:, :, i], full_matrices=False)
        
        # 更新奇异值
        for j in range(m):
            shat[j] = frac_update_sigma(shat[j], lambda_val, a)
        
        # 计算目标值
        objV += xp.sum((a * xp.abs(shat)) / (1 + a * xp.abs(shat)))
        
        # 重构矩阵
        # 注意：uhat和vhat是复数类型（来自FFT后的SVD），shat是实数
        # NumPy/CuPy会自动处理类型提升（实数shat与复数uhat/vhat相乘）
        diag_shat = xp.diag(shat)
        Yhat[:, :, i] = uhat @ diag_shat @ vhat
    
    # 逆FFT变换（在GPU上执行）
    Y = xp.fft.ifft(Yhat, axis=2).real
    
    if mode == 3:
        Y = xp.moveaxis(Y, 1, 0)
        X = xp.moveaxis(Y, -1, 0)
    else:
        X = Y
    
    # 将标量转换为Python类型
    if hasattr(objV, 'item'):
        objV = objV.item()
    
    return X, objV

