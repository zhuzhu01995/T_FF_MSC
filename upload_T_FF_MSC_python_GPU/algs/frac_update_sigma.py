"""Fractional更新sigma函数 - GPU加速版本"""
import sys
import os

# 处理相对导入问题
try:
    # 尝试相对导入（当作为包的一部分导入时）
    from .glu import glu
except ImportError:
    # 如果相对导入失败（直接运行时），使用绝对导入
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from algs.glu import glu

# 导入gpu_utils
sys.path.append('..')
from gpu_utils import xp


def frac_update_sigma(sigma, lambda_val, a):
    """
    更新奇异值（GPU加速）
    
    参数:
        sigma: 输入奇异值
        lambda_val: lambda参数
        a: alpha参数
    
    返回:
        s_sigma: 更新后的奇异值
    """
    lambda_mu = lambda_val
    
    # 计算阈值
    a_sq = a * a
    if lambda_mu <= 1 / a_sq:
        t_star = (lambda_mu * a) / 2
    else:
        t_star = xp.sqrt(lambda_mu) - 1 / (2 * a)
    
    # 应用阈值规则
    if xp.abs(sigma) <= t_star:
        s_sigma = 0
    else:
        # 调用glu函数计算非零解
        s_sigma = glu(lambda_mu, a, sigma)
    
    return s_sigma

