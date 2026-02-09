"""分数阈值函数 - GPU加速版本"""
import sys
sys.path.append('..')
from gpu_utils import xp


def glu(la, a, x):
    """
    分数阈值函数的表达式（GPU加速）
    
    参数:
        la: lambda参数
        a: alpha参数
        x: 输入值
    
    返回:
        x0: 输出值
    """
    # 分数阈值函数的表达式
    f = xp.arccos(-1 + (27 * la * a * a) / (4.0 * (1 + a * xp.abs(x)) ** 3.0))
    x0 = xp.sign(x) * (((1 + a * xp.abs(x)) * (1 + 2 * xp.cos(f / 3 - xp.pi / 3)) - 3) / (3.0 * a))
    
    return x0

