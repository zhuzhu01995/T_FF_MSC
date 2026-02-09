"""列联表计算函数 - GPU加速版本"""
import numpy as np


def contingency(Mem1, Mem2):
    """
    为两个向量构建列联矩阵
    
    参数:
        Mem1: 第一个成员向量（numpy数组）
        Mem2: 第二个成员向量（numpy数组）
    
    返回:
        Cont: 列联矩阵（numpy数组）
    
    注：此函数在CPU上运行，因为涉及索引操作
    """
    if Mem1.ndim > 1 or Mem2.ndim > 1:
        raise ValueError('Contingency: Requires two vector arguments')
    
    if len(Mem1) != len(Mem2):
        raise ValueError('Contingency: Vectors must have the same length')
    
    Cont = np.zeros((int(np.max(Mem1)), int(np.max(Mem2))))
    
    for i in range(len(Mem1)):
        Cont[int(Mem1[i])-1, int(Mem2[i])-1] += 1
    
    return Cont

