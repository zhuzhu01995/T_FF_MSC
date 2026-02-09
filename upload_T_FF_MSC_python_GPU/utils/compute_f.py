"""F-score、精确率和召回率计算函数 - CPU版本"""
import numpy as np


def compute_f(T, H):
    """
    计算F-score、精确率和召回率
    
    参数:
        T: 真实标签（numpy数组）
        H: 聚类结果（numpy数组）
    
    返回:
        f: F-score
        p: 精确率
        r: 召回率
    
    注：此函数在CPU上运行
    """
    if len(T) != len(H):
        raise ValueError('T and H must have the same length')
    
    N = len(T)
    numT = 0
    numH = 0
    numI = 0
    
    for n in range(N):
        Tn = (T[n+1:] == T[n])
        Hn = (H[n+1:] == H[n])
        numT = numT + np.sum(Tn)
        numH = numH + np.sum(Hn)
        numI = numI + np.sum(Tn & Hn)
    
    p = 1
    r = 1
    f = 1
    
    if numH > 0:
        p = numI / numH
    
    if numT > 0:
        r = numI / numT
    
    if (p + r) == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)
    
    return f, p, r

