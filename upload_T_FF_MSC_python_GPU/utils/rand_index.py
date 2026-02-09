"""Rand指数计算函数 - CPU版本"""
import numpy as np
from scipy.special import comb
from .contingency import contingency


def rand_index(c1, c2):
    """
    计算Rand指数来比较两个分区
    
    参数:
        c1: 第一个聚类成员向量（numpy数组）
        c2: 第二个聚类成员向量（numpy数组）
    
    返回:
        AR: 调整后的Rand指数（Hubert & Arabie）
        RI: 未调整的Rand指数
        MI: Mirkin指数
        HI: Hubert指数
    
    注：此函数在CPU上运行
    """
    if c1.ndim > 1 or c2.ndim > 1:
        raise ValueError('RandIndex: Requires two vector arguments')
    
    if len(c1) != len(c2):
        raise ValueError('RandIndex: Vectors must have the same length')
    
    # 形成列联矩阵
    C = contingency(c1, c2)
    
    n = np.sum(C)
    nis = np.sum(np.sum(C, axis=1) ** 2)  # 行和的平方和
    njs = np.sum(np.sum(C, axis=0) ** 2)  # 列和的平方和
    
    t1 = comb(int(n), 2, exact=True)  # 实体对的总数
    t2 = np.sum(C ** 2)  # nij^2的和
    t3 = 0.5 * (nis + njs)
    
    # 期望指数（用于调整）
    nc = (n * (n**2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))
    
    A = t1 + t2 - t3  # 一致数
    D = -t2 + t3  # 不一致数
    
    if t1 == nc:
        AR = 0  # 避免除以零；如果k=1，定义Rand = 0
    else:
        AR = (A - nc) / (t1 - nc)  # 调整后的Rand - Hubert & Arabie 1985
    
    RI = A / t1  # Rand 1971 - 一致概率
    MI = D / t1  # Mirkin 1970 - 不一致概率
    HI = (A - D) / t1  # Hubert 1977 - p(一致) - p(不一致)
    
    return AR, RI, MI, HI

