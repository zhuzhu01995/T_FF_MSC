"""准确率计算函数 - CPU版本"""
import numpy as np
from .best_map import best_map


def accuracy(C, gt):
    """
    计算聚类准确率
    
    参数:
        C: 聚类结果（numpy数组）
        gt: 真实标签（numpy数组）
    
    返回:
        ACC: 准确率
    
    注：此函数在CPU上运行
    """
    C = best_map(gt, C)
    ACC = np.sum(gt == C) / len(gt)
    return ACC

