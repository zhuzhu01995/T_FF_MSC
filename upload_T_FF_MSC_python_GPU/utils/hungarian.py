"""匈牙利算法（使用scipy实现）- CPU版本"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian(A):
    """
    使用匈牙利方法解决分配问题
    
    参数:
        A: 方形成本矩阵（numpy数组）
    
    返回:
        C: 最优分配（numpy数组），C[i]表示行i被分配到列C[i]
        T: 最优分配的成本
    
    注：此函数在CPU上运行，因为使用scipy
    """
    # scipy的linear_sum_assignment返回行和列索引
    row_ind, col_ind = linear_sum_assignment(A)
    
    # 创建分配向量（MATLAB风格，从0开始）
    # C[i]表示行i被分配到列C[i]
    # 注意：row_ind和col_ind的长度可能小于A的行数（如果矩阵不是方阵或某些行/列没有匹配）
    # 我们需要创建一个完整的分配向量，长度为A的行数
    n_rows = A.shape[0]
    C = np.zeros(n_rows, dtype=int)
    
    # 将匹配的分配填入C
    for i, row in enumerate(row_ind):
        if row < n_rows:
            C[row] = col_ind[i]
    
    # 计算总成本
    T = A[row_ind, col_ind].sum()
    
    return C, T

