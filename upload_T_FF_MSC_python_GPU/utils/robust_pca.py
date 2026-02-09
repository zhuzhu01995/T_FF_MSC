"""
Robust PCA (RPCA) 实现
用于预处理去除噪声，作为naive baseline

基于：Candès et al., "Robust Principal Component Analysis?"
使用ADMM方法求解：min ||L||_* + lambda||S||_1 s.t. X = L + S
"""

import numpy as np
from gpu_utils import xp, to_cpu, to_gpu


def robust_pca(X, lambda_val=None, max_iter=100, tol=1e-7, mu=1.0, rho=1.2, max_mu=1e10):
    """
    Robust PCA: 将矩阵X分解为低秩部分L和稀疏部分S
    X = L + S
    
    使用ADMM方法求解：
    min ||L||_* + lambda||S||_1  s.t. X = L + S
    
    参数:
        X: 输入矩阵 (d × N)，numpy或cupy数组
        lambda_val: 正则化参数（默认：1/sqrt(max(d, N))）
        max_iter: 最大迭代次数
        tol: 收敛容差
        mu: 初始惩罚参数
        rho: 惩罚参数增长因子
        max_mu: 最大惩罚参数
    
    返回:
        L: 低秩部分（去噪后的数据）
        S: 稀疏部分（噪声）
    """
    # 转移到GPU（如果可用）
    X_gpu = to_gpu(X) if isinstance(X, np.ndarray) else X
    
    d, N = X_gpu.shape
    
    # 默认lambda值
    if lambda_val is None:
        lambda_val = 1.0 / np.sqrt(max(d, N))
    
    # 初始化
    L = xp.zeros_like(X_gpu)
    S = xp.zeros_like(X_gpu)
    Y = xp.zeros_like(X_gpu)  # 拉格朗日乘子
    
    # 主循环
    for iter_num in range(max_iter):
        # 更新L: min ||L||_* + (mu/2)||L - (X - S + Y/mu)||_F^2
        # 使用奇异值阈值（SVT）
        M = X_gpu - S + Y / mu
        U, sigma, Vt = xp.linalg.svd(M, full_matrices=False)
        # 软阈值：sigma -> max(sigma - 1/mu, 0)
        sigma_shrunk = xp.maximum(sigma - 1.0 / mu, 0)
        L = U @ xp.diag(sigma_shrunk) @ Vt
        
        # 更新S: min lambda||S||_1 + (mu/2)||S - (X - L + Y/mu)||_F^2
        # 使用软阈值算子
        T = X_gpu - L + Y / mu
        threshold = lambda_val / mu
        S = xp.sign(T) * xp.maximum(xp.abs(T) - threshold, 0)
        
        # 更新拉格朗日乘子
        Y = Y + mu * (X_gpu - L - S)
        
        # 更新惩罚参数
        mu = min(mu * rho, max_mu)
        
        # 收敛检查
        residual = xp.linalg.norm(X_gpu - L - S, 'fro')
        if residual < tol:
            break
    
    # 返回去噪后的数据L（转移到CPU如果是numpy输入）
    if isinstance(X, np.ndarray):
        L = to_cpu(L)
        S = to_cpu(S)
    
    return L, S


def robust_pca_multi_view(X_list, lambda_val=None, **kwargs):
    """
    对多视图数据分别应用Robust PCA
    
    参数:
        X_list: 多视图数据列表 [X1, X2, ..., XK]
        lambda_val: 正则化参数（可以为每个视图设置不同的值，或使用默认值）
        **kwargs: 传递给robust_pca的其他参数
    
    返回:
        L_list: 去噪后的数据列表 [L1, L2, ..., LK]
        S_list: 噪声列表 [S1, S2, ..., SK]
    """
    K = len(X_list)
    L_list = []
    S_list = []
    
    # 如果lambda_val是列表，为每个视图使用不同的值
    if isinstance(lambda_val, (list, tuple)):
        lambda_vals = lambda_val
    else:
        lambda_vals = [lambda_val] * K
    
    for k in range(K):
        L_k, S_k = robust_pca(X_list[k], lambda_val=lambda_vals[k], **kwargs)
        L_list.append(L_k)
        S_list.append(S_k)
    
    return L_list, S_list

