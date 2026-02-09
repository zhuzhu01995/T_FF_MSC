"""T-FF-MSC算法主函数 - GPU加速版本"""
import sys
import os

# 处理相对导入问题
try:
    # 尝试相对导入（当作为包的一部分导入时）
    from .frac_shrink import frac_shrink
    from .solve_e_problem import solve_e_problem
except ImportError:
    # 如果相对导入失败（直接运行时），使用绝对导入
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from algs.frac_shrink import frac_shrink
    from algs.solve_e_problem import solve_e_problem

# 导入其他模块
sys.path.append('..')
from gpu_utils import xp, to_cpu, to_gpu
from utils import spectral_clustering, compute_nmi, accuracy, compute_f, rand_index


def alg_t_ff_msc(X, cls_num, gt, opts=None):
    """
    T-FF-MSC: 基于张量Log-determinant模型的多视图子空间聚类（GPU加速）
    
    参数:
        X: 数据特征（列表，每个元素是一个视图，numpy数组或cupy数组）
        cls_num: 聚类数
        gt: 真实标签（numpy数组）
        opts: 可选参数
            - maxIter: 最大迭代次数
            - mu: 惩罚参数
            - rho: 惩罚参数
            - epsilon: 停止容差
    
    返回:
        C: 聚类结果（numpy数组）
        S: 亲和矩阵（numpy数组）
        Out: 其他输出信息，如指标、历史记录
    """
    # 将数据转移到GPU
    X_gpu = to_gpu(X)
    
    # 参数设置
    N = X_gpu[0].shape[1]
    K = len(X_gpu)  # 视图数
    
    # 默认参数
    maxIter = 200
    epsilon = 1e-7
    lambda_val = 0.2
    mu = 1e-5
    rho = 1e-5
    eta = 2
    max_mu = 1e10
    max_rho = 1e10
    flag_debug = 0
    Frac_alpha = 5000
    
    # 从opts中读取参数
    if opts is not None:
        if 'maxIter' in opts:
            maxIter = opts['maxIter']
        if 'epsilon' in opts:
            epsilon = opts['epsilon']
        if 'lambda' in opts:
            lambda_val = opts['lambda']
        if 'mu' in opts:
            mu = opts['mu']
        if 'rho' in opts:
            rho = opts['rho']
        if 'eta' in opts:
            eta = opts['eta']
        if 'max_mu' in opts:
            max_mu = opts['max_mu']
        if 'max_rho' in opts:
            max_rho = opts['max_rho']
        if 'flag_debug' in opts:
            flag_debug = opts['flag_debug']
        if 'Frac_alpha' in opts:
            Frac_alpha = opts['Frac_alpha']
    
    # 在GPU上初始化
    Z = [xp.zeros((N, N)) for _ in range(K)]
    W = [xp.zeros((N, N)) for _ in range(K)]
    G = [xp.zeros((N, N)) for _ in range(K)]
    E = [xp.zeros((X_gpu[k].shape[0], N)) for k in range(K)]
    Y = [xp.zeros((X_gpu[k].shape[0], N)) for k in range(K)]
    
    # 初始化历史结构
    history = {
        'norm_Z_G': [],  # 恢复 ||Z - G||_∞
        'norm_Z': [],
        'objval': []
    }
    
    iter_num = 0
    Isconverg = False
    
    # 迭代
    while not Isconverg:
        if flag_debug:
            print(f'----processing iter {iter_num + 1}--------')
        
        # 更新 Z^k（在GPU上进行矩阵运算）
        for k in range(K):
            tmp = (X_gpu[k] - E[k]).T @ Y[k] + mu * (X_gpu[k] - E[k]).T @ (X_gpu[k] - E[k]) - W[k] + rho * G[k]
            Z[k] = xp.linalg.solve(mu * (X_gpu[k] - E[k]).T @ (X_gpu[k] - E[k]) + rho * xp.eye(N), tmp)
        
        # 更新 E^k
        # 准备近端梯度法的输入
        C_cell = [Z[k] - xp.eye(N) for k in range(K)]
        D_cell = [X_gpu[k] - X_gpu[k] @ Z[k] + Y[k] / mu for k in range(K)]
        
        # 设置近端梯度求解器的参数
        max_iter_inner = 50  # 内循环最大迭代次数
        tol_inner = 1e-5  # 内循环容差
        
        # 从当前E创建初始E_stacked
        E_stacked = xp.vstack(E)
        
        # 使用近端梯度法求解（在GPU上）
        E_stacked = solve_e_problem(C_cell, D_cell, lambda_val, mu,
                                     max_iter_inner, tol_inner, E_stacked)
        
        # 将堆叠的E分割回视图特定的矩阵
        start_idx = 0
        for k in range(K):
            d_k = X_gpu[k].shape[0]
            E[k] = E_stacked[start_idx:start_idx + d_k, :]
            start_idx = start_idx + d_k
        
        # 更新 G（在GPU上进行张量FFT和SVD）
        Z_tensor = xp.stack(Z, axis=2)
        W_tensor = xp.stack(W, axis=2)
        
        G_tensor, objV = frac_shrink(Z_tensor + W_tensor / rho, 6 / rho, 3, Frac_alpha)
        
        # 更新辅助变量（在GPU上）
        W_tensor = W_tensor + rho * (Z_tensor - G_tensor)
        for k in range(K):
            Y[k] = Y[k] + mu * ((X_gpu[k] - E[k]) - (X_gpu[k] - E[k]) @ Z[k])
            G[k] = G_tensor[:, :, k]
            W[k] = W_tensor[:, :, k]
        
        # 记录迭代信息
        history['objval'].append(objV)
        
        # 收敛条件（在GPU上计算）
        Isconverg = True
        
        # 检查收敛性
        residual_list = [(X_gpu[k] - E[k]) - (X_gpu[k] - E[k]) @ Z[k] for k in range(K)]
        norm_Z = float(max([xp.max(xp.abs(residual)) for residual in residual_list]))
        history['norm_Z'].append(norm_Z)
        
        if norm_Z > epsilon:
            if flag_debug:
                print(f'norm_Z   {norm_Z:.10f}')
            Isconverg = False
        
        norm_Z_G = float(max([xp.max(xp.abs(Z[k] - G[k])) for k in range(K)]))
        history['norm_Z_G'].append(norm_Z_G)
        
        if norm_Z_G > epsilon:
            if flag_debug:
                print(f'norm_Z_G   {norm_Z_G:.10f}')
            Isconverg = False
        
        # 检查最大迭代次数
        if iter_num > maxIter:
            Isconverg = True
        
        # 更新惩罚参数
        mu = min(mu * eta, max_mu)
        rho = min(rho * eta, max_rho)
        
        iter_num += 1
    
    # 聚类（在GPU上计算亲和矩阵）
    S_gpu = xp.zeros((N, N))
    for k in range(K):
        S_gpu = S_gpu + xp.abs(Z[k]) + xp.abs(Z[k].T)
    
    # 将亲和矩阵转移回CPU进行谱聚类
    S = to_cpu(S_gpu)
    C = spectral_clustering(S, cls_num)
    
    # 计算评估指标（在CPU上）
    _, nmi, _ = compute_nmi(gt, C)
    ACC = accuracy(C, gt)
    f, p, r = compute_f(gt, C)
    AR, _, _, _ = rand_index(gt, C)
    
    # 记录输出
    Out = {
        'NMI': nmi,
        'AR': AR,
        'ACC': ACC,
        'recall': r,
        'precision': p,
        'fscore': f,
        'history': history
    }
    
    return C, S, Out

