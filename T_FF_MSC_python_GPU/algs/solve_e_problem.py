"""求解E子问题的FISTA加速近端梯度法 - GPU加速版本"""
import sys
sys.path.append('..')
from gpu_utils import xp


def solve_e_problem(C_cell, D_cell, lambda_val, mu, max_iter, epsilon, E_init=None):
    """
    使用FISTA加速近端梯度法求解多视图E子问题（GPU加速）
    
    参数:
        C_cell: 列表 {C^{(1)}, C^{(2)}, ..., C^{(V)}}，每个C{v}的大小为N × N（GPU数组）
        D_cell: 列表 {D^{(1)}, D^{(2)}, ..., D^{(V)}}，每个D{v}的大小为d_v × N（GPU数组）
        lambda_val: 正则化参数λ
        mu: 惩罚参数μ（在ADMM中使用）
        max_iter: 最大迭代次数
        epsilon: 收敛容差
        E_init: 初始堆叠矩阵（可选）
    
    返回:
        E_stacked: 堆叠矩阵 [E^{(1)}; E^{(2)}; ...; E^{(V)}]（GPU数组）
    """
    # 获取视图数和维度信息
    V = len(C_cell)  # 视图数
    dims = [D.shape[0] for D in D_cell]  # 每个视图的维度
    N = D_cell[0].shape[1]  # 样本数
    total_dim = sum(dims)  # 总维度
    
    # 预计算每个视图的起始行索引
    row_start = [0] + list(xp.cumsum(xp.array(dims[:-1])).tolist())
    row_end = list(xp.cumsum(xp.array(dims)).tolist())
    
    # 计算Lipschitz常数（在GPU上）
    L_val = 0
    for v in range(V):
        CCT = C_cell[v] @ C_cell[v].T
        norm_CCT = float(xp.linalg.norm(CCT, 2))  # 转换为CPU标量
        
        if norm_CCT > L_val:
            L_val = norm_CCT
    
    L_val = mu * L_val  # 最终的Lipschitz常数
    eta = 1 / L_val  # 步长
    
    # 初始化变量（在GPU上）
    if E_init is not None:
        E_k = E_init.copy()
    else:
        E_k = xp.zeros((total_dim, N))
    
    W_k = E_k.copy()  # 辅助变量（用于FISTA加速）
    t_k = 1  # FISTA动量参数
    
    # 主迭代循环
    for iter_num in range(max_iter):
        # 存储前一次迭代的解
        E_prev = E_k.copy()
        
        # 计算梯度（视图并行，在GPU上）
        G_cell = []  # 存储每个视图的梯度
        
        for v in range(V):
            # 提取对应于当前视图的W_k部分
            W_v = W_k[row_start[v]:row_end[v], :]
            
            # 计算梯度：G(v) = μ * (W_v * C{v} + D{v}) * C{v}'
            term = W_v @ C_cell[v] + D_cell[v]
            G_v = mu * (term @ C_cell[v].T)
            G_cell.append(G_v)
        
        # 堆叠所有视图的梯度（在GPU上）
        G_stacked = xp.vstack(G_cell)
        
        # 梯度步：U = W_k - η * G_stacked
        U = W_k - eta * G_stacked
        
        # 近端算子：逐列收缩（在GPU上向量化）
        threshold = eta * lambda_val
        
        # 向量化版本：计算所有列的范数
        norms = xp.linalg.norm(U, axis=0)  # 每列的L2范数
        
        # 收缩因子
        shrink_factors = xp.maximum(0, 1 - threshold / (norms + 1e-12))
        
        # 应用收缩
        E_next = U * shrink_factors[xp.newaxis, :]
        
        # FISTA加速
        t_next = (1 + xp.sqrt(1 + 4 * t_k ** 2)) / 2
        momentum = (t_k - 1) / t_next
        
        # 更新辅助变量：W_{k+1} = E_{k+1} + momentum * (E_{k+1} - E_k)
        W_next = E_next + momentum * (E_next - E_k)
        
        # 更新迭代变量
        E_k = E_next
        W_k = W_next
        t_k = t_next
        
        # 收敛检查（在GPU上计算范数，转换为CPU标量进行比较）
        diff_norm = float(xp.linalg.norm(E_k - E_prev, 'fro'))
        if diff_norm < epsilon:
            break
    
    # 返回最终解
    return E_k

