"""谱聚类算法 - GPU加速版本"""
import numpy as np
from sklearn.cluster import KMeans
import warnings
import sys
sys.path.append('..')
from gpu_utils import to_cpu, to_gpu, xp

warnings.filterwarnings('ignore')


def spectral_clustering(CKSym, n, random_state=42):
    """
    使用 Ng, Jordan 和 Weiss 的谱聚类算法对图的节点进行聚类（GPU加速）
    
    参数:
        CKSym: NxN 邻接矩阵（numpy或cupy数组）
        n: 聚类的组数
        random_state: KMeans的随机种子（默认42，设为None使用系统随机）
    
    返回:
        groups: N维向量，包含N个点到n个组的成员关系（numpy数组）
    """
    # 转移到GPU（如果还没有）
    CKSym_gpu = to_gpu(CKSym) if isinstance(CKSym, np.ndarray) else CKSym
    
    N = CKSym_gpu.shape[0]
    MAXiter = 1000  # KMeans的最大迭代次数
    REPlic = 20  # KMeans的重复次数
    
    # 使用归一化对称拉普拉斯矩阵 L = I - D^{-1/2} W D^{-1/2}（在GPU上计算）
    DN = xp.diag(1.0 / xp.sqrt(xp.sum(CKSym_gpu, axis=1) + xp.finfo(float).eps))
    LapN = xp.eye(N) - DN @ CKSym_gpu @ DN
    
    # 使用eigh进行特征分解（对称矩阵专用，返回特征值按升序排列）
    # 选择最小的n个特征值对应的特征向量用于聚类（在GPU上）
    eigvals, eigvecs = xp.linalg.eigh(LapN)
    kerN = eigvecs[:, :n]  # 取前n个特征向量（对应最小的n个特征值）
    
    # 归一化特征向量（按行归一化）（在GPU上向量化）
    norms = xp.linalg.norm(kerN, axis=1, keepdims=True)
    kerNS = kerN / (norms + xp.finfo(float).eps)
    
    # 将数据转移回CPU进行KMeans聚类（scikit-learn在CPU上运行）
    kerNS_cpu = to_cpu(kerNS)
    
    # KMeans聚类
    kmeans = KMeans(n_clusters=n, max_iter=MAXiter, n_init=REPlic, random_state=random_state)
    groups = kmeans.fit_predict(kerNS_cpu) + 1  # +1 使索引从1开始，匹配MATLAB
    
    return groups

