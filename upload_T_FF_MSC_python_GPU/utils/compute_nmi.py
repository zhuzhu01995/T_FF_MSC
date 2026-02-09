"""归一化互信息（NMI）计算函数 - CPU版本"""
import numpy as np


def compute_nmi(T, H):
    """
    计算归一化互信息
    
    参数:
        T: 真实标签（numpy数组）
        H: 聚类结果（numpy数组）
    
    返回:
        A: 混淆矩阵
        nmi: 归一化互信息
        avgent: 平均熵
    
    注：此函数在CPU上运行
    """
    N = len(T)
    classes = np.unique(T)
    clusters = np.unique(H)
    num_class = len(classes)
    num_clust = len(clusters)
    
    # 计算每个类中的点数
    D = np.zeros(num_class)
    for j in range(num_class):
        index_class = (T == classes[j])
        D[j] = np.sum(index_class)
    
    # 计算互信息
    mi = 0
    A = np.zeros((num_clust, num_class))
    avgent = 0
    B = np.zeros(num_clust)
    miarr = np.zeros((num_clust, num_class))
    
    for i in range(num_clust):
        # 聚类i中的点数
        index_clust = (H == clusters[i])
        B[i] = np.sum(index_clust)
        
        for j in range(num_class):
            index_class = (T == classes[j])
            # 计算类j中最终在聚类i中的点数
            A[i, j] = np.sum(index_class & index_clust)
            
            if A[i, j] != 0:
                miarr[i, j] = A[i, j] / N * np.log2(N * A[i, j] / (B[i] * D[j]))
                # 平均熵计算
                avgent = avgent - (B[i] / N) * (A[i, j] / B[i]) * np.log2(A[i, j] / B[i])
            else:
                miarr[i, j] = 0
            
            mi = mi + miarr[i, j]
    
    # 类熵
    class_ent = 0
    for i in range(num_class):
        if D[i] > 0:
            class_ent = class_ent + D[i] / N * np.log2(N / D[i])
    
    # 聚类熵
    clust_ent = 0
    for i in range(num_clust):
        if B[i] > 0:
            clust_ent = clust_ent + B[i] / N * np.log2(N / B[i])
    
    # 归一化互信息
    if (clust_ent + class_ent) > 0:
        nmi = 2 * mi / (clust_ent + class_ent)
    else:
        nmi = 0
    
    return A, nmi, avgent

