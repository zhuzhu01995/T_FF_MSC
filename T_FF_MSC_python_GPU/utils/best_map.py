"""最佳标签映射函数 - CPU版本"""
import numpy as np
from .hungarian import hungarian


def best_map(L1, L2):
    """
    将L2的标签重新排列以尽可能匹配L1
    
    参数:
        L1: 第一个标签向量（numpy数组）
        L2: 第二个标签向量（numpy数组）
    
    返回:
        newL2: 重新映射后的L2标签（numpy数组）
    
    注：此函数在CPU上运行
    """
    L1 = L1.flatten()
    L2 = L2.flatten()
    
    if L1.shape != L2.shape:
        raise ValueError('size(L1) must == size(L2)')
    
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    
    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    
    for i in range(nClass1):
        for j in range(nClass2):
            G[i, j] = np.sum((L1 == Label1[i]) & (L2 == Label2[j]))
    
    c, _ = hungarian(-G)
    newL2 = np.zeros_like(L2)
    
    # c[i]表示分配给真实类别i的聚类类别索引
    # 所以应该将聚类类别Label2[c[i]]映射到真实类别Label1[i]
    # 注意：hungarian返回的c长度等于min(nClass1, nClass2)
    for i in range(nClass1):
        if i < len(c) and c[i] < len(Label2):  # 防止索引越界
            newL2[L2 == Label2[c[i]]] = Label1[i]
    
    return newL2

