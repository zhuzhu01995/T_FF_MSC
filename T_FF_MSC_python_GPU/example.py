"""T-FF-MSC算法GPU版本使用示例"""
import numpy as np
from scipy.io import loadmat
from utils import normalize_data
from algs import alg_t_ff_msc
from gpu_utils import get_gpu_info, GPU_AVAILABLE, to_gpu


def simple_example():
    """简单示例：在Yale数据集上运行一次（GPU加速）"""
    print("=" * 60)
    print("T-FF-MSC 算法GPU加速版本 - 简单示例")
    print("=" * 60)
    
    # 显示GPU信息
    print("\n" + get_gpu_info())
    print()
    
    # 加载数据
    print("加载Yale数据集...")
    data = loadmat('../T_FF_MSC_python/data/yale.mat')
    
    # 提取多视图数据
    X = [data['X1'].astype(float), 
         data['X2'].astype(float), 
         data['X3'].astype(float)]
    gt = data['gt'].flatten()
    
    print(f"数据集信息:")
    print(f"  视图数: {len(X)}")
    print(f"  样本数: {X[0].shape[1]}")
    print(f"  特征维度: {[x.shape[0] for x in X]}")
    print(f"  类别数: {len(np.unique(gt))}")
    
    # 数据归一化并转移到GPU
    print(f"\n数据归一化{'并转移到GPU' if GPU_AVAILABLE else ''}...")
    Y = [to_gpu(normalize_data(x)) for x in X]
    
    # 设置参数
    opts = {
        'Frac_alpha': 5000,
        'maxIter': 200,
        'epsilon': 1e-4,
        'flag_debug': 1,  # 启用调试输出
        'mu': 1e-5,
        'rho': 1e-5,
        'eta': 2,
        'max_mu': 1e10,
        'max_rho': 1e10,
        'lambda': 0.221  # Yale数据集的最佳参数
    }
    
    cls_num = len(np.unique(gt))
    
    # 运行算法
    print(f"\n运行T-FF-MSC算法（{'GPU加速' if GPU_AVAILABLE else 'CPU模式'}）...")
    import time
    start_time = time.time()
    
    C, S, Out = alg_t_ff_msc(Y, cls_num, gt, opts)
    
    elapsed_time = time.time() - start_time
    
    # 显示结果
    print("\n" + "=" * 60)
    print("结果:")
    print("=" * 60)
    print(f"运行时间: {elapsed_time:.2f} 秒")
    print(f"NMI (归一化互信息): {Out['NMI']:.4f}")
    print(f"ACC (准确率): {Out['ACC']:.4f}")
    print(f"AR (调整Rand指数): {Out['AR']:.4f}")
    print(f"Precision (精确率): {Out['precision']:.4f}")
    print(f"Recall (召回率): {Out['recall']:.4f}")
    print(f"F-score: {Out['fscore']:.4f}")
    print("=" * 60)


def benchmark_example():
    """基准测试：比较GPU和CPU性能"""
    print("\n" + "=" * 60)
    print("GPU vs CPU 性能基准测试")
    print("=" * 60)
    
    if not GPU_AVAILABLE:
        print("⚠ GPU不可用，跳过基准测试")
        return
    
    # 加载数据
    print("\n加载Yale数据集...")
    data = loadmat('../T_FF_MSC_python/data/yale.mat')
    X = [data['X1'].astype(float), 
         data['X2'].astype(float), 
         data['X3'].astype(float)]
    gt = data['gt'].flatten()
    
    # 数据归一化
    Y = [normalize_data(x) for x in X]
    cls_num = len(np.unique(gt))
    
    # 设置参数（减少迭代次数以加快测试）
    opts = {
        'Frac_alpha': 5000,
        'maxIter': 50,
        'epsilon': 1e-4,
        'flag_debug': 0,
        'mu': 1e-5,
        'rho': 1e-5,
        'eta': 2,
        'max_mu': 1e10,
        'max_rho': 1e10,
        'lambda': 0.221
    }
    
    # GPU版本
    print("\n运行GPU版本...")
    Y_gpu = [to_gpu(y) for y in Y]
    import time
    start_time = time.time()
    C_gpu, S_gpu, Out_gpu = alg_t_ff_msc(Y_gpu, cls_num, gt, opts)
    gpu_time = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print(f"GPU运行时间: {gpu_time:.2f} 秒")
    print(f"GPU准确率: {Out_gpu['ACC']:.4f}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    # 运行简单示例
    simple_example()
    
    # 如果想运行基准测试，取消下面的注释
    # benchmark_example()

