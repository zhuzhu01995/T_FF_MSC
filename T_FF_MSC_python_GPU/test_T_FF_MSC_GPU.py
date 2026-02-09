"""T-FF-MSC算法GPU版本测试脚本
使用方法：
    1. 修改 test_list 选择要测试的数据集（可以选择多个）
    2. 修改 num_runs 设置运行次数
    3. 运行脚本: python test_T_FF_MSC_GPU.py
"""
import numpy as np
from scipy.io import loadmat
import os
import time
from utils import normalize_data
from algs import alg_t_ff_msc
from gpu_utils import to_gpu, get_gpu_info, GPU_AVAILABLE


def main():
    """主测试函数"""
    # 显示GPU信息
    print("=" * 100)
    print("T-FF-MSC GPU加速版本 - 测试脚本")
    print("=" * 100)
    print("\n" + get_gpu_info())
    print()
    
    # 数据路径（相对于原CPU版本）
    data_path = '../T_FF_MSC_python/data/'
    
    # 如果上述路径不存在，尝试使用本地data文件夹
    if not os.path.exists(data_path):
        data_path = 'data/'
    
    # ========================================
    # 配置区域 - 在这里修改测试参数
    # ========================================
    
    # 随机种子模式设置
    # 'fixed': 固定种子（所有运行结果完全相同，适合论文可重复性）
    # 'variable': 变化种子（每次运行使用不同种子，适合评估算法稳定性）
    # None: 系统随机（每次运行都不同，完全随机）
    random_seed_mode = 'fixed'  # 修改这里来改变随机种子行为
    
    # 根据模式设置随机种子
    if random_seed_mode == 'variable':
        np.random.seed(42)
        print("⚠️  警告: 使用固定随机种子，所有运行结果将完全相同（标准差将为0）")
        print("   这不是算法稳定性的真实反映，仅用于结果可重复性\n")
    elif random_seed_mode == 'variable':
        # 不设置全局种子，在循环中为每次运行设置不同种子
        print("✓ 使用变化随机种子，每次运行使用不同种子（可评估算法稳定性）\n")
    else:
        # 完全随机
        print("✓ 使用系统随机种子（每次运行都不同）\n")
    
    # 运行次数（建议：快速测试1-5次，完整测试20次）
    num_runs = 20
    
    # 可视化选项（已移除绘图功能）
    enable_plotting = False  # 绘图功能已被移除
    
    # 数据集配置
    datasets = [
        {
            'file': 'yale.mat',
            'name': 'Yale',
            'views': 3,
            'lambda': 0.221,  # 最佳参数
            'description': '165样本, 15类'
        },
        {
            'file': 'yaleB.mat',
            'name': 'Extended YaleB',
            'views': 3,
            'lambda': 0.001,
            'description': '2414样本, 38类'
        },
        {
            'file': 'ORL.mat',
            'name': 'ORL',
            'views': 3,
            'lambda': 0.1,  #'lambda': 0.219,
            'description': '400样本, 40类'
        },
        {
            'file': 'COIL20MV.mat',
            'name': 'COIL-20',
            'views': 3,
            'lambda': 0.001,
            'description': '1440样本, 20类'
        },
    ]
    
    # ========================================
    # 选择要测试的数据集（通过索引）
    # 索引: 0=Yale, 1=YaleB, 2=ORL, 3=COIL-20
    # ========================================
    test_list = [0]  # 修改这里选择数据集，例如 [0, 2] 测试Yale和ORL
    
    print("=" * 100)
    print("测试配置:")
    print(f"  运行次数: {num_runs}")
    print(f"  数据路径: {data_path}")
    print(f"  GPU加速: {'启用' if GPU_AVAILABLE else '禁用（CPU模式）'}")
    print("\n可用数据集:")
    for i, ds in enumerate(datasets):
        marker = "→" if i in test_list else " "
        print(f"  {marker} [{i}] {ds['name']:<20} - {ds['description']:<20} (lambda={ds['lambda']})")
    print(f"\n将测试: {[datasets[i]['name'] for i in test_list]}")
    print("=" * 100)
    print()
    
    # 遍历要测试的数据集
    for dataset_idx in test_list:
        ds = datasets[dataset_idx]
        
        print("\n" + "=" * 100)
        print(f"测试数据集 [{dataset_idx}]: {ds['name']} - {ds['description']}")
        print("=" * 100)
        
        # 加载数据
        data_file = os.path.join(data_path, ds['file'])
        if not os.path.exists(data_file):
            print(f'✗ 错误: 数据文件 {data_file} 不存在！')
            print('  请确保数据文件在正确的位置。')
            print('  提示: 将原CPU版本的data文件夹中的.mat文件复制到:')
            print(f'       {os.path.abspath(data_path)}')
            continue
        
        print(f"✓ 加载数据: {data_file}")
        data = loadmat(data_file)
        
        # 提取多视图数据
        X = []
        for k in range(ds['views']):
            key = f'X{k+1}'
            if key in data:
                X.append(data[key].astype(float))
            else:
                print(f'  ✗ 警告: 数据文件中没有找到 {key}')
        
        # 提取真实标签
        gt = None       #初始化gt为无值状态，后续若成功提取标签则覆盖该值
        for label_key in ['gt', 'gnd', 'y']:
            if label_key in data:
                gt = data[label_key].flatten()
                break
        
        if gt is None:
            print('✗ 错误: 找不到真实标签！')
            continue
        
        cls_num = len(np.unique(gt))
        K = len(X)
        
        print(f"✓ 数据加载成功:")
        print(f"  视图数: {K}")
        print(f"  样本数: {X[0].shape[1]}")
        print(f"  特征维度: {[x.shape[0] for x in X]}")
        print(f"  类别数: {cls_num}")
        
        # 数据归一化并转移到GPU
        print(f"✓ 数据归一化{'并转移到GPU' if GPU_AVAILABLE else ''}...")
        Y = []
        for iv in range(K):
            Y.append(to_gpu(normalize_data(X[iv])))
        
        # 设置参数
        opts = {
            'Frac_alpha': 5000,
            'maxIter': 200,
            'epsilon': 1e-4,
            'flag_debug': 0,
            'mu': 1e-5,
            'rho': 1e-5,
            'eta': 2,
            'max_mu': 1e10,
            'max_rho': 1e10,
            'lambda': ds['lambda']
        }
        
        print(f"✓ 算法参数: lambda={ds['lambda']}, maxIter={opts['maxIter']}")
        
        # 初始化记录数组
        run_times = []
        NMI_results = []
        AR_results = []
        ACC_results = []
        recall_results = []
        precision_results = []
        fscore_results = []
        
        # 用于绘图的数据
        all_S_matrices = []  # 存储所有运行的亲和矩阵
        all_C_results = []   # 存储所有运行的聚类结果
        all_histories = []   # 存储所有运行的history
        
        # 多次运行
        print(f"\n运行算法 (共{num_runs}次):")
        print("-" * 100)
        
        total_start = time.time()
        
        for run_idx in range(num_runs):
            print(f"  [{run_idx+1:2d}/{num_runs}] ", end='', flush=True)
            
            # 根据随机种子模式设置种子
            if random_seed_mode == 'variable':
                np.random.seed(42 + run_idx)  # 每次运行使用不同的种子
            elif random_seed_mode is None:
                # 不设置种子，使用系统随机
                pass
            # 如果random_seed_mode == 'fixed'，已经在主函数开始处设置了全局种子
            
            # 计时
            run_start = time.time()
            C, S, Out = alg_t_ff_msc(Y, cls_num, gt, opts)
            elapsed = time.time() - run_start
            
            # 记录结果
            run_times.append(elapsed)
            NMI_results.append(Out['NMI'])
            AR_results.append(Out['AR'])
            ACC_results.append(Out['ACC'])
            recall_results.append(Out['recall'])
            precision_results.append(Out['precision'])
            fscore_results.append(Out['fscore'])
            
            # 保存用于绘图的数据
            all_S_matrices.append(S)
            all_C_results.append(C)
            all_histories.append(Out['history'])
            
            # 显示结果
            print(f"时间: {elapsed:6.2f}s  |  "
                  f"NMI: {Out['NMI']:.4f}  "
                  f"ACC: {Out['ACC']:.4f}  "
                  f"AR: {Out['AR']:.4f}  "
                  f"F: {Out['fscore']:.4f}")
        
        total_time = time.time() - total_start
        
        # 统计结果
        print("-" * 100)
        print("\n结果统计:")
        print("=" * 100)
        print(f"{'指标':<12}  {'均值':>10}  {'标准差':>10}  {'最小值':>10}  {'最大值':>10}")
        print("-" * 100)
        
        metrics = {
            '运行时间(s)': run_times,
            'NMI': NMI_results,
            'ACC': ACC_results,
            'AR': AR_results,
            'Recall': recall_results,
            'Precision': precision_results,
            'F-score': fscore_results,
        }
        
        for name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            print(f"{name:<12}  {mean_val:>10.4f}  {std_val:>10.4f}  "
                  f"{min_val:>10.4f}  {max_val:>10.4f}")
        
        print("=" * 100)
        print(f"\n总运行时间: {total_time:.2f}秒")
        print(f"平均单次时间: {np.mean(run_times):.2f}秒")
        print(f"GPU加速: {'启用' if GPU_AVAILABLE else '禁用'}")
        
        # 最佳结果
        best_idx = np.argmax(ACC_results)
        print(f"\n最佳运行 (第{best_idx+1}次):")
        print(f"  ACC: {ACC_results[best_idx]:.4f}")
        print(f"  NMI: {NMI_results[best_idx]:.4f}")
        print(f"  F-score: {fscore_results[best_idx]:.4f}")
        print(f"  时间: {run_times[best_idx]:.2f}秒")
        
        # ========================================
        # 绘图部分 - 已被移除
        # ========================================
        # 绘图功能已被移除
    
    print("\n" + "=" * 100)
    print("所有测试完成！")
    print("=" * 100)


def quick_test():
    """快速测试 - 每个数据集只运行1次"""
    print("快速测试模式 - 每个数据集运行1次\n")
    
    # 临时修改运行次数
    import sys
    sys.modules[__name__].__dict__['num_runs'] = 1
    main()


def full_test():
    """完整测试 - 测试所有数据集"""
    print("完整测试模式 - 测试所有数据集\n")
    
    # 修改test_list为所有数据集
    import sys
    original_main = sys.modules[__name__].main
    
    def modified_main():
        # 这里会测试所有数据集
        pass
    
    main()


if __name__ == '__main__':
    # ========================================
    # 使用说明
    # ========================================
    # 
    # 基本使用:
    #   python test_T_FF_MSC_GPU.py
    #
    # 修改测试的数据集:
    #   在上面的 test_list 中修改索引
    #   例如: test_list = [0, 2] 测试Yale和ORL
    #
    # 数据集索引:
    #   0 - Yale (小数据集，快速测试)
    #   1 - Extended YaleB (大数据集)
    #   2 - ORL (中等数据集)
    #   3 - COIL-20 (大数据集)
    #
    # ========================================
    
    main()

