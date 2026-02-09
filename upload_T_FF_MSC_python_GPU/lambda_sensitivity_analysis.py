"""Lambda参数敏感性分析脚本

使用方法：
    1. 修改数据集配置和最优lambda值
    2. 设置测试范围和步长
    3. 运行脚本: python lambda_sensitivity_analysis.py
"""
import numpy as np
from scipy.io import loadmat
import os
import time
# 绘图功能已被移除

from utils import normalize_data
from algs import alg_t_ff_msc
from gpu_utils import to_gpu, get_gpu_info, GPU_AVAILABLE


def lambda_sensitivity_analysis(dataset_file, dataset_name, optimal_lambda, 
                                cls_num, views, num_runs=10, 
                                lambda_range=0.2, lambda_step=0.01,
                                data_path='data/'):
    """
    进行lambda参数敏感性分析
    
    参数:
        dataset_file: 数据文件名
        dataset_name: 数据集名称
        optimal_lambda: 最优lambda值
        cls_num: 类别数
        views: 视图数
        num_runs: 每个lambda值运行的次数
        lambda_range: lambda测试范围（在最优值±lambda_range范围内）
        lambda_step: lambda测试步长
        data_path: 数据文件路径
    
    返回:
        results: 结果字典，包含所有lambda值的性能指标
    """
    print("=" * 100)
    print(f"Lambda敏感性分析 - {dataset_name}")
    print("=" * 100)
    print(f"最优lambda: {optimal_lambda}")
    print(f"测试范围: [{optimal_lambda - lambda_range:.4f}, {optimal_lambda + lambda_range:.4f}]")
    print(f"测试步长: {lambda_step}")
    print(f"每个lambda运行次数: {num_runs}")
    print("=" * 100)
    print()
    
    # 加载数据
    data_file = os.path.join(data_path, dataset_file)
    if not os.path.exists(data_file):
        print(f'✗ 错误: 数据文件 {data_file} 不存在！')
        return None
    
    print(f"✓ 加载数据: {data_file}")
    data = loadmat(data_file)
    
    # 提取多视图数据
    X = []
    for k in range(views):
        key = f'X{k+1}'
        if key in data:
            X.append(data[key].astype(float))
        else:
            print(f'  ✗ 警告: 数据文件中没有找到 {key}')
            return None
    
    # 提取真实标签
    gt = None
    for label_key in ['gt', 'gnd', 'y']:
        if label_key in data:
            gt = data[label_key].flatten()
            break
    
    if gt is None:
        print('✗ 错误: 找不到真实标签！')
        return None
    
    print(f"✓ 数据加载成功:")
    print(f"  视图数: {views}")
    print(f"  样本数: {X[0].shape[1]}")
    print(f"  特征维度: {[x.shape[0] for x in X]}")
    print(f"  类别数: {cls_num}")
    
    # 数据归一化（保持在CPU上，每次lambda测试前再转移到GPU）
    # 这样可以避免大数据集在GPU上长时间复用导致的状态问题
    print(f"✓ 数据归一化...")
    X_normalized = []
    for iv in range(views):
        X_normalized.append(normalize_data(X[iv]))  # 保持在CPU上
    
    # 生成lambda测试值列表
    lambda_min = max(0, optimal_lambda - lambda_range)
    lambda_max = optimal_lambda + lambda_range
    lambda_values = np.arange(lambda_min, lambda_max + lambda_step, lambda_step)
    lambda_values = np.round(lambda_values, 4)  # 保留4位小数
    
    # 确保最优lambda在测试列表中（如果不在，则添加）
    tolerance = 1e-5  # 容差，用于浮点数比较
    if not np.any(np.abs(lambda_values - optimal_lambda) < tolerance):
        lambda_values = np.append(lambda_values, optimal_lambda)
        lambda_values = np.sort(lambda_values)
        lambda_values = np.unique(lambda_values)  # 去重
    
    print(f"\n将测试 {len(lambda_values)} 个lambda值")
    print(f"Lambda值: {lambda_values[:5]} ... {lambda_values[-5:]}")
    print()
    
    # 存储结果
    results = {
        'lambda_values': lambda_values,
        'ACC': [],
        'NMI': [],
        'AR': [],
        'F_score': [],
        'Recall': [],
        'Precision': [],
        'run_times': []
    }
    
    # 对每个lambda值进行测试
    total_lambdas = len(lambda_values)
    for lambda_idx, lambda_val in enumerate(lambda_values):
        print(f"[{lambda_idx+1:3d}/{total_lambdas}] Lambda = {lambda_val:.4f} ", end='', flush=True)
        
        # 存储当前lambda的所有运行结果
        acc_list = []
        nmi_list = []
        ar_list = []
        f_list = []
        recall_list = []
        precision_list = []
        time_list = []
        
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
            'lambda': lambda_val
        }
        
        # 清理GPU内存（每个lambda测试前）
        if GPU_AVAILABLE:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
        
        # 多次运行
        for run_idx in range(num_runs):
            # 设置随机种子（可选，用于可重复性）
            np.random.seed(42 + run_idx)
            
            # 重要：对于大数据集（如YaleB），每次运行前重新从CPU数据创建GPU数据
            # 这样可以避免GPU内存状态累积导致的问题
            Y = []
            for iv in range(views):
                Y.append(to_gpu(X_normalized[iv]))  # 每次运行前重新转移到GPU
            
            # 清理GPU内存（如果使用GPU）
            if GPU_AVAILABLE:
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass
            
            # 运行算法
            run_start = time.time()
            C, S, Out = alg_t_ff_msc(Y, cls_num, gt, opts)
            elapsed = time.time() - run_start
            
            # 清理GPU内存（运行后）
            if GPU_AVAILABLE:
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass
            
            # 记录结果
            acc_list.append(Out['ACC'])
            nmi_list.append(Out['NMI'])
            ar_list.append(Out['AR'])
            f_list.append(Out['fscore'])
            recall_list.append(Out['recall'])
            precision_list.append(Out['precision'])
            time_list.append(elapsed)
        
        # 计算平均值
        results['ACC'].append(np.mean(acc_list))
        results['NMI'].append(np.mean(nmi_list))
        results['AR'].append(np.mean(ar_list))
        results['F_score'].append(np.mean(f_list))
        results['Recall'].append(np.mean(recall_list))
        results['Precision'].append(np.mean(precision_list))
        results['run_times'].append(np.mean(time_list))
        
        # 显示结果
        print(f"| ACC: {np.mean(acc_list):.4f}  NMI: {np.mean(nmi_list):.4f}  "
              f"F: {np.mean(f_list):.4f}  Time: {np.mean(time_list):.2f}s")
    
    print()
    print("=" * 100)
    print("敏感性分析完成！")
    print("=" * 100)
    
    return results


def plot_sensitivity_results(results, dataset_name, optimal_lambda, save_folder='plots'):
    """
    绘制敏感性分析结果
    
    注意：绘图功能已被移除，只保存文本结果
    """
    if results is None:
        return
    
    # 创建保存文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    lambda_values = results['lambda_values']
    
    # 保存详细结果到文本文件
    save_txt_path = os.path.join(save_folder, f'lambda_sensitivity_{dataset_name}.txt')
    with open(save_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Lambda Sensitivity Analysis Results - {dataset_name}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Optimal Lambda: {optimal_lambda:.4f}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Lambda':<12} {'ACC':<10} {'NMI':<10} {'AR':<10} {'F-score':<10} {'Recall':<10} {'Precision':<10}\n")
        f.write("-" * 80 + "\n")
        for i, lam in enumerate(lambda_values):
            f.write(f"{lam:<12.4f} {results['ACC'][i]:<10.4f} {results['NMI'][i]:<10.4f} "
                   f"{results['AR'][i]:<10.4f} {results['F_score'][i]:<10.4f} "
                   f"{results['Recall'][i]:<10.4f} {results['Precision'][i]:<10.4f}\n")
    print(f"✓ 详细结果已保存: {save_txt_path}")
    print("  注意：绘图功能已被移除")


def main():
    """主函数"""
    # 显示GPU信息
    print("=" * 100)
    print("Lambda参数敏感性分析")
    print("=" * 100)
    print("\n" + get_gpu_info())
    print()
    
    # 设置随机种子
    np.random.seed(42)
    
    # 数据路径
    data_path = 'data/'
    
    # ========================================
    # 配置区域 - 在这里修改测试参数
    # ========================================
    
    # 数据集配置
    datasets = [
        {
            'file': 'yale.mat',
            'name': 'Yale',
            'views': 3,
            'optimal_lambda': 0.221,  # 最优lambda值
            'cls_num': 15,
            'description': '165样本, 15类'
        },
        {
            'file': 'yaleB.mat',
            'name': 'Extended YaleB',
            'views': 3,
            'optimal_lambda': 0.001,
            'cls_num': 10,
            'description': '2414样本, 38类'
        },
        {
            'file': 'ORL.mat',
            'name': 'ORL',
            'views': 3,
            'optimal_lambda': 0.1,
            'cls_num': 40,
            'description': '400样本, 40类'
        },
        {
            'file': 'COIL20MV.mat',
            'name': 'COIL-20',
            'views': 3,
            'optimal_lambda': 0.001,
            'cls_num': 20,
            'description': '1440样本, 20类'
        },
    ]
    
    # 选择要分析的数据集（通过索引）
    test_list = [3]  # 0=Yale, 1=YaleB, 2=ORL, 3=COIL-20
    
    # 敏感性分析参数
    num_runs = 10  # 每个lambda值运行的次数（建议10-20次）
    lambda_range = 0.1  # 在最优值±lambda_range范围内测试
    lambda_step = 0.01  # lambda测试步长（建议0.01或0.02）
    
    print("=" * 100)
    print("测试配置:")
    print(f"  数据路径: {data_path}")
    print(f"  每个lambda运行次数: {num_runs}")
    print(f"  Lambda测试范围: ±{lambda_range}")
    print(f"  Lambda测试步长: {lambda_step}")
    print(f"  GPU加速: {'启用' if GPU_AVAILABLE else '禁用（CPU模式）'}")
    print("\n可用数据集:")
    for i, ds in enumerate(datasets):
        marker = "→" if i in test_list else " "
        print(f"  {marker} [{i}] {ds['name']:<20} - {ds['description']:<20} (最优λ={ds['optimal_lambda']})")
    print(f"\n将分析: {[datasets[i]['name'] for i in test_list]}")
    print("=" * 100)
    print()
    
    # 遍历要分析的数据集
    for dataset_idx in test_list:
        ds = datasets[dataset_idx]
        
        # 进行敏感性分析
        results = lambda_sensitivity_analysis(
            dataset_file=ds['file'],
            dataset_name=ds['name'],
            optimal_lambda=ds['optimal_lambda'],
            cls_num=ds['cls_num'],
            views=ds['views'],
            num_runs=num_runs,
            lambda_range=lambda_range,
            lambda_step=lambda_step,
            data_path=data_path
        )
        
        if results is not None:
            # 绘制结果
            plot_sensitivity_results(results, ds['name'], ds['optimal_lambda'])
            
            # 打印统计信息
            print("\n" + "=" * 100)
            print("结果统计:")
            print("=" * 100)
            
            # 找到最优lambda对应的索引
            optimal_idx = np.argmin(np.abs(results['lambda_values'] - ds['optimal_lambda']))
            
            print(f"\n最优lambda ({ds['optimal_lambda']:.4f}) 的性能:")
            print(f"  ACC: {results['ACC'][optimal_idx]:.4f}")
            print(f"  NMI: {results['NMI'][optimal_idx]:.4f}")
            print(f"  F-score: {results['F_score'][optimal_idx]:.4f}")
            
            # 找到最佳性能对应的lambda
            best_acc_idx = np.argmax(results['ACC'])
            best_nmi_idx = np.argmax(results['NMI'])
            best_f_idx = np.argmax(results['F_score'])
            
            print(f"\n各指标最佳lambda:")
            print(f"  最佳ACC: λ = {results['lambda_values'][best_acc_idx]:.4f}, ACC = {results['ACC'][best_acc_idx]:.4f}")
            print(f"  最佳NMI: λ = {results['lambda_values'][best_nmi_idx]:.4f}, NMI = {results['NMI'][best_nmi_idx]:.4f}")
            print(f"  最佳F-score: λ = {results['lambda_values'][best_f_idx]:.4f}, F = {results['F_score'][best_f_idx]:.4f}")
            
            # 计算性能变化范围
            acc_range = np.max(results['ACC']) - np.min(results['ACC'])
            nmi_range = np.max(results['NMI']) - np.min(results['NMI'])
            f_range = np.max(results['F_score']) - np.min(results['F_score'])
            
            print(f"\n性能变化范围:")
            print(f"  ACC范围: [{np.min(results['ACC']):.4f}, {np.max(results['ACC']):.4f}], 变化幅度: {acc_range:.4f}")
            print(f"  NMI范围: [{np.min(results['NMI']):.4f}, {np.max(results['NMI']):.4f}], 变化幅度: {nmi_range:.4f}")
            print(f"  F-score范围: [{np.min(results['F_score']):.4f}, {np.max(results['F_score']):.4f}], 变化幅度: {f_range:.4f}")
            
            print("=" * 100)
    
    print("\n" + "=" * 100)
    print("所有分析完成！")
    print("=" * 100)


if __name__ == '__main__':
    main()

