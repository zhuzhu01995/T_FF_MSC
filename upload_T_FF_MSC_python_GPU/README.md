# T-FF-MSC Python GPU加速实现

这是 **T-FF-MSC (Tensor-based Fractional-order Function Multi-view Subspace Clustering)** 算法的 **GPU加速版本**，使用 CuPy 实现显著的性能提升。

## ⚠️ 重要说明：代码与论文的符号对应关系

**请注意**：代码中的符号与论文中的符号存在一些差异，具体对应关系如下：

| 代码中的符号 | 论文中的符号 | 说明 |
|------------|------------|------|
| `mu` | $\rho$ | 惩罚参数 |
| `rho` | $\mu$ | 惩罚参数 |
| `W` | $\mathbf{M}$ | 辅助变量（拉格朗日乘数） |

**示例**：
- 代码中的 `mu` 参数对应论文中的 $\rho$
- 代码中的 `rho` 参数对应论文中的 $\mu$
- 代码中的变量 `W` 对应论文中的矩阵 $\mathbf{M}$

在使用代码时，请根据上述对应关系理解参数含义。

---

## ✨ 特性

- 🚀 **GPU加速**: 使用CuPy在GPU上进行张量运算、FFT、SVD等计算
- 🔄 **自动回退**: 无GPU时自动回退到NumPy CPU模式
- 📦 **模块化设计**: 易于扩展和维护
- 🎯 **支持多视图子空间聚类**: 完整的多视图学习算法
- 📊 **完整的评估指标**: NMI、ACC、F-score等
- ⚡ **显著性能提升**: 在大型数据集上可获得10-50倍加速
- 📈 **可视化工具**: 包含混淆矩阵、收敛曲线、亲和矩阵等可视化功能

## 🆚 GPU vs CPU版本对比

| 特性 | CPU版本 | GPU版本 |
|------|---------|---------|
| 矩阵运算 | NumPy (CPU) | CuPy (GPU) |
| FFT/IFFT | NumPy FFT | CuPy FFT (cuFFT) |
| SVD分解 | NumPy/SciPy | CuPy (cuSOLVER) |
| 线性求解 | NumPy | CuPy (cuSOLVER) |
| 性能 | 基线 | 10-50x 加速* |

*性能提升取决于数据集大小和GPU型号

## 📁 项目结构

```
T_FF_MSC_python_GPU/
├── algs/                    # GPU加速算法模块
│   ├── alg_t_ff_msc.py     # 主算法（GPU）
│   ├── frac_shrink.py      # Fractional收缩算子（GPU）
│   ├── frac_update_sigma.py # 奇异值更新（GPU）
│   ├── glu.py              # 分数阈值函数（GPU）
│   └── solve_e_problem.py  # E子问题求解器（GPU）
├── utils/                   # 工具函数模块
│   ├── normalize_data.py   # 数据归一化（GPU）
│   ├── spectral_clustering.py # 谱聚类（混合GPU/CPU）
│   ├── accuracy.py         # 准确率计算
│   ├── best_map.py         # 标签映射
│   ├── hungarian.py        # 匈牙利算法
│   ├── compute_nmi.py      # NMI计算
│   ├── compute_f.py        # F-score计算
│   ├── contingency.py      # 列联表
│   └── rand_index.py       # Rand指数
├── data/                    # 数据文件夹
├── plots/                   # 图片保存文件夹（自动生成）
├── gpu_utils.py            # GPU工具模块（自动检测和回退）
├── test_T_FF_MSC_GPU.py    # 完整测试脚本
├── plot_results.py         # 绘图结果模块
├── visualization.py        # 可视化函数模块
├── example.py              # 使用示例
├── quick_test.py           # 快速测试
├── requirements.txt        # Python依赖
└── README.md              # 项目说明
```

## 🚀 快速开始

### 1. 安装依赖

#### 基础依赖（必需）
```bash
pip install numpy scipy scikit-learn matplotlib
```

#### GPU支持（可选，强烈推荐）

**要求**: NVIDIA GPU + CUDA Toolkit

根据您的CUDA版本安装CuPy：

```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x
```

详细安装说明: https://docs.cupy.dev/en/stable/install.html

> **注意**: 如果不安装CuPy，程序会自动使用CPU模式运行。

#### 可选依赖（用于交互式可视化）
```bash
pip install plotly  # 用于生成交互式3D收敛曲线HTML
```

### 2. 准备数据

将数据文件（.mat格式）放在 `data/` 目录下。支持的数据集包括：
- Yale
- Extended YaleB
- ORL
- COIL-20
- 其他多视图数据集

### 3. 运行快速测试

```bash
python quick_test.py
```

### 4. 运行完整测试

```bash
python test_T_FF_MSC_GPU.py
```

在 `test_T_FF_MSC_GPU.py` 中可以配置：
- `test_list`: 选择要测试的数据集
- `num_runs`: 运行次数
- `enable_plotting`: 是否启用绘图功能
- `save_plots`: 是否保存图片（True）或显示图片（False）
- `plots_folder`: 图片保存文件夹名称

### 5. 运行示例

```bash
python example.py
```

## 💡 使用示例

```python
import numpy as np
from scipy.io import loadmat
from utils import normalize_data
from algs import alg_t_ff_msc
from gpu_utils import to_gpu, get_gpu_info

# 显示GPU信息
print(get_gpu_info())

# 加载数据
data = loadmat('data/yale.mat')
X = [data['X1'], data['X2'], data['X3']]
gt = data['gt'].flatten()

# 数据归一化并转移到GPU
Y = [to_gpu(normalize_data(x)) for x in X]

# 设置参数
# 注意：代码中的mu对应论文中的ρ，代码中的rho对应论文中的μ
opts = {
    'Frac_alpha': 5000,
    'maxIter': 200,
    'epsilon': 1e-4,
    'lambda': 0.221,  # Yale数据集的最佳参数
    'mu': 1e-5,       # 对应论文中的ρ
    'rho': 1e-5,      # 对应论文中的μ
    'eta': 2,
    'max_mu': 1e10,
    'max_rho': 1e10
}

# 运行算法（自动在GPU上运行）
cls_num = len(np.unique(gt))
C, S, Out = alg_t_ff_msc(Y, cls_num, gt, opts)

# 查看结果
print(f"NMI: {Out['NMI']:.4f}")
print(f"ACC: {Out['ACC']:.4f}")
print(f"F-score: {Out['fscore']:.4f}")
```

## 📊 可视化功能

项目包含完整的可视化工具，可以生成以下图表：

1. **混淆矩阵** (Confusion Matrix)
   - 显示预测类别与真实类别的对应关系
   - 值为0的单元格保持空白
   - 包含网格线

2. **收敛曲线** (Convergence History)
   - 3D曲线图显示算法收敛过程
   - 显示两个停止准则：||Z - G||_∞ 和 ||(X-E) - (X-E)Z||_∞
   - 支持交互式HTML格式（可拖动旋转）

3. **亲和矩阵** (Affinity Matrix)
   - 热力图显示样本间的相似度
   - 颜色从深蓝紫色渐变到黄色
   - 颜色条刻度为0.5间隔

### 使用可视化功能

```python
from plot_results import plot_all_results_for_dataset

# 在测试完成后调用
plot_all_results_for_dataset(
    gt=gt,
    best_C=best_C,
    best_history=best_history,
    all_S_matrices=all_S_matrices,
    dataset_name='Yale',
    save_folder='plots',
    save_plots=True
)
```

所有图片会保存在 `plots/` 文件夹中。

## 🎯 数据集

项目支持以下数据集：

| 数据集 | 样本数 | 视图数 | 类别数 | 推荐lambda | GPU加速 |
|--------|--------|--------|--------|-----------|---------|
| Yale | 165 | 3 | 15 | 0.221 | ~5-10x |
| Extended YaleB | 2414 | 3 | 38 | 0.001 | ~20-30x |
| ORL | 400 | 3 | 40 | 0.219 | ~10-15x |
| COIL-20 | 1440 | 3 | 20 | 0.001 | ~30-50x |

*加速比基于NVIDIA RTX 3090测试

## 🔧 GPU工具API

### 自动GPU/CPU切换

```python
from gpu_utils import xp, to_gpu, to_cpu, GPU_AVAILABLE

# xp 自动指向 cupy 或 numpy
arr = xp.zeros((100, 100))  # 在GPU上创建（如果可用）

# 手动转换
gpu_arr = to_gpu(cpu_arr)   # CPU -> GPU
cpu_arr = to_cpu(gpu_arr)   # GPU -> CPU

# 检查GPU可用性
if GPU_AVAILABLE:
    print("使用GPU加速")
else:
    print("使用CPU模式")
```

### 获取GPU信息

```python
from gpu_utils import get_gpu_info

print(get_gpu_info())
```

## 📊 性能优化

GPU版本的主要优化包括：

1. **张量运算**: 所有矩阵乘法、求逆在GPU上进行
2. **FFT/IFFT**: 使用cuFFT库加速频域变换
3. **SVD分解**: 使用cuSOLVER加速奇异值分解
4. **向量化操作**: 减少Python循环，充分利用GPU并行性
5. **内存管理**: 智能的CPU-GPU数据传输

## 🔬 GPU加速算法详解

### 核心加速部分

1. **Fractional收缩算子** (`frac_shrink.py`)
   - GPU FFT/IFFT
   - 并行SVD分解
   - 张量运算

2. **E子问题求解** (`solve_e_problem.py`)
   - 向量化近端算子
   - 批量矩阵运算
   - FISTA加速

3. **主算法** (`alg_t_ff_msc.py`)
   - GPU上的矩阵求解
   - 张量堆叠和切片
   - 并行更新

## 🐛 故障排除

### CuPy安装问题

```bash
# 检查CUDA版本
nvcc --version

# 或
nvidia-smi

# 根据CUDA版本安装对应的CuPy
pip install cupy-cuda11x  # 或 cupy-cuda12x
```

### 内存不足

如果遇到GPU内存不足：

1. 减少批量大小
2. 使用较小的数据集
3. 关闭其他GPU程序

### CPU回退

如果GPU不可用，程序会自动使用CPU模式。检查输出：

```
⚠ 未检测到CuPy，将使用NumPy（CPU模式）运行
```

### 绘图问题

如果遇到matplotlib后端问题：

- 程序默认会保存图片到 `plots/` 文件夹
- 如果显示失败，会自动保存为PNG文件
- 确保已安装 matplotlib: `pip install matplotlib`

## 📝 参数说明

### 主要参数

- `lambda`: 正则化参数，控制稀疏性
- `mu`: 惩罚参数（对应论文中的 $\rho$）
- `rho`: 惩罚参数（对应论文中的 $\mu$）
- `maxIter`: 最大迭代次数
- `epsilon`: 收敛容差
- `Frac_alpha`: Fractional阶参数
- `eta`: 惩罚参数增长因子

### 随机种子设置

在 `test_T_FF_MSC_GPU.py` 中可以设置：

- `random_seed_mode = 'fixed'`: 固定种子（结果可重复，适合论文实验）
- `random_seed_mode = 'variable'`: 变化种子（可评估算法稳定性）
- `random_seed_mode = None`: 系统随机（每次运行都不同）

## 📝 许可证

MIT License

## 🙏 致谢

本项目基于原始MATLAB实现和CPU版Python实现，通过CuPy实现GPU加速。

## 📧 问题反馈

如果您在使用过程中遇到任何问题，欢迎提出Issue。

---

**GPU加速效果**: 在配备NVIDIA GPU的系统上，性能可提升10-50倍（取决于数据集大小）。即使没有GPU，也可以在CPU模式下正常运行。

**符号对应关系**: 请务必注意代码与论文中的符号差异，详见本文档开头的"重要说明"部分。
