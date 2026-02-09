# T-FF-MSC Python GPU Accelerated Implementation

**T-FF-MSC (Tensor-based Fractional-order Function Multi-view Subspace Clustering)** GPU-accelerated version using CuPy for significant performance improvement.

## ⚠️ Important Note: Symbol Correspondence

There are differences between the symbols in the code and those in the paper:

| Code Symbol | Paper Symbol | Description |
|------------|------------|-------------|
| `mu` | $\rho$ | Penalty parameter |
| `rho` | $\mu$ | Penalty parameter |
| `W` | $\mathbf{M}$ | Auxiliary variable (Lagrange multiplier) |

## Installation

### Basic Dependencies
```bash
pip install numpy scipy scikit-learn
```

### GPU Support (Optional, Highly Recommended)
```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x
```

> If CuPy is not installed, the program will automatically run in CPU mode.

## Quick Start

### 1. Prepare Data
Place data files (.mat format) in the `data/` directory.

### 2. Run Tests
```bash
python test_T_FF_MSC_GPU.py
```

Configure in `test_T_FF_MSC_GPU.py`:
- `test_list`: Select datasets to test (0=Yale, 1=YaleB, 2=ORL, 3=COIL-20)
- `num_runs`: Number of runs

### 3. Run Example
```bash
python example.py
```

## Usage Example

```python
import numpy as np
from scipy.io import loadmat
from utils import normalize_data
from algs import alg_t_ff_msc
from gpu_utils import to_gpu

# Load data
data = loadmat('data/yale.mat')
X = [data['X1'], data['X2'], data['X3']]
gt = data['gt'].flatten()

# Normalize data and transfer to GPU
Y = [to_gpu(normalize_data(x)) for x in X]

# Set parameters
opts = {
    'Frac_alpha': 5000,
    'maxIter': 200,
    'epsilon': 1e-4,
    'lambda': 0.221,  # Optimal parameter for Yale dataset
    'mu': 1e-5,       # Corresponds to ρ in the paper
    'rho': 1e-5,      # Corresponds to μ in the paper
    'eta': 2,
    'max_mu': 1e10,
    'max_rho': 1e10
}

# Run algorithm
cls_num = len(np.unique(gt))
C, S, Out = alg_t_ff_msc(Y, cls_num, gt, opts)

# View results
print(f"NMI: {Out['NMI']:.4f}")
print(f"ACC: {Out['ACC']:.4f}")
print(f"F-score: {Out['fscore']:.4f}")
```

## Datasets

| Dataset | Samples | Views | Classes | Recommended lambda |
|---------|---------|-------|---------|-------------------|
| Yale | 165 | 3 | 15 | 0.221 |
| Extended YaleB | 2414 | 3 | 38 | 0.001 |
| ORL | 400 | 3 | 40 | 0.1 |
| COIL-20 | 1440 | 3 | 20 | 0.001 |

## Project Structure

```
T_FF_MSC_python_GPU/
├── algs/                    # GPU-accelerated algorithm modules
├── utils/                   # Utility function modules
├── data/                    # Data directory
├── gpu_utils.py             # GPU utility module
├── test_T_FF_MSC_GPU.py     # Test script
├── example.py               # Usage example
└── requirements.txt         # Dependencies list
```

## License

MIT License
