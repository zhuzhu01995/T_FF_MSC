"""T-FF-MSC GPU加速版本"""
from .gpu_utils import GPU_AVAILABLE, get_gpu_info

__version__ = '1.0.0-gpu'
__all__ = ['GPU_AVAILABLE', 'get_gpu_info']

# 打印GPU状态
if GPU_AVAILABLE:
    print(f"T-FF-MSC GPU版本已加载 (v{__version__})")
else:
    print(f"T-FF-MSC CPU版本已加载 (v{__version__}) - GPU不可用")

