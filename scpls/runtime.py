"""Numerical settings for frozen FP32 inference."""

import os


def configure_numeric_runtime():
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        'inference_dtype': 'float32',
        'nvidia_tf32_override': os.environ['NVIDIA_TF32_OVERRIDE'],
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
    }
