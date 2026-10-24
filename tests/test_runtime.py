"""Checks for numerical configuration and CUDA device selection."""

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

from scpls.inference import configure_device
from scpls.runtime import configure_numeric_runtime


class RuntimeTests(unittest.TestCase):
    def test_tf32_is_disabled_before_torch_import(self):
        program = '''
import builtins
import os
original_import = builtins.__import__
observed = []
def checked_import(name, *args, **kwargs):
    if name == 'torch' and not observed:
        observed.append(os.environ.get('NVIDIA_TF32_OVERRIDE'))
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from scpls.runtime import configure_numeric_runtime
configure_numeric_runtime()
assert observed == ['0'], observed
'''
        env = dict(os.environ, NVIDIA_TF32_OVERRIDE='1')
        subprocess.run([sys.executable, '-c', program],
                       cwd=Path(__file__).resolve().parent.parent,
                       env=env, check=True, capture_output=True, text=True)

    def test_fp32_flags_and_report_agree(self):
        previous = (torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32)
        try:
            with patch.dict(os.environ, {'NVIDIA_TF32_OVERRIDE': '1'}):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                report = configure_numeric_runtime()
                self.assertEqual(os.environ['NVIDIA_TF32_OVERRIDE'], '0')
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                self.assertEqual(report, {
                    'inference_dtype': 'float32',
                    'nvidia_tf32_override': '0',
                    'cuda_matmul_allow_tf32': False,
                    'cudnn_allow_tf32': False,
                })
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous[0]
            torch.backends.cudnn.allow_tf32 = previous[1]

    def test_cuda_without_index_selects_zero(self):
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device') as set_device, \
             patch('torch.cuda.manual_seed_all'):
            device = configure_device('cuda')
        self.assertEqual(device, torch.device('cuda:0'))
        set_device.assert_called_once_with(torch.device('cuda:0'))


if __name__ == '__main__':
    unittest.main()
