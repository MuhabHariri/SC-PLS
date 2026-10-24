#!/usr/bin/env python3
"""Run SC-PLS evaluation or frozen pseudo-label selection."""

import os
os.environ['NVIDIA_TF32_OVERRIDE']='0'
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

from scpls.cli import main

if __name__=='__main__':
    raise SystemExit(main())
