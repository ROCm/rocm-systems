#!/usr/bin/env python3
"""Thin wrapper to invoke mnctl as a module.

Usage:
    python3 run_mnctl.py [OPTIONS] [ROCM_IMAGE]

Equivalent to:
    python3 -m mnctl [OPTIONS] [ROCM_IMAGE]

This script exists so that remote nodes can be invoked with a single
path (``python3 /path/to/docker/run_mnctl.py --run ...``) without
needing PYTHONPATH or ``-m`` syntax.
"""

import os
import sys

# Ensure the docker/ directory is on sys.path so that
# ``import mnctl`` resolves to the local package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mnctl.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
