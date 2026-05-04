#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

"""
NCCL4Py: Python bindings for NVIDIA Collective Communications Library (NCCL).

NCCL4Py provides Pythonic access to NCCL for efficient multi-GPU and multi-node
communication. It supports all NCCL collective operations, point-to-point
communication, and advanced features like buffer registration and custom reduction
operators.
"""

from nccl._version import __version__

# Register the local HIP-backed cuda.core shim under the `cuda.core`
# namespace as a fallback for the case where the user reaches nccl
# before importing cuda.core (e.g. via `from nccl.core import ...`).
# A real on-disk `cuda/core/__init__.py` shipped alongside this package
# also handles the public path `from cuda.core import ...`; whichever
# loads first wins, both routes are idempotent.
from nccl._hip_compat.cuda_core_shim import _register_as_cuda_core as _register_cuda_core_shim

_register_cuda_core_shim()
del _register_cuda_core_shim


# Re-export get_version() lazily. Importing nccl alone does not load
# librccl.so — that happens only on first access of ``nccl.get_version``
# or any ``nccl.bindings.*`` symbol. Keeps `import nccl` cheap so it can
# safely run as a side-effect of cuda.core resolution
# (`cuda/core/__init__.py` imports nccl._hip_compat).
def __getattr__(name):
    if name == "get_version":
        from nccl.bindings import get_version

        return get_version
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "get_version",
]
