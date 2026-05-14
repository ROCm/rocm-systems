#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

"""
nccl4py: Python bindings for RCCL on ROCm (AMD fork of NVIDIA nccl4py).

nccl4py provides Pythonic access to RCCL (the ROCm Communication Collectives
Library, AMD's drop-in replacement for NVIDIA NCCL) for efficient multi-GPU
and multi-node communication. It supports all NCCL/RCCL collective operations,
point-to-point communication, and advanced features like buffer registration
and custom reduction operators.
"""

from nccl._version import __version__

# Register the local HIP-backed cuda.core shim under the `cuda.core`
# namespace via sys.modules. Idempotent and gated on
# `"cuda.core" not in sys.modules`, so it is a no-op if a real
# cuda-core / cuda-bindings package has already populated the
# namespace. After this runs, `from cuda.core import ...` resolves
# to the HIP shim. The shim is **not** shipped as a regular
# top-level `cuda` package on disk (a regular non-PEP-420
# `cuda/__init__.py` would shadow co-installed distributions that
# legitimately contribute to `cuda.*`), so users must `import nccl`
# before reaching `cuda.core`.
from nccl._hip_compat.cuda_core_shim import _register_as_cuda_core as _register_cuda_core_shim

_register_cuda_core_shim()
del _register_cuda_core_shim


# Re-export get_version() lazily. Importing nccl alone does not load
# librccl.so — that happens only on first access of ``nccl.get_version``
# or any ``nccl.bindings.*`` symbol. Keeps `import nccl` cheap.
def __getattr__(name):
    if name == "get_version":
        from nccl.bindings import get_version

        return get_version
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "get_version",
]
