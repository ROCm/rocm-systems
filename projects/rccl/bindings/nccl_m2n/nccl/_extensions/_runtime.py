# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

"""Process-wide runtime state shared by extension facades and bindings."""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

NATIVE_CALL_LOCK = threading.RLock()


def is_rocm() -> bool:
    """Return True when nccl4py is the RCCL port (ships ``nccl._hip_compat``).

    This checks which nccl4py is installed, not the GPU runtime or the loaded
    library: an RCCL-port nccl4py selects the ROCm paths (``lib/rocm``, no
    ``cuda.*`` imports, the RCCL load-time check).
    """
    return importlib.util.find_spec("nccl._hip_compat") is not None


def cuda_major() -> int:
    """Return the CUDA userspace major selected by ``cuda.bindings``."""
    from cuda import bindings

    major = bindings.__version__.split(".", 1)[0]
    if major not in {"12", "13"}:
        raise RuntimeError("cuda.bindings 12.x or 13.x must be installed")
    return int(major)


def bundled_library(libname: str) -> str | None:
    """Return the bundled library matching the installed runtime (``cu12``/``cu13``/``rocm``)."""
    package_dir = Path(__file__).resolve().parent.parent / libname.removeprefix("nccl_")
    variant = "rocm" if is_rocm() else f"cu{cuda_major()}"
    library = package_dir / "lib" / variant / f"lib{libname}.so"
    return str(library) if library.is_file() else None


__all__ = ["NATIVE_CALL_LOCK", "bundled_library", "cuda_major", "is_rocm"]
