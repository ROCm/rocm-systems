# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep a single librccl in the process when running nccl.m2n with torch on ROCm.

TheRock torch wheels preload their bundled ``librccl.so.1`` by full path through
``rocm_sdk.preload_libraries``, which maps a second RCCL even when another one is
already loaded with the same SONAME (``LD_PRELOAD``). When
``NCCL_M2N_RCCL_LIBRARY`` names the RCCL that ``libnccl_m2n`` links, load it
first and register it in rocm_sdk's preload cache so torch reuses it.
Active only when this directory is on ``PYTHONPATH``.

Test-only: ``rocm_sdk._ALL_CDLLS`` is private and a TheRock update can break
this without warning. It goes away once torch's RCCL has the APIs nccl_m2n needs.
"""

import os

_rccl_path = os.environ.get("NCCL_M2N_RCCL_LIBRARY")
if _rccl_path:
    import ctypes

    _rccl = ctypes.CDLL(_rccl_path, mode=ctypes.RTLD_GLOBAL)
    try:
        import rocm_sdk
    except ImportError:
        pass
    else:
        rocm_sdk._ALL_CDLLS.setdefault("rccl", _rccl)
