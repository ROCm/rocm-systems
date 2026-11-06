# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load-time checks of ``libnccl_m2n.so`` on ROCm: the RCCL it binds to and the GPU archs.

A torch wheel from TheRock loads its own ``librccl.so.1`` (RCCL 2.28.9), which
lacks APIs that nccl_m2n calls. If that copy is loaded first, libnccl_m2n binds
to it, or the process ends up with two RCCLs, and fails much later. A GPU arch
missing from the library fails only at the first kernel launch. Check right
after ``dlopen`` and raise a clear error instead.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

# Mirrors NCCL_M2N_MIN_NCCL_VERSION_CODE in nccl_m2n.h.
MIN_RCCL_VERSION_CODE = 23005
REQUIRED_SYMBOLS = (
    "ncclPutSignal",
    "ncclSignal",
    "ncclWaitSignal",
    "ncclGetLsaDevicePointer",
    "ncclCommQueryProperties",
)

_HINT = (
    "nccl.m2n needs the RCCL that libnccl_m2n.so was built against, loaded before torch. "
    "Load it first with LD_PRELOAD=<librccl.so.1> and NCCL_M2N_RCCL_LIBRARY=<same path> with "
    "projects/rccl/bindings/nccl_m2n/rocm on PYTHONPATH (test-only sitecustomize.py); "
    "see projects/rccl/contrib/nccl_m2n/scripts/README.md."
)


# Written next to the bundled library by the build (contrib/nccl_m2n/scripts/build.sh).
GPU_ARCHS_FILE = "gpu_archs.txt"


def _version_string(code: int) -> str:
    return f"{code // 10000}.{code // 100 % 100}.{code % 100}"


def loaded_rccl_paths() -> list[str]:
    """Distinct ``librccl.so*`` files mapped into this process, from ``/proc/self/maps``."""
    seen: dict[tuple[int, int] | str, str] = {}
    try:
        with open("/proc/self/maps") as maps:
            lines = maps.readlines()
    except OSError:
        return []
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        path = fields[5].strip()
        if not os.path.basename(path).startswith("librccl.so"):
            continue
        try:
            st = os.stat(path)
            key: tuple[int, int] | str = (st.st_dev, st.st_ino)
        except OSError:
            key = path
        seen.setdefault(key, path)
    return list(seen.values())


def describe_loaded_rccl() -> str:
    paths = loaded_rccl_paths()
    return "loaded librccl: " + (", ".join(paths) if paths else "none")


def load_failure_detail() -> str:
    """Context for a failed ``dlopen`` of libnccl_m2n (e.g. undefined RCCL symbols)."""
    return f"{describe_loaded_rccl()}. {_HINT}"


def _mapped_paths() -> list[str]:
    try:
        with open("/proc/self/maps") as maps:
            lines = maps.readlines()
    except OSError:
        return []
    paths = []
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            paths.append(fields[5].strip())
    return paths


def loaded_library_path(library: str) -> str | None:
    """Real path of the mapped object ``library`` names; a bare name is looked up in /proc/self/maps."""
    if os.path.isabs(library):
        return os.path.realpath(library)
    name = os.path.basename(library)
    for path in _mapped_paths():
        if os.path.basename(path) == name:
            return os.path.realpath(path)
    return None


def _base_arch(target: str) -> str:
    return target.split(":", 1)[0]


def library_gpu_archs(m2n_library: str) -> list[str] | None:
    """Base GPU archs ``m2n_library`` was built for, from ``gpu_archs.txt`` next to the loaded file."""
    path = loaded_library_path(m2n_library)
    if path is None:
        return None
    try:
        text = (Path(path).parent / GPU_ARCHS_FILE).read_text()
    except OSError:
        return None
    return sorted({_base_arch(arch) for arch in text.replace(";", " ").split() if arch})


def visible_gpu_archs() -> list[str] | None:
    """Base arch (e.g. ``gfx950``) of every visible GPU, via hip-python (None if unavailable)."""
    try:
        from hip import hip

        err, count = hip.hipGetDeviceCount()
        if int(err) != 0:
            return None
        archs = []
        for device in range(count):
            props = hip.hipDeviceProp_t()
            (err,) = hip.hipGetDeviceProperties(props, device)
            if int(err) != 0:
                return None
            name = props.gcnArchName
            if isinstance(name, bytes):
                name = name.decode()
            archs.append(_base_arch(name))
        return archs
    except Exception:
        # The arch check is advisory: never block loading on a hip-python API difference.
        return None


def check_gpu_archs(m2n_library: str) -> None:
    """Raise RuntimeError if a visible GPU has an arch ``m2n_library`` was not built for."""
    built = library_gpu_archs(m2n_library)
    if not built:
        return
    devices = visible_gpu_archs()
    if devices is None:
        return
    unsupported = sorted({arch for arch in devices if arch not in built})
    if unsupported:
        raise RuntimeError(
            f"{m2n_library} was built for {', '.join(built)}, but the visible GPUs include "
            f"{', '.join(unsupported)}. Rebuild libnccl_m2n (and RCCL) with these archs "
            "(OFFLOAD_ARCH / M2N_GPU_ARCHS in projects/rccl/contrib/nccl_m2n/scripts/build.sh)."
        )


def check_loaded_library(m2n_library: str) -> None:
    """All load-time checks: GPU archs, then the RCCL libnccl_m2n binds to."""
    check_gpu_archs(m2n_library)
    check_rccl(m2n_library)


def check_rccl(m2n_library: str) -> None:
    """Raise RuntimeError unless ``m2n_library`` resolves a suitable, single RCCL."""
    lib = ctypes.CDLL(m2n_library, mode=os.RTLD_NOLOAD | ctypes.RTLD_GLOBAL)
    problems = []

    paths = loaded_rccl_paths()
    if len(paths) > 1:
        problems.append(f"{len(paths)} RCCL libraries are loaded: {', '.join(paths)}")

    try:
        get_version = lib.ncclGetVersion
    except AttributeError:
        problems.append("ncclGetVersion not found")
    else:
        version = ctypes.c_int(0)
        get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        if get_version(ctypes.byref(version)) != 0:
            problems.append("ncclGetVersion failed")
        elif version.value < MIN_RCCL_VERSION_CODE:
            problems.append(
                f"RCCL {_version_string(version.value)} is older than "
                f"{_version_string(MIN_RCCL_VERSION_CODE)}"
            )

    missing = [name for name in REQUIRED_SYMBOLS if not hasattr(lib, name)]
    if missing:
        problems.append("RCCL lacks " + ", ".join(missing))

    if problems:
        raise RuntimeError(
            f"{m2n_library}: unsuitable RCCL: " + "; ".join(problems)
            + f" ({describe_loaded_rccl()}). " + _HINT
        )


__all__ = [
    "GPU_ARCHS_FILE",
    "MIN_RCCL_VERSION_CODE",
    "REQUIRED_SYMBOLS",
    "check_gpu_archs",
    "check_loaded_library",
    "check_rccl",
    "describe_loaded_rccl",
    "library_gpu_archs",
    "loaded_library_path",
    "load_failure_detail",
    "loaded_rccl_paths",
    "visible_gpu_archs",
]
