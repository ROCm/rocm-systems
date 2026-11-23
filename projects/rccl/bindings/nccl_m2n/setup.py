# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

import importlib.util
import os
import sys
from pathlib import Path

from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent
M2N_PACKAGE = ROOT / "nccl" / "m2n"
BUILDING_SDIST = "sdist" in sys.argv
REQUIRE_NATIVE_LIBS = (
    os.environ.get("NCCL_EXTENSIONS_REQUIRE_NATIVE_LIBS", "0") != "0"
)


def _check_staged_libraries(package_dir: Path, library: str, package: str) -> None:
    staged = package_dir / "lib" / "rocm" / library
    if staged.is_file():
        return
    if REQUIRE_NATIVE_LIBS:
        raise SystemExit(f"Error: wheels with native libraries require {staged}")
    message = (
        f"{staged} was not found. The built wheel will require a compatible "
        "external library at runtime (NCCL_M2N_LIBRARY or NCCL_M2N_HOME)."
    )
    print(f"WARNING: {message}", file=sys.stderr)


def _rccl_nccl4py_installed() -> bool:
    # Look for nccl/_hip_compat without importing nccl: its __init__ needs numpy and hip-python,
    # which a wheel build environment (cibuildwheel) does not have.
    spec = importlib.util.find_spec("nccl")
    locations = spec.submodule_search_locations if spec is not None else None
    return any(Path(location, "_hip_compat").is_dir() for location in locations or [])


# Built against the RCCL port of nccl4py (projects/rccl/bindings/nccl4py, ships nccl._hip_compat),
# installed beforehand and used with --no-build-isolation; no CUDA headers, nccl_m2n only.
if not BUILDING_SDIST:
    from Cython.Build import cythonize

    if not _rccl_nccl4py_installed():
        raise SystemExit("Error: install the RCCL nccl4py port first and build with --no-build-isolation")
    _check_staged_libraries(M2N_PACKAGE, "libnccl_m2n.so", "nccl.m2n")

PACKAGE = "nccl._extensions.bindings"
LIBNAMES = ["nccl_m2n"]


def _ext(module: str, source: str, *, include_dirs: list[str] | None = None) -> Extension:
    return Extension(
        module,
        sources=[source],
        include_dirs=[*(include_dirs or [])],
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["dl"],
    )


def libname_extensions(libname: str) -> list[Extension]:
    """Three extensions per library: lowpp, cy variant, _internal loader.

    For libname="nccl_m2n":
        nccl._extensions.bindings.nccl_m2n           <- nccl/_extensions/bindings/nccl_m2n.pyx
        nccl._extensions.bindings.cynccl_m2n         <- nccl/_extensions/bindings/cynccl_m2n.pyx
        nccl._extensions.bindings._internal.nccl_m2n <- nccl/_extensions/bindings/_internal/nccl_m2n_linux.pyx
    """
    pkg_dir = os.path.join(*PACKAGE.split("."))
    return [
        _ext(f"{PACKAGE}.{libname}", os.path.join(pkg_dir, f"{libname}.pyx")),
        _ext(f"{PACKAGE}.cy{libname}", os.path.join(pkg_dir, f"cy{libname}.pyx")),
        _ext(
            f"{PACKAGE}._internal.{libname}",
            os.path.join(pkg_dir, "_internal", f"{libname}_linux.pyx"),
        ),
    ]


ext_modules = []
if not BUILDING_SDIST:
    pkg_dir = os.path.join(*PACKAGE.split("."))
    ext_modules.append(
        _ext(
            f"{PACKAGE}._internal.utils",
            os.path.join(pkg_dir, "_internal", "utils.pyx"),
        )
    )
    for libname in LIBNAMES:
        ext_modules.extend(libname_extensions(libname))
    ext_modules = cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        compiler_directives={
            "embedsignature": True,
            "show_performance_hints": True,
            "freethreading_compatible": True,
        },
    )

setup(
    ext_modules=ext_modules,
    zip_safe=False,
    options={"build_ext": {"inplace": False}},
)
