"""
CMake-based build for the _rocshmem4py C++ extension.

Metadata (name, version, classifiers, etc.) lives in pyproject.toml.

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import re
import sys
import subprocess
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Base release version of the binding's own Python API. The full version
# reported to pip is this plus a PEP 440 local segment recording the linked
# rocSHMEM library version (e.g. "0.1.0+rocshmem3.6.0"), so a built wheel is
# self-documenting about which rocSHMEM it was compiled against.
BASE_VERSION = "0.1.0"


def _rocshmem_prefixes():
    """Candidate rocSHMEM install prefixes, in the same precedence order that
    CMakeLists uses for find_package: CMAKE_PREFIX_PATH first, then
    ROCSHMEM_HOME, then ROCM_PATH, then the /opt/rocm default."""
    prefixes = []
    if os.environ.get("CMAKE_PREFIX_PATH"):
        prefixes += os.environ["CMAKE_PREFIX_PATH"].split(os.pathsep)
    for var in ("ROCSHMEM_HOME", "ROCM_PATH"):
        if os.environ.get(var):
            prefixes.append(os.environ[var])
    prefixes.append("/opt/rocm")
    return [p for p in prefixes if p]


def _detect_rocshmem_version():
    """Parse ROCSHMEM_VERSION from the installed rocshmem_config.h so the wheel
    records the linked library version. Returns None if it cannot be found, in
    which case the version falls back to BASE_VERSION (CMake still enforces the
    minimum-version gate at configure time)."""
    pattern = re.compile(r'#define\s+ROCSHMEM_VERSION\s+"([^"]+)"')
    for prefix in _rocshmem_prefixes():
        header = os.path.join(prefix, "include", "rocshmem", "rocshmem_config.h")
        try:
            with open(header) as fh:
                for line in fh:
                    m = pattern.search(line)
                    if m:
                        return m.group(1)
        except OSError:
            continue
    return None


def _package_version():
    lib_version = _detect_rocshmem_version()
    if not lib_version:
        return BASE_VERSION
    # PEP 440 local version label: alphanumerics and periods only.
    local = "rocshmem" + re.sub(r"[^0-9A-Za-z.]", ".", lib_version)
    return f"{BASE_VERSION}+{local}"


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        # Forward CMAKE_PREFIX_PATH as a cache variable, not just via the
        # environment. find_package searches the cache-variable CMAKE_PREFIX_PATH
        # (which CMakeLists augments with ROCM_PATH for HIP) *before* the
        # environment variable, so an env-only CMAKE_PREFIX_PATH is shadowed by
        # any rocSHMEM that ships under /opt/rocm. Passing it as -D keeps the
        # caller's rocSHMEM install at the front of the search order. Convert
        # os.pathsep (":") to CMake's list separator (";").
        if os.environ.get("CMAKE_PREFIX_PATH"):
            prefix_path = os.environ["CMAKE_PREFIX_PATH"].replace(os.pathsep, ";")
            cmake_args.append(f"-DCMAKE_PREFIX_PATH={prefix_path}")
        if "ROCM_PATH" in os.environ:
            cmake_args.append(f'-DROCM_PATH={os.environ["ROCM_PATH"]}')
        if "THEROCK_TOOLCHAIN_ROOT" in os.environ:
            cmake_args.append(
                f'-DTHEROCK_TOOLCHAIN_ROOT={os.environ["THEROCK_TOOLCHAIN_ROOT"]}'
            )
        if "ROCSHMEM_HOME" in os.environ:
            cmake_args.append(f'-DROCSHMEM_HOME={os.environ["ROCSHMEM_HOME"]}')
        # Optional override of the GPU architectures the device code objects are
        # built for. When unset, CMake auto-detects them from the installed
        # rocSHMEM device bitcode so the wheel matches that rocSHMEM build.
        if os.environ.get("ROCSHMEM_GPU_TARGETS"):
            cmake_args.append(
                f'-DROCSHMEM_GPU_TARGETS={os.environ["ROCSHMEM_GPU_TARGETS"]}'
            )

        build_args = ["--config", cfg, "--", f"-j{os.cpu_count() or 4}"]

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=self.build_temp,
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=self.build_temp,
        )


setup(
    version=_package_version(),
    ext_modules=[CMakeExtension("_rocshmem4py", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
)
