"""
CMake-based build for the _rocshmem4py C++ extension.

Metadata (name, version, classifiers, etc.) lives in pyproject.toml.

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import sys
import subprocess
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


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
        if "ROCM_PATH" in os.environ:
            cmake_args.append(f'-DROCM_PATH={os.environ["ROCM_PATH"]}')
        if "THEROCK_TOOLCHAIN_ROOT" in os.environ:
            cmake_args.append(
                f'-DTHEROCK_TOOLCHAIN_ROOT={os.environ["THEROCK_TOOLCHAIN_ROOT"]}'
            )
        if "ROCSHMEM_HOME" in os.environ:
            cmake_args.append(f'-DROCSHMEM_HOME={os.environ["ROCSHMEM_HOME"]}')

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
    ext_modules=[CMakeExtension("_rocshmem4py", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
)
