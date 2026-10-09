# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
import os
from pathlib import Path
import subprocess

import pytest


def pytest_addoption(parser):
    parser.addoption("--helper", required=True)
    parser.addoption("--launcher", required=True)
    parser.addoption(
        "--gpu", action="store_true", help="run ROCgdb/HIP integration tests"
    )
    parser.addoption("--rocm", default="/opt/rocm")
    parser.addoption("--gpu-arch", default="native")


@pytest.fixture(scope="session")
def helper(request):
    return str(Path(request.config.getoption("--helper")).resolve())


@pytest.fixture(scope="session")
def launcher(request):
    return str(Path(request.config.getoption("--launcher")).resolve())


@pytest.fixture(scope="session")
def fake_roctx(tmp_path_factory):
    root = tmp_path_factory.mktemp("fake-roctx")
    library = root / "libfake-roctx.so"
    subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            "-shared",
            "-fPIC",
            "-std=c++17",
            str(Path(__file__).parent / "fake_roctx.cpp"),
            "-o",
            str(library),
        ],
        check=True,
    )
    return str(library)


def compile_hip_app(request, tmp_path_factory, extra_flags=()):
    if not request.config.getoption("--gpu"):
        pytest.skip("use --gpu to run the ROCgdb/HIP integration tests")
    root = tmp_path_factory.mktemp("hip-app")
    binary = root / "application with spaces"
    rocm = Path(request.config.getoption("--rocm"))
    subprocess.run(
        [
            str(rocm / "bin/hipcc"),
            "-g",
            "-O1",
            *extra_flags,
            "--offload-arch=" + request.config.getoption("--gpu-arch"),
            str(Path(__file__).parent / "application.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    return str(binary)


@pytest.fixture(scope="session")
def hip_app(request, tmp_path_factory):
    return compile_hip_app(request, tmp_path_factory)


@pytest.fixture(scope="session")
def hip_app_noinline(request, tmp_path_factory):
    # A symbol for the stub may remain even when optimized callers inline it.
    # This separate fixture tests only the supported out-of-line launch case.
    return compile_hip_app(request, tmp_path_factory, ["-fno-inline"])
