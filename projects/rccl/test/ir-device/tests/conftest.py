# *************************************************************************
#  * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Shared fixtures for the librccl_device.bc (IR) device tests.

This mirrors test/ext-plugins: the test artifact (here, the GoogleTest
binary built from bindings/ir/test/IR_test.cpp against librccl_device.bc) is
built on demand from a session-scoped fixture and the whole suite is skipped
with a clear reason when its prerequisites are missing — instead of failing
the CI job on machines that did not build with -DEMIT_LLVM_IR=ON.

Prerequisites (set as environment variables, with sensible defaults):

  RCCL_DIR     RCCL source root        (default: repo root, derived)
  RCCL_BUILD   RCCL CMake build dir    (default: $RCCL_DIR/build/release)
  ROCM_PATH    ROCm install root       (default: /opt/rocm)
  ARCH         offload / bitcode arch  (default: gfx942)
  GTEST_ROOT   GoogleTest prefix       (default: $RCCL_BUILD/gtest)
  IR_OUTDIR    dir for the compiled exe(default: <workdir>/ir_test_build)

The build requires that RCCL was configured/built with:
  cmake -DEMIT_LLVM_IR=ON -DBITCODE_LIB_ARCH=<arch> -DBUILD_TESTS=ON ...
so that librccl_device.bc, the hipify-staged headers, generated nccl.h/rccl.h
and GoogleTest are all present.
"""

import glob
import logging
import os
import subprocess
from types import SimpleNamespace

import pytest

logger = logging.getLogger(__name__)

WORKDIR = os.getcwd()

# RCCL source root: this file is test/ir-device/tests/conftest.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RCCL_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

RCCL_DIR = os.path.abspath(os.environ.get("RCCL_DIR", _DEFAULT_RCCL_DIR))
RCCL_BUILD = os.path.abspath(
    os.environ.get("RCCL_BUILD", os.path.join(RCCL_DIR, "build", "release"))
)
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
ARCH = os.environ.get("ARCH", "gfx942")
GTEST_ROOT = os.environ.get("GTEST_ROOT", os.path.join(RCCL_BUILD, "gtest"))
IR_OUTDIR = os.environ.get("IR_OUTDIR", os.path.join(WORKDIR, "ir_test_build"))

IR_DIR = os.path.join(RCCL_DIR, "bindings", "ir")
RUN_SCRIPT = os.path.join(IR_DIR, "test", "run_IR_test.sh")
BITCODE = os.path.join(RCCL_BUILD, "lib", "librccl_device.bc")
HIPIFY_INC = os.path.join(RCCL_BUILD, "hipify", "src", "include")
TEST_EXE = os.path.join(IR_OUTDIR, "IR_test.exe")

LOGDIR = os.path.join(WORKDIR, "logs")
os.makedirs(LOGDIR, exist_ok=True)


def _missing_prerequisite():
    """Return a human-readable reason string if a prerequisite is missing.

    Note: librccl_device.bc is intentionally NOT required here — it is built
    on demand from the hipify-staged headers (see _build_bitcode). The real
    gate is the hipify staging dir, which a prior RCCL CMake build produces
    and which is needed both to emit the bitcode and to compile the test.
    """
    if not os.path.isfile(RUN_SCRIPT):
        return f"run_IR_test.sh not found at {RUN_SCRIPT}"
    if not os.path.isdir(HIPIFY_INC):
        return (
            f"hipify staging dir not found at {HIPIFY_INC} "
            "(build RCCL once so the staged nccl_device headers exist)"
        )
    gtest_inc = os.path.join(GTEST_ROOT, "include")
    lib_globs = (
        os.path.join(GTEST_ROOT, "lib", "libgtest.*"),
        os.path.join(GTEST_ROOT, "lib64", "libgtest.*"),
        os.path.join(GTEST_ROOT, "lib", "*-linux-gnu", "libgtest.*"),
    )
    has_lib = any(glob.glob(p) for p in lib_globs)
    if not os.path.isfile(os.path.join(gtest_inc, "gtest", "gtest.h")) or not has_lib:
        return f"GoogleTest not found under GTEST_ROOT={GTEST_ROOT}"
    if not os.path.isdir("/dev/dri") and not os.path.exists("/dev/kfd"):
        return "no AMD GPU device nodes (/dev/kfd, /dev/dri) present"
    return None


def _build_bitcode():
    """Emit librccl_device.bc via the standalone bindings/ir Makefile.

    Equivalent to building RCCL with -DEMIT_LLVM_IR=ON, but without a full
    rebuild: the Makefile reuses the hipify-staged headers an earlier RCCL
    build already produced. Returns the build log path; raises on failure.
    """
    env = os.environ.copy()
    env.update({"ROCM_PATH": ROCM_PATH})
    args = [
        "make", "-C", IR_DIR,
        "EMIT_LLVM_IR=1",
        f"BITCODE_LIB_ARCH={ARCH}",
        f"BUILDDIR={RCCL_BUILD}",
        f"ROCM_PATH={ROCM_PATH}",
        "llvm_ir",
    ]
    build_log = os.path.join(LOGDIR, "ir_bitcode_build.log")
    with open(build_log, "w") as log:
        proc = subprocess.run(
            args, env=env, stdout=log, stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    assert proc.returncode == 0, (
        f"Failed to build librccl_device.bc (see {build_log})"
    )
    assert os.path.isfile(BITCODE), f"Bitcode not produced at {BITCODE}"
    return build_log


@pytest.fixture(scope="session")
def ir_test_binary():
    """Build the bitcode if needed, then compile IR_test.cpp against it once.

    Skips the entire suite if prerequisites are missing. Returns the path to
    the compiled GoogleTest executable.
    """
    reason = _missing_prerequisite()
    if reason:
        pytest.skip(f"IR device tests skipped: {reason}")

    # Emit librccl_device.bc on demand (the EMIT_LLVM_IR step) when absent.
    if os.path.isfile(BITCODE):
        logger.info("librccl_device.bc PRESENT at %s — reusing it", BITCODE)
    else:
        logger.info(
            "librccl_device.bc MISSING at %s — building it (EMIT_LLVM_IR step, "
            "arch=%s)...", BITCODE, ARCH)
        bc_log = _build_bitcode()
        logger.info("librccl_device.bc BUILT at %s (log: %s)", BITCODE, bc_log)

    os.makedirs(IR_OUTDIR, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "ARCH": ARCH,
            "ROCM_PATH": ROCM_PATH,
            "BUILD": RCCL_BUILD,
            "BC": BITCODE,
            "GTEST_ROOT": GTEST_ROOT,
            "OUTDIR": IR_OUTDIR,
            "BUILD_ONLY": "1",
        }
    )

    build_log = os.path.join(LOGDIR, "ir_test_build.log")
    with open(build_log, "w") as log:
        proc = subprocess.run(
            ["bash", RUN_SCRIPT],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    assert proc.returncode == 0, (
        f"Failed to build IR_test (see {build_log})"
    )
    assert os.path.isfile(TEST_EXE), f"Test binary not produced at {TEST_EXE}"
    return TEST_EXE


@pytest.fixture(scope="session")
def paths():
    return SimpleNamespace(
        WORKDIR=WORKDIR,
        RCCL_DIR=RCCL_DIR,
        RCCL_BUILD=RCCL_BUILD,
        ROCM_PATH=ROCM_PATH,
        ARCH=ARCH,
        GTEST_ROOT=GTEST_ROOT,
        IR_OUTDIR=IR_OUTDIR,
        BITCODE=BITCODE,
        TEST_EXE=TEST_EXE,
        LOGDIR=LOGDIR,
    )


@pytest.fixture(scope="session")
def run_gtest(ir_test_binary):
    """Return a helper that runs the IR_test binary for a --gtest_filter.

    The helper returns (subprocess.CompletedProcess, log_file_path). Output is
    tee'd to a log file under logs/ for post-mortem inspection.
    """

    def _run(gtest_filter, log_name, gpu=None):
        env = os.environ.copy()
        env.setdefault("HSA_NO_SCRATCH_RECLAIM", "1")
        # Default to a single GPU for determinism unless the caller overrides.
        if gpu is not None:
            env["HIP_VISIBLE_DEVICES"] = str(gpu)
        elif "HIP_VISIBLE_DEVICES" not in env:
            env["HIP_VISIBLE_DEVICES"] = "0"

        args = [ir_test_binary, f"--gtest_filter={gtest_filter}",
                "--gtest_color=no"]
        log_file = os.path.join(LOGDIR, log_name)
        with open(log_file, "w") as log:
            proc = subprocess.run(
                args,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                timeout=120,
            )
        return proc, log_file

    return _run
