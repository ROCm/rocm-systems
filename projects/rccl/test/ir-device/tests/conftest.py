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
import shutil
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
IR_TEST_SRC = os.path.join(IR_DIR, "test", "IR_test.cpp")
# [D] typed ReduceCopy cases, linked into the same IR_test.exe.
IR_REDUCE_COPY_SRC = os.path.join(_THIS_DIR, "..", "IR_reduce_copy_test.cpp")
BITCODE = os.path.join(RCCL_BUILD, "lib", "librccl_device.bc")
HIPIFY_INC = os.path.join(RCCL_BUILD, "hipify", "src", "include")
GENERATED_INC = os.path.join(RCCL_BUILD, "include")
AMDCLANGPP = os.path.join(ROCM_PATH, "bin", "amdclang++")
TEST_EXE = os.path.join(IR_OUTDIR, "IR_test.exe")

# ROCm LLVM tools for the bitcode build and its export check.
LLVM_TOOLS = ("clang++", "opt", "llvm-dis", "llvm-as")


def _find_llvm_bin():
    """Return the first LLVM bin dir holding all LLVM_TOOLS, or None, trying
    amdclang++'s own dir before $ROCM_PATH/llvm/bin and lib/llvm/bin."""
    dirs = [os.path.dirname(os.path.realpath(AMDCLANGPP)),
            os.path.join(ROCM_PATH, "llvm", "bin"),
            os.path.join(ROCM_PATH, "lib", "llvm", "bin")]
    for d in dirs:
        if all(os.path.isfile(os.path.join(d, t)) for t in LLVM_TOOLS):
            return os.path.normpath(d)
    return None


LLVM_BIN = _find_llvm_bin()

# --- Multi-rank GIN/composite barrier functional test (separate MPI binary) ---
# This one needs the host RCCL library + MPI in addition to the bitcode, so it
# is gated independently and skipped (not failed) when those are absent.
GIN_MPI_TEST_SRC = os.path.join(IR_DIR, "test", "IR_gin_mpi_test.cpp")
GIN_MPI_TEST_EXE = os.path.join(IR_OUTDIR, "IR_gin_mpi_test.exe")
NRANKS = int(os.environ.get("IR_GIN_NRANKS", "2"))
# Devices the MPI run may use, one per rank. Defaults to the first NRANKS GPUs
# (0,1,...) so each rank's in-binary hipSetDevice(localRank) lands on its own
# GPU. Override with IR_GIN_GPUS="2,3" to target specific devices.
IR_GIN_GPUS = os.environ.get("IR_GIN_GPUS", ",".join(str(i) for i in range(NRANKS)))

LOGDIR = os.path.join(WORKDIR, "logs")
os.makedirs(LOGDIR, exist_ok=True)


def _find_gtest_libdir():
    """Return the dir holding libgtest.{a,so} under GTEST_ROOT, or None.

    Mirrors the RCCL CMake gtest install layout (lib or lib64) and Debian
    multiarch installs (lib/<triple>-linux-gnu).
    """
    candidates = [
        os.path.join(GTEST_ROOT, "lib"),
        os.path.join(GTEST_ROOT, "lib64"),
    ]
    candidates += glob.glob(os.path.join(GTEST_ROOT, "lib", "*-linux-gnu"))
    for d in candidates:
        if os.path.isfile(os.path.join(d, "libgtest.a")) or os.path.isfile(
            os.path.join(d, "libgtest.so")
        ):
            return d
    return None


def _has_amd_gpu():
    """Return True when ROCm can see an AMD GPU.

    Asks rocm_agent_enumerator, which lists one gfx target per GPU agent plus
    gfx000 for the CPU. Probing device nodes alone misses ROCm under WSL2,
    which reaches the GPU through /dev/dxg rather than /dev/kfd or /dev/dri,
    so the node check is only the fallback when the enumerator cannot run.
    """
    enumerator = os.path.join(ROCM_PATH, "bin", "rocm_agent_enumerator")
    try:
        proc = subprocess.run(
            [enumerator], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            universal_newlines=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        proc = None
    if proc is not None and proc.returncode == 0:
        return any(
            line.strip().startswith("gfx") and line.strip() != "gfx000"
            for line in proc.stdout.splitlines()
        )
    return any(os.path.exists(n) for n in ("/dev/kfd", "/dev/dri", "/dev/dxg"))


def _missing_bitcode_prerequisite():
    """Return a reason string if librccl_device.bc cannot be built here.

    Only the LLVM tools and the hipify-staged headers: no GPU, GoogleTest or
    amdclang++, so bitcode-only checks also run on a CPU-only build host.
    """
    if LLVM_BIN is None:
        return (
            f"ROCm LLVM tools ({' '.join(LLVM_TOOLS)}) not found under "
            f"{ROCM_PATH}/llvm/bin, {ROCM_PATH}/lib/llvm/bin or beside "
            "amdclang++ (install ROCm llvm tools for IR device tests)"
        )
    if not os.path.isdir(HIPIFY_INC):
        return (
            f"hipify staging dir not found at {HIPIFY_INC} "
            "(build RCCL once so the staged nccl_device headers exist)"
        )
    return None


def _missing_prerequisite():
    """Return a human-readable reason string if a prerequisite is missing.

    Note: librccl_device.bc is intentionally NOT required here — it is built
    on demand from the hipify-staged headers (see _build_bitcode). The real
    gate is the hipify staging dir, which a prior RCCL CMake build produces
    and which is needed both to emit the bitcode and to compile the test.
    """
    if not os.path.isfile(IR_TEST_SRC):
        return f"IR_test.cpp not found at {IR_TEST_SRC}"
    if not os.path.isfile(AMDCLANGPP):
        return f"amdclang++ not found at {AMDCLANGPP} (set ROCM_PATH)"
    reason = _missing_bitcode_prerequisite()
    if reason:
        return reason
    gtest_inc = os.path.join(GTEST_ROOT, "include")
    if (
        not os.path.isfile(os.path.join(gtest_inc, "gtest", "gtest.h"))
        or _find_gtest_libdir() is None
    ):
        return f"GoogleTest not found under GTEST_ROOT={GTEST_ROOT}"
    if not _has_amd_gpu():
        return (
            "no AMD GPU found (rocm_agent_enumerator lists no gfx agent, or "
            "it is unavailable and /dev/kfd, /dev/dri, /dev/dxg are absent)"
        )
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
        # The Makefile defaults to $(ROCM_PATH)/llvm, which not every ROCm has.
        f"ROCM_LLVM={os.path.dirname(LLVM_BIN)}",
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


def _build_test_binary():
    """Compile IR_test.cpp into a GoogleTest binary linked against the bitcode.

    Ports the hipcc invocation that previously lived in run_IR_test.sh:

      * -Xoffload-linker <bc>   routes librccl_device.bc into the AMDGPU
        device-side LTO link.
      * -plugin-opt=-amdgpu-internalize-symbols=false keeps the exported
        device thunks from being re-internalized by AMDGPU LTO.
      * -O0 is deliberate: a ROCm 7.x AMDGPU codegen bug breaks the indirect
        (vtable) dispatch this test exercises at -O1/-O2 (hang or error 700).
        This is a correctness test, not a benchmark, so -O0 is fine.

    Returns the build log path; raises on failure.
    """
    os.makedirs(IR_OUTDIR, exist_ok=True)
    gtest_inc = os.path.join(GTEST_ROOT, "include")
    gtest_libdir = _find_gtest_libdir()
    args = [
        AMDCLANGPP,
        "-x", "hip",
        f"--offload-arch={ARCH}", "-O0",
        "-D__HIP_PLATFORM_AMD__=1",
        # Source wrapper first: cmake -DEMIT_LLVM_IR=OFF does not restage
        # build/include/nccl_device_wrapper.h, so a generated copy can be stale.
        # ROCm system headers come after project-specific includes so freshly
        # hipified headers in HIPIFY_INC take precedence over installed copies.
        f"-I{IR_DIR}",
        f"-I{HIPIFY_INC}",
        f"-I{os.path.join(HIPIFY_INC, 'nccl_device')}",
        f"-I{GENERATED_INC}",
        f"-I{gtest_inc}",
        f"-I{os.path.join(ROCM_PATH, 'include')}",
        IR_TEST_SRC,
        IR_REDUCE_COPY_SRC,
        "-Xoffload-linker", BITCODE,
        "-Xoffload-linker", "-plugin-opt=-amdgpu-internalize-symbols=false",
        f"-L{os.path.join(ROCM_PATH, 'lib')}", "-lamdhip64",
        f"-L{gtest_libdir}", "-lgtest_main", "-lgtest", "-lpthread",
        "-o", TEST_EXE,
    ]
    build_log = os.path.join(LOGDIR, "ir_test_build.log")
    with open(build_log, "w") as log:
        log.write("$ " + " ".join(args) + "\n\n")
        log.flush()
        proc = subprocess.run(
            args, env=os.environ.copy(), stdout=log,
            stderr=subprocess.STDOUT, universal_newlines=True,
        )
    assert proc.returncode == 0, f"Failed to compile IR_test (see {build_log})"
    assert os.path.isfile(TEST_EXE), f"Test binary not produced at {TEST_EXE}"
    return build_log


def _find_rccl_libdir():
    """Return the dir holding the host librccl.so under RCCL_BUILD, or None."""
    candidates = [
        RCCL_BUILD,
        os.path.join(RCCL_BUILD, "lib"),
        os.path.join(RCCL_BUILD, "lib64"),
    ]
    for d in candidates:
        if glob.glob(os.path.join(d, "librccl.so*")):
            return d
    return None


def _missing_mpi_prerequisite():
    """Reason string if the MPI GIN functional test cannot be built/run.

    Builds on _missing_prerequisite() (hipcc / hipify / gtest / GPU) and
    adds the host-library + MPI requirements unique to the multi-rank test. The
    GIN-runtime gates (>=2 GPUs, IB GIN, NCCL_GIN_TYPE/CUMEM/INTRANET env) are
    enforced inside the test itself via GTEST_SKIP, so they are not duplicated
    here — this only covers what is needed to *build and launch* the binary.
    """
    base = _missing_prerequisite()
    if base:
        return base
    if not os.path.isfile(GIN_MPI_TEST_SRC):
        return f"IR_gin_mpi_test.cpp not found at {GIN_MPI_TEST_SRC}"
    if shutil.which("mpirun") is None or shutil.which("mpicxx") is None:
        return "MPI not found (mpirun/mpicxx not in PATH)"
    if _find_rccl_libdir() is None:
        return (
            f"host librccl.so not found under {RCCL_BUILD} "
            "(build RCCL's host library, not just -DEMIT_LLVM_IR=ON)"
        )
    return None


def _build_gin_mpi_binary():
    """Compile IR_gin_mpi_test.cpp into an MPI GoogleTest binary.

    Same device-side bitcode link as IR_test (-Xoffload-linker librccl_device.bc
    + amdgpu-internalize-symbols=false), plus the host RCCL library and MPI.
    Returns the build log path; raises on failure.
    """
    os.makedirs(IR_OUTDIR, exist_ok=True)
    gtest_inc = os.path.join(GTEST_ROOT, "include")
    gtest_libdir = _find_gtest_libdir()
    rccl_libdir = _find_rccl_libdir()
    mpi_cxx = shutil.which("mpicxx")
    # Use the mpicxx wrapper as the compiler so it automatically injects the
    # correct MPI include paths and link flags for whichever MPI implementation
    # (OpenMPI, MPICH, etc.) is installed.  OMPI_CXX / MPICH_CXX redirect the
    # wrapper to amdclang++ so HIP device-code compilation works transparently.
    args = [
        mpi_cxx,
        "-x", "hip",
        f"--offload-arch={ARCH}", "-O0",
        "-D__HIP_PLATFORM_AMD__=1",
        f"-I{IR_DIR}",
        f"-I{HIPIFY_INC}",
        f"-I{os.path.join(HIPIFY_INC, 'nccl_device')}",
        f"-I{GENERATED_INC}",
        f"-I{gtest_inc}",
        f"-I{os.path.join(ROCM_PATH, 'include')}",
        GIN_MPI_TEST_SRC,
        "-Xoffload-linker", BITCODE,
        "-Xoffload-linker", "-plugin-opt=-amdgpu-internalize-symbols=false",
        f"-L{gtest_libdir}", "-lgtest_main", "-lgtest", "-lpthread",
        f"-L{rccl_libdir}", f"-Wl,-rpath,{rccl_libdir}", "-lrccl",
        f"-L{os.path.join(ROCM_PATH, 'lib')}", "-lamdhip64",
        "-o", GIN_MPI_TEST_EXE,
    ]
    env = os.environ.copy()
    env.update({"OMPI_CXX": AMDCLANGPP, "MPICH_CXX": AMDCLANGPP})
    build_log = os.path.join(LOGDIR, "ir_gin_mpi_build.log")
    with open(build_log, "w") as log:
        log.write("$ " + " ".join(args) + "\n\n")
        log.flush()
        proc = subprocess.run(
            args, env=env, stdout=log,
            stderr=subprocess.STDOUT, universal_newlines=True,
        )
    assert proc.returncode == 0, (
        f"Failed to compile IR_gin_mpi_test (see {build_log})"
    )
    assert os.path.isfile(GIN_MPI_TEST_EXE), (
        f"MPI test binary not produced at {GIN_MPI_TEST_EXE}"
    )
    return build_log


@pytest.fixture(scope="session")
def ir_gin_mpi_binary():
    """Build the bitcode if needed, then compile the MPI GIN test once.

    Skips the whole multi-rank suite when its build/launch prerequisites
    (host librccl.so, MPI, hipify staging, GoogleTest, GPU nodes) are missing.
    """
    reason = _missing_mpi_prerequisite()
    if reason:
        pytest.skip(f"IR GIN multi-rank tests skipped: {reason}")

    if not os.path.isfile(BITCODE):
        logger.info("librccl_device.bc MISSING — building it (arch=%s)...", ARCH)
        _build_bitcode()
    logger.info("Compiling IR_gin_mpi_test.cpp -> %s ...", GIN_MPI_TEST_EXE)
    _build_gin_mpi_binary()
    return GIN_MPI_TEST_EXE


@pytest.fixture(scope="session")
def run_gin_mpi_gtest(ir_gin_mpi_binary):
    """Return a helper that launches the MPI GIN test under `mpirun -np N`.

    The helper returns (subprocess.CompletedProcess, log_file_path). GIN env
    defaults (NCCL_GIN_TYPE=2, NCCL_CUMEM_ENABLE=1, RCCL_ENABLE_INTRANET=1) are
    seeded if unset so a single-node 2-GPU run can establish GIN; the binary
    itself GTEST_SKIPs cleanly if GIN still cannot initialize. HIP_VISIBLE_DEVICES
    is pinned to IR_GIN_GPUS (default the first NRANKS devices) so each rank gets
    its own GPU regardless of any ambient single-GPU restriction.
    """

    def _run(gtest_filter, log_name):
        env = os.environ.copy()
        env.setdefault("NCCL_GIN_ENABLE", "1")
        env.setdefault("NCCL_GIN_TYPE", "2")
        env.setdefault("NCCL_CUMEM_ENABLE", "1")
        env.setdefault("RCCL_ENABLE_INTRANET", "1")
        env.setdefault("HSA_NO_SCRATCH_RECLAIM", "1")
        # Pin GPUs per rank: expose NRANKS distinct devices to the whole run so
        # each rank's hipSetDevice(localRank) selects its own. Set explicitly
        # (not setdefault) so a restrictive ambient HIP_VISIBLE_DEVICES=0 from
        # the shell doesn't collapse every rank onto one GPU and force a skip.
        env["HIP_VISIBLE_DEVICES"] = IR_GIN_GPUS
        args = [
            "mpirun", "-np", str(NRANKS),
            ir_gin_mpi_binary,
            f"--gtest_filter={gtest_filter}", "--gtest_color=no",
        ]
        log_file = os.path.join(LOGDIR, log_name)
        with open(log_file, "w") as log:
            log.write("$ " + " ".join(args) + "\n\n")
            log.flush()
            proc = subprocess.run(
                args, env=env, stdout=log, stderr=subprocess.STDOUT,
                universal_newlines=True, timeout=300,
            )
        return proc, log_file

    return _run


@pytest.fixture(scope="session")
def ir_bitcode():
    """Update librccl_device.bc once and return its path.

    Needs no GPU, so bitcode-only tests (D0) do not skip on a build host.
    """
    reason = _missing_bitcode_prerequisite()
    if reason:
        pytest.skip(f"IR bitcode tests skipped: {reason}")

    # Always invoke make so dependency timestamps decide whether the artifact
    # is current. Merely checking that BITCODE exists can reuse an artifact
    # from before a wrapper/implementation edit when RCCL itself was rebuilt
    # with EMIT_LLVM_IR=OFF.
    logger.info("Updating librccl_device.bc at %s (arch=%s)...", BITCODE, ARCH)
    bc_log = _build_bitcode()
    logger.info("librccl_device.bc READY at %s (log: %s)", BITCODE, bc_log)
    return BITCODE


@pytest.fixture(scope="session")
def ir_test_binary(request):
    """Update the bitcode, then compile IR_test.cpp against it once.

    Skips the entire suite if prerequisites are missing. Returns the path to
    the compiled GoogleTest executable.
    """
    reason = _missing_prerequisite()
    if reason:
        pytest.skip(f"IR device tests skipped: {reason}")
    # Requested only now, so a host that cannot run IR_test skips before
    # spending a bitcode build on it.
    request.getfixturevalue("ir_bitcode")

    logger.info(
        "Compiling IR_test.cpp -> %s (amdclang++ -O0, arch=%s)...", TEST_EXE, ARCH)
    test_log = _build_test_binary()
    logger.info("IR_test binary BUILT at %s (log: %s)", TEST_EXE, test_log)
    return TEST_EXE


@pytest.fixture(scope="session")
def paths():
    return SimpleNamespace(
        WORKDIR=WORKDIR,
        RCCL_DIR=RCCL_DIR,
        RCCL_BUILD=RCCL_BUILD,
        ROCM_PATH=ROCM_PATH,
        LLVM_BIN=LLVM_BIN,
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
