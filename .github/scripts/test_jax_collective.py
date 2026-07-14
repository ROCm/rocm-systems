#!/usr/bin/env python3
"""Run JAX collective smoke tests against CI-built RCCL.

This script handles:
  1. Discovering the CI-built librccl.so in the artifact directory
  2. Setting LD_LIBRARY_PATH to override JAX's bundled RCCL
  3. Cloning the matching JAX test sources (sparse checkout of ROCm/jax)
  4. Setting XLA environment variables for ROCm
  5. Running 10 collective smoke tests via pytest

Usage from GitHub Actions:
  python .github/scripts/test_jax_collective.py \
      --artifact-dir ./build \
      --jax-src ./jax-src \
      --results-log ./jax_collective_results.log

Local prototyping (no RCCL override):
  python .github/scripts/test_jax_collective.py \
      --jax-src ./jax-src \
      --results-log ./jax_collective_results.log \
      --skip-rccl-override
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

JAX_REPO = "https://github.com/ROCm/jax.git"
DEFAULT_BRANCH = "jax-v0.10.2-testing"

SMOKE_TESTS = [
    "tests/pmap_test.py::PythonPmapTest::testBasic",
    "tests/pmap_test.py::PythonPmapTest::testGather",
    "tests/pmap_test.py::PythonPmapTest::testReduceScatter",
    "tests/pmap_test.py::PythonPmapTest::testCollectivePermute",
    "tests/pmap_test.py::PythonPmapTest::testAllToAll0",
    "tests/shard_map_test.py::ShardMapTest::test_all_gather",
    "tests/shard_map_test.py::ShardMapTest::test_matmul_reduce_scatter",
    "tests/shard_map_test.py::ShardMapTest::test_collective_permute",
    "tests/shard_map_test.py::ShardMapTest::test_axis_index",
    "tests/shard_map_test.py::ShardMapTest::test_all_to_all_multiple_axis_names",
]

XLA_ENV = {
    "XLA_PYTHON_CLIENT_ALLOCATOR": "default",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_FLAGS": (
        "--xla_gpu_force_compilation_parallelism=1 "
        "--xla_gpu_enable_nccl_comm_splitting=false "
        "--xla_gpu_enable_command_buffer= "
        "--xla_gpu_enable_cublaslt=false"
    ),
}


def find_rccl_library(artifact_dir: Path) -> Path:
    """Find librccl.so in the artifact directory tree."""
    matches = list(artifact_dir.rglob("librccl.so"))
    if not matches:
        so_files = list(artifact_dir.rglob("*.so"))[:20]
        log.error("librccl.so not found in %s", artifact_dir)
        log.error("Shared libraries found: %s", [str(f) for f in so_files])
        sys.exit(1)
    lib_path = matches[0]
    log.info("Found librccl.so at: %s", lib_path)
    return lib_path


def find_rocm_lib_dir(artifact_dir: Path) -> Path | None:
    """Find the dist/rocm/lib directory in artifacts."""
    for d in artifact_dir.rglob("dist/rocm/lib"):
        if d.is_dir():
            log.info("Found ROCm lib dir: %s", d)
            return d
    return None


def setup_ld_library_path(rccl_lib_dir: Path, rocm_lib_dir: Path | None) -> str:
    """Prepend RCCL and ROCm lib dirs to LD_LIBRARY_PATH."""
    parts = [str(rccl_lib_dir.resolve())]
    if rocm_lib_dir:
        parts.append(str(rocm_lib_dir.resolve()))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    new_path = ":".join(parts)
    os.environ["LD_LIBRARY_PATH"] = new_path
    log.info("LD_LIBRARY_PATH=%s", new_path)
    return new_path


def verify_rccl_override(rccl_lib_dir: Path) -> None:
    """Verify that the CI-built librccl.so exists on disk."""
    ci_rccl = rccl_lib_dir.resolve() / "librccl.so"
    if not ci_rccl.exists():
        log.error("CI-built librccl.so not found at %s", ci_rccl)
        sys.exit(1)
    log.info("CI-built RCCL: %s (%d bytes)", ci_rccl, ci_rccl.stat().st_size)


def setup_xla_environment() -> None:
    """Set XLA environment variables required for JAX on ROCm."""
    for key, value in XLA_ENV.items():
        os.environ[key] = value
        log.info("Set %s=%s", key, value)


def clone_jax_test_sources(jax_src: Path, branch: str) -> None:
    """Sparse-clone ROCm/jax to get test sources and test-requirements.txt."""
    if jax_src.exists() and (jax_src / "tests" / "pmap_test.py").exists():
        log.info("JAX test sources already present at %s, skipping clone", jax_src)
        return

    log.info("Cloning ROCm/jax (branch=%s, sparse) into %s", branch, jax_src)
    subprocess.run(
        [
            "git", "clone",
            "--depth=1",
            f"--branch={branch}",
            "--filter=blob:none",
            "--sparse",
            JAX_REPO,
            str(jax_src),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", "tests/", "build/", "jax/"],
        cwd=jax_src,
        check=True,
    )

    test_file = jax_src / "tests" / "pmap_test.py"
    if not test_file.exists():
        log.error("pmap_test.py not found after clone")
        sys.exit(1)
    log.info("JAX test sources ready: %s", jax_src)


def _pip_install(args: list[str]) -> None:
    """Install packages using uv pip (preferred) or pip."""
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install"] + args
    else:
        cmd = [sys.executable, "-m", "pip", "install"] + args
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def install_test_requirements(jax_src: Path) -> None:
    """Install test dependencies from build/test-requirements.txt."""
    req_file = jax_src / "build" / "test-requirements.txt"
    if not req_file.exists():
        log.warning("build/test-requirements.txt not found, installing minimal deps")
        _pip_install(["pytest", "pytest-timeout", "hypothesis", "numpy", "absl-py"])
        return

    log.info("Installing test requirements from %s", req_file)
    _pip_install(["-r", str(req_file)])


def print_environment_info() -> None:
    """Print GPU and environment details for CI logs."""
    log.info("--- Environment Info ---")
    try:
        import jax
        log.info("JAX version: %s", jax.__version__)
        log.info("Devices: %s", jax.devices())
        log.info("Device count: %d", jax.device_count())
        log.info("Local device count: %d", jax.local_device_count())
    except Exception as e:
        log.error("Failed to query JAX devices: %s", e)
        sys.exit(1)

    log.info("LD_LIBRARY_PATH: %s", os.environ.get("LD_LIBRARY_PATH", ""))
    for key in sorted(XLA_ENV):
        log.info("%s=%s", key, os.environ.get(key, "<unset>"))
    log.info("--- End Environment Info ---")


def run_tests(jax_src: Path, results_log: Path, timeout: int) -> int:
    """Run pytest on the 10 collective smoke tests and return exit code."""
    cmd = [
        sys.executable, "-m", "pytest",
        "-sv",
        f"--timeout={timeout}",
        "--tb=short",
    ] + SMOKE_TESTS

    log.info("Running: %s", " ".join(cmd))

    with open(results_log, "w") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=jax_src,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_file.write(proc.stdout)
        print(proc.stdout, end="")

    log.info("Test exit code: %d", proc.returncode)
    log.info("Results written to: %s", results_log)
    return proc.returncode


def set_github_output(key: str, value: str) -> None:
    """Write a key=value pair to GITHUB_OUTPUT if available."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory containing CI-built RCCL artifacts",
    )
    parser.add_argument(
        "--jax-src",
        type=Path,
        required=True,
        help="Directory to clone JAX test sources into",
    )
    parser.add_argument(
        "--jax-branch",
        default=DEFAULT_BRANCH,
        help=f"ROCm/jax branch for test sources (default: {DEFAULT_BRANCH})",
    )
    parser.add_argument(
        "--results-log",
        type=Path,
        default=Path("jax_collective_results.log"),
        help="Path for test results log file",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-test timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--skip-rccl-override",
        action="store_true",
        help="Skip RCCL library override (use system/wheel-bundled RCCL)",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover library paths and set GITHUB_OUTPUT, then exit",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip cloning JAX sources (assume already present)",
    )
    parser.add_argument(
        "--skip-install-deps",
        action="store_true",
        help="Skip installing test dependencies",
    )

    args = parser.parse_args()

    # Step 1: RCCL library discovery and override
    if not args.skip_rccl_override:
        if args.artifact_dir is None:
            log.error("--artifact-dir required unless --skip-rccl-override is set")
            sys.exit(1)
        rccl_lib = find_rccl_library(args.artifact_dir)
        rccl_lib_dir = rccl_lib.parent
        rocm_lib_dir = find_rocm_lib_dir(args.artifact_dir)

        set_github_output("RCCL_LIB_DIR", str(rccl_lib_dir))
        if rocm_lib_dir:
            set_github_output("ROCM_LIB_DIR", str(rocm_lib_dir))

        if args.discover_only:
            return

        setup_ld_library_path(rccl_lib_dir, rocm_lib_dir)
        verify_rccl_override(rccl_lib_dir)
    elif args.discover_only:
        return

    # Step 2: Set XLA environment variables
    setup_xla_environment()

    # Step 3: Clone JAX test sources
    if not args.skip_clone:
        clone_jax_test_sources(args.jax_src, args.jax_branch)

    # Step 4: Install test dependencies
    if not args.skip_install_deps:
        install_test_requirements(args.jax_src)

    # Step 5: Print environment info (also validates JAX can see GPUs)
    print_environment_info()

    # Step 6: Run smoke tests
    exit_code = run_tests(args.jax_src, args.results_log, args.timeout)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
