#!/usr/bin/env python3
"""Run PyTorch c10d NCCL distributed tests against CI-built RCCL.

This script handles:
  1. Discovering the CI-built librccl.so in the artifact directory
  2. Verifying that LD_LIBRARY_PATH overrides PyTorch's bundled RCCL
  3. Cloning the matching PyTorch test sources (sparse checkout)
  4. Running pytest on test_c10d_nccl.py

Usage from GitHub Actions:
  python .github/scripts/test_pytorch_c10d.py \
      --artifact-dir ./build \
      --pytorch-src ./pytorch-src \
      --results-log ./pytorch_c10d_results.log
"""

import argparse
import ctypes
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


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
    parts = [str(rccl_lib_dir)]
    if rocm_lib_dir:
        parts.append(str(rocm_lib_dir))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    new_path = ":".join(parts)
    os.environ["LD_LIBRARY_PATH"] = new_path
    log.info("LD_LIBRARY_PATH=%s", new_path)
    return new_path


def verify_rccl_override(rccl_lib_dir: Path) -> None:
    """Verify that the CI-built librccl.so is loadable via LD_LIBRARY_PATH."""
    try:
        ctypes.CDLL("librccl.so")
        log.info("Successfully loaded librccl.so via LD_LIBRARY_PATH")
    except OSError as e:
        log.error("Failed to load librccl.so: %s", e)
        sys.exit(1)

    ci_rccl = rccl_lib_dir / "librccl.so"
    log.info("CI-built RCCL: %s (%d bytes)", ci_rccl, ci_rccl.stat().st_size)

    try:
        import torch

        torch_rccl = Path(torch.__file__).parent / "lib" / "librccl.so"
        if torch_rccl.exists():
            log.info(
                "PyTorch bundled RCCL: %s (%d bytes)",
                torch_rccl,
                torch_rccl.stat().st_size,
            )
        else:
            log.info("PyTorch bundled RCCL: not found (device package may not bundle it)")
    except ImportError:
        log.warning("torch not yet installed, skipping bundled RCCL check")


def clone_pytorch_test_sources(pytorch_src: Path) -> None:
    """Sparse-clone PyTorch test sources matching the installed torch version."""
    import torch

    torch_version = torch.__version__.split("+")[0]
    log.info("PyTorch version: %s", torch_version)

    git_ref = f"v{torch_version}"
    result = subprocess.run(
        ["git", "ls-remote", "--tags", "https://github.com/pytorch/pytorch.git", git_ref],
        capture_output=True,
        text=True,
    )
    if not result.stdout.strip():
        log.info("Tag %s not found, using nightly branch", git_ref)
        git_ref = "nightly"

    log.info("Cloning PyTorch (ref=%s, sparse) into %s", git_ref, pytorch_src)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth=1",
            f"--branch={git_ref}",
            "--filter=blob:none",
            "--sparse",
            "https://github.com/pytorch/pytorch.git",
            str(pytorch_src),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", "test/", "torch/testing/"],
        cwd=pytorch_src,
        check=True,
    )

    test_file = pytorch_src / "test" / "distributed" / "test_c10d_nccl.py"
    if not test_file.exists():
        log.error("test_c10d_nccl.py not found after clone")
        sys.exit(1)
    log.info("Test sources ready: %s", test_file)


def print_environment_info() -> None:
    """Print GPU and environment details for CI logs."""
    import torch

    log.info("PyTorch: %s", torch.__version__)
    log.info("CUDA/HIP available: %s", torch.cuda.is_available())
    log.info("GPU count: %s", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        log.info("  GPU %d: %s", i, torch.cuda.get_device_name(i))
    log.info("LD_LIBRARY_PATH: %s", os.environ.get("LD_LIBRARY_PATH", ""))


def run_tests(pytorch_src: Path, results_log: Path) -> int:
    """Run pytest on test_c10d_nccl.py and return the exit code."""
    miopen_cache = tempfile.mkdtemp(prefix="miopen_cache_")
    os.environ["MIOPEN_USER_DB_PATH"] = miopen_cache

    env = os.environ.copy()
    env["PYTHONPATH"] = str(pytorch_src / "test") + ":" + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(pytorch_src / "test" / "distributed" / "test_c10d_nccl.py"),
        "-v",
        "--timeout=600",
        "--tb=short",
        "-k",
        "not NCCLTraceTestDumpOnTimeout",
    ]
    log.info("Running: %s", " ".join(cmd))

    with open(results_log, "w") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=pytorch_src,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_file.write(proc.stdout)
        # Also print to stdout for CI log visibility
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory containing CI-built artifacts",
    )
    parser.add_argument(
        "--pytorch-src",
        type=Path,
        required=True,
        help="Directory to clone PyTorch test sources into",
    )
    parser.add_argument(
        "--results-log",
        type=Path,
        default=Path("pytorch_c10d_results.log"),
        help="Path for test results log file",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover library paths and set GITHUB_OUTPUT, then exit",
    )

    args = parser.parse_args()

    # Step 1: Discover RCCL library path
    rccl_lib = find_rccl_library(args.artifact_dir)
    rccl_lib_dir = rccl_lib.parent
    rocm_lib_dir = find_rocm_lib_dir(args.artifact_dir)

    set_github_output("RCCL_LIB_DIR", str(rccl_lib_dir))
    if rocm_lib_dir:
        set_github_output("ROCM_LIB_DIR", str(rocm_lib_dir))

    if args.discover_only:
        return

    # Step 2: Set up LD_LIBRARY_PATH and verify override
    setup_ld_library_path(rccl_lib_dir, rocm_lib_dir)
    verify_rccl_override(rccl_lib_dir)

    # Step 3: Clone PyTorch test sources
    clone_pytorch_test_sources(args.pytorch_src)

    # Step 4: Print environment info and run tests
    print_environment_info()
    exit_code = run_tests(args.pytorch_src, args.results_log)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
