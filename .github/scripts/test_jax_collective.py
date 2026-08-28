#!/usr/bin/env python3
"""Run JAX collective smoke tests against CI-built RCCL.

This script handles:
  1. Discovering the CI-built librccl.so in the artifact directory
  2. Prepending its directory to LD_LIBRARY_PATH so JAX loads it
  3. Cloning the matching JAX test sources (sparse checkout)
  4. Running pytest on pmap_test.py and shard_map_test.py

Usage from GitHub Actions:
  python .github/scripts/test_jax_collective.py \
      --artifact-dir ./build \
      --jax-src ./jax-src \
      --results-log ./jax_collective_results.log
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from rccl_ci_utils import (
    find_rccl_library,
    parse_junit_xml,
    send_email_report,
    send_teams_webhook,
    set_github_output,
    verify_rccl_override,
    write_github_summary,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

JAX_REPO = "https://github.com/ROCm/jax.git"

SMOKE_TEST_FILES = [
    "tests/pmap_test.py",
    "tests/shard_map_test.py",
]

SMOKE_TEST_KEYWORDS = [
    "PythonPmapTest and testBasic",
    "PythonPmapTest and testGather and not testGatherBool and not testGatherNeg and not testGatherTiled and not testGatherReplica",
    "PythonPmapTest and testReduceScatter and not Tiled and not Replica",
    "PythonPmapTest and testCollectivePermute and not Grad and not Cyclic",
    "PythonPmapTest and testAllToAll and not Replica and not Vmap and not Grad",
    "ShardMapTest and test_all_gather and not invariant and not axis_index",
    "ShardMapTest and test_matmul_reduce_scatter",
    "ShardMapTest and test_collective_permute and not multiple",
    "ShardMapTest and test_axis_index and not basic and not twoaxes and not eager",
    "ShardMapTest and test_all_to_all and not axis_index and not grad",
]

XLA_ENV = {
    "XLA_PYTHON_CLIENT_ALLOCATOR": "default",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_FLAGS": (
        "--xla_gpu_force_compilation_parallelism=1 "
        "--xla_gpu_enable_nccl_comm_splitting=false "
        # Empty value disables all command buffer types (HIP graphs) on ROCm.
        "--xla_gpu_enable_command_buffer= "
        "--xla_gpu_enable_cublaslt=false"
    ),
}


def find_lib_dirs(artifact_dir: Path) -> list[Path]:
    """Find all directories containing .so files in the artifact tree."""
    lib_dirs: set[Path] = set()
    for so_file in artifact_dir.rglob("*.so"):
        lib_dirs.add(so_file.parent.resolve())
    for so_file in artifact_dir.rglob("*.so.*"):
        lib_dirs.add(so_file.parent.resolve())
    sorted_dirs = sorted(lib_dirs)
    for d in sorted_dirs:
        count = sum(1 for f in d.iterdir() if ".so" in f.name)
        log.info("Found lib dir: %s (%d libs)", d, count)
    return sorted_dirs


def populate_rocm_lib_dir(lib_dirs: list[Path]) -> None:
    """Populate /opt/rocm/lib with symlinks to artifact libraries.

    JAX's xla_rocm_plugin.so has RUNPATH including /opt/rocm/lib. With ELF
    RUNPATH semantics, transitive dependencies are resolved via RUNPATH
    rather than LD_LIBRARY_PATH. In a container with no system ROCm, we
    populate /opt/rocm/lib so the loader can find them.
    """
    rocm_lib = Path("/opt/rocm/lib")
    try:
        rocm_lib.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        log.warning("Cannot create %s — not running as root", rocm_lib)
        return

    count = 0
    for d in lib_dirs:
        for so_file in d.iterdir():
            if ".so" not in so_file.name:
                continue
            target = rocm_lib / so_file.name
            if target.exists() or target.is_symlink():
                continue
            target.symlink_to(so_file.resolve())
            count += 1
    log.info("Created %d symlinks in %s", count, rocm_lib)


def setup_ld_library_path(lib_dirs: list[Path]) -> str:
    """Prepend all artifact lib dirs to LD_LIBRARY_PATH."""
    parts = [str(d) for d in lib_dirs]
    rocm_lib = Path("/opt/rocm/lib")
    if rocm_lib.is_dir():
        parts.insert(0, str(rocm_lib))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    new_path = ":".join(parts)
    os.environ["LD_LIBRARY_PATH"] = new_path
    log.info("LD_LIBRARY_PATH=%s", new_path)
    return new_path


def setup_xla_environment() -> None:
    """Set XLA environment variables required for JAX on ROCm."""
    for key, value in XLA_ENV.items():
        os.environ[key] = value
        log.info("Set %s=%s", key, value)


def clone_jax_test_sources(jax_src: Path) -> None:
    """Sparse-clone ROCm/jax to get test sources matching the installed version.

    Tries to find a tag matching the installed JAX version (e.g. jax-v0.5.3).
    Falls back to the default branch HEAD for nightly/dev builds.
    """
    if jax_src.exists() and (jax_src / "tests" / "pmap_test.py").exists():
        log.info("JAX test sources already present at %s, skipping clone", jax_src)
        return

    import jax
    jax_version = jax.__version__
    base_version = jax_version.split(".dev")[0].split("+")[0]
    git_ref = f"jax-v{base_version}"
    log.info("JAX version: %s, trying tag: %s", jax_version, git_ref)

    result = subprocess.run(
        ["git", "ls-remote", "--tags", JAX_REPO, git_ref],
        capture_output=True, text=True,
    )
    if not result.stdout.strip():
        log.info("Tag %s not found, using default branch HEAD", git_ref)
        git_ref = None

    clone_cmd = [
        "git", "clone",
        "--depth=1",
        "--filter=blob:none",
        "--sparse",
        JAX_REPO,
        str(jax_src),
    ]
    if git_ref:
        clone_cmd.insert(-1, f"--branch={git_ref}")

    log.info("Cloning ROCm/jax (ref=%s, sparse) into %s", git_ref or "HEAD", jax_src)
    subprocess.run(clone_cmd, check=True)
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


def print_environment_info() -> None:
    """Print GPU and environment details for CI logs."""
    log.info("--- Environment Info ---")
    try:
        import jax
        log.info("JAX version: %s", jax.__version__)
        devices = jax.devices()
        log.info("Devices: %s", devices)
        log.info("Device count: %d", jax.device_count())
        log.info("Local device count: %d", jax.local_device_count())
        gpu_devices = [d for d in devices if d.platform != "cpu"]
        if not gpu_devices:
            log.error("No GPU devices found — JAX fell back to CPU only")
            log.error("Check that ROCm libraries are on LD_LIBRARY_PATH")
            sys.exit(1)
    except Exception as e:
        log.error("Failed to query JAX devices: %s", e)
        sys.exit(1)

    log.info("LD_LIBRARY_PATH: %s", os.environ.get("LD_LIBRARY_PATH", ""))
    for key in sorted(XLA_ENV):
        log.info("%s=%s", key, os.environ.get(key, "<unset>"))
    log.info("--- End Environment Info ---")


def run_tests(jax_src: Path, results_log: Path) -> tuple[int, dict]:
    """Run pytest on the 10 collective smoke tests and return (exit_code, summary)."""
    junit_xml = results_log.parent / "jax_collective_results.xml"

    k_expr = " or ".join(f"({kw})" for kw in SMOKE_TEST_KEYWORDS)
    cmd = [
        sys.executable, "-m", "pytest",
        "-sv",
        "--timeout=120",
        "--tb=short",
        f"--junitxml={junit_xml}",
        "-k", k_expr,
    ] + SMOKE_TEST_FILES

    log.info("Running: %s", " ".join(cmd))

    with open(results_log, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=jax_src,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        proc.wait()

    log.info("Test exit code: %d", proc.returncode)
    log.info("Results written to: %s", results_log)

    exit_code = proc.returncode
    passed_tests = []
    failed_tests = []
    tests_run = 0
    summary_line = ""

    if junit_xml.exists():
        log.info("Parsing JUnit XML: %s", junit_xml)
        junit = parse_junit_xml(junit_xml)
        passed_tests = junit["passed"]
        failed_tests = junit["failed"]
        tests_run = junit["tests_run"]
        parts = []
        if passed_tests:
            parts.append(f"{len(passed_tests)} passed")
        if failed_tests:
            parts.append(f"{len(failed_tests)} failed")
        summary_line = ", ".join(parts)

        if junit["error_details"]:
            log.info("Failure/error details from JUnit XML:")
            for detail in junit["error_details"]:
                log.info("  %s", detail)
    else:
        log.warning("JUnit XML not found at %s, falling back to exit code only", junit_xml)

    if tests_run < len(SMOKE_TEST_KEYWORDS):
        log.error(
            "Expected at least %d smoke tests but only %d were collected — "
            "tests may have been skipped or deselected",
            len(SMOKE_TEST_KEYWORDS),
            tests_run,
        )
        exit_code = 1

    summary = {
        "exit_code": exit_code,
        "passed": passed_tests,
        "failed": failed_tests,
        "summary_line": summary_line,
        "tests_run": tests_run,
        "expected_tests": len(SMOKE_TEST_KEYWORDS),
    }
    return exit_code, summary


def generate_summary_report(summary: dict, rccl_lib: Path) -> str:
    """Generate a plain-text summary report."""
    import jax

    status = "PASSED" if summary["exit_code"] == 0 else "FAILED"
    devices = jax.devices()
    gpu_name = str(devices[0]) if devices else "unknown"

    lines = [
        "RCCL JAX Collective Test Report",
        "=" * 40,
        f"Status:     {status}",
        f"Date:       {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        f"JAX:        {jax.__version__}",
        f"RCCL:       {rccl_lib}",
        f"GPUs:       {len(devices)}x {gpu_name}",
        "",
        f"Results:    {summary['summary_line']}",
    ]

    if summary.get("expected_tests") is not None:
        lines.append(f"Collected:  {summary['tests_run']}/{summary['expected_tests']} expected smoke tests")
    lines.append("")

    if summary["failed"]:
        lines.append(f"FAILED tests ({len(summary['failed'])}):")
        for name in summary["failed"]:
            lines.append(f"  FAIL  {name}")
        lines.append("")

    if summary["passed"]:
        lines.append(f"PASSED tests ({len(summary['passed'])}):")
        for name, duration in summary["passed"]:
            lines.append(f"  OK    {name:60s} {duration}")
        lines.append("")

    run_url = os.environ.get("GITHUB_SERVER_URL", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if run_url and repo and run_id:
        lines.append(f"CI run: {run_url}/{repo}/actions/runs/{run_id}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory containing CI-built artifacts",
    )
    parser.add_argument(
        "--jax-src",
        type=Path,
        required=True,
        help="Directory to clone JAX test sources into",
    )
    parser.add_argument(
        "--results-log",
        type=Path,
        default=Path("jax_collective_results.log"),
        help="Path for test results log file",
    )
    parser.add_argument(
        "--notify-email",
        type=str,
        default="",
        help="Send summary report to this email address",
    )
    parser.add_argument(
        "--notify-webhook",
        type=str,
        default="",
        help="Send summary report to this Teams webhook URL",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover library paths and set GITHUB_OUTPUT, then exit",
    )

    args = parser.parse_args()

    # Step 1: Discover RCCL library and all lib dirs in artifacts
    rccl_lib = find_rccl_library(args.artifact_dir)
    rccl_lib_dir = rccl_lib.parent
    lib_dirs = find_lib_dirs(args.artifact_dir)

    set_github_output("RCCL_LIB_DIR", str(rccl_lib_dir))

    if args.discover_only:
        return

    # Step 2: Set up library paths and verify override
    populate_rocm_lib_dir(lib_dirs)
    setup_ld_library_path(lib_dirs)
    verify_rccl_override(rccl_lib_dir)

    # Step 3: Set XLA environment variables
    setup_xla_environment()

    # Step 4: Clone JAX test sources
    clone_jax_test_sources(args.jax_src)

    # Step 5: Print environment info and run tests
    print_environment_info()
    exit_code, summary = run_tests(args.jax_src, args.results_log)

    # Step 6: Generate and distribute summary report
    report = generate_summary_report(summary, rccl_lib)
    log.info("\n%s", report)
    write_github_summary(report)

    summary_path = args.results_log.parent / "jax_collective_summary.txt"
    summary_path.write_text(report)
    log.info("Summary written to: %s", summary_path)

    status = "PASSED" if exit_code == 0 else "FAILED"
    if args.notify_email:
        send_email_report(report, args.notify_email, status,
                          subject_prefix="RCCL JAX Collective Test")
    if args.notify_webhook:
        send_teams_webhook(report, args.notify_webhook, status,
                           subject_prefix="RCCL JAX Collective Test")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
