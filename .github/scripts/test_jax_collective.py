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
import re
import smtplib
import subprocess
import sys
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

JAX_REPO = "https://github.com/ROCm/jax.git"
JAX_REF = "66918cf7a6adef25e8f71dbebb954e6dd5393109"  # jax-v0.10.2-testing

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
        # Empty value disables all command buffer types (HIP graphs) on ROCm.
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
    lib_path = matches[0].resolve()
    log.info("Found librccl.so at: %s", lib_path)
    return lib_path


def find_lib_dirs(artifact_dir: Path) -> list[Path]:
    """Find all directories containing .so files in the artifact tree."""
    lib_dirs: set[Path] = set()
    for so_file in artifact_dir.rglob("*.so"):
        lib_dirs.add(so_file.parent.resolve())
    for so_file in artifact_dir.rglob("*.so.*"):
        lib_dirs.add(so_file.parent.resolve())
    sorted_dirs = sorted(lib_dirs)
    for d in sorted_dirs:
        log.info("Found lib dir: %s", d)
    return sorted_dirs


def create_soname_symlinks(lib_dirs: list[Path]) -> None:
    """Create missing SONAME symlinks (e.g. libfoo.so.1 -> libfoo.so.1.2.3).

    Artifact flattening can strip versioned symlinks. JAX wheels link
    against SONAME versions (librocprofiler-sdk.so.1) that may only
    exist as librocprofiler-sdk.so.1.0.0 after flattening.
    """
    for d in lib_dirs:
        for so_file in d.glob("*.so.*"):
            name = so_file.name
            # Match libfoo.so.X.Y.Z — create libfoo.so.X symlink
            parts = name.split(".so.")
            if len(parts) != 2:
                continue
            version_parts = parts[1].split(".")
            if len(version_parts) <= 1:
                continue
            soname = f"{parts[0]}.so.{version_parts[0]}"
            soname_path = d / soname
            if not soname_path.exists():
                soname_path.symlink_to(so_file.name)
                log.info("Created symlink: %s -> %s", soname_path, so_file.name)


def setup_ld_library_path(lib_dirs: list[Path]) -> str:
    """Prepend all artifact lib dirs to LD_LIBRARY_PATH."""
    create_soname_symlinks(lib_dirs)

    parts = [str(d) for d in lib_dirs]
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


def clone_jax_test_sources(jax_src: Path) -> None:
    """Sparse-clone ROCm/jax at a pinned commit to get test sources."""
    if jax_src.exists() and (jax_src / "tests" / "pmap_test.py").exists():
        log.info("JAX test sources already present at %s, skipping clone", jax_src)
        return

    log.info("Cloning ROCm/jax (commit=%s, sparse) into %s", JAX_REF[:12], jax_src)
    subprocess.run(
        [
            "git", "clone",
            "--depth=1",
            "--filter=blob:none",
            "--sparse",
            JAX_REPO,
            str(jax_src),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "fetch", "--depth=1", "origin", JAX_REF],
        cwd=jax_src,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", JAX_REF],
        cwd=jax_src,
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
    cmd = [
        sys.executable, "-m", "pytest",
        "-sv",
        "--timeout=120",
        "--tb=short",
    ] + SMOKE_TESTS

    log.info("Running: %s", " ".join(cmd))

    passed_tests = []
    failed_tests = []
    summary_line = ""
    current_test = ""

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
            test_header = re.match(r"tests/.*?::([\w:]+)", line)
            if test_header:
                current_test = test_header.group(1)
            if "PASSED" in line and re.search(r"PASSED\s+\[", line):
                dur_match = re.search(r"\[(\d+\.\d+s)\]", line)
                duration = dur_match.group(1) if dur_match else ""
                passed_tests.append((current_test, duration))
                current_test = ""
            elif "FAILED" in line and re.search(r"FAILED\s+\[", line):
                failed_tests.append(current_test)
                current_test = ""
            elif line.strip().startswith("=") and "passed" in line:
                summary_line = line.strip()
        proc.wait()

    log.info("Test exit code: %d", proc.returncode)
    log.info("Results written to: %s", results_log)

    summary = {
        "exit_code": proc.returncode,
        "passed": passed_tests,
        "failed": failed_tests,
        "summary_line": summary_line.strip("= "),
    }
    return proc.returncode, summary


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
        "",
    ]

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


def write_github_summary(report: str) -> None:
    """Write report to GITHUB_STEP_SUMMARY if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write("```\n")
            f.write(report)
            f.write("\n```\n")
        log.info("Summary written to GITHUB_STEP_SUMMARY")


def send_email_report(report: str, recipient: str, status: str) -> None:
    """Send the summary report via email."""
    subject = f"RCCL JAX Collective Test: {status}"
    msg = MIMEText(report)
    msg["Subject"] = subject
    msg["From"] = "rccl-ci@amd.com"
    msg["To"] = recipient

    smtp_servers = ["smtp.amd.com", "aussmtp.amd.com", "mail.amd.com", "localhost"]
    for server in smtp_servers:
        try:
            with smtplib.SMTP(server, timeout=10) as s:
                s.sendmail(msg["From"], [recipient], msg.as_string())
            log.info("Email sent to %s via %s", recipient, server)
            return
        except Exception as e:
            log.debug("SMTP %s failed: %s", server, e)
            continue
    log.warning("Could not send email to %s (tried: %s)", recipient, ", ".join(smtp_servers))


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

    # Step 2: Set up LD_LIBRARY_PATH and verify override
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

    if args.notify_email:
        status = "PASSED" if exit_code == 0 else "FAILED"
        send_email_report(report, args.notify_email, status)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
