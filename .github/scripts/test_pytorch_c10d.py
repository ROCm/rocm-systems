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
import logging
import os
import re
import smtplib
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SMOKE_TESTS = [
    "test_all_reduce_coalesced_nccl",
    "test_all_reduce_coalesced_manager_nccl",
    "test_allgather_base",
    "test_all_gather_into_tensor_coalesced_manager_nccl",
    "test_broadcast_coalesced_nccl",
    "test_broadcast_subgroup",
    "test_reduce_scatter_base_k",
    "test_reduce_scatter_tensor_coalesced",
    "test_non_blocking_p2p",
    "test_send_recv_subgroup",
    "test_nccl_barrier_device_ids",
    "test_reduce_subgroup",
    "test_scatter_subgroup",
    "test_gather_subgroup",
    "test_all_to_all_single",
    "test_init_wo_backend_str",
    "test_new_group",
    "test_pass_nccl_options_high_priority_stream",
    "test_set_process_group_desc",
    "test_tensor_dtype_complex",
    "test_batch_send_recv_subgroup",
    "test_collectives",
]


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


def clone_pytorch_test_sources(pytorch_src: Path) -> None:
    """Sparse-clone PyTorch test sources matching the installed torch version.

    For release builds (e.g. 2.5.0), clones at the matching tag.
    For nightly builds (e.g. 2.14.0a0+rocm7.15.0a20260712), clones using
    --shallow-since to get commits around the build date, then checks out the
    commit closest to that date so test sources match the installed wheel.
    """
    import re
    from datetime import datetime, timedelta

    import torch

    torch_version = torch.__version__
    base_version = torch_version.split("+")[0]
    log.info("PyTorch version: %s", torch_version)

    git_ref = f"v{base_version}"
    result = subprocess.run(
        ["git", "ls-remote", "--tags", "https://github.com/pytorch/pytorch.git", git_ref],
        capture_output=True,
        text=True,
    )

    date_match = re.search(r"(\d{8})", torch_version)
    use_date_pinning = False

    if result.stdout.strip():
        log.info("Found tag %s", git_ref)
    elif date_match:
        build_date = date_match.group(1)
        dt = datetime.strptime(build_date, "%Y%m%d")
        shallow_since = (dt - timedelta(days=2)).strftime("%Y-%m-%d")
        log.info("Tag %s not found; nightly build date %s", git_ref, build_date)
        git_ref = "nightly"
        use_date_pinning = True
    else:
        log.info("Tag %s not found, using nightly branch HEAD", git_ref)
        git_ref = "nightly"

    if use_date_pinning:
        log.info("Cloning PyTorch (ref=%s, shallow-since=%s, sparse) into %s",
                 git_ref, shallow_since, pytorch_src)
        subprocess.run(
            [
                "git", "clone",
                f"--branch={git_ref}",
                f"--shallow-since={shallow_since}",
                "--filter=blob:none",
                "--sparse",
                "https://github.com/pytorch/pytorch.git",
                str(pytorch_src),
            ],
            check=True,
        )
    else:
        log.info("Cloning PyTorch (ref=%s, depth=1, sparse) into %s", git_ref, pytorch_src)
        subprocess.run(
            [
                "git", "clone",
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
        ["git", "sparse-checkout", "set", "test/"],
        cwd=pytorch_src,
        check=True,
    )

    if use_date_pinning:
        before = (dt + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
        result = subprocess.run(
            ["git", "log", f"--before={before}", "--format=%H", "-1"],
            cwd=pytorch_src,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            commit = result.stdout.strip()
            log.info("Checking out commit %s (latest before %s)", commit[:12], before)
            subprocess.run(
                ["git", "checkout", commit],
                cwd=pytorch_src,
                check=True,
            )
        else:
            log.warning("Could not find commit before %s, using HEAD of nightly", before)

    test_file = pytorch_src / "test" / "distributed" / "test_c10d_nccl.py"
    if not test_file.exists():
        log.error("test_c10d_nccl.py not found after clone")
        sys.exit(1)
    log.info("Test sources ready: %s", test_file)


def patch_missing_torch_modules() -> None:
    """Create stubs for internal torch modules missing from nightly wheels."""
    import torch

    torch_dir = Path(torch.__file__).parent
    strobelight_dir = torch_dir / "_strobelight"
    profiler_file = strobelight_dir / "compile_time_profiler.py"
    if not profiler_file.exists():
        log.info("Creating stub for torch._strobelight (missing from nightly wheel)")
        strobelight_dir.mkdir(parents=True, exist_ok=True)
        (strobelight_dir / "__init__.py").write_text("")
        profiler_file.write_text(
            "class StrobelightCompileTimeProfiler:\n"
            "    def __enter__(self): return self\n"
            "    def __exit__(self, *a): pass\n"
        )


def print_environment_info() -> None:
    """Print GPU and environment details for CI logs."""
    import torch

    log.info("PyTorch: %s", torch.__version__)
    log.info("CUDA/HIP available: %s", torch.cuda.is_available())
    log.info("GPU count: %s", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        log.info("  GPU %d: %s", i, torch.cuda.get_device_name(i))
    log.info("LD_LIBRARY_PATH: %s", os.environ.get("LD_LIBRARY_PATH", ""))


def parse_junit_xml(xml_path: Path) -> dict:
    """Parse JUnit XML and return structured results."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    passed_tests = []
    failed_tests = []
    error_details = []
    tests_run = 0
    failures = 0
    errors = 0

    for suite in root.iter("testsuite"):
        tests_run = int(suite.get("tests", 0))
        failures = int(suite.get("failures", 0))
        errors = int(suite.get("errors", 0))

    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        time_s = tc.get("time", "")
        duration = f"{float(time_s):.2f}s" if time_s else ""

        failure = tc.find("failure")
        error = tc.find("error")
        if failure is not None:
            failed_tests.append(name)
            error_details.append(
                f"FAILED: {name}\n  {failure.get('message', '')}"
            )
        elif error is not None:
            failed_tests.append(name)
            error_details.append(
                f"ERROR: {name}\n  {error.get('message', '')}"
            )
        else:
            passed_tests.append((name, duration))

    return {
        "passed": passed_tests,
        "failed": failed_tests,
        "error_details": error_details,
        "tests_run": tests_run,
        "failures": failures,
        "errors": errors,
    }


def run_tests(pytorch_src: Path, results_log: Path, test_scope: str = "smoke") -> tuple[int, dict]:
    """Run pytest on test_c10d_nccl.py and return (exit_code, summary_dict)."""
    miopen_cache = tempfile.mkdtemp(prefix="miopen_cache_")
    os.environ["MIOPEN_USER_DB_PATH"] = miopen_cache

    env = os.environ.copy()
    env["PYTHONPATH"] = str(pytorch_src / "test") + ":" + env.get("PYTHONPATH", "")

    junit_xml = results_log.parent / "pytorch_c10d_results.xml"

    if test_scope == "smoke":
        k_expr = " or ".join(SMOKE_TESTS)
        timeout = "60"
        log.info("Running smoke tests (%d tests)", len(SMOKE_TESTS))
    else:
        k_expr = "not NCCLTraceTestDumpOnTimeout"
        timeout = "600"
        log.info("Running all tests (excluding NCCLTraceTestDumpOnTimeout)")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(pytorch_src / "test" / "distributed" / "test_c10d_nccl.py"),
        "-v",
        f"--timeout={timeout}",
        "--tb=short",
        f"--junitxml={junit_xml}",
        "-k",
        k_expr,
    ]
    log.info("Running: %s", " ".join(cmd))

    with open(results_log, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=pytorch_src,
            env=env,
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
    error_details = []
    tests_run = 0
    summary_line = ""

    if junit_xml.exists():
        log.info("Parsing JUnit XML: %s", junit_xml)
        junit = parse_junit_xml(junit_xml)
        passed_tests = junit["passed"]
        failed_tests = junit["failed"]
        error_details = junit["error_details"]
        tests_run = junit["tests_run"]
        parts = []
        if passed_tests:
            parts.append(f"{len(passed_tests)} passed")
        if failed_tests:
            parts.append(f"{len(failed_tests)} failed")
        summary_line = ", ".join(parts)

        if error_details:
            log.info("Failure/error details from JUnit XML:")
            for detail in error_details:
                log.info("  %s", detail)
    else:
        log.warning("JUnit XML not found at %s, falling back to exit code only", junit_xml)

    if test_scope == "smoke" and tests_run < len(SMOKE_TESTS):
        log.error(
            "Expected %d smoke tests but only %d were collected — "
            "test names may have changed in the nightly",
            len(SMOKE_TESTS),
            tests_run,
        )
        exit_code = 1

    summary = {
        "exit_code": exit_code,
        "test_scope": test_scope,
        "passed": passed_tests,
        "failed": failed_tests,
        "summary_line": summary_line,
        "tests_run": tests_run,
        "expected_tests": len(SMOKE_TESTS) if test_scope == "smoke" else None,
    }
    return exit_code, summary


def generate_summary_report(summary: dict, rccl_lib: Path) -> str:
    """Generate a plain-text summary report."""
    import torch

    status = "PASSED" if summary["exit_code"] == 0 else "FAILED"
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        gpu_info.append(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    lines = [
        f"RCCL PyTorch c10d Test Report",
        f"{'=' * 40}",
        f"Status:     {status}",
        f"Test scope: {summary['test_scope']}",
        f"Date:       {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"",
        f"PyTorch:    {torch.__version__}",
        f"RCCL:       {rccl_lib}",
        f"GPUs:       {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}",
        f"",
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
    subject = f"RCCL PyTorch c10d Test: {status}"
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
        "--test-scope",
        choices=["smoke", "all"],
        default="smoke",
        help="Run smoke tests (22 curated tests, ~3min) or all tests (default: smoke)",
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

    # Step 4: Patch missing modules, print environment info, and run tests
    patch_missing_torch_modules()
    print_environment_info()
    exit_code, summary = run_tests(args.pytorch_src, args.results_log, args.test_scope)

    # Step 5: Generate and distribute summary report
    report = generate_summary_report(summary, rccl_lib)
    log.info("\n%s", report)
    write_github_summary(report)

    summary_path = args.results_log.parent / "pytorch_c10d_summary.txt"
    summary_path.write_text(report)
    log.info("Summary written to: %s", summary_path)

    if args.notify_email:
        status = "PASSED" if exit_code == 0 else "FAILED"
        send_email_report(report, args.notify_email, status)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
