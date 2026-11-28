#!/usr/bin/env python3
"""Run PyTorch c10d NCCL distributed tests against CI-built RCCL.

This script handles:
  1. Discovering the CI-built librccl.so in the artifact directory
  2. Replacing PyTorch's pip-bundled librccl.so with the CI-built version
  3. Cloning the matching PyTorch test sources (sparse checkout)
  4. Running pytest on the c10d tests that use the default NCCL backend

Usage from GitHub Actions:
  python projects/rccl/ci/scripts/test_pytorch_c10d.py \
      --artifact-dir ./build \
      --pytorch-src ./pytorch-src \
      --results-log ./pytorch_c10d_results.log
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rccl_ci_utils import (
    SetupFailureReport,
    find_pip_sdk_lib_dirs,
    find_rccl_library,
    override_bundled_hip_runtime,
    override_bundled_hsa_runtime,
    override_bundled_rccl,
    parse_junit_xml,
    quarantine_rocm_sysdeps,
    reconcile_soname_versions,
    send_email_report,
    send_teams_webhook,
    set_github_output,
    setup_kpack_device_code,
    write_github_summary,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PYTORCH_REPO = "https://github.com/pytorch/pytorch.git"

# OSS c10d tests that still reach RCCL via the default nccl backend
# (AICOMRCCL-4001); "file::Class" runs one class.
SMOKE_TARGETS = (
    "test_c10d_nccl.py",
    "test_c10d_ops_nccl.py",
)
ALL_TARGETS = SMOKE_TARGETS + (
    # The rest of the file is CPU, fake process group or compile-only.
    "test_c10d_functional_native.py::TestWithNCCL",
    "test_c10d_spawn_nccl.py",
    "test_c10d_object_collectives.py",
)

# Each must run under exactly this name; -k matches substrings, so more run.
SMOKE_TESTS = [
    # test_c10d_nccl.py
    "test_init_wo_backend_str",
    "test_init_with_idx",
    "test_set_nccl_pg_timeout_backend0",
    # test_c10d_ops_nccl.py
    "test_empty_tensors",
    "test_broadcast_ops",
    "test_allreduce_ops",
    "test_reduce_ops",
    "test_allgather_ops",
    "test_allgather_base_ops",
    "test_gather_ops",
    "test_scatter_ops",
    "test_reduce_scatter_ops",
    "test_reduce_scatter_base_ops",
    "test_all_gather_v",
    "test_reduce_scatter_v",
    "test_alltoall_ops_with_cudafree_race",
    "test_barrier",
    "test_send_recv",
    "test_reduce_op_premul_sum",
]

# Fewest tests that must run (pass or fail) per file, so a file upstream
# empties or breaks fails the all scope.
ALL_MIN_RAN = {
    "test_c10d_nccl.py": 5,
    "test_c10d_ops_nccl.py": 28,
    "test_c10d_functional_native.py": 16,
    "test_c10d_spawn_nccl.py": 9,
    "test_c10d_object_collectives.py": 13,
}

REPORT_SUBJECT = "RCCL PyTorch c10d Test"

setup_report = SetupFailureReport(REPORT_SUBJECT)


def target_file(target: str) -> str:
    """The file part of a pytest target: "f.py::Class" -> "f.py"."""
    return target.split("::", 1)[0]


def case_key(case: dict) -> str:
    """"Class.test" for a JUnit testcase, without the module path."""
    cls = case["classname"].rsplit(".", 1)[-1]
    return f"{cls}.{case['name']}" if cls else case["name"]


def is_known(case: dict, known_set: set[str]) -> bool:
    """Known failures are keyed by bare name, or by Class.test where a name
    occurs in more than one file."""
    return case["name"] in known_set or case_key(case) in known_set


def file_of(case: dict, files) -> str | None:
    """Which of `files` a JUnit testcase came from.

    Uses the testcase's file attribute (junit_family=xunit1), else the module
    in its dotted classname, e.g. "test.distributed.test_c10d_ops_nccl.T".
    """
    path = case.get("file", "").replace("\\", "/")
    if path:
        return next((f for f in files if path == f or path.endswith("/" + f)), None)
    dotted = f".{case['classname']}."
    return next((f for f in files if f".{f.removesuffix('.py')}." in dotted), None)


def ran(case: dict) -> bool:
    return case["outcome"] in ("passed", "failed")


def distinct_tests(cases: list[dict]) -> list[dict]:
    """One testcase per test: pytest writes two for a test whose call fails
    and whose teardown then errors."""
    return list({(c.get("file", ""), c["classname"], c["name"]): c for c in cases}.values())


def missing_smoke_tests(smoke_tests: list[str], cases: list[dict]) -> list[str]:
    """Smoke tests with no testcase of exactly that name that ran; a skipped
    test gives no coverage."""
    ran_names = {c["name"] for c in cases if ran(c)}
    return [t for t in smoke_tests if t not in ran_names]


def short_files(min_ran: dict[str, int], cases: list[dict]) -> list[str]:
    """Files in which fewer tests ran than their floor."""
    counts = Counter(file_of(c, min_ran) for c in cases if ran(c))
    return [f"{f}: {counts[f]} test(s) ran, expected at least {n}"
            for f, n in min_ran.items() if counts[f] < n]


def judge(test_scope: str, cases: list[dict], known_set: set[str],
          returncode: int) -> dict:
    """Turn the JUnit testcases of one run into a verdict."""
    errors: list[str] = []
    cases = distinct_tests(cases)
    failed = [c for c in cases if c["outcome"] == "failed"]
    known_failed = [case_key(c) for c in failed if is_known(c, known_set)]
    unexpected_failed = [case_key(c) for c in failed if not is_known(c, known_set)]

    if test_scope == "smoke":
        missing = missing_smoke_tests(SMOKE_TESTS, cases)
        if missing:
            errors.append(f"{len(missing)} smoke test(s) did not run "
                          f"(missing, renamed or skipped): {', '.join(missing)}")
        smoke_ran = len(SMOKE_TESTS) - len(missing)
    else:
        errors.extend(short_files(ALL_MIN_RAN, cases))
        smoke_ran = None
    if unexpected_failed:
        errors.append(f"{len(unexpected_failed)} unexpected failure(s)")

    # pytest exits 1 for test failures; anything else (interrupted, internal
    # or usage error, nothing collected) fails the run on its own.
    if returncode not in (0, 1):
        errors.append(f"pytest exited {returncode}")
    elif returncode == 1 and not failed:
        errors.append("pytest exited 1 but the JUnit XML records no failure")

    passed = [c for c in cases if c["outcome"] == "passed"]
    now_passing = sorted(k for k in known_set if any(is_known(c, {k}) for c in passed))
    return {
        "exit_code": 1 if errors else 0,
        "errors": errors,
        "known_failed": known_failed,
        "unexpected_failed": unexpected_failed,
        "now_passing": now_passing,
        "smoke_ran": smoke_ran,
        "ran_per_file": Counter(file_of(c, ALL_MIN_RAN) for c in cases if ran(c)),
    }


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
    sysdeps_lib = rccl_lib_dir.resolve() / "rocm_sysdeps" / "lib"
    if sysdeps_lib.is_dir():
        parts.append(str(sysdeps_lib))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    new_path = ":".join(parts)
    os.environ["LD_LIBRARY_PATH"] = new_path
    log.info("LD_LIBRARY_PATH=%s", new_path)
    return new_path


def clone_pytorch_test_sources(pytorch_src: Path, targets: tuple[str, ...]) -> None:
    """Sparse-check-out test/ at the commit the installed torch was built from.

    The date in a nightly version string is the ROCm build date, not the
    PyTorch source date, so use torch.version.git_version instead.
    """
    import re

    import torch

    commit = torch.version.git_version
    log.info("PyTorch version: %s, git_version: %s", torch.__version__, commit)
    if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
        log.error("torch.version.git_version is not a commit SHA; "
                  "cannot match test sources to the installed torch")
        sys.exit(1)

    log.info("Fetching PyTorch test sources at %s (sparse) into %s", commit, pytorch_src)
    pytorch_src.mkdir(parents=True, exist_ok=True)
    for cmd in (
        ["git", "init", "-q"],
        ["git", "remote", "add", "origin", PYTORCH_REPO],
        ["git", "sparse-checkout", "set", "test/"],
        ["git", "fetch", "--depth=1", "--filter=blob:none", "origin", commit],
        ["git", "checkout", "-q", "FETCH_HEAD"],
    ):
        subprocess.run(cmd, cwd=pytorch_src, check=True)

    dist_dir = pytorch_src / "test" / "distributed"
    missing = [f for f in map(target_file, targets) if not (dist_dir / f).exists()]
    if missing:
        log.error("Test files not found after clone: %s", ", ".join(missing))
        sys.exit(1)
    log.info("Test sources ready: %d file(s) under %s", len(targets), dist_dir)


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



def nccl_backend_description() -> str:
    """What the installed torch builds the default "nccl" backend from."""
    from torch.distributed import distributed_c10d as c10d

    aliased = getattr(c10d, "ProcessGroupNCCL", None) is getattr(c10d, "ProcessGroupNCCL2", object())
    legacy = getattr(c10d, "_NCCL_LEGACY_AVAILABLE", None)
    return (f"ProcessGroupNCCL is ProcessGroupNCCL2: {aliased}, "
            f"_NCCL_LEGACY_AVAILABLE: {legacy}")


def print_environment_info() -> None:
    """Print GPU and environment details for CI logs."""
    import torch

    log.info("PyTorch: %s", torch.__version__)
    log.info("CUDA/HIP available: %s", torch.cuda.is_available())
    log.info("GPU count: %s", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        log.info("  GPU %d: %s", i, torch.cuda.get_device_name(i))
    log.info("c10d NCCL backend: %s", nccl_backend_description())
    log.info("LD_LIBRARY_PATH: %s", os.environ.get("LD_LIBRARY_PATH", ""))


def run_tests(pytorch_src: Path, results_log: Path, test_scope: str = "smoke") -> tuple[int, dict]:
    """Run pytest on the scope's targets and return (exit_code, summary_dict)."""
    miopen_cache = tempfile.mkdtemp(prefix="miopen_cache_")
    os.environ["MIOPEN_USER_DB_PATH"] = miopen_cache

    env = os.environ.copy()
    env["PYTHONPATH"] = str(pytorch_src / "test") + ":" + env.get("PYTHONPATH", "")

    junit_xml = results_log.parent / "pytorch_c10d_results.xml"

    if test_scope == "smoke":
        targets = SMOKE_TARGETS
        k_expr = " or ".join(SMOKE_TESTS)
        # --timeout must exceed MultiProcContinuousTest's wait for its workers, or
        # a hung test's result is reported under the next test.
        env["DISTRIBUTED_TESTS_DEFAULT_TIMEOUT"] = "120"
        timeout = "300"
        log.info("Running smoke tests (%d tests)", len(SMOKE_TESTS))
    else:
        targets = ALL_TARGETS
        k_expr = "not NCCLTraceTestDumpOnTimeout"
        timeout = "600"
        log.info("Running all tests (excluding NCCLTraceTestDumpOnTimeout)")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *(str(pytorch_src / "test" / "distributed" / t) for t in targets),
        "-v",
        f"--timeout={timeout}",
        "--tb=short",
        f"--junitxml={junit_xml}",
        # Record each testcase's file, for the per-file floors.
        "-o", "junit_family=xunit1",
        # PyTorch's conftest otherwise leaves tests skipped at collection out
        # of the report; it asks JUnit runs to turn the plugin off.
        "-p", "no:prune_skips_plugin",
        # test_c10d_nccl.py asserts at import that its five tests exist; if
        # one is renamed, the other files must still run.
        "--continue-on-collection-errors",
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
        setup_report.tests_started()
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        proc.wait()

    log.info("Test exit code: %d", proc.returncode)
    log.info("Results written to: %s", results_log)

    junit = {"passed": [], "cases": [], "error_details": []}
    if junit_xml.exists():
        log.info("Parsing JUnit XML: %s", junit_xml)
        junit = parse_junit_xml(junit_xml)
        if junit["error_details"]:
            log.info("Failure/error details from JUnit XML:")
            for detail in junit["error_details"]:
                log.info("  %s", detail)
    else:
        # No XML means no evidence any test ran; judge() fails the run.
        log.error("JUnit XML not found at %s", junit_xml)

    known_failures_file = Path(__file__).parent / "known_failures.json"
    known_set: set[str] = set()
    if known_failures_file.exists():
        with open(known_failures_file) as f:
            known_data = json.load(f)
        known_set = set(known_data.get("tests", {}).keys())
        log.info("Loaded %d known failures from %s", len(known_set), known_failures_file.name)

    cases = junit["cases"]
    verdict = judge(test_scope, cases, known_set, proc.returncode)
    for err in verdict["errors"]:
        log.error("%s", err)
    if verdict["unexpected_failed"]:
        log.error("Unexpected failures: %s", verdict["unexpected_failed"])
    elif verdict["known_failed"] and verdict["exit_code"] == 0:
        log.info("All %d failures are known — treating as PASSED", len(verdict["known_failed"]))
    if verdict["now_passing"]:
        log.warning(
            "%d known failure(s) now PASSING — consider removing from "
            "known_failures.json: %s", len(verdict["now_passing"]), verdict["now_passing"],
        )

    passed_tests = junit["passed"]
    skipped = sum(1 for c in cases if c["outcome"] == "skipped")
    parts = []
    if passed_tests:
        parts.append(f"{len(passed_tests)} passed")
    if verdict["unexpected_failed"]:
        parts.append(f"{len(verdict['unexpected_failed'])} unexpected failures")
    if verdict["known_failed"]:
        parts.append(f"{len(verdict['known_failed'])} known failures")
    if skipped:
        parts.append(f"{skipped} skipped")
    summary_line = ", ".join(parts) or "no tests ran"

    smoke = test_scope == "smoke"
    scope_files = [target_file(t) for t in targets]
    summary = {
        "exit_code": verdict["exit_code"],
        "test_scope": test_scope,
        "passed": passed_tests,
        "known_failed": verdict["known_failed"],
        "unexpected_failed": verdict["unexpected_failed"],
        "errors": verdict["errors"],
        "summary_line": summary_line,
        # For smoke, how many of SMOKE_TESTS actually ran; the dashboard reads
        # it from the "Collected:" line. Otherwise distinct testcases.
        "tests_run": (verdict["smoke_ran"] if smoke
                      else len({(c["classname"], c["name"]) for c in cases})),
        "expected_tests": len(SMOKE_TESTS) if smoke else None,
        "ran_per_file": {f: verdict["ran_per_file"][f] for f in scope_files},
    }
    return verdict["exit_code"], summary


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
        f"GPUs:       {torch.cuda.device_count()}x {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'}",
        f"c10d NCCL:  {nccl_backend_description()}",
        f"",
        f"Results:    {summary['summary_line']}",
    ]

    if summary.get("expected_tests") is not None:
        lines.append(f"Collected:  {summary['tests_run']}/{summary['expected_tests']} expected smoke tests")
    lines.append("")

    if summary.get("ran_per_file"):
        lines.append("Tests run per file:")
        for f, n in summary["ran_per_file"].items():
            lines.append(f"  {n:4d}  {f}")
        lines.append("")

    if summary.get("errors"):
        lines.append("Verdict errors:")
        for err in summary["errors"]:
            lines.append(f"  {err}")
        lines.append("")

    unexpected = summary.get("unexpected_failed", [])
    known = summary.get("known_failed", [])

    if unexpected:
        lines.append(f"UNEXPECTED failures ({len(unexpected)}):")
        for name in unexpected:
            lines.append(f"  FAIL  {name}")
        lines.append("")

    if known:
        lines.append(f"Known failures ({len(known)}) — excluded from verdict:")
        for name in known:
            lines.append(f"  KNOWN {name}")
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
        help="Run the curated smoke tests or all tests (default: smoke)",
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
    setup_report.notify(args.notify_email, args.notify_webhook)

    # Step 1: Discover RCCL library path
    setup_report.step("Discover RCCL library")
    rccl_lib = find_rccl_library(args.artifact_dir)
    rccl_lib_dir = rccl_lib.parent
    rocm_lib_dir = find_rocm_lib_dir(args.artifact_dir)

    set_github_output("RCCL_LIB_DIR", str(rccl_lib_dir))
    if rocm_lib_dir:
        set_github_output("ROCM_LIB_DIR", str(rocm_lib_dir))

    if args.discover_only:
        return

    # Step 2: Create symlinks for soname version mismatches between
    # CI-built libraries and pip-installed packages (must run before
    # quarantine so pip dirs get compatibility symlinks)
    setup_report.step("Reconcile library soname versions")
    pip_lib_dirs = find_pip_sdk_lib_dirs()
    reconcile_soname_versions([rccl_lib_dir] + pip_lib_dirs)

    # Step 3: Quarantine TheRock-bundled libamd_smi and rocm_sysdeps to
    # prevent the nl_genl destructor crash (SIGSEGV in containers).
    # After this, libamd_smi resolves from pip dirs (via symlinks above).
    setup_report.step("Quarantine TheRock amd-smi and sysdeps")
    quarantine_rocm_sysdeps(rccl_lib_dir)

    # Step 4: Replace pip-bundled librccl.so with CI-built version
    setup_report.step("Replace pip-bundled RCCL")
    override_bundled_rccl(rccl_lib_dir)

    # Step 4b: Replace pip-bundled libamdhip64.so with TheRock's version
    # so the HIP runtime matches the compiler that built RCCL's device
    # code objects (prevents findSymbol → guarantee → abort)
    setup_report.step("Replace pip-bundled HIP runtime")
    override_bundled_hip_runtime(rccl_lib_dir)

    # Step 4b-2: ROCr must be replaced as a pair with HIP above, and both
    # must complete before the first `import torch` in step 5.
    setup_report.step("Replace pip-bundled ROCr runtime")
    override_bundled_hsa_runtime(rccl_lib_dir)

    # Step 4c: Configure kpack device code loading — TheRock-built libraries
    # have device code stripped into separate .kpack archives per GPU arch
    setup_report.step("Install CI-built kpack device code")
    setup_kpack_device_code(args.artifact_dir)

    setup_report.step("Set up library paths")
    setup_ld_library_path(rccl_lib_dir, rocm_lib_dir)

    # Step 5: Clone PyTorch test sources
    setup_report.step("Clone PyTorch test sources")
    clone_pytorch_test_sources(
        args.pytorch_src, SMOKE_TARGETS if args.test_scope == "smoke" else ALL_TARGETS)

    # Step 6: Patch missing modules, print environment info, and run tests
    setup_report.step("Check PyTorch environment")
    patch_missing_torch_modules()
    print_environment_info()
    setup_report.step("Start pytest")
    exit_code, summary = run_tests(args.pytorch_src, args.results_log, args.test_scope)

    # Step 7: Generate and distribute summary report
    report = generate_summary_report(summary, rccl_lib)
    log.info("\n%s", report)
    write_github_summary(report)

    summary_path = args.results_log.parent / "pytorch_c10d_summary.txt"
    summary_path.write_text(report)
    log.info("Summary written to: %s", summary_path)

    status = "PASSED" if exit_code == 0 else "FAILED"
    if args.notify_email:
        send_email_report(report, args.notify_email, status,
                          subject_prefix=REPORT_SUBJECT)
    if args.notify_webhook:
        send_teams_webhook(report, args.notify_webhook, status,
                           subject_prefix=REPORT_SUBJECT)

    sys.exit(exit_code)


if __name__ == "__main__":
    setup_report.run(main)
