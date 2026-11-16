#!/usr/bin/env python3
"""Run PyTorch distributed test suites against CI-built RCCL.

This script handles:
  1. Discovering the CI-built librccl.so in the artifact directory
  2. Replacing PyTorch's pip-bundled librccl.so with the CI-built version
  3. Cloning the matching PyTorch test sources (sparse checkout)
  4. Running pytest on one suite of PyTorch distributed test files (SUITES)

Usage from GitHub Actions:
  python projects/rccl/ci/scripts/test_pytorch_c10d.py \
      --artifact-dir ./build \
      --pytorch-src ./pytorch-src \
      --suite c10d_nccl \
      --results-log ./pytorch_c10d_results.log
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rccl_ci_utils import (
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


@dataclass(frozen=True)
class Suite:
    """One pytest invocation over a fixed set of PyTorch distributed test files.

    Each suite runs as its own CI job, with its own known-failure list and its
    own floor on how many tests must be collected.
    """

    name: str
    # Paths under test/distributed in the PyTorch tree.
    files: tuple[str, ...]
    # Minimum testcases in the JUnit XML for the full ("all") run. A run that
    # collects fewer is a failure: a renamed or deleted file must not read as
    # a quiet pass.
    min_tests: int
    known_failures_file: str
    # Stem for the log, JUnit XML and summary written next to --results-log.
    artifact_stem: str
    # Per-test pytest-timeout, seconds, for the full run.
    timeout_s: int = 600
    # -k expression for the full run, or None for every collected test.
    k_expr: str | None = None
    # Curated smoke subset, or None when the suite has no smoke tier.
    smoke_tests: tuple[str, ...] | None = None
    # Extra environment for pytest, as (name, value) pairs.
    env: tuple[tuple[str, str], ...] = ()
    # Known failures are keyed "Class.test" instead of bare "test" names.
    # The multi-file suites need it: the same method name occurs in several
    # files. c10d_nccl keeps bare names so known_failures.json is unchanged.
    qualified_ids: bool = True


SUITES: dict[str, Suite] = {
    s.name: s
    for s in (
        Suite(
            name="c10d_nccl",
            files=("test_c10d_nccl.py",),
            min_tests=200,
            known_failures_file="known_failures.json",
            artifact_stem="pytorch_c10d",
            k_expr="not NCCLTraceTestDumpOnTimeout",
            smoke_tests=tuple(SMOKE_TESTS),
            qualified_ids=False,
        ),
        # Nonblocking comms, comm split/shrink, per-collective config, the
        # enqueue re-architecture and cost simulation: newer NCCL APIs and
        # flags PyTorch is turning on. ROCM-32626, 33626, 33627 (AIPYTORCH-1369).
        Suite(
            name="c10d_nccl2",
            files=("test_c10d_nccl2.py",),
            # 107 collected on torch 2.16 nightlies 20261113 and 20261115.
            min_tests=100,
            known_failures_file="known_failures_c10d_nccl2.json",
            artifact_stem="pytorch_c10d_nccl2",
        ),
        # Symmetric-memory allocation, import and teardown. ROCM-33022
        # (AIPYTORCH-1501).
        Suite(
            name="symmetric_memory",
            files=("test_symmetric_memory.py",),
            # 107 collected on gfx942 (102 on gfx950), torch 20261113/20261115.
            min_tests=100,
            known_failures_file="known_failures_symmetric_memory.json",
            artifact_stem="pytorch_symmetric_memory",
        ),
        # Many communicators alive at once (device meshes, FSDP/HSDP groups)
        # and RCCL's INFO log read as a contract. ROCM-33640, 33092. The files
        # are those named in AIPYTORCH-1567 and changed by the PyTorch
        # /dev/shm workaround PRs (#198299, #198301, #198302, #200632,
        # #202272, #202274).
        Suite(
            name="fsdp_dtensor",
            files=(
                "_composable/fsdp/test_fully_shard_comm.py",
                "_composable/fsdp/test_fully_shard_frozen.py",
                "_composable/fsdp/test_fully_shard_mixed_precision.py",
                "_composable/fsdp/test_fully_shard_training.py",
                "_composable/test_composability/test_pp_composability.py",
                "fsdp/test_fsdp_hybrid_shard.py",
                "fsdp/test_fsdp_tp_integration.py",
                "fsdp/test_hsdp_dtensor_state_dict.py",
                "tensor/test_redistribute.py",
                "tensor/test_utils.py",
                "tensor/test_view_ops.py",
                "test_composability.py",
            ),
            # 552 collected on torch 20261113, 556 on 20261115.
            min_tests=520,
            known_failures_file="known_failures_fsdp_dtensor.json",
            artifact_stem="pytorch_fsdp_dtensor",
            # TestFullyShard1DTrainingCore.test_train_parity_multi_group needs
            # 308-316 s (measured on MI350X with a longer limit), just over
            # PyTorch's default 300 s per-test limit, so it timed out in every
            # run. It is the suite's most communicator-heavy test; listing it
            # as a known failure would hide a regression in it. Raise the
            # limit instead (read by common_distributed at import; it also
            # sets the teardown "ranks did not exit" limit), and keep
            # pytest-timeout above it so PyTorch's own timeout fires first.
            env=(("DISTRIBUTED_TESTS_DEFAULT_TIMEOUT", "600"),),
            timeout_s=900,
        ),
    )
}


def case_id(case: dict, qualified: bool) -> str:
    """Known-failure key for a JUnit testcase: "test" or "Class.test"."""
    if not qualified:
        return case["name"]
    cls = case["classname"].rsplit(".", 1)[-1]
    return f"{cls}.{case['name']}" if cls else case["name"]


def file_of(case: dict, suite: Suite) -> str | None:
    """The suite file a JUnit testcase came from.

    Uses the testcase's ``file`` attribute when pytest wrote one (we ask for
    junit_family=xunit1). Otherwise falls back to ``classname``, the dotted
    path of the module relative to pytest's rootdir plus the class, e.g.
    "test.distributed.tensor.test_utils.UtilTest". The rootdir decides how
    much of the prefix is present, so match the module on dot boundaries and
    take the longest: "test_composability" is both a file and the directory
    holding test_pp_composability.py.
    """
    path = case.get("file", "").replace("\\", "/")
    by_length = sorted(suite.files, key=len, reverse=True)
    if path:
        for f in by_length:
            if path == f or path.endswith("/" + f):
                return f
        return None
    dotted = f".{case['classname']}."
    for f in by_length:
        module = f.removesuffix(".py").replace("/", ".")
        if f".{module}." in dotted:
            return f
    return None


def judge(suite: Suite, test_scope: str, cases: list[dict], tests_run: int,
          known_set: set[str], returncode: int) -> dict:
    """Turn JUnit testcases into a verdict for one suite.

    Fails on any unexpected failure, on collecting fewer tests than the
    suite's floor, and (full runs) on any suite file with no test that
    actually ran: a file that is all skips or fails to import gives no
    coverage, and must not count as a pass.
    """
    errors: list[str] = []
    qualified = suite.qualified_ids
    # A test whose call fails and whose teardown then errors appears as two
    # testcases with the same id; count it once.
    failed = list(dict.fromkeys(
        case_id(c, qualified) for c in cases if c["outcome"] == "failed"))
    failed_set = set(failed)
    # PyTorch's persistent multi-process test classes skip every remaining
    # test in the class after one fails ("Previous test failed, skipping").
    # Those tests did not run. Report them apart from capability skips, so a
    # known failure cannot silently hide the rest of its class.
    not_run = list(dict.fromkeys(
        case_id(c, qualified) for c in cases
        if c["outcome"] == "skipped"
        and c.get("message", "").startswith("Previous test failed")
        and case_id(c, qualified) not in failed_set))
    passed_ids = {case_id(c, qualified) for c in cases if c["outcome"] == "passed"}

    smoke = test_scope == "smoke" and suite.smoke_tests is not None
    if smoke:
        if tests_run < len(suite.smoke_tests):
            errors.append(
                f"expected {len(suite.smoke_tests)} smoke tests but only "
                f"{tests_run} were collected; test names may have changed")
    else:
        if tests_run < suite.min_tests:
            errors.append(
                f"expected at least {suite.min_tests} tests but only "
                f"{tests_run} were collected; check test discovery")
        ran = {file_of(c, suite) for c in cases if c["outcome"] in ("passed", "failed")}
        for f in suite.files:
            if f not in ran:
                errors.append(f"{f}: no test ran (all skipped, missing, or not importable)")

    known_failed = [t for t in failed if t in known_set]
    unexpected_failed = [t for t in failed if t not in known_set]
    if unexpected_failed:
        errors.append(f"{len(unexpected_failed)} unexpected failure(s)")

    # pytest exits 1 for test failures; anything else (interrupted, internal
    # error, usage error, no tests collected) is a failure on its own.
    if returncode not in (0, 1):
        errors.append(f"pytest exited {returncode}")
    elif returncode == 1 and not failed:
        errors.append("pytest exited 1 but the JUnit XML records no failure")

    return {
        "exit_code": 1 if errors else 0,
        "errors": errors,
        "failed": failed,
        "known_failed": known_failed,
        "unexpected_failed": unexpected_failed,
        "now_passing": sorted(known_set & passed_ids),
        "not_run_after_failure": not_run,
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


def clone_pytorch_test_sources(pytorch_src: Path, suite: Suite) -> None:
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
    missing = [f for f in suite.files if not (dist_dir / f).exists()]
    if missing:
        log.error("Suite %s: test files not found after clone: %s", suite.name, missing)
        sys.exit(1)
    log.info("Test sources ready: %d file(s) under %s", len(suite.files), dist_dir)


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



def dev_shm_gib() -> float | None:
    """Size of /dev/shm in GiB, or None if it is not mounted."""
    try:
        st = os.statvfs("/dev/shm")
    except OSError:
        return None
    return st.f_blocks * st.f_frsize / 2**30


def check_dev_shm(expect_gib: float | None) -> float | None:
    """Log /dev/shm's size; exit if it is not the size the job asked for.

    The workflow sizes /dev/shm like PyTorch CI's, so that RCCL's per-
    communicator shared-memory use (ROCM-33640) hits the same ceiling here.
    If a runner or container change drops that limit, the job would still
    pass while testing nothing about it; refuse to start instead.
    """
    size = dev_shm_gib()
    log.info("/dev/shm: %s", f"{size:.2f} GiB" if size is not None else "not mounted")
    if expect_gib is not None:
        if size is None or abs(size - expect_gib) > 0.05 * expect_gib:
            log.error("/dev/shm is %s, expected %.1f GiB; is the container started "
                      "with --shm-size and without --ipc host?",
                      "missing" if size is None else f"{size:.2f} GiB", expect_gib)
            sys.exit(1)
    return size


def print_environment_info() -> None:
    """Print GPU and environment details for CI logs."""
    import torch

    log.info("PyTorch: %s", torch.__version__)
    log.info("CUDA/HIP available: %s", torch.cuda.is_available())
    log.info("GPU count: %s", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        log.info("  GPU %d: %s", i, torch.cuda.get_device_name(i))
    log.info("LD_LIBRARY_PATH: %s", os.environ.get("LD_LIBRARY_PATH", ""))


def run_tests(pytorch_src: Path, results_log: Path, suite: Suite,
              test_scope: str = "smoke") -> tuple[int, dict]:
    """Run pytest on one suite and return (exit_code, summary_dict)."""
    miopen_cache = tempfile.mkdtemp(prefix="miopen_cache_")
    os.environ["MIOPEN_USER_DB_PATH"] = miopen_cache

    env = os.environ.copy()
    env["PYTHONPATH"] = str(pytorch_src / "test") + ":" + env.get("PYTHONPATH", "")
    for name, value in suite.env:
        env[name] = value
        log.info("Suite %s: %s=%s", suite.name, name, value)

    junit_xml = results_log.parent / f"{suite.artifact_stem}_results.xml"

    smoke = test_scope == "smoke" and suite.smoke_tests is not None
    if smoke:
        k_expr = " or ".join(suite.smoke_tests)
        timeout = "60"
        log.info("Suite %s: running smoke tests (%d tests)", suite.name, len(suite.smoke_tests))
    else:
        k_expr = suite.k_expr
        timeout = str(suite.timeout_s)
        if test_scope == "smoke":
            log.info("Suite %s has no smoke tier; running all of it", suite.name)
        log.info("Suite %s: running all tests%s", suite.name,
                 f" matching -k {k_expr!r}" if k_expr else "")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *(str(pytorch_src / "test" / "distributed" / f) for f in suite.files),
        "-v",
        f"--timeout={timeout}",
        "--tb=short",
        f"--junitxml={junit_xml}",
        # xunit1 writes each testcase's file, which tells the per-file
        # coverage check where a test came from.
        "-o", "junit_family=xunit1",
    ]
    if k_expr:
        cmd += ["-k", k_expr]
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

    junit = {"passed": [], "cases": [], "tests_run": 0, "error_details": []}
    if junit_xml.exists():
        log.info("Parsing JUnit XML: %s", junit_xml)
        junit = parse_junit_xml(junit_xml)
        if junit["error_details"]:
            log.info("Failure/error details from JUnit XML:")
            for detail in junit["error_details"]:
                log.info("  %s", detail)
    else:
        # No XML means no evidence any test ran; judge() fails on the floor.
        log.error("JUnit XML not found at %s", junit_xml)

    known_failures_file = Path(__file__).parent / suite.known_failures_file
    known_set: set[str] = set()
    if known_failures_file.exists():
        with open(known_failures_file) as f:
            known_set = set(json.load(f).get("tests", {}).keys())
        log.info("Loaded %d known failures from %s", len(known_set), known_failures_file.name)
    else:
        log.info("No known-failure list for suite %s (%s)", suite.name, known_failures_file.name)

    # Distinct testcases, not the <testsuite tests=> attribute: pytest counts
    # subtest reports there too (1276 for 556 fsdp_dtensor testcases), and a
    # test with a teardown error appears twice.
    tests_run = len(dict.fromkeys((c["classname"], c["name"]) for c in junit["cases"]))
    verdict = judge(suite, test_scope, junit["cases"], tests_run,
                    known_set, proc.returncode)
    for err in verdict["errors"]:
        log.error("%s", err)
    if verdict["unexpected_failed"]:
        log.error("Unexpected failures: %s", verdict["unexpected_failed"])
    if verdict["now_passing"]:
        log.warning(
            "%d known failure(s) now PASSING — consider removing from %s: %s",
            len(verdict["now_passing"]), suite.known_failures_file, verdict["now_passing"],
        )

    passed_tests = junit["passed"]
    not_run = verdict["not_run_after_failure"]
    skipped = sum(1 for c in junit["cases"] if c["outcome"] == "skipped") - len(not_run)
    parts = []
    if passed_tests:
        parts.append(f"{len(passed_tests)} passed")
    if verdict["unexpected_failed"]:
        parts.append(f"{len(verdict['unexpected_failed'])} unexpected failures")
    if verdict["known_failed"]:
        parts.append(f"{len(verdict['known_failed'])} known failures")
    if not_run:
        parts.append(f"{len(not_run)} not run after an earlier failure in their class")
    if skipped:
        parts.append(f"{skipped} skipped")

    summary = {
        "exit_code": verdict["exit_code"],
        "suite": suite.name,
        "files": list(suite.files),
        "test_scope": test_scope if suite.smoke_tests is not None else "all",
        "passed": passed_tests,
        "failed": verdict["failed"],
        "known_failed": verdict["known_failed"],
        "unexpected_failed": verdict["unexpected_failed"],
        "errors": verdict["errors"],
        "not_run_after_failure": not_run,
        "summary_line": ", ".join(parts) or "no tests ran",
        "tests_run": tests_run,
        "expected_tests": len(suite.smoke_tests) if smoke else None,
    }
    return verdict["exit_code"], summary


def generate_summary_report(summary: dict, rccl_lib: Path,
                            shm_gib: float | None = None) -> str:
    """Generate a plain-text summary report."""
    import torch

    status = "PASSED" if summary["exit_code"] == 0 else "FAILED"
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        gpu_info.append(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    lines = [
        f"RCCL PyTorch distributed Test Report: {summary['suite']}",
        f"{'=' * 40}",
        f"Status:     {status}",
        f"Suite:      {summary['suite']} ({len(summary['files'])} file(s))",
        f"Test scope: {summary['test_scope']}",
        f"Date:       {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"",
        f"PyTorch:    {torch.__version__}",
        f"RCCL:       {rccl_lib}",
        f"GPUs:       {torch.cuda.device_count()}x {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'}",
        f"/dev/shm:   {f'{shm_gib:.2f} GiB' if shm_gib is not None else 'not mounted'}",
        f"",
        f"Results:    {summary['summary_line']}",
    ]

    if summary.get("expected_tests") is not None:
        lines.append(f"Collected:  {summary['tests_run']}/{summary['expected_tests']} expected smoke tests")
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

    not_run = summary.get("not_run_after_failure", [])
    if not_run:
        lines.append(f"NOT RUN ({len(not_run)}): skipped by PyTorch after an earlier "
                     "failure in the same test class, so these give no coverage:")
        for name in not_run:
            lines.append(f"  NOTRUN {name}")
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
        default=None,
        help="Path for the test log; the JUnit XML and summary go next to it "
             "(default: <pytorch-src>/<suite stem>_results.log)",
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="c10d_nccl",
        help="Which PyTorch distributed test suite to run (default: c10d_nccl)",
    )
    parser.add_argument(
        "--test-scope",
        choices=["smoke", "all"],
        default="smoke",
        help="Run a suite's curated smoke tests or all of it (default: smoke). "
             "Suites with no smoke tier always run all of it.",
    )
    parser.add_argument(
        "--expect-shm-gib",
        type=float,
        default=None,
        help="Fail before running tests unless /dev/shm is this size (GiB)",
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
    suite = SUITES[args.suite]
    if args.results_log is None:
        args.results_log = args.pytorch_src / f"{suite.artifact_stem}_results.log"
    shm_gib = check_dev_shm(args.expect_shm_gib)

    # Step 1: Discover RCCL library path
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
    pip_lib_dirs = find_pip_sdk_lib_dirs()
    reconcile_soname_versions([rccl_lib_dir] + pip_lib_dirs)

    # Step 3: Quarantine TheRock-bundled libamd_smi and rocm_sysdeps to
    # prevent the nl_genl destructor crash (SIGSEGV in containers).
    # After this, libamd_smi resolves from pip dirs (via symlinks above).
    quarantine_rocm_sysdeps(rccl_lib_dir)

    # Step 4: Replace pip-bundled librccl.so with CI-built version
    override_bundled_rccl(rccl_lib_dir)

    # Step 4b: Replace pip-bundled libamdhip64.so with TheRock's version
    # so the HIP runtime matches the compiler that built RCCL's device
    # code objects (prevents findSymbol → guarantee → abort)
    override_bundled_hip_runtime(rccl_lib_dir)

    # Step 4b-2: ROCr must be replaced as a pair with HIP above, and both
    # must complete before the first `import torch` in step 5.
    override_bundled_hsa_runtime(rccl_lib_dir)

    # Step 4c: Configure kpack device code loading — TheRock-built libraries
    # have device code stripped into separate .kpack archives per GPU arch
    setup_kpack_device_code(args.artifact_dir)

    setup_ld_library_path(rccl_lib_dir, rocm_lib_dir)

    # Step 5: Clone PyTorch test sources
    clone_pytorch_test_sources(args.pytorch_src, suite)

    # Step 6: Patch missing modules, print environment info, and run tests
    patch_missing_torch_modules()
    print_environment_info()
    exit_code, summary = run_tests(args.pytorch_src, args.results_log, suite, args.test_scope)

    # Step 7: Generate and distribute summary report
    report = generate_summary_report(summary, rccl_lib, shm_gib)
    log.info("\n%s", report)
    write_github_summary(report)

    summary_path = args.results_log.parent / f"{suite.artifact_stem}_summary.txt"
    summary_path.write_text(report)
    log.info("Summary written to: %s", summary_path)

    status = "PASSED" if exit_code == 0 else "FAILED"
    if args.notify_email:
        send_email_report(report, args.notify_email, status,
                          subject_prefix=f"RCCL PyTorch {suite.name} Test")
    if args.notify_webhook:
        send_teams_webhook(report, args.notify_webhook, status,
                           subject_prefix=f"RCCL PyTorch {suite.name} Test")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
