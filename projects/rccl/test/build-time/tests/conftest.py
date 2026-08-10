# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Fixtures and CLI options for the RCCL build-time regression harness.

The measurement itself lives in ../build_time.py, which is also runnable
standalone; this file only adapts it to pytest so CI gets one result per GPU
target with proper skip/fail semantics. The division of responsibility is:

  * an arch this ROCm toolchain cannot build       -> skip (not our bug)
  * git history unavailable (shallow CI checkout)  -> skip (environment)
  * a broken toolchain or an unresolvable explicit
    base revision                                  -> fail (real breakage)
  * a build-time regression past the tolerance     -> assertion failure

Every option below also reads a default from the matching RCCL_BUILD_TIME_*
environment variable, so a runner config can drive this through either
`test_filter` arguments or `env_variables`.
"""

import os
import sys
from dataclasses import dataclass

import pytest

# build_time.py sits at the harness root, which pytest's importlib import mode
# does not put on sys.path. conftest is imported before collection, so adding
# it here is enough for the test module to import it too.
_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import build_time as bt  # noqa: E402  (needs the sys.path entry above)


def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default) or default)


def pytest_addoption(parser):
    group = parser.getgroup("build-time", "RCCL build-time regression gate")
    group.addoption(
        "--targets", action="store", default=_env_str("RCCL_BUILD_TIME_TARGETS"),
        help="space-separated GPU targets to time (default: DEFAULT_GPUS from CMakeLists.txt)")
    group.addoption(
        "--local-gpu", action="store_true", default=False,
        help="time only the arch(es) of the GPUs in this machine")
    group.addoption(
        "--compare-base", action="store_true", default=False,
        help="gate on the ratio against the PR's base commit instead of a wall-clock budget")
    group.addoption(
        "--base-rev", action="store", default="",
        help="explicit base revision for --compare-base (default: merge-base with origin/develop)")
    group.addoption(
        "--max-regression-pct", action="store", type=float,
        default=_env_float("RCCL_BUILD_TIME_MAX_REGRESSION_PCT", 10.0),
        help="fail if head is more than this %% slower than base (default: 10)")
    group.addoption(
        "--repeat", action="store", type=int,
        default=_env_int("RCCL_BUILD_TIME_REPEAT", 2),
        help="alternating base/head rounds; the minimum of each wins (default: 2)")
    group.addoption(
        "--jobs", action="store", type=int,
        default=_env_int("RCCL_BUILD_TIME_JOBS", bt.cpu_count()),
        help="build parallelism (default: CPU count)")
    group.addoption(
        "--threshold-sec", action="store", type=float,
        default=_env_float("RCCL_BUILD_TIME_THRESHOLD_SEC", bt.DEFAULT_THRESHOLD_SEC),
        help="per-target wall-clock budget for the absolute gate (default: %d)"
             % bt.DEFAULT_THRESHOLD_SEC)
    group.addoption(
        "--build-root", action="store", default=_env_str("RCCL_BUILD_TIME_ROOT"),
        help="where to put scratch build dirs (default: a temp dir)")
    group.addoption(
        "--keep", action="store_true", default=False,
        help="keep build dirs for inspection")
    # Not --strict: pytest already defines that as an alias for --strict-markers.
    group.addoption(
        "--strict-targets", action="store_true",
        default=_env_str("RCCL_BUILD_TIME_STRICT") == "1",
        help="fail, rather than skip, targets the compiler cannot build")


@dataclass
class Settings:
    """Resolved options for one harness run."""

    targets: str
    local_gpu: bool
    compare: bool
    base_rev: str
    max_regression_pct: float
    repeat: int
    jobs: int
    threshold_sec: float
    build_root: str
    keep: bool
    strict_targets: bool


def _settings(config) -> Settings:
    return Settings(
        targets=config.getoption("--targets"),
        local_gpu=config.getoption("--local-gpu"),
        compare=config.getoption("--compare-base"),
        base_rev=config.getoption("--base-rev"),
        max_regression_pct=config.getoption("--max-regression-pct"),
        repeat=config.getoption("--repeat"),
        jobs=config.getoption("--jobs"),
        threshold_sec=config.getoption("--threshold-sec"),
        build_root=config.getoption("--build-root"),
        keep=config.getoption("--keep"),
        strict_targets=config.getoption("--strict-targets"),
    )


def _resolve_targets(config) -> tuple[list[str], str]:
    """Pick the target list and report where it came from.

    Mirrors build_time.resolve_targets(), but reads pytest options instead of
    an argparse namespace so the two entry points stay in sync.
    """
    settings = _settings(config)
    if settings.targets.split():
        return settings.targets.split(), "--targets"
    if settings.local_gpu:
        return bt.local_gpu_targets(), "local GPUs (rocm_agent_enumerator)"
    return bt.targets_from_cmake(), "DEFAULT_GPUS in CMakeLists.txt"


def pytest_report_header(config):
    """Record how the gate was configured; wall-clock numbers need the context."""
    settings = _settings(config)
    mode = ("comparison against %s" % (settings.base_rev or "merge-base with origin/develop")
            if settings.compare else
            "absolute threshold %s" % bt.fmt(settings.threshold_sec))
    return [
        "build-time gate: %s" % mode,
        "build-time opts: jobs=%d repeat=%d tolerance=%.1f%% rocm=%s"
        % (settings.jobs, settings.repeat, settings.max_regression_pct, bt.rocm_path()),
    ]


def pytest_generate_tests(metafunc):
    """Parametrize over GPU targets so each arch is its own reported result.

    Resolution runs at collection time and can legitimately fail (no GPU, no
    rocm_agent_enumerator). Rather than turning that into a collection error
    with a traceback, the reason is stashed on the config and surfaced by the
    gpu_target fixture as an ordinary, readable test failure.
    """
    if "target" not in metafunc.fixturenames:
        return
    try:
        targets, source = _resolve_targets(metafunc.config)
    except Exception as exc:  # noqa: BLE001 - reported through gpu_target instead
        metafunc.config.build_time_target_error = str(exc)
        metafunc.parametrize("target", [pytest.param(None, id="unresolved")])
        return
    metafunc.config.build_time_targets_from = source
    metafunc.parametrize("target", targets, ids=list(targets))


@pytest.fixture
def gpu_target(target, pytestconfig, settings, hipcc) -> str:
    """The arch under test, or skip/fail when it cannot be measured.

    An arch this toolchain cannot build is expected -- the RCCL default list
    runs ahead of released ROCm -- so it skips unless --strict-targets says the
    toolchain is meant to cover everything. A target list that could not be
    resolved at all is a broken runner, so it fails.
    """
    if target is None:
        pytest.fail("cannot determine which GPU targets to time: %s"
                    % getattr(pytestconfig, "build_time_target_error", "unknown error"))
    if not bt.compiler_supports(target, hipcc):
        reason = "compiler does not support --offload-arch=%s" % target
        if settings.strict_targets:
            pytest.fail(reason + " (--strict-targets)")
        pytest.skip(reason)
    return target


@pytest.fixture(scope="session")
def settings(pytestconfig) -> Settings:
    return _settings(pytestconfig)


@pytest.fixture(scope="session")
def hipcc() -> str:
    """A hipcc that can compile HIP, or fail the suite explaining why not."""
    try:
        return bt.resolve_hipcc()
    except RuntimeError as exc:
        pytest.fail(str(exc))


@pytest.fixture(scope="session")
def build_root(settings: Settings, tmp_path_factory) -> str:
    """Scratch directory for the nested builds and their logs.

    Left in place when --keep or an explicit --build-root is given; otherwise
    pytest's tmp_path_factory owns the lifetime and its usual retention policy
    keeps the last few runs' logs around for post-mortem.
    """
    if settings.build_root:
        os.makedirs(settings.build_root, exist_ok=True)
        return settings.build_root
    return str(tmp_path_factory.mktemp("rccl-build-time"))


@pytest.fixture(scope="session")
def base_source_dir(settings: Settings, build_root: str):
    """RCCL source tree of the PR's base commit, checked out into a worktree.

    Skips (rather than fails) when the base cannot be reached: on a shallow CI
    checkout the history simply is not there, and reporting that as a build-time
    regression would be a lie. An explicit --base-rev that does not resolve is
    the caller's mistake, so that one fails.
    """
    if not settings.compare:
        pytest.skip("not running the comparison gate (pass --compare-base)")

    try:
        base_sha, label, err = bt.resolve_base_sha(settings.base_rev or None)
    except bt.CommandError as exc:
        pytest.fail("--base-rev %r does not resolve: %s" % (settings.base_rev, exc))
    if err or not base_sha:
        pytest.skip("cannot resolve base commit: %s" % (err or "unknown error"))

    head_sha, err = bt.git_err("rev-parse", "HEAD")
    if err or not head_sha:
        pytest.skip("cannot resolve HEAD: %s" % (err or "unknown error"))
    if base_sha == head_sha:
        pytest.skip("base and head are the same commit (%s); nothing to compare"
                    % base_sha[:10])

    print("\nbuild-time comparison: base %s (%s) vs head %s"
          % (base_sha[:10], label, head_sha[:10]))
    try:
        worktree, base_dir = bt.add_base_worktree(base_sha, build_root)
    except bt.CommandError as exc:
        pytest.skip("cannot check out base commit %s: %s" % (base_sha[:10], exc))

    try:
        yield base_dir
    finally:
        bt.remove_base_worktree(worktree)
