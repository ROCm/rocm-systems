#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Host tests for TestExecutor's init-pipeline wiring (Phase 4b-4).

The env-correctness test is portable; the end-to-end test drives the real
PipelineScheduler + Rendezvous + a fake entry through _run_suite_init_pipeline
and is POSIX-only (uses _spawn_captured's process groups).
"""

import os
import shlex
import subprocess
import sys
import types

import pytest

_RUNNER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER_ROOT not in sys.path:
    sys.path.insert(0, _RUNNER_ROOT)

from lib.test_executor import LaunchSpec, TestExecutor as Executor  # noqa: E402
from lib.pipeline_runner import PlannedEntry  # noqa: E402
from lib.pipeline import KIND_PIPELINE, KIND_SERIAL  # noqa: E402

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fake_entry.py")

POSIX_ONLY = pytest.mark.skipif(os.name != "posix",
                                reason="uses process groups (start_new_session/killpg)")


def _make_executor(tmp_path, **arg_over):
    ex = Executor.__new__(Executor)
    args = types.SimpleNamespace(
        exec_mode="init-pipeline", test_name=None, skip_mpi_check=False,
        init_pool=2, init_timeout=0, phase_timings=False, verbose=False,
    )
    for k, v in arg_over.items():
        setattr(args, k, v)
    ex.args = args
    ex.log_dir = str(tmp_path)
    ex.emit_enabled = True
    ex._emit_log_counter = 0
    ex.test_names, ex.test_results, ex.test_durations, ex.test_suites = [], [], [], []
    ex.test_records = []
    ex.interval_validation = []
    ex.interval_validation_failed = False
    return ex


# --------------------------------------------------------------------------- #
# Env wiring (portable)
# --------------------------------------------------------------------------- #
def test_build_pipeline_spec_sets_warmup_and_rendezvous_for_pipeline(tmp_path):
    ex = _make_executor(tmp_path)
    captured = {}

    def fake_build(tcfg, suite):
        captured[tcfg["name"]] = dict(tcfg["env_variables"])
        return LaunchSpec(name=tcfg["name"], cmd="true", run_cwd=".",
                          env={}, timeout=0, is_gtest=True), None

    ex._build_launch_spec = fake_build
    base = {"AR": {"name": "AR", "env_variables": {"FOO": "1"}}}

    from lib.pipeline import Rendezvous
    rdv = Rendezvous.for_entry(str(tmp_path), "run", 0)
    p_pipe = PlannedEntry("AR", "AR.g8_mp_r1", KIND_PIPELINE, "fork_coll",
                          {"UT_MIN_GPUS": "8", "UT_RANKS_PER_GPU": "1"}, True)
    ex._build_pipeline_spec(p_pipe, {}, base, rdv)
    env = captured["AR.g8_mp_r1"]
    assert env["FOO"] == "1"                       # parent env preserved
    assert env["UT_MIN_GPUS"] == "8"               # pinned sweep env merged
    assert env["RCCL_TEST_READY_GO"] == "1"        # warmup enabled
    assert env["RCCL_TEST_WARMUP_PROFILE"] == "fork_coll"  # canonical profile (v10 §5.1)
    assert env["RCCL_TEST_RENDEZVOUS_DIR"] == os.path.abspath(rdv.dir)
    assert os.path.isabs(env["RCCL_TEST_RENDEZVOUS_DIR"])  # ranks run with a different cwd


def test_build_pipeline_spec_go_timeout_env(tmp_path):
    ex = _make_executor(tmp_path, go_timeout=45)
    captured = {}

    def fake_build(tcfg, suite):
        captured[tcfg["name"]] = dict(tcfg["env_variables"])
        return LaunchSpec(name=tcfg["name"], cmd="true", run_cwd=".",
                          env={}, timeout=0, is_gtest=True), None

    ex._build_launch_spec = fake_build
    from lib.pipeline import Rendezvous
    rdv = Rendezvous.for_entry(str(tmp_path), "run", 0)
    p = PlannedEntry("M", "M", KIND_PIPELINE, "mpi_coll", {}, False)
    ex._build_pipeline_spec(p, {}, {"M": {"name": "M", "env_variables": {}}}, rdv)
    assert captured["M"]["RCCL_TEST_GO_TIMEOUT_SEC"] == "45"
    assert captured["M"]["RCCL_TEST_WARMUP_PROFILE"] == "mpi_coll"


def test_build_pipeline_spec_serial_has_no_rendezvous(tmp_path):
    ex = _make_executor(tmp_path)
    captured = {}

    def fake_build(tcfg, suite):
        captured[tcfg["name"]] = dict(tcfg["env_variables"])
        return LaunchSpec(name=tcfg["name"], cmd="true", run_cwd=".",
                          env={}, timeout=0, is_gtest=True), None

    ex._build_launch_spec = fake_build
    base = {"S": {"name": "S", "env_variables": {}}}
    p_serial = PlannedEntry("S", "S", KIND_SERIAL, "none", {}, False)
    ex._build_pipeline_spec(p_serial, {}, base, None)
    env = captured["S"]
    assert "RCCL_TEST_READY_GO" not in env
    assert "RCCL_TEST_RENDEZVOUS_DIR" not in env


# --------------------------------------------------------------------------- #
# End-to-end through the real scheduler + rendezvous (POSIX)
# --------------------------------------------------------------------------- #
@POSIX_ONLY
def test_run_suite_init_pipeline_end_to_end(tmp_path):
    ex = _make_executor(tmp_path)
    timeline = os.path.join(str(tmp_path), "tl.txt")

    def fake_build(tcfg, suite):
        name = tcfg["name"]
        env = {**os.environ, **{k: str(v) for k, v in tcfg.get("env_variables", {}).items()}}
        code = 1 if "FAILME" in name else 0
        cmd = (f"{shlex.quote(sys.executable)} {shlex.quote(FAKE)} 0.05 0.05 "
               f"{code} {shlex.quote(name)} {shlex.quote(timeline)}")
        spec = LaunchSpec(name=name, cmd=cmd, run_cwd=str(tmp_path), env=env,
                          timeout=0, is_gtest=False, gtest_json_path="",
                          emit_log_path=None)
        return spec, None

    ex._build_launch_spec = fake_build

    suite_config = {"suite_details": {"name": "S"}}
    tests = [
        {"name": "AR", "binary": "rccl-UnitTests", "warmup_profile": "fork_coll",
         "test_filter": "AllReduce.OutOfPlace",
         "fork_expand": {"num_gpus": [8], "process_mask": 3, "max_ranks_per_gpu": 1}},
        {"name": "M", "binary": "rccl-UnitTestsMPI", "warmup_profile": "mpi_coll", "test_filter": "M.C"},
        {"name": "Sx"},  # serial
    ]
    records = ex._run_suite_init_pipeline(suite_config, "S", tests)

    by = {r["name"]: r for r in records}
    # Fork AR split into 2 sub-entries (sp/mp, r1) + 1 parent_summary.
    assert by["AR.g8_sp_r1"]["record_type"] == "sub_entry"
    assert by["AR.g8_mp_r1"]["record_type"] == "sub_entry"
    assert by["AR"]["record_type"] == "parent_summary" and by["AR"]["result"] == "PASSED"
    assert by["M"]["record_type"] == "entry" and by["M"]["result"] == "PASSED"
    assert by["Sx"]["record_type"] == "entry" and by["Sx"]["result"] == "PASSED"
    # Top-line tracking got exactly 3 entries (AR once via parent_summary).
    assert len(ex.test_results) == 3
    assert ex.test_results.count("PASSED") == 3
    assert ex.interval_validation_failed is False
    # A pipeline entry's duration EXCLUDES its READY->GO park, so it tracks post-GO execution, not the --init-pool-inflated lifetime.
    for nm in ("AR.g8_sp_r1", "AR.g8_mp_r1", "M"):
        pt = by[nm]["phase_timings"]
        assert by[nm]["duration"] == pytest.approx(pt["execution_time"])
        assert by[nm]["duration"] <= pt["total"]
    # A serial entry has no queue wait, so it keeps its full lifetime.
    assert by["Sx"]["duration"] == pytest.approx(by["Sx"]["phase_timings"]["total"])


@POSIX_ONLY
def test_run_wide_scheduler_spans_two_suites(tmp_path):
    """v11 CR-6: pipeline entries in DIFFERENT suites share one scheduler; their
    combined execution intervals never overlap and per-suite reports are correct."""
    ex = _make_executor(tmp_path)
    timeline = os.path.join(str(tmp_path), "tl.txt")

    def fake_build(tcfg, suite):
        name = tcfg["name"]
        env = {**os.environ, **{k: str(v) for k, v in tcfg.get("env_variables", {}).items()}}
        cmd = (f"{shlex.quote(sys.executable)} {shlex.quote(FAKE)} 0.1 0.15 0 "
               f"{shlex.quote(name)} {shlex.quote(timeline)}")
        return LaunchSpec(name=name, cmd=cmd, run_cwd=str(tmp_path), env=env,
                          timeout=0, is_gtest=False, gtest_json_path="", emit_log_path=None), None

    ex._build_launch_spec = fake_build
    test_suites = [
        {"suite_details": {"name": "suiteA", "enabled": True},
         "tests": [{"name": "M", "binary": "rccl-UnitTestsMPI", "warmup_profile": "mpi_coll",
                    "test_filter": "M.C"}]},
        {"suite_details": {"name": "suiteB", "enabled": True},
         "tests": [{"name": "F", "binary": "rccl-UnitTests", "warmup_profile": "fork_coll",
                    "test_filter": "F.C",
                    "fork_expand": {"num_gpus": [8], "process_mask": 1, "ranks_per_gpu": 1}}]},
    ]
    records = ex.run_all_suites_init_pipeline(test_suites)

    suites = {r["suite"] for r in records if r.get("counts_toward_topline")}
    assert suites == {"suiteA", "suiteB"}                       # both suites reported
    assert all(r["result"] == "PASSED" for r in records if r.get("counts_toward_topline"))
    # Interval validation ran run-wide and found no overlap across suites.
    assert ex.interval_validation[-1]["max_executing"] == 1
    assert ex.interval_validation[-1]["ok"]


def test_serial_expanded_emits_per_config_rows(tmp_path):
    """--expand-sweeps runs each pinned sub-entry serially via run_test (no warmup)
    and emits per-config sub_entry rows + a parent_summary."""
    ex = _make_executor(tmp_path, exec_mode="serial", expand_sweeps=True)
    seen = {}

    def fake_run_test(tcfg, suite):
        # Capture the pinned env + name each sub-entry runs with, and NO warmup.
        seen[tcfg["name"]] = dict(tcfg.get("env_variables", {}))
        return {"name": tcfg["name"], "result": "PASSED", "duration": 1.0}

    ex.run_test = fake_run_test
    suite_config = {"suite_details": {"name": "S"}}
    tests = [{"name": "AR", "warmup_profile": "fork_coll",
              "fork_expand": {"num_gpus": [8], "process_mask": 3, "max_ranks_per_gpu": 1}}]
    records = ex._run_suite_serial_expanded(suite_config, "S", tests)

    by = {r["name"]: r for r in records}
    assert by["AR.g8_sp_r1"]["record_type"] == "sub_entry"
    assert by["AR.g8_mp_r1"]["record_type"] == "sub_entry"
    assert by["AR"]["record_type"] == "parent_summary" and by["AR"]["result"] == "PASSED"
    # Pinned per-config selectors were applied, and NO warmup/rendezvous env.
    assert seen["AR.g8_sp_r1"]["UT_PROCESS_MASK"] == "1"
    assert seen["AR.g8_mp_r1"]["UT_PROCESS_MASK"] == "2"
    assert "RCCL_TEST_READY_GO" not in seen["AR.g8_sp_r1"]
    assert "RCCL_TEST_RENDEZVOUS_DIR" not in seen["AR.g8_sp_r1"]
    # One top-line row (the parent), counted once.
    assert ex.test_results == ["PASSED"]


@POSIX_ONLY
def test_rerun_failed_converges_split_sweep(tmp_path):
    """A fork sweep whose first sub-entry fails (fail-fast cancels the rest) must,
    under --rerun-failed with a flip env, rerun the failed+cancelled sub-entries in
    order and roll the parent up to PASSED."""
    ex = _make_executor(tmp_path, rerun_failed=True)
    timeline = os.path.join(str(tmp_path), "tl.txt")

    def fake_build(tcfg, suite):
        name = tcfg["name"]
        env = {**os.environ, **{k: str(v) for k, v in tcfg.get("env_variables", {}).items()}}
        # First sibling (g8_sp) fails on the initial pass; FORCE_PASS (injected by
        # the rerun env) makes the fake exit 0 on the rerun.
        code = 1 if "g8_sp" in name else 0
        cmd = (f"{shlex.quote(sys.executable)} {shlex.quote(FAKE)} 0.05 0.05 "
               f"{code} {shlex.quote(name)} {shlex.quote(timeline)}")
        return LaunchSpec(name=name, cmd=cmd, run_cwd=str(tmp_path), env=env,
                          timeout=0, is_gtest=False, gtest_json_path="", emit_log_path=None), None

    ex._build_launch_spec = fake_build
    suite_config = {"suite_details": {"name": "S"},
                    "rerun_env_variables": {"FORCE_PASS": "1"}}
    tests = [{"name": "AR", "binary": "rccl-UnitTests", "warmup_profile": "fork_coll",
              "test_filter": "AllReduce.OutOfPlace",
              "fork_expand": {"num_gpus": [8], "process_mask": 3, "max_ranks_per_gpu": 1}}]
    records = ex._run_suite_init_pipeline(suite_config, "S", tests)

    by = {r["name"]: r for r in records}
    # After rerun both sub-entries pass and the parent rolls up PASSED.
    assert by["AR.g8_sp_r1"]["result"] == "PASSED"
    assert by["AR.g8_mp_r1"]["result"] == "PASSED"
    assert by["AR"]["record_type"] == "parent_summary" and by["AR"]["result"] == "PASSED"
    assert ex.test_results == ["PASSED"]  # one top-line entry, now passing


# --------------------------------------------------------------------------- #
# Ambient env isolation, exit accounting, planning robustness
# --------------------------------------------------------------------------- #
def _spec_executor(tmp_path):
    """Minimal TestExecutor able to run the real _build_launch_spec."""
    ex = Executor.__new__(Executor)
    ex.args = types.SimpleNamespace(
        verbose=False, coverage_report=False, mpi_args="", system=None,
        exec_mode="serial", test_name=None, skip_mpi_check=False, suite_name=None,
    )
    ex.global_env = {}
    ex._gpus_per_node = 8
    ex._gpus_per_node_detected = True
    ex.mpi_hosts = ""
    ex.paths = {"mpi_path": ""}
    ex.build_dir = str(tmp_path)
    ex.log_dir = str(tmp_path)
    ex.emit_enabled = False
    ex._emit_log_counter = 0
    ex._rocm_root = lambda: "/opt/rocm"

    def _resolve(binary, tc):
        path = os.path.join(str(tmp_path), binary)
        open(path, "a").close()   # _build_launch_spec rejects a nonexistent binary
        return path
    ex._resolve_binary_path = _resolve
    return ex


def test_ambient_runner_env_is_scrubbed_from_a_serial_launch(tmp_path, monkeypatch):
    """A stray `export RCCL_TEST_READY_GO=1` used to reach every SERIAL test through
    os.environ.copy(); the C++ contract then _exit(42)s and it reports as a plain
    gtest FAILURE with no hint of the cause."""
    monkeypatch.setenv("RCCL_TEST_READY_GO", "1")
    monkeypatch.setenv("RCCL_TEST_WARMUP_PROFILE", "fork_coll")
    monkeypatch.setenv("RCCL_TEST_RENDEZVOUS_DIR", str(tmp_path))
    monkeypatch.setenv("RCCL_TEST_INJECT_READY_FAIL", "1")
    ex = _spec_executor(tmp_path)
    spec, early = ex._build_launch_spec(
        {"name": "T", "binary": "rccl-UnitTests", "test_filter": "A.B", "num_ranks": 1},
        {"suite_details": {"name": "S"}})
    assert early is None and spec is not None
    for k in ("RCCL_TEST_READY_GO", "RCCL_TEST_WARMUP_PROFILE",
              "RCCL_TEST_RENDEZVOUS_DIR", "RCCL_TEST_INJECT_READY_FAIL"):
        assert k not in spec.env, k


def test_pipeline_entry_still_receives_its_own_rendezvous_env(tmp_path, monkeypatch):
    """The scrub must not defeat the feature: config-supplied values still win."""
    monkeypatch.setenv("RCCL_TEST_READY_GO", "stale")
    ex = _spec_executor(tmp_path)
    spec, _ = ex._build_launch_spec(
        {"name": "T", "binary": "rccl-UnitTests", "test_filter": "A.B", "num_ranks": 1,
         "env_variables": {"RCCL_TEST_READY_GO": "1",
                           "RCCL_TEST_RENDEZVOUS_DIR": str(tmp_path)}},
        {"suite_details": {"name": "S"}})
    assert spec.env["RCCL_TEST_READY_GO"] == "1"
    assert spec.env["RCCL_TEST_RENDEZVOUS_DIR"] == str(tmp_path)


def test_missing_binary_is_a_clean_preflight_rejection(tmp_path):
    """_resolve_binary_path does no existence check and subprocess.run raises
    FileNotFoundError (an OSError): it must become a rejected row, not a traceback."""
    ex = _make_executor(tmp_path, allow_serial_only=True, suite_name=None)
    ex._resolve_binary_path = lambda b, tc: os.path.join(str(tmp_path), "does-not-exist")
    suites = [{"suite_details": {"name": "S", "enabled": True},
               "tests": [{"name": "T", "binary": "rccl-UnitTests", "test_filter": "A.B",
                          "warmup_profile": "fork_coll",
                          "fork_expand": {"num_gpus": [8], "process_mask": 1,
                                          "ranks_per_gpu": 1}}]}]
    resolved, errors = ex.plan_init_pipeline_run(suites)
    assert resolved[0]["disposition"] == "rejected"
    assert "filter preflight" in resolved[0]["error"]
    assert errors, "a rejected entry must abort the run before any spawn"


def test_pipeline_result_strings_reach_the_exit_code():
    """TIMED_OUT / CANCELLED / INFRA_ERROR are scheduler spellings the serial
    TestResult enum does not have; counting only FAILED+TIMEOUT exited 0 on them."""
    from lib.test_executor import count_failures
    assert count_failures(["PASSED", "SKIPPED"]) == (0, 0)
    assert count_failures(["FAILED"]) == (1, 0)
    assert count_failures(["TIMEOUT"]) == (0, 1)
    assert count_failures(["TIMED_OUT"]) == (0, 1)
    assert count_failures(["CANCELLED"]) == (1, 0)
    assert count_failures(["INFRA_ERROR"]) == (1, 0)


def test_interval_validation_failure_is_recorded_not_discarded(tmp_path, monkeypatch):
    """v11 CR-8 was computed, printed and thrown away: two overlapping executions
    still exited 0. The result must now reach a flag the runner can act on."""
    import lib.test_executor as te
    monkeypatch.setattr(te, "validate_execution_intervals",
                        lambda recs: (False, {"max_executing": 2, "executed_entries": 2,
                                              "total_records": len(recs), "ok": False},
                                      ["max EXECUTING == 2: execution intervals overlap"]))
    ex = _make_executor(tmp_path, suite_name=None)
    ex._build_launch_spec = lambda tcfg, suite: (None, {"result": "PASSED", "duration": 0})
    ex.run_all_suites_init_pipeline([
        {"suite_details": {"name": "S", "enabled": True},
         "tests": [{"name": "M", "binary": "rccl-UnitTestsMPI",
                    "warmup_profile": "mpi_coll", "test_filter": "M.C"}]}])
    assert ex.interval_validation_failed is True
    assert ex.interval_validation[-1]["max_executing"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
