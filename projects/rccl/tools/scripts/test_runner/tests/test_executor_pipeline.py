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
    assert env["RCCL_TEST_RENDEZVOUS_DIR"] == os.path.abspath(rdv.dir)
    assert os.path.isabs(env["RCCL_TEST_RENDEZVOUS_DIR"])  # ranks run with a different cwd


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
        {"name": "AR", "warmup_profile": "fork_coll",
         "fork_expand": {"num_gpus": [8], "process_mask": 3, "max_ranks_per_gpu": 1}},
        {"name": "M", "warmup_profile": "mpi_coll"},
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
    tests = [{"name": "AR", "warmup_profile": "fork_coll",
              "fork_expand": {"num_gpus": [8], "process_mask": 3, "max_ranks_per_gpu": 1}}]
    records = ex._run_suite_init_pipeline(suite_config, "S", tests)

    by = {r["name"]: r for r in records}
    # After rerun both sub-entries pass and the parent rolls up PASSED.
    assert by["AR.g8_sp_r1"]["result"] == "PASSED"
    assert by["AR.g8_mp_r1"]["result"] == "PASSED"
    assert by["AR"]["record_type"] == "parent_summary" and by["AR"]["result"] == "PASSED"
    assert ex.test_results == ["PASSED"]  # one top-line entry, now passing


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
