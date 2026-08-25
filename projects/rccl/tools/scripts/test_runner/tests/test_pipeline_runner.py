#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Host tests for init-pipeline entry planning + record_type roll-up (Phase 4b-3).
"""

import os
import sys

import pytest

_RUNNER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER_ROOT not in sys.path:
    sys.path.insert(0, _RUNNER_ROOT)

from lib.pipeline import KIND_PIPELINE, KIND_SERIAL, RESULT_FAILED, RESULT_TIMED_OUT  # noqa: E402
from lib.pipeline_runner import (  # noqa: E402
    aggregate_phase_timings,
    assemble_records,
    classify_errors,
    eligibility_from_test,
    max_concurrent_execution,
    validate_execution_intervals,
    plan_entries,
    planning_summary,
    resolve_test,
    rollup_result,
    run_guards,
    topline_counts,
)


# --------------------------------------------------------------------------- #
# eligibility / planning
# --------------------------------------------------------------------------- #
def test_serial_mode_forces_serial():
    e = eligibility_from_test({"name": "T", "warmup_profile": "fork_coll"}, exec_mode="serial")
    assert not e.is_pipeline


def test_plan_serial_for_unprofiled():
    plan = plan_entries([{"name": "T"}], exec_mode="init-pipeline")
    assert len(plan) == 1 and plan[0].kind == KIND_SERIAL and not plan[0].is_sub_entry


def test_plan_mpi_single_pipeline_entry():
    plan = plan_entries([{"name": "M", "warmup_profile": "mpi_coll"}], exec_mode="init-pipeline")
    assert len(plan) == 1 and plan[0].kind == KIND_PIPELINE and not plan[0].is_sub_entry


def test_plan_fork_expands_to_sub_entries():
    tests = [{"name": "AllReduce.InPlace", "warmup_profile": "fork_coll",
              "fork_expand": {"num_gpus": [8], "process_mask": 3, "max_ranks_per_gpu": 2}}]
    plan = plan_entries(tests, exec_mode="init-pipeline")
    assert len(plan) == 4
    assert all(p.kind == KIND_PIPELINE and p.is_sub_entry for p in plan)
    assert plan[0].env_overrides["UT_MIN_GPUS"] == "8"
    assert {p.name for p in plan} == {
        "AllReduce.InPlace.g8_sp_r1", "AllReduce.InPlace.g8_sp_r2",
        "AllReduce.InPlace.g8_mp_r1", "AllReduce.InPlace.g8_mp_r2",
    }


def test_plan_single_generation_fork_not_split():
    tests = [{"name": "T", "warmup_profile": "fork_coll",
              "fork_expand": {"num_gpus": [8], "process_mask": 2, "ranks_per_gpu": 1,
                              "max_ranks_per_gpu": 1}}]
    plan = plan_entries(tests, exec_mode="init-pipeline")
    assert len(plan) == 1 and not plan[0].is_sub_entry and plan[0].name == "T"


# --------------------------------------------------------------------------- #
# roll-up
# --------------------------------------------------------------------------- #
def test_rollup_any_fail_is_failed():
    assert rollup_result(["PASSED", RESULT_FAILED, "PASSED"]) == RESULT_FAILED
    assert rollup_result(["PASSED", RESULT_TIMED_OUT]) == RESULT_FAILED


def test_rollup_all_skipped_is_skipped():
    assert rollup_result(["SKIPPED", "SKIPPED"]) == "SKIPPED"


def test_rollup_default_passed():
    assert rollup_result(["PASSED", "SKIPPED", "PASSED"]) == "PASSED"


# --------------------------------------------------------------------------- #
# assemble_records / no double count
# --------------------------------------------------------------------------- #
def test_assemble_split_has_parent_summary_and_no_double_count():
    tests = [{"name": "AR", "warmup_profile": "fork_coll",
              "fork_expand": {"num_gpus": [8], "process_mask": 3, "max_ranks_per_gpu": 1}}]
    plan = plan_entries(tests, exec_mode="init-pipeline")  # AR.g8_sp_r1, AR.g8_mp_r1
    results = {
        "AR.g8_sp_r1": {"result": "PASSED", "suite": "S"},
        "AR.g8_mp_r1": {"result": RESULT_FAILED, "suite": "S"},
    }
    recs = assemble_records(plan, results)
    subs = [r for r in recs if r["record_type"] == "sub_entry"]
    parents = [r for r in recs if r["record_type"] == "parent_summary"]
    assert len(subs) == 2 and len(parents) == 1
    assert all(not r["counts_toward_topline"] for r in subs)
    assert parents[0]["counts_toward_topline"] and parents[0]["result"] == RESULT_FAILED
    # Top-line counts the parent once, never the sub-entries.
    assert topline_counts(recs) == {"PASSED": 0, "FAILED": 1, "SKIPPED": 0}


def test_assemble_single_entries_count_directly():
    plan = plan_entries(
        [{"name": "A", "warmup_profile": "mpi_coll"}, {"name": "B"}],
        exec_mode="init-pipeline")
    results = {"A": {"result": "PASSED", "suite": "S"}, "B": {"result": "SKIPPED", "suite": "S"}}
    recs = assemble_records(plan, results)
    assert all(r["record_type"] == "entry" for r in recs)
    assert topline_counts(recs) == {"PASSED": 1, "FAILED": 0, "SKIPPED": 1}


def test_assemble_mixed_suite_topline():
    tests = [
        {"name": "AR", "warmup_profile": "fork_coll",
         "fork_expand": {"num_gpus": [8], "process_mask": 3, "max_ranks_per_gpu": 1}},
        {"name": "M", "warmup_profile": "mpi_coll"},
        {"name": "S"},  # serial
    ]
    plan = plan_entries(tests, exec_mode="init-pipeline")
    results = {
        "AR.g8_sp_r1": {"result": "PASSED", "suite": "S"},
        "AR.g8_mp_r1": {"result": "PASSED", "suite": "S"},
        "M": {"result": "PASSED", "suite": "S"},
        "S": {"result": "PASSED", "suite": "S"},
    }
    recs = assemble_records(plan, results)
    # 3 top-line: AR parent_summary + M + S
    assert sum(1 for r in recs if r["counts_toward_topline"]) == 3
    assert topline_counts(recs) == {"PASSED": 3, "FAILED": 0, "SKIPPED": 0}


# --------------------------------------------------------------------------- #
# planner classification validation (v10 §5.1/§5.4)
# --------------------------------------------------------------------------- #
def test_classify_unknown_profile_rejected():
    errs = classify_errors([{"name": "T", "warmup_profile": "bogus"}], exec_mode="init-pipeline")
    assert len(errs) == 1 and "unknown warmup_profile" in errs[0][1]


def test_classify_netib_rejected():
    errs = classify_errors([{"name": "N", "warmup_profile": "netib_plugin",
                             "binary": "rccl-UnitTestsMPI"}], exec_mode="init-pipeline")
    assert len(errs) == 1 and "netib_plugin" in errs[0][1]


def test_classify_mpi_profile_needs_mpi_binary():
    bad = classify_errors([{"name": "M", "warmup_profile": "mpi_coll",
                            "binary": "rccl-UnitTests", "test_filter": "M.C"}],
                          exec_mode="init-pipeline")
    assert any("requires an MPI binary" in m for _, m in bad)
    ok = classify_errors([{"name": "M", "warmup_profile": "mpi_coll",
                           "binary": "rccl-UnitTestsMPI", "test_filter": "M.C"}],
                         exec_mode="init-pipeline")
    assert ok == []


def test_classify_fork_profile_needs_fork_binary():
    bad = classify_errors([{"name": "F", "warmup_profile": "fork_coll",
                            "binary": "rccl-UnitTestsMPI", "test_filter": "F.C"}],
                          exec_mode="init-pipeline")
    assert any("requires a fork" in m for _, m in bad)
    ok = classify_errors([{"name": "F", "warmup_profile": "fork_coll",
                           "binary": "rccl-UnitTests", "test_filter": "F.C"}],
                         exec_mode="init-pipeline")
    assert ok == []


def test_classify_pipeline_requires_nonwildcard_filter():
    # Missing filter -> rejected (would run the whole binary).
    bad = classify_errors([{"name": "AR", "binary": "rccl-UnitTests",
                            "warmup_profile": "fork_coll"}], exec_mode="init-pipeline")
    assert any("test_filter" in m for _, m in bad)
    # Wildcard filter -> rejected.
    bad2 = classify_errors([{"name": "AR", "binary": "rccl-UnitTests",
                             "warmup_profile": "fork_coll", "test_filter": "*"}],
                           exec_mode="init-pipeline")
    assert any("test_filter" in m for _, m in bad2)
    # Explicit exact filter -> accepted.
    ok = classify_errors([{"name": "AR", "binary": "rccl-UnitTests",
                           "warmup_profile": "fork_coll", "test_filter": "AllReduce.OutOfPlace"}],
                         exec_mode="init-pipeline")
    assert ok == []


def test_classify_rejects_runner_owned_env_override():
    for var in ("RCCL_TEST_READY_GO", "RCCL_TEST_WARMUP_PROFILE",
                "RCCL_TEST_RENDEZVOUS_DIR", "RCCL_TEST_GO_TIMEOUT_SEC"):
        bad = classify_errors([{"name": "M", "binary": "rccl-UnitTestsMPI",
                                "warmup_profile": "mpi_coll", "test_filter": "M.C",
                                "env_variables": {var: "x"}}], exec_mode="init-pipeline")
        assert any("runner-owned" in m for _, m in bad), var


def test_classify_serial_mode_only_checks_unknown():
    # In serial mode, profile/binary compat isn't enforced (nothing is piped).
    assert classify_errors([{"name": "M", "warmup_profile": "mpi_coll",
                             "binary": "rccl-UnitTests"}], exec_mode="serial") == []
    assert len(classify_errors([{"name": "X", "warmup_profile": "bogus"}],
                               exec_mode="serial")) == 1


# --------------------------------------------------------------------------- #
# resolution / provenance / guards / summary (v10 §7)
# --------------------------------------------------------------------------- #
def test_resolve_serial_and_provenance():
    r = resolve_test({"name": "S"}, exec_mode="init-pipeline")
    assert r["disposition"] == "serial" and r["provenance"] == "fallback"
    assert r["effective_profile"] == "absent" and r["exclusion_reason"] == "no_rendezvous_hook"


def test_resolve_pipeline_fork_counts_subentries():
    r = resolve_test({"name": "AR", "binary": "rccl-UnitTests", "warmup_profile": "fork_coll",
                      "test_filter": "AllReduce.OutOfPlace",
                      "fork_expand": {"num_gpus": [8], "process_mask": 3, "max_ranks_per_gpu": 1}},
                     exec_mode="init-pipeline")
    assert r["disposition"] == "pipeline" and r["provenance"] == "entry"
    assert r["effective_profile"] == "fork_coll" and r["pipeline_subentries"] == 2


def test_resolve_rejected_netib():
    r = resolve_test({"name": "N", "binary": "rccl-UnitTestsMPI", "warmup_profile": "netib_plugin"},
                     exec_mode="init-pipeline")
    assert r["disposition"] == "rejected" and r["exclusion_reason"] == "netib_boundary_unimplemented"


def test_zero_pipeline_guard():
    resolved = [resolve_test({"name": "S"}, exec_mode="init-pipeline")]
    assert run_guards(resolved, exec_mode="init-pipeline")           # zero pipeline -> error
    assert not run_guards(resolved, exec_mode="init-pipeline", allow_serial_only=True)


def test_rejected_entry_is_globally_fatal():
    # A rejected NetIb entry alongside a valid pipeline entry aborts the whole run.
    resolved = [
        resolve_test({"name": "M", "binary": "rccl-UnitTestsMPI", "warmup_profile": "mpi_coll",
                      "test_filter": "AllReduceMPITest.OutOfPlace"}, exec_mode="init-pipeline"),
        resolve_test({"name": "N", "binary": "rccl-UnitTestsMPI", "warmup_profile": "netib_plugin",
                      "test_filter": "NetIb.Foo"}, exec_mode="init-pipeline"),
    ]
    errs = run_guards(resolved, exec_mode="init-pipeline")
    assert any("N" in e and "rejected" in e for e in errs)


def test_duplicate_stable_id_is_fatal():
    from lib.pipeline_runner import stable_id
    a = resolve_test({"name": "M", "binary": "rccl-UnitTestsMPI", "warmup_profile": "mpi_coll",
                      "test_filter": "M.C"}, exec_mode="init-pipeline")
    b = resolve_test({"name": "M", "binary": "rccl-UnitTestsMPI", "warmup_profile": "mpi_coll",
                      "test_filter": "M.C"}, exec_mode="init-pipeline")
    a["suite"] = b["suite"] = "S"
    assert stable_id(a) == stable_id(b)
    assert any("duplicate stable id" in e for e in run_guards([a, b], exec_mode="init-pipeline"))


def test_serial_entry_not_fatal_in_mixed_run():
    resolved = [
        resolve_test({"name": "M", "binary": "rccl-UnitTestsMPI", "warmup_profile": "mpi_coll",
                      "test_filter": "AllReduceMPITest.OutOfPlace"}, exec_mode="init-pipeline"),
        resolve_test({"name": "S"}, exec_mode="init-pipeline"),  # serial by policy
    ]
    assert run_guards(resolved, exec_mode="init-pipeline") == []


def test_overbroad_default_guard():
    # A pipeline entry whose profile came from an implicit fallback is rejected.
    row = resolve_test({"name": "M", "binary": "rccl-UnitTestsMPI", "warmup_profile": "mpi_coll",
                        "test_filter": "M.C"}, exec_mode="init-pipeline")
    row["provenance"] = "fallback"  # simulate an implicit non-none default
    errs = run_guards([row], exec_mode="init-pipeline")
    assert any("implicit" in e for e in errs)


def test_max_concurrent_execution_combined_kinds():
    # pipeline [10,12], serial [12,14], pipeline [14,16] -> never 2 (serial slots in).
    recs = [
        {"entry_kind": "pipeline", "exec_start": 10.0, "exec_end": 12.0},
        {"entry_kind": "serial", "exec_start": 12.0, "exec_end": 14.0},
        {"entry_kind": "pipeline", "exec_start": 14.0, "exec_end": 16.0},
        {"entry_kind": "pipeline", "exec_start": None, "exec_end": None},  # rejected -> skipped
    ]
    assert max_concurrent_execution(recs) == 1


def test_max_concurrent_execution_detects_overlap():
    recs = [
        {"exec_start": 10.0, "exec_end": 13.0},
        {"exec_start": 12.0, "exec_end": 15.0},  # overlaps the first
    ]
    assert max_concurrent_execution(recs) == 2


def test_validate_execution_intervals_ok():
    recs = [
        {"entry_kind": "pipeline", "name": "P", "exec_start": 12.0, "exec_end": 14.0,
         "launch_ts": 10.0, "ready_ts": 11.5, "go_ts": 12.0, "exit_ts": 14.0},
        {"entry_kind": "serial", "name": "S", "exec_start": 14.0, "exec_end": 15.0,
         "launch_ts": 14.0, "exit_ts": 15.0},
    ]
    ok, report, errors = validate_execution_intervals(recs)
    assert ok and report["max_executing"] == 1 and report["executed_entries"] == 2


def test_validate_execution_intervals_detects_overlap():
    recs = [
        {"entry_kind": "pipeline", "name": "A", "exec_start": 10.0, "exec_end": 13.0,
         "launch_ts": 9.0, "ready_ts": 9.5, "go_ts": 10.0, "exit_ts": 13.0},
        {"entry_kind": "pipeline", "name": "B", "exec_start": 12.0, "exec_end": 15.0,
         "launch_ts": 11.0, "ready_ts": 11.5, "go_ts": 12.0, "exit_ts": 15.0},  # overlaps A
    ]
    ok, _, errors = validate_execution_intervals(recs)
    assert not ok and any("overlap" in e for e in errors)


def test_validate_execution_intervals_detects_missing_reap():
    # exec_start set but exec_end absent -> executed without a confirmed reap.
    recs = [{"entry_kind": "serial", "name": "C", "exec_start": 20.0, "exec_end": None,
             "launch_ts": 20.0, "exit_ts": None}]
    ok, _, errors = validate_execution_intervals(recs)
    assert not ok and any("reap" in e for e in errors)


def test_validate_execution_intervals_detects_unordered():
    recs = [{"entry_kind": "pipeline", "name": "X", "exec_start": 10.0, "exec_end": 11.0,
             "launch_ts": 10.0, "ready_ts": 9.0, "go_ts": 10.5, "exit_ts": 11.0}]  # ready < launch
    ok, _, errors = validate_execution_intervals(recs)
    assert not ok and any("ordered" in e for e in errors)


def test_planning_summary_counts_subentries():
    resolved = [
        resolve_test({"name": "AR", "binary": "rccl-UnitTests", "warmup_profile": "fork_coll",
                      "test_filter": "AllReduce.OutOfPlace",
                      "fork_expand": {"num_gpus": [8], "process_mask": 3, "max_ranks_per_gpu": 1}},
                     exec_mode="init-pipeline"),
        resolve_test({"name": "M", "binary": "rccl-UnitTestsMPI", "warmup_profile": "mpi_coll",
                      "test_filter": "M.C"}, exec_mode="init-pipeline"),
        resolve_test({"name": "S"}, exec_mode="init-pipeline"),
    ]
    s = planning_summary(resolved)
    assert s["fork_resolved_subentries"] == 2 and s["mpi_pipeline_entries"] == 1
    assert s["serial_entries"] == 1 and s["executable_pipeline_entries"] == 3


# --------------------------------------------------------------------------- #
# phase-timing aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_phase_timings_basic():
    records = [
        {"phase_timings": {"time_to_ready": 1.0, "ready_queue_wait": 0.0,
                           "execution_time": 2.0, "total": 3.0}},
        {"phase_timings": {"time_to_ready": 3.0, "ready_queue_wait": 4.0,
                           "execution_time": 6.0, "total": 13.0}},
        {"record_type": "parent_summary"},  # no phase_timings -> skipped
    ]
    agg = aggregate_phase_timings(records)
    assert agg["time_to_ready"]["count"] == 2
    assert agg["time_to_ready"]["min"] == 1.0
    assert agg["time_to_ready"]["max"] == 3.0
    assert agg["time_to_ready"]["median"] == 2.0
    assert agg["execution_time"]["sum"] == 8.0


def test_aggregate_skips_unavailable_phases():
    records = [
        {"phase_timings": {"time_to_ready": 1.0, "ready_queue_wait": None,
                           "execution_time": None, "total": None}},
    ]
    agg = aggregate_phase_timings(records)
    assert agg["time_to_ready"]["count"] == 1
    assert agg["ready_queue_wait"] is None  # never reached -> not counted as 0
    assert agg["execution_time"] is None


def test_aggregate_empty():
    assert aggregate_phase_timings([]) == {
        "time_to_ready": None, "ready_queue_wait": None,
        "execution_time": None, "total": None,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
