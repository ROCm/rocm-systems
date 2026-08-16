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
    eligibility_from_test,
    plan_entries,
    rollup_result,
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
