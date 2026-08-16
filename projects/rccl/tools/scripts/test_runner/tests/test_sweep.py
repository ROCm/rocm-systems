#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Host tests for the pure sweep enumerator + fork eligibility (init-pipeline 4.1/4.1a).

These pin down that ``enumerate_sweep`` reproduces the C++ RunSimpleSweep outer
loops exactly (coverage AND order), that pinning collapses to one generation, and
that bad selectors are rejected the same way the C++ harness rejects them.
"""

import itertools
import os
import sys

import pytest

_RUNNER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER_ROOT not in sys.path:
    sys.path.insert(0, _RUNNER_ROOT)

from lib.sweep import (  # noqa: E402
    Eligibility,
    PROCESS_MULTI,
    PROCESS_SINGLE,
    PROFILE_FORK,
    PROFILE_MPI,
    PROFILE_NONE,
    SweepConfig,
    build_manifest,
    enumerate_sweep,
    expand_fork_entry,
    is_pipeline_profile,
)


def _reference_sweep(num_gpus_list, process_mask, max_rpg, rpg, only_pow2):
    """Independent re-implementation of the C++ nested loops for cross-check."""
    mp = []
    if process_mask & PROCESS_SINGLE:
        mp.append(False)
    if process_mask & PROCESS_MULTI:
        mp.append(True)
    lo = rpg if rpg > 0 else 1
    hi = rpg if rpg > 0 else max_rpg
    out = []
    for g in num_gpus_list:
        if only_pow2 and (g & (g - 1)) != 0:
            continue
        for m in mp:
            for r in range(lo, hi + 1):
                out.append((g, m, r))
    return out


# --------------------------------------------------------------------------- #
# enumerate_sweep
# --------------------------------------------------------------------------- #
def test_unset_matches_nested_loops():
    cfgs = enumerate_sweep([1, 2, 4, 8], process_mask=3, max_ranks_per_gpu=3)
    got = [(c.num_gpus, c.multi_process, c.ranks_per_gpu) for c in cfgs]
    assert got == _reference_sweep([1, 2, 4, 8], 3, 3, 0, False)


def test_order_is_gpus_then_process_then_rpg():
    cfgs = enumerate_sweep([2, 4], process_mask=3, max_ranks_per_gpu=2)
    got = [(c.num_gpus, c.multi_process, c.ranks_per_gpu) for c in cfgs]
    # num_gpus outer, single-before-multi, ranks_per_gpu inner
    assert got == [
        (2, False, 1), (2, False, 2), (2, True, 1), (2, True, 2),
        (4, False, 1), (4, False, 2), (4, True, 1), (4, True, 2),
    ]


def test_pinned_ranks_per_gpu_is_single_generation():
    cfgs = enumerate_sweep([8], process_mask=PROCESS_MULTI, max_ranks_per_gpu=4, ranks_per_gpu=2)
    assert len(cfgs) == 1
    assert cfgs[0] == SweepConfig(8, True, 2)


def test_process_mask_single_only():
    cfgs = enumerate_sweep([8], process_mask=PROCESS_SINGLE, max_ranks_per_gpu=1)
    assert [c.multi_process for c in cfgs] == [False]


def test_only_pow2_filters():
    cfgs = enumerate_sweep([1, 2, 3, 5, 8], process_mask=PROCESS_SINGLE, only_pow2=True)
    assert [c.num_gpus for c in cfgs] == [1, 2, 8]


def test_negative_ranks_per_gpu_rejected():
    with pytest.raises(ValueError):
        enumerate_sweep([8], ranks_per_gpu=-1)


def test_ranks_per_gpu_above_max_rejected():
    with pytest.raises(ValueError):
        enumerate_sweep([8], max_ranks_per_gpu=2, ranks_per_gpu=3)


def test_env_pins_all_three_dimensions():
    env = SweepConfig(8, True, 2).env()
    assert env["UT_MIN_GPUS"] == "8" and env["UT_MAX_GPUS"] == "8"
    assert env["UT_PROCESS_MASK"] == str(PROCESS_MULTI)
    assert env["UT_RANKS_PER_GPU"] == "2"
    assert env["UT_POW2_GPUS"] == "0"


def test_suffix_is_stable_and_safe():
    assert SweepConfig(8, False, 1).suffix() == "g8_sp_r1"
    assert SweepConfig(4, True, 2).suffix() == "g4_mp_r2"


# --------------------------------------------------------------------------- #
# Eligibility / expansion / manifest
# --------------------------------------------------------------------------- #
def test_profile_routing():
    assert is_pipeline_profile(PROFILE_FORK)
    assert is_pipeline_profile(PROFILE_MPI)
    assert not is_pipeline_profile(PROFILE_NONE)
    assert not is_pipeline_profile("bogus")


def test_expand_fork_entry_single_generation_each():
    elig = Eligibility(warmup_profile=PROFILE_FORK, num_gpus=[8],
                       process_mask=3, max_ranks_per_gpu=2)
    subs = expand_fork_entry(elig)
    assert len(subs) == 4  # {sp,mp} x {r1,r2}
    assert all(s["expected_generations"] == 1 for s in subs)
    assert {s["suffix"] for s in subs} == {"g8_sp_r1", "g8_sp_r2", "g8_mp_r1", "g8_mp_r2"}


def test_non_fork_profiles_not_expanded():
    assert expand_fork_entry(Eligibility(warmup_profile=PROFILE_MPI)) == []
    assert expand_fork_entry(Eligibility(warmup_profile=PROFILE_NONE)) == []


def test_manifest_fork_rows():
    elig = Eligibility(warmup_profile=PROFILE_FORK, num_gpus=[8], process_mask=PROCESS_MULTI,
                       max_ranks_per_gpu=2, ranks_per_gpu=2)
    rows = build_manifest("AllReduce.InPlace", elig)
    assert rows == [{
        "parent_test": "AllReduce.InPlace", "num_gpus": 8,
        "multi_process": True, "ranks_per_gpu": 2, "expected_generations": 1,
    }]


def test_manifest_mpi_single_row():
    rows = build_manifest("NetIbMPI.General", Eligibility(warmup_profile=PROFILE_MPI))
    assert len(rows) == 1 and rows[0]["expected_generations"] == 1


def test_manifest_serial_empty():
    assert build_manifest("X", Eligibility(warmup_profile=PROFILE_NONE)) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
