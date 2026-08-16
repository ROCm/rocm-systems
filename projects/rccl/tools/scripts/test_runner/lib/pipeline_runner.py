#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Init-pipeline entry planning + result roll-up (plan 4.1a / 9).

Pure glue between the config (warmup_profile / fork_expand) and the scheduler:

  * ``eligibility_from_test``  -- read a test's init-pipeline eligibility.
  * ``plan_entries``           -- expand a suite's tests into the units the
                                  scheduler runs (serial units, single pipeline
                                  entries, or per-generation fork sub-entries).
  * ``rollup_parent`` / ``assemble_records`` -- fold sub-entry results back into a
    parent summary with ``record_type`` + ``counts_toward_topline`` so totals,
    DB rows and perf comparisons never double-count a split sweep test.

Everything here is side-effect free (no process launch), so it is fully unit
tested; the scheduler + TestExecutor supply the actual process I/O separately.
"""

from dataclasses import dataclass, field

try:
    from lib.sweep import (
        Eligibility, PROFILE_FORK, PROFILE_NONE, expand_fork_entry,
    )
    from lib.pipeline import (
        KIND_PIPELINE, KIND_SERIAL,
        RESULT_FAILED, RESULT_TIMED_OUT, RESULT_INFRA_ERROR,
    )
except ImportError:  # allow "import lib.*" from the runner root
    from sweep import (
        Eligibility, PROFILE_FORK, PROFILE_NONE, expand_fork_entry,
    )
    from pipeline import (
        KIND_PIPELINE, KIND_SERIAL,
        RESULT_FAILED, RESULT_TIMED_OUT, RESULT_INFRA_ERROR,
    )

RESULT_PASSED = "PASSED"
RESULT_SKIPPED = "SKIPPED"

# Results that make a parent roll-up FAILED.
_FAIL_RESULTS = frozenset({RESULT_FAILED, RESULT_TIMED_OUT, RESULT_INFRA_ERROR})


@dataclass
class PlannedEntry:
    """One unit the scheduler will run."""
    parent_name: str        # config test name -- the roll-up key
    name: str               # display name (parent, or parent + config suffix)
    kind: str               # KIND_PIPELINE | KIND_SERIAL
    warmup_profile: str
    env_overrides: dict = field(default_factory=dict)  # pinned sweep env (fork)
    is_sub_entry: bool = False   # True iff this parent was split into >1 sub-entries


def eligibility_from_test(test, *, exec_mode):
    """Resolve a test's init-pipeline eligibility. Under --exec-mode=serial every
    test is serial; otherwise it comes from the config warmup_profile/fork_expand."""
    if exec_mode != "init-pipeline":
        return Eligibility(warmup_profile=PROFILE_NONE)
    fe = test.get("fork_expand", {}) or {}
    return Eligibility(
        warmup_profile=test.get("warmup_profile", PROFILE_NONE),
        num_gpus=list(fe.get("num_gpus", [])),
        process_mask=fe.get("process_mask", 3),
        max_ranks_per_gpu=fe.get("max_ranks_per_gpu", 1),
        ranks_per_gpu=fe.get("ranks_per_gpu", 0),
        only_pow2=fe.get("only_pow2", False),
    )


def plan_entries(tests, *, exec_mode):
    """Expand a suite's tests into scheduler units.

    * non-pipeline (warmup_profile none / serial mode) -> one KIND_SERIAL unit;
    * mpi_coll / netib_plugin                          -> one KIND_PIPELINE unit;
    * fork_coll                                        -> one KIND_PIPELINE unit
      per single generation (a lone generation stays one un-split entry).

    Raises ValueError (via expand_fork_entry) on a bad fork selector.
    """
    planned = []
    for t in tests:
        name = t.get("name")
        elig = eligibility_from_test(t, exec_mode=exec_mode)
        if not elig.is_pipeline:
            planned.append(PlannedEntry(name, name, KIND_SERIAL, PROFILE_NONE))
            continue
        if elig.warmup_profile == PROFILE_FORK:
            subs = expand_fork_entry(elig)
            if len(subs) <= 1:
                env = subs[0]["env"] if subs else {}
                planned.append(PlannedEntry(name, name, KIND_PIPELINE, PROFILE_FORK, env, False))
            else:
                for s in subs:
                    planned.append(PlannedEntry(
                        name, f"{name}.{s['suffix']}", KIND_PIPELINE,
                        PROFILE_FORK, s["env"], True))
        else:
            planned.append(PlannedEntry(name, name, KIND_PIPELINE, elig.warmup_profile))
    return planned


def rollup_result(sub_results):
    """Parent roll-up from sub-entry result strings (plan 9): FAILED if any
    sub failed/timed-out/infra-errored, else SKIPPED if all skipped, else PASSED."""
    results = list(sub_results)
    if any(r in _FAIL_RESULTS for r in results):
        return RESULT_FAILED
    if results and all(r == RESULT_SKIPPED for r in results):
        return RESULT_SKIPPED
    return RESULT_PASSED


def assemble_records(planned, results):
    """Turn planned units + their results into final records.

    ``results`` maps a PlannedEntry.name -> result dict (must contain 'result';
    may carry duration/phase timings). Output rows carry ``record_type`` and
    ``counts_toward_topline`` so the top-line total = sum of rows where
    counts_toward_topline is True (each single entry once, each split sweep once
    via its parent_summary -- sub-entries never counted).
    """
    # Group planned units by parent, preserving order.
    order = []
    by_parent = {}
    for p in planned:
        if p.parent_name not in by_parent:
            by_parent[p.parent_name] = []
            order.append(p.parent_name)
        by_parent[p.parent_name].append(p)

    records = []
    for parent in order:
        group = by_parent[parent]
        split = len(group) > 1 or any(p.is_sub_entry for p in group)
        if not split:
            p = group[0]
            r = results.get(p.name, {})
            rec = {
                "record_type": "entry",
                "suite": r.get("suite"),
                "test_name": parent,
                "name": p.name,
                "result": r.get("result"),
                "kind": p.kind,
                "warmup_profile": p.warmup_profile,
                "counts_toward_topline": True,
            }
            rec.update({k: v for k, v in r.items() if k not in rec})
            records.append(rec)
            continue

        sub_results = []
        for p in group:
            r = results.get(p.name, {})
            sub_results.append(r.get("result"))
            rec = {
                "record_type": "sub_entry",
                "suite": r.get("suite"),
                "test_name": parent,
                "name": p.name,
                "result": r.get("result"),
                "kind": p.kind,
                "warmup_profile": p.warmup_profile,
                "counts_toward_topline": False,
            }
            rec.update({k: v for k, v in r.items() if k not in rec})
            records.append(rec)
        records.append({
            "record_type": "parent_summary",
            "suite": next((results.get(p.name, {}).get("suite") for p in group), None),
            "test_name": parent,
            "name": parent,
            "result": rollup_result(sub_results),
            "sub_entry_count": len(group),
            "counts_toward_topline": True,
        })
    return records


def topline_counts(records):
    """Count PASSED/FAILED/SKIPPED over only the top-line records."""
    counts = {RESULT_PASSED: 0, RESULT_FAILED: 0, RESULT_SKIPPED: 0}
    for r in records:
        if not r.get("counts_toward_topline"):
            continue
        res = r.get("result")
        if res in _FAIL_RESULTS:
            counts[RESULT_FAILED] += 1
        elif res == RESULT_SKIPPED:
            counts[RESULT_SKIPPED] += 1
        elif res == RESULT_PASSED:
            counts[RESULT_PASSED] += 1
    return counts
