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
    from lib.gtest_preflight import is_wildcard_filter
except ImportError:
    from gtest_preflight import is_wildcard_filter

try:
    from lib.sweep import (
        Eligibility, PROFILE_FORK, PROFILE_MPI, PROFILE_NETIB, PROFILE_NONE,
        expand_fork_entry,
    )
    from lib.pipeline import (
        KIND_PIPELINE, KIND_SERIAL,
        RESULT_FAILED, RESULT_TIMED_OUT, RESULT_INFRA_ERROR,
    )
except ImportError:  # allow "import lib.*" from the runner root
    from sweep import (
        Eligibility, PROFILE_FORK, PROFILE_MPI, PROFILE_NETIB, PROFILE_NONE,
        expand_fork_entry,
    )
    from pipeline import (
        KIND_PIPELINE, KIND_SERIAL,
        RESULT_FAILED, RESULT_TIMED_OUT, RESULT_INFRA_ERROR,
    )

RESULT_PASSED = "PASSED"
RESULT_SKIPPED = "SKIPPED"

# Distinct exit code the C++ binaries use for a profile/role mismatch (must match
# RcclUnitTesting::RCCL_TEST_CONFIG_ERROR in test/common/Rendezvous.hpp).
CONFIG_ERROR_EXIT_CODE = 42
_VALID_PROFILES = frozenset({PROFILE_FORK, PROFILE_MPI, PROFILE_NETIB, PROFILE_NONE})


# Exclusion-reason enum (v10 §7): why a test is not a pipeline entry.
EXCLUSION_REASONS = frozenset({
    "no_rendezvous_hook", "first_init_semantics", "intentional_init_failure",
    "fault_injection", "device_visibility_mismatch", "multiple_generations",
    "runtime_or_profiler_lifecycle", "netib_boundary_unimplemented",
    "unknown_binary", "unclassified",
})


def resolve_test(test, *, exec_mode):
    """Resolve one test's init-pipeline disposition (pure, pre-launch).

    Returns a manifest row: name, binary, planner_profile, provenance
    (entry | suite | fallback), disposition (pipeline | serial | rejected),
    effective_profile (the value the launched process gets, or 'absent'/'disabled'
    for serial), pipeline_subentries, exclusion_reason, error.
    """
    name = test.get("name")
    binary = str(test.get("binary", "") or "")
    explicit = "warmup_profile" in test
    # suite-level provenance is honored only if the config marks it explicitly.
    suite_scoped = bool(test.get("_warmup_profile_from_suite"))
    prof = test.get("warmup_profile", PROFILE_NONE)
    provenance = "entry" if explicit and not suite_scoped else ("suite" if suite_scoped else "fallback")

    row = {
        "name": name, "binary": binary, "planner_profile": prof,
        "provenance": provenance, "disposition": "serial",
        "effective_profile": "absent", "pipeline_subentries": 0,
        "exclusion_reason": None, "error": None,
    }
    if exec_mode != "init-pipeline":
        return row

    errs = classify_errors([test], exec_mode="init-pipeline")
    if errs:
        row["disposition"] = "rejected"
        row["error"] = errs[0][1]
        if prof == PROFILE_NETIB:
            row["exclusion_reason"] = "netib_boundary_unimplemented"
        elif prof not in _VALID_PROFILES:
            row["exclusion_reason"] = "unclassified"
        else:
            row["exclusion_reason"] = "unknown_binary"
        return row

    if prof == PROFILE_NONE:
        row["exclusion_reason"] = "no_rendezvous_hook"
        return row

    subs = plan_entries([test], exec_mode="init-pipeline")
    row["disposition"] = "pipeline"
    row["effective_profile"] = prof
    row["pipeline_subentries"] = sum(1 for p in subs if p.kind == KIND_PIPELINE)
    return row


def planning_summary(resolved):
    """Counts for the pre-execution planning summary (executable sub-entries, not
    parent descriptors)."""
    def _pipe(r):
        return r["disposition"] == "pipeline"
    mpi_subs = sum(r["pipeline_subentries"] for r in resolved if _pipe(r) and r["planner_profile"] == PROFILE_MPI)
    fork_parents = sum(1 for r in resolved if _pipe(r) and r["planner_profile"] == PROFILE_FORK)
    fork_subs = sum(r["pipeline_subentries"] for r in resolved if _pipe(r) and r["planner_profile"] == PROFILE_FORK)
    netib_rejected = sum(1 for r in resolved if r["disposition"] == "rejected" and r["planner_profile"] == PROFILE_NETIB)
    rejected = sum(1 for r in resolved if r["disposition"] == "rejected")
    serial = sum(1 for r in resolved if r["disposition"] == "serial")
    executable_pipeline = mpi_subs + fork_subs
    return {
        "mpi_pipeline_entries": mpi_subs,
        "fork_parent_descriptors": fork_parents,
        "fork_resolved_subentries": fork_subs,
        "netib_rejected": netib_rejected,
        "rejected_total": rejected,
        "serial_entries": serial,
        "executable_pipeline_entries": executable_pipeline,
    }


def run_guards(resolved, *, exec_mode, allow_serial_only=False):
    """Whole-run guardrails (v10 §7 / v11 CR-2). Returns a deduped list of fatal
    error messages; a non-empty list means the runner must exit BEFORE spawning
    anything. A serial-by-policy ('none') entry is never fatal."""
    errors = []
    if exec_mode != "init-pipeline":
        return errors

    # CR-2: ANY rejected entry aborts the whole run before spawn.
    for r in resolved:
        if r["disposition"] == "rejected":
            ident = f"{r.get('suite', '?')}/{r.get('name')}"
            reason = r.get("error") or r.get("exclusion_reason") or "rejected"
            errors.append(f"{ident}: rejected -- {reason}")

    executable = sum(r["pipeline_subentries"] for r in resolved if r["disposition"] == "pipeline")
    if executable == 0 and not allow_serial_only:
        errors.append("--exec-mode init-pipeline resolved ZERO pipeline entries over the whole run; "
                      "pass --allow-serial-only if a serial-only run is intended")
    # Overbroad-default guard: a pipeline entry from an implicit (non-explicit)
    # fallback is rejected -- classification must be entry- or approved-suite-level.
    for r in resolved:
        if r["disposition"] == "pipeline" and r["provenance"] not in ("entry", "suite"):
            errors.append(f"{r.get('suite', '?')}/{r['name']}: pipeline classification came from an "
                          f"implicit '{r['provenance']}' default; classification must be explicit "
                          f"(entry or approved suite level)")

    # Deduplicate while preserving order + suite/test identity.
    seen = set()
    deduped = []
    for e in errors:
        if e not in seen:
            seen.add(e)
            deduped.append(e)
    return deduped


def classify_errors(tests, *, exec_mode):
    """Planner-side (pre-spawn) classification checks (v10 §5.1/§5.4).

    Returns a list of (test_name, message). A non-empty list means the runner must
    NOT spawn those entries; it records them as INFRA_ERROR/configuration_error
    instead. Checks: unknown warmup_profile; netib_plugin rejected until its READY
    boundary exists; and binary/profile compatibility (mpi_coll needs an MPI
    binary; fork_coll needs a fork rccl-UnitTests binary).
    """
    errors = []
    for t in tests:
        name = t.get("name")
        prof = t.get("warmup_profile", PROFILE_NONE)
        if prof not in _VALID_PROFILES:
            errors.append((name, f"unknown warmup_profile '{prof}' "
                                 f"(expected one of {sorted(_VALID_PROFILES)})"))
            continue
        if exec_mode != "init-pipeline":
            continue
        # The runner owns the pipeline env vars; a test/suite may not override them.
        user_env = t.get("env_variables", {}) or {}
        for k in ("RCCL_TEST_WARMUP_PROFILE", "RCCL_TEST_READY_GO",
                  "RCCL_TEST_RENDEZVOUS_DIR", "RCCL_TEST_GO_TIMEOUT_SEC"):
            if k in user_env:
                errors.append((name, f"env_variables may not set runner-owned '{k}' "
                                     f"(the init-pipeline runner constructs it)"))
        if prof == PROFILE_NETIB:
            errors.append((name, "warmup_profile 'netib_plugin' is rejected until its "
                                 "READY boundary is implemented (v10 §5.4)"))
            continue
        binary = str(t.get("binary", "") or "")
        if prof == PROFILE_MPI and "MPI" not in binary:
            errors.append((name, f"warmup_profile 'mpi_coll' requires an MPI binary, "
                                 f"got '{binary}'"))
        elif prof == PROFILE_FORK and ("MPI" in binary or "rccl-UnitTests" not in binary):
            errors.append((name, f"warmup_profile 'fork_coll' requires a fork "
                                 f"rccl-UnitTests binary, got '{binary}'"))
        # Pipeline gtest entries must select exactly one case: a missing/wildcard
        # filter runs the whole binary (multiple generations) -- the job-216183
        # failure. name is the runner label, NOT the gtest filter (v11 CR-1).
        if prof in (PROFILE_FORK, PROFILE_MPI) and t.get("is_gtest", True):
            if is_wildcard_filter(t.get("test_filter")):
                errors.append((name, "pipeline gtest entry requires an explicit "
                                     f"non-wildcard test_filter (got {t.get('test_filter')!r}); "
                                     "'name' is the runner label, not the gtest filter"))
    return errors

# Results that make a parent roll-up FAILED. Includes both the scheduler's
# "TIMED_OUT" and the gtest-inference "TIMEOUT" spellings.
_FAIL_RESULTS = frozenset({RESULT_FAILED, RESULT_TIMED_OUT, RESULT_INFRA_ERROR, "TIMEOUT"})


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


_PHASES = ("time_to_ready", "ready_queue_wait", "execution_time", "total")


def _percentile(sorted_vals, q):
    """Linear-interpolated percentile (q in [0,1]) of a pre-sorted list."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def aggregate_phase_timings(records):
    """Aggregate per-entry ``phase_timings`` into count/min/median/p95/max/sum for
    each phase (init-pipeline §9). Records without phase_timings (e.g. parent
    summaries, serial-path rows) and unavailable (None) phases are skipped, so the
    counts reflect only entries that actually reached that phase."""
    acc = {p: [] for p in _PHASES}
    for r in records:
        pt = r.get("phase_timings")
        if not pt:
            continue
        for p in _PHASES:
            v = pt.get(p)
            if v is not None:
                acc[p].append(v)
    out = {}
    for p, vals in acc.items():
        if not vals:
            out[p] = None
            continue
        s = sorted(vals)
        out[p] = {
            "count": len(s), "min": s[0], "median": _percentile(s, 0.5),
            "p95": _percentile(s, 0.95), "max": s[-1], "sum": sum(s),
        }
    return out


def max_concurrent_execution(records):
    """Max overlap of execution intervals across BOTH entry kinds (review
    amendment 1): pipeline exec = [go, reap], serial exec = [spawn, reap]. The
    run-wide invariant is EXECUTING <= 1, so this must be 1. Records without an
    execution interval (rejected/cancelled before exec) are skipped. Uses the
    ``exec_start``/``exec_end`` monotonic stamps the runner records."""
    events = []
    for r in records:
        a = r.get("exec_start")
        b = r.get("exec_end")
        if a is None or b is None or b <= a:
            continue
        events.append((a, 1))
        events.append((b, -1))
    events.sort(key=lambda x: (x[0], x[1]))  # close before open at a tie
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def validate_execution_intervals(records):
    """Production validity check run automatically before a run is declared valid
    (v11 CR-8). Returns (ok, report, errors).

    Fails if: any two execution intervals overlap (max EXECUTING > 1); an executed
    entry lacks a confirmed reap endpoint; or a pipeline entry's phase timestamps
    are not ordered launch <= ready <= go <= exit (serial: launch <= exit). Missing
    events must remain absent (None), never zero-filled.
    """
    errors = []
    peak = max_concurrent_execution(records)
    if peak > 1:
        errors.append(f"max EXECUTING == {peak}: execution intervals overlap")

    executed = 0
    for r in records:
        start = r.get("exec_start")
        if start is None:
            continue  # never executed (rejected / cancelled pre-exec)
        executed += 1
        if r.get("exec_end") is None:
            errors.append(f"{r.get('name')}: executed entry has no confirmed reap endpoint")
            continue
        kind = r.get("entry_kind")
        if kind == "pipeline":
            seq = [t for t in (r.get("launch_ts"), r.get("ready_ts"),
                               r.get("go_ts"), r.get("exit_ts")) if t is not None]
            if seq != sorted(seq):
                errors.append(f"{r.get('name')}: pipeline timestamps not ordered "
                              f"(launch<=ready<=go<=exit)")
        elif kind == "serial":
            lo, hi = r.get("launch_ts"), r.get("exit_ts")
            if lo is not None and hi is not None and hi < lo:
                errors.append(f"{r.get('name')}: serial exit precedes launch")

    report = {
        "max_executing": peak,
        "executed_entries": executed,
        "total_records": len(records),
        "ok": not errors,
    }
    return (not errors, report, errors)


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
