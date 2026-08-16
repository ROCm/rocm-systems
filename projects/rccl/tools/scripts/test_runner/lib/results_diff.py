#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Per-configuration results diff for the init-pipeline correctness gate (plan §12).

The decisive gate is "identical per-configuration pass/fail/skip vs the serial
baseline" -- compared on the per-config rows, NOT the parent roll-up. This module
indexes each run's top-line records by (suite, test) and reports:

  * regressions      -- passed/skipped in baseline, failed in candidate (gate-fail)
  * fixes            -- failed in baseline, passed in candidate
  * only_baseline    -- coverage the candidate dropped (also a gate concern)
  * only_candidate   -- configs the candidate added
  * changed          -- other result changes (e.g. FAILED<->TIMED_OUT)

Known pre-existing failures (CuMem signal-11, POSIX-FD env cluster, ...) can be
excluded via glob patterns so they don't mask real regressions.

Pure + host-tested; a thin CLI (tools/compare_results.py) wraps it over the
tests.jsonl files the runner emits under --emit-results.
"""

import fnmatch

# Any result that is not a clean pass/skip is a "failure" for gate purposes.
_FAIL = frozenset({"FAILED", "TIMED_OUT", "TIMEOUT", "INFRA_ERROR", "CANCELLED"})
_PASSLIKE = frozenset({"PASSED", "SKIPPED"})


def _is_topline(rec):
    """A serial record has no record_type -> it is top-line; a pipeline record is
    top-line iff counts_toward_topline is set (parent_summary / single entry)."""
    if "counts_toward_topline" in rec:
        return bool(rec["counts_toward_topline"])
    return True


def _key(rec):
    return (rec.get("suite"), rec.get("test_name") or rec.get("name"))


def index_results(records):
    """Map (suite, test) -> result over the top-line records (last one wins)."""
    out = {}
    for r in records:
        if _is_topline(r):
            out[_key(r)] = r.get("result")
    return out


def _excluded(key, patterns):
    if not patterns:
        return False
    name = key[1] or ""
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def diff_results(baseline_records, candidate_records, exclude=None):
    """Diff two runs per-configuration. Returns a dict of finding lists plus the
    matched count and a `gate_ok` flag (True iff no regressions and no dropped
    coverage, after applying the exclude globs)."""
    A = index_results(baseline_records)
    B = index_results(candidate_records)
    keys = set(A) | set(B)

    regressions, fixes, changed, only_baseline, only_candidate = [], [], [], [], []
    matched = 0
    for k in sorted(keys, key=lambda x: (str(x[0]), str(x[1]))):
        if _excluded(k, exclude):
            continue
        ra, rb = A.get(k), B.get(k)
        if k not in B:
            only_baseline.append((k, ra))
            continue
        if k not in A:
            only_candidate.append((k, rb))
            continue
        if ra == rb:
            matched += 1
            continue
        a_fail, b_fail = ra in _FAIL, rb in _FAIL
        if not a_fail and b_fail:
            regressions.append((k, ra, rb))
        elif a_fail and not b_fail:
            fixes.append((k, ra, rb))
        else:
            changed.append((k, ra, rb))

    return {
        "regressions": regressions,
        "fixes": fixes,
        "changed": changed,
        "only_baseline": only_baseline,
        "only_candidate": only_candidate,
        "matched": matched,
        "gate_ok": (not regressions and not only_baseline),
    }


def format_report(diff, baseline_label="baseline", candidate_label="candidate"):
    """Render a human-readable per-config diff report."""
    lines = []
    lines.append(f"Per-config diff: {candidate_label} vs {baseline_label}")
    lines.append(f"  matched:        {diff['matched']}")
    lines.append(f"  regressions:    {len(diff['regressions'])}")
    lines.append(f"  fixes:          {len(diff['fixes'])}")
    lines.append(f"  changed:        {len(diff['changed'])}")
    lines.append(f"  only {baseline_label:9s}: {len(diff['only_baseline'])} (dropped coverage)")
    lines.append(f"  only {candidate_label:9s}: {len(diff['only_candidate'])}")

    def _section(title, rows, three):
        if not rows:
            return
        lines.append(f"\n  {title}:")
        for row in rows[:200]:
            if three:
                (suite, name), ra, rb = row
                lines.append(f"    [{suite}] {name}: {ra} -> {rb}")
            else:
                (suite, name), r = row
                lines.append(f"    [{suite}] {name}: {r}")

    _section("REGRESSIONS (gate-fail)", diff["regressions"], True)
    _section("dropped coverage (only in baseline)", diff["only_baseline"], False)
    _section("fixes", diff["fixes"], True)
    _section("other changes", diff["changed"], True)
    _section("new configs (only in candidate)", diff["only_candidate"], False)

    lines.append(f"\n  GATE: {'PASS' if diff['gate_ok'] else 'FAIL'} "
                 f"(identical per-config pass/fail/skip, no dropped coverage)")
    return "\n".join(lines)
