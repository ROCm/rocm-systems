#!/usr/bin/env python3
"""
check_matrix.py — the HRR API matrix reporter.

Three jobs, one source of truth:

  --emit-cxx [PATH] Generate the C++ expectations header the Catch2 tests
                    compile against, so the tests and this reporter cannot
                    disagree about what an API is supposed to do. Defaults to
                    the header in projects/hrr/tests/integration.

  --summary         Print the tier table without running anything. Useful for
                    seeing what the matrix claims to cover.

  --check-workloads Read the tier workload lists against the Catch2 sources:
                    every workload a tier names must exist, and every workload
                    in the matrix's own sources must be named by a tier.

  --results DIR     Read the observation files the Catch2 tests wrote during a
                    run, join them against the manifest, and emit
                    matrix_report.json plus a markdown coverage report.

An "observation" is what one tier's test actually saw: which APIs appeared in
the archive's Event Type Breakdown, and which of them emitted a NOOP or
ERROR_STUB warning during replay. The tests write these as JSON so this script
does not have to re-run a GPU workload to produce a report.

Verdicts per API:

  PASS           observed class == expected class.
  XFAIL          expected to be broken and still broken. Payload-loss APIs sit
                 here; they turn into XPASS when P2 fixes them, which is the
                 signal to update api_matrix.yaml.
  XPASS          expected to be broken and isn't. Not a failure, but it means
                 the matrix is now describing the past.
  FAIL           observed class != expected class. Something was reclassified
                 without the matrix being updated.
  NOT_CAPTURED   declared as never reaching the archive under its own name, and
                 it did not. HIP lowers hipExtModuleLaunchKernel and chevron
                 launches to hipModuleLaunchKernel before HRR sees them, so no
                 workload can ever make those rows appear; asserting the absence
                 is the only honest thing to check.
  NOT_EXERCISED  nothing reached it. Fine only if api_matrix.yaml's
                 unreachable: section explains why, with a reason that was
                 measured on this stack; an undeclared one fails the run,
                 because a workload that quietly stops reaching an API is
                 exactly how this matrix would rot into a green no-op.
  SKIPPED        the overlay asks for no assertion at all, or the tier did not
                 run in this pass.

Exit codes: 0 all good, 1 at least one FAIL, an undeclared coverage gap, or a
tier below its coverage floor, 2 usage/tool error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matrix_common as mc  # noqa: E402

# The header the Catch2 tests compile against, so --emit-cxx needs no argument
# from inside the tree: projects/hrr/tools/api-matrix -> projects/hrr.
DEFAULT_HEADER = (Path(__file__).resolve().parents[2] / "tests" /
                  "integration" / "hrr_api_matrix_expectations.h")
DEFAULT_TEST_DIR = DEFAULT_HEADER.parent

# These two files exist only to supply tier workloads, so a workload defined in
# one of them that no tier names is dead weight: it compiles, it passes when
# invoked by name, and the APIs it calls are simply never counted. That is how
# the _spt coverage went missing when those calls were split out of the
# original families into their own translation unit. Workloads elsewhere are
# driven by their own roundtrip tests and are not expected to appear in a tier.
MATRIX_WORKLOAD_SOURCES = ("hrr_api_matrix_workload_test.cc",
                           "hrr_spt_workload_test.cc")

_DIRECT_CASE_RE = re.compile(r'TEST_CASE\(\s*"(Unit_HRR_[A-Za-z0-9_]*_Direct)"')

VERDICT_PASS = "PASS"
VERDICT_XFAIL = "XFAIL"
VERDICT_XPASS = "XPASS"
VERDICT_FAIL = "FAIL"
VERDICT_NOT_CAPTURED = "NOT_CAPTURED"
VERDICT_NOT_EXERCISED = "NOT_EXERCISED"
VERDICT_SKIPPED = "SKIPPED"

VERDICT_ORDER = [VERDICT_FAIL, VERDICT_XPASS, VERDICT_PASS, VERDICT_XFAIL,
                 VERDICT_NOT_CAPTURED, VERDICT_NOT_EXERCISED, VERDICT_SKIPPED]

# Verdicts that mean "the matrix checked something and it held". Everything
# else is either a problem or an admitted gap.
VERDICT_COVERED = {VERDICT_PASS, VERDICT_XFAIL, VERDICT_NOT_CAPTURED}

CLASS_CODE = {mc.CLASS_REAL: 0, mc.CLASS_NOOP: 1, mc.CLASS_ERROR_STUB: 2,
              mc.CLASS_HANDLER_ERROR: 3, mc.CLASS_CRASH: 4,
              mc.CLASS_UNREPLAYABLE: 5}

# Must match HrrAppliesWhen in the emitted header.
WHEN_CODE = {mc.UNREACHABLE_ALWAYS: 0, mc.UNREACHABLE_NO_IMAGE_SUPPORT: 1}

# Classes that mean the API ran a real handler and did not come back cleanly.
# Both are known-broken rather than misclassified, so they report as XFAIL and
# flip to XPASS the day the handler is fixed.
BROKEN_CLASSES = (mc.CLASS_HANDLER_ERROR, mc.CLASS_CRASH)

# Placeholder for "two tiers saw this API do different things". Not a class an
# API can have; it is the report saying the observations do not agree.
CLASS_CONFLICT = "CONFLICT"


# ---------------------------------------------------------------------------
# C++ header generation
# ---------------------------------------------------------------------------

CXX_HEADER = '''/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * GENERATED FILE — do not edit by hand.
 *
 * Regenerate with:
 *   projects/hrr/tools/api-matrix/check_matrix.py --emit-cxx \\
 *       projects/hrr/tests/integration/{basename}
 *
 * Source of truth is the pair api_classes.json (derived from HRR's generator)
 * and api_matrix.yaml (the authored tier and expectation overlay). Generating
 * this header rather than transcribing it is what keeps the Catch2 tests and
 * the matrix reporter from drifting into disagreeing about what an API is
 * supposed to do at replay.
 *
 * Generated {timestamp} from {api_count} HIP APIs.
 */

#pragma once

#include <cstddef>

// Mirrors HrrReplayClass in hrr_test_common.hh. Kept as a plain int so this
// header stays self-contained and can be included in either order.
enum HrrExpectCode {{
  kHrrExpectReal = 0,       // a real handler runs: GENERATED, MANUAL or CUSTOM
  kHrrExpectNoop = 1,       // NOOP_PLAYBACK_APIS
  kHrrExpectErrorStub = 2,  // ERROR_STUB_PLAYBACK_APIS
  kHrrExpectHandlerError = 3,  // a real handler runs and returns a HIP error
  kHrrExpectCrash = 4,      // a real handler runs and kills the replay process
  kHrrExpectUnreplayable = 5,  // UNREPLAYABLE_PLAYBACK_APIS: refused by name,
                               // with the reason, instead of being attempted
}};

enum HrrAppliesWhen {{
  kHrrWhenAlways = 0,          // holds on any part
  kHrrWhenNoImageSupport = 1,  // only where hipDeviceAttributeImageSupport = 0
}};

struct HrrApiExpectation {{
  const char* api;
  const char* tier;
  int expect;          // HrrExpectCode
  bool payload_loss;   // counted as faithful but cannot replay (section 8.3)
  bool expect_captured;  // false: HIP folds it away before HRR records it
  // The handler is real and works, but for the arguments one of this tier's
  // own workloads records it returns an error. Observing HANDLER_ERROR is
  // then a known failure rather than the API having been reclassified.
  bool handler_error_ok;
  bool skip;           // not assertable from a single-process Catch2 workload
  // Device capability this API's unreachable: declaration was measured under.
  // kHrrWhenAlways for everything that holds on any part. Otherwise the matrix
  // has only ever seen the API on a part with that capability, and on any
  // other one it is unmeasured: whether a texture or array handle can exist is
  // a property of the device, so a declaration taken where none can is not
  // evidence about a part where they can.
  int applies_when;
}};

// A tier's workloads and the coverage floor its run must clear.
//
// min_covered is the number of APIs the tier must actually reach for its
// result to mean anything: a tier that exercises fewer than this has lost
// call sites, most likely because a workload started failing early and took
// the rest of its API calls with it. Without the floor, a workload that
// aborted on its second line would report a clean sweep of zero failures.
//
// workloads is null-terminated so the test can walk it without a second count.
struct HrrTierFloor {{
  const char* tier;
  int min_covered;
  int gpus;
  bool skip_by_default;
  const char* const* workloads;
}};

inline constexpr HrrApiExpectation kHrrApiMatrix[] = {{
{rows}
}};

inline constexpr size_t kHrrApiMatrixCount =
    sizeof(kHrrApiMatrix) / sizeof(kHrrApiMatrix[0]);

{workload_arrays}
inline constexpr HrrTierFloor kHrrTierFloors[] = {{
{floors}
}};

inline constexpr size_t kHrrTierFloorCount =
    sizeof(kHrrTierFloors) / sizeof(kHrrTierFloors[0]);
'''


def emit_cxx(rows: List[mc.ApiRow], overlay: Dict[str, Any], path: Path) -> None:
    body = []
    for r in sorted(rows, key=lambda x: (x.tier, x.api)):
        body.append(
            '    {{"{api}", "{tier}", {expect}, {ploss}, {cap}, {herr}, '
            '{skip}, {when}}},'.format(
                api=r.api, tier=r.tier, expect=CLASS_CODE[r.expect_class],
                ploss="true" if r.payload_loss else "false",
                cap="true" if r.expect_captured else "false",
                herr="true" if r.tier in r.handler_error_in else "false",
                skip="true" if r.skip else "false",
                when=WHEN_CODE[r.unreachable_when]))

    floors = []
    workload_arrays = []
    for tier in mc.TIER_ORDER:
        meta = (overlay.get("tiers") or {}).get(tier)
        if meta is None:
            continue
        names = mc.tier_workloads(overlay, tier)
        array = f"kHrrWorkloads{tier}"
        entries = "".join(f'\n    "{w}",' for w in names)
        workload_arrays.append(
            f"inline constexpr const char* const {array}[] = {{{entries}\n"
            "    nullptr,\n};\n")
        floors.append(
            '    {{"{t}", {n}, {g}, {s}, {a}}},'.format(
                t=tier, n=int(meta.get("min_covered", 0)),
                g=int(meta.get("gpus", 1)),
                s="true" if meta.get("skip_by_default") else "false",
                a=array))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CXX_HEADER.format(
        basename=path.name,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        api_count=len(rows),
        rows="\n".join(body),
        workload_arrays="\n".join(workload_arrays),
        floors="\n".join(floors),
    ))


# ---------------------------------------------------------------------------
# Observation loading and verdicts
# ---------------------------------------------------------------------------

def load_observations(results_dir: Path) -> Dict[str, Any]:
    """Merge the per-tier JSON files the Catch2 tests wrote.

    Shape of each file:
        {"tier": "T0",
         "workloads": ["Unit_HRR_..."],
         "captured": {"hipMalloc": 4, ...},
         "observed": {"hipMalloc": "REAL", ...},
         "replay_exit": 0}
    """
    if not results_dir.is_dir():
        raise SystemExit(f"error: results dir not found: {results_dir}")

    merged: Dict[str, Any] = {"tiers": {}, "captured": {}, "observed": {},
                              "by_tier": {}}
    # Matched by tier name rather than by taking every .json and excluding the
    # ones we know about: this directory also holds the matrix report and the
    # test-coverage sweep's output, and a new neighbour should not have to be
    # added to an exclusion list to avoid being read as an observation.
    files = [f for f in sorted(results_dir.glob("T*.json"))
             if f.stem in mc.TIER_ORDER]
    if not files:
        raise SystemExit(
            f"error: no observation files in {results_dir}.\n"
            "The Catch2 tests write these when HRR_MATRIX_RESULTS_DIR is set; "
            "run them through run-api-matrix.sh.")

    for f in files:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"error: {f} is not valid JSON: {exc}")
        tier = data.get("tier")
        if not tier:
            raise SystemExit(f"error: {f} has no 'tier' field")
        merged["tiers"][tier] = {
            "workloads": data.get("workloads", []),
            "replay_exit": data.get("replay_exit"),
            "source_file": f.name,
        }
        for api, count in (data.get("captured") or {}).items():
            merged["captured"][api] = merged["captured"].get(api, 0) + int(count)
        for api, klass in (data.get("observed") or {}).items():
            merged["by_tier"].setdefault(api, {})[tier] = klass
            # If two tiers both reached an API they must agree; a disagreement
            # is itself worth surfacing rather than silently taking the last.
            prev = merged["observed"].get(api)
            if prev is not None and prev != klass:
                merged["observed"][api] = CLASS_CONFLICT
            else:
                merged["observed"][api] = klass
    return merged


def verdict_for(row: mc.ApiRow, observations: Dict[str, Any]) -> Dict[str, Any]:
    captured = observations["captured"].get(row.api, 0)
    observed = observations["observed"].get(row.api)

    result: Dict[str, Any] = {
        "api": row.api,
        "tier": row.tier,
        "expect": row.expect_class,
        "observed": observed,
        "captured": captured,
        "payload_loss": row.payload_loss,
        "note": row.note,
    }

    if row.skip:
        result["verdict"] = VERDICT_SKIPPED
        result["detail"] = row.skip_reason or "skipped by the overlay"
        return result

    # Declared as never reaching the archive under its own name. The assertion
    # is the absence: if it starts being captured, HIP's lowering changed and
    # the matrix should say so rather than quietly gaining a row.
    if not row.expect_captured:
        if captured:
            result["verdict"] = VERDICT_XPASS
            result["detail"] = (f"declared as never captured, but the archive "
                                f"has {captured} event(s) for it")
        else:
            result["verdict"] = VERDICT_NOT_CAPTURED
            result["detail"] = row.note or "not recorded under its own name"
        return result

    if observed is None or not captured:
        result["verdict"] = VERDICT_NOT_EXERCISED
        # Being captured but unobserved is a different situation from never
        # being called: the workload did reach it, and the replay then stopped
        # short. Saying so points at the aborting API rather than sending the
        # reader off to add a call site that already exists.
        result["detail"] = (
            "captured, but the replay stopped before reaching it"
            if observed is None and captured
            else "no workload in this tier called it")
        # A gap the overlay has measured and explained is expected; one it has
        # not is the matrix quietly losing an API, so build_report counts it as
        # a coverage problem and the run fails.
        if row.unreachable_reason:
            result["expected_gap"] = True
            result["unreachable_group"] = row.unreachable_group
            result["detail"] = row.unreachable_reason
        return result

    # Declared unreachable, and a tier reached it anyway. Whatever blocked it —
    # a device without image support, a missing SDK declaration, a crash
    # earlier in the archive — no longer does, so the declaration is stale and
    # the API now belongs in a tier's assertions.
    if row.unreachable_reason:
        result["verdict"] = VERDICT_XPASS
        result["unreachable_group"] = row.unreachable_group
        result["detail"] = (
            f"declared unreachable ({row.unreachable_group}), but replay "
            f"observed it as {observed} — drop the declaration and let the "
            "tier assert on it")
        return result

    # A real handler can succeed on one archive's arguments and fail on
    # another's — a pointer it cannot translate, a handle that meant something
    # only in the capturing process. That is not the API being reclassified, so
    # the overlay names the tiers where it is known to happen and the report
    # keeps it as a known failure rather than a contradiction. This applies
    # whether one tier saw it or several disagreed.
    by_tier = observations["by_tier"].get(row.api, {})
    unexpected = {
        tier: klass for tier, klass in by_tier.items()
        if klass != row.expect_class
        and not (klass == mc.CLASS_HANDLER_ERROR and tier in row.handler_error_in)
    }
    # Only when a tier actually saw the error. A run that did not include the
    # failing tier's workload must report the plain PASS it measured rather
    # than an XFAIL for something it never observed.
    declared_error = not unexpected and any(
        klass == mc.CLASS_HANDLER_ERROR and tier in row.handler_error_in
        for tier, klass in by_tier.items())

    if observed == CLASS_CONFLICT or declared_error:
        if declared_error:
            where = ", ".join(sorted(row.handler_error_in))
            result["verdict"] = VERDICT_XFAIL
            result["observed"] = row.expect_class
            result["detail"] = (
                f"real handler, but it still returns an error for the "
                f"arguments {where} records")
            return result
        result["verdict"] = VERDICT_FAIL
        result["detail"] = ("tiers disagree: " + ", ".join(
            f"{t}={k}" for t, k in sorted(by_tier.items())))
        return result

    if observed != row.expect_class:
        result["verdict"] = VERDICT_FAIL
        result["detail"] = (f"expected {row.expect_class} at replay, "
                            f"observed {observed}")
        return result

    # A declared handler error or crash is a known-broken API that is still
    # broken, which is what XFAIL is for. It flips to XPASS the day it replays.
    if row.expect_class in BROKEN_CLASSES:
        result["verdict"] = VERDICT_XFAIL
        result["detail"] = (
            "handler still takes the replay process down with a signal"
            if row.expect_class == mc.CLASS_CRASH
            else "handler still returns a HIP error at replay")
        return result

    # Declared unreplayable and observed refusing by name. The API is excluded
    # from replay on purpose and says so at both capture and replay, which is
    # the contract being asserted — not a gap the matrix is tolerating.
    #
    # Checked ahead of payload_loss because an unreplayable API is usually also
    # one whose arguments do not survive the process boundary — that is why it
    # is unreplayable. The loss is what XFAIL exists to flag only while it is
    # silent; here both ends say so out loud, so there is nothing left hidden.
    if row.expect_class == mc.CLASS_UNREPLAYABLE:
        result["verdict"] = VERDICT_PASS
        result["detail"] = ("replay refuses it by name and reason: a declared "
                            "scope exclusion")
        return result

    # Class matches. For a payload-loss API that is not the end of the story:
    # the handler runs, which is exactly why the loss is invisible.
    if row.payload_loss:
        result["verdict"] = VERDICT_XFAIL
        result["detail"] = ("real handler runs but the recorded arguments are "
                            "incomplete (section 8.3)")
        return result

    result["verdict"] = VERDICT_PASS
    result["detail"] = ""
    return result


def build_report(rows: List[mc.ApiRow], overlay: Dict[str, Any],
                 observations: Dict[str, Any]) -> Dict[str, Any]:
    ran_tiers = set(observations["tiers"])
    results = []
    for row in rows:
        if row.tier not in ran_tiers:
            results.append({
                "api": row.api, "tier": row.tier, "expect": row.expect_class,
                "observed": None, "captured": 0,
                "payload_loss": row.payload_loss, "note": row.note,
                "verdict": VERDICT_SKIPPED,
                "detail": f"tier {row.tier} was not run",
            })
            continue
        results.append(verdict_for(row, observations))

    per_tier: Dict[str, Dict[str, Any]] = {}
    for r in results:
        bucket = per_tier.setdefault(r["tier"], {v: 0 for v in VERDICT_ORDER})
        bucket[r["verdict"]] += 1

    # Coverage is counted per tier from that tier's own workloads, which is the
    # same thing the C++ roundtrip asserts and the reason both read one
    # min_covered. An API its owning tier misses but another tier happens to
    # reach still gets a verdict — the class is the class, whoever observed it
    # — but it does not count towards the owner's floor, because the floor is
    # there to catch that tier's workloads quietly reaching less than before.
    floors = {}
    problems = []
    for tier, bucket in per_tier.items():
        meta = (overlay.get("tiers") or {}).get(tier, {}) or {}
        floor = int(meta.get("min_covered", 0))
        covered = sum(
            1 for r in results
            if r["tier"] == tier
            # An API the overlay skips is asserted on by neither side, so it
            # cannot count towards a floor. Excluding it here is what keeps
            # this number identical to the one the C++ roundtrip prints.
            and r["verdict"] != VERDICT_SKIPPED
            and (tier in observations["by_tier"].get(r["api"], {})
                 # An API declared as never captured is asserted on by the
                 # absence of its row, which the tier's own archive shows.
                 or r["verdict"] == VERDICT_NOT_CAPTURED))
        elsewhere = sum(bucket[v] for v in VERDICT_COVERED) + \
            bucket[VERDICT_XPASS] + bucket[VERDICT_FAIL] - covered
        floors[tier] = {"covered": covered, "min_covered": floor,
                        "covered_by_another_tier": elsewhere,
                        "ran": tier in ran_tiers}
        if tier in ran_tiers and covered < floor:
            problems.append(
                f"{tier}: its own workloads exercised {covered} APIs, "
                f"floor is {floor}")

    # An API in a tier that ran, that nothing reached, and that the overlay
    # does not explain. Left unreported this is how a matrix rots: a workload
    # stops early, the APIs behind it quietly stop being asserted on, and the
    # run still says PASS.
    undeclared = [r for r in results
                  if r["verdict"] == VERDICT_NOT_EXERCISED
                  and not r.get("expected_gap")]
    for r in sorted(undeclared, key=lambda x: (x["tier"], x["api"])):
        problems.append(
            f"{r['tier']}: {r['api']} was not exercised and has no "
            f"unreachable: declaration ({r['detail']})")

    failures = [r for r in results if r["verdict"] == VERDICT_FAIL]

    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tiers_run": sorted(ran_tiers),
        "tier_meta": observations["tiers"],
        "per_tier": {t: per_tier[t] for t in mc.TIER_ORDER if t in per_tier},
        "coverage": floors,
        "failures": failures,
        "coverage_problems": problems,
        "results": results,
        "ok": not failures and not problems,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_markdown(report: Dict[str, Any], overlay: Dict[str, Any]) -> str:
    out: List[str] = []
    out.append("# HRR per-API playback matrix")
    out.append("")
    out.append(f"Generated {report['generated']}. "
               f"Tiers run: {', '.join(report['tiers_run']) or 'none'}.")
    out.append("")
    out.append("| Tier | PASS | XFAIL | XPASS | FAIL | not captured "
               "| not exercised | skipped | covered / floor |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for tier, bucket in report["per_tier"].items():
        cov = report["coverage"].get(tier, {})
        out.append(
            f"| {tier} | {bucket[VERDICT_PASS]} | {bucket[VERDICT_XFAIL]} | "
            f"{bucket[VERDICT_XPASS]} | {bucket[VERDICT_FAIL]} | "
            f"{bucket[VERDICT_NOT_CAPTURED]} | "
            f"{bucket[VERDICT_NOT_EXERCISED]} | {bucket[VERDICT_SKIPPED]} | "
            f"{cov.get('covered', 0)} / {cov.get('min_covered', 0)} |")
    out.append("")

    if report["failures"]:
        out.append("## Failures")
        out.append("")
        out.append("An API behaved differently at replay from what "
                   "`api_matrix.yaml` declares. Either the generator was "
                   "changed without updating the matrix, or replay regressed.")
        out.append("")
        out.append("| API | Tier | Expected | Observed | Detail |")
        out.append("|---|---|---|---|---|")
        for r in report["failures"]:
            out.append(f"| `{r['api']}` | {r['tier']} | {r['expect']} | "
                       f"{r['observed']} | {r['detail']} |")
        out.append("")

    xfails = [r for r in report["results"] if r["verdict"] == VERDICT_XFAIL]
    if xfails:
        out.append("## Known payload loss (XFAIL)")
        out.append("")
        out.append("These have a real playback handler and are counted in the "
                   "\"faithfully replayed\" figure, but the arguments they "
                   "need never reached the archive. They flip to XPASS when "
                   "P2 lands, which is the signal to update the overlay.")
        out.append("")
        out.append("| API | Tier | Why |")
        out.append("|---|---|---|")
        for r in sorted(xfails, key=lambda x: (x["tier"], x["api"])):
            why = " ".join((r["note"] or r["detail"]).split())
            out.append(f"| `{r['api']}` | {r['tier']} | {why} |")
        out.append("")

    not_run = [r for r in report["results"]
               if r["verdict"] == VERDICT_NOT_EXERCISED]
    declared = [r for r in not_run if r.get("expected_gap")]
    undeclared = [r for r in not_run if not r.get("expected_gap")]

    if undeclared:
        out.append("## Not exercised, and not explained")
        out.append("")
        out.append(f"{len(undeclared)} APIs in a tier that ran had no call "
                   "site reach them and no `unreachable:` group claiming "
                   "them. Either add the call site or, if it cannot be made "
                   "to write an event, add it to a group with the measurement "
                   "that says so. Until then the matrix is not asserting "
                   "anything about them.")
        out.append("")
        by_tier: Dict[str, List[str]] = {}
        for r in undeclared:
            by_tier.setdefault(r["tier"], []).append(r["api"])
        for tier in mc.TIER_ORDER:
            if tier not in by_tier:
                continue
            names = ", ".join(f"`{a}`" for a in sorted(by_tier[tier]))
            out.append(f"- **{tier}** ({len(by_tier[tier])}): {names}")
        out.append("")

    if declared:
        out.append("## Unreachable on this stack")
        out.append("")
        out.append(f"{len(declared)} APIs cannot be made to write an event "
                   "here, each for a reason that was measured rather than "
                   "assumed. The assertion is that they stay unreachable: any "
                   "one of them showing up in an archive is reported as XPASS "
                   "and the group's reason no longer holds.")
        out.append("")
        by_group: Dict[str, List[Dict[str, Any]]] = {}
        for r in declared:
            by_group.setdefault(r.get("unreachable_group", "?"), []).append(r)
        for group in sorted(by_group):
            rows_in = by_group[group]
            out.append(f"### {group} ({len(rows_in)})")
            out.append("")
            out.append(rows_in[0]["detail"])
            out.append("")
            names = ", ".join(f"`{r['api']}`"
                              for r in sorted(rows_in, key=lambda x: x["api"]))
            out.append(names)
            out.append("")

    out.append("## Tier definitions")
    out.append("")
    for tier in mc.TIER_ORDER:
        meta = (overlay.get("tiers") or {}).get(tier)
        if not meta:
            continue
        out.append(f"### {tier} — {meta.get('title', '')}")
        out.append("")
        out.append(" ".join((meta.get("rationale") or "").split()))
        out.append("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------

def print_summary(rows: List[mc.ApiRow], overlay: Dict[str, Any]) -> None:
    summary = mc.tier_summary(rows)
    print(f"{'tier':5s} {'apis':>5s} {'REAL':>5s} {'NOOP':>5s} {'STUB':>5s} "
          f"{'unrep':>6s} {'herr':>6s} {'crash':>6s} {'ploss':>6s} "
          f"{'skip':>5s} {'gpus':>5s}  title")
    for tier, bucket in summary.items():
        meta = (overlay.get("tiers") or {}).get(tier, {}) or {}
        print(f"{tier:5s} {bucket['apis']:5d} {bucket[mc.CLASS_REAL]:5d} "
              f"{bucket[mc.CLASS_NOOP]:5d} {bucket[mc.CLASS_ERROR_STUB]:5d} "
              f"{bucket[mc.CLASS_UNREPLAYABLE]:6d} "
              f"{bucket[mc.CLASS_HANDLER_ERROR]:6d} "
              f"{bucket[mc.CLASS_CRASH]:6d} "
              f"{bucket['payload_loss']:6d} {bucket['skipped']:5d} "
              f"{meta.get('gpus', 1):5d}  {meta.get('title', '')}")
    print(f"{'':5s} {sum(b['apis'] for b in summary.values()):5d}  total")


def check_workloads(overlay: Dict[str, Any], test_dir: Path) -> int:
    """Cross-check the tier workload lists against the Catch2 sources.

    The coverage floors catch a tier that has lost call sites, but only after a
    GPU run and only as a number. This reads the two sides against each other
    on the host: a tier naming a workload nobody defines is a typo or a rename,
    and a matrix workload no tier names is coverage that was written and then
    never counted.
    """
    if not test_dir.is_dir():
        print(f"error: test sources not found: {test_dir}", file=sys.stderr)
        return 2

    defined: Dict[str, str] = {}
    for src in sorted(test_dir.glob("*.cc")):
        for match in _DIRECT_CASE_RE.finditer(src.read_text()):
            defined[match.group(1)] = src.name

    referenced: Dict[str, List[str]] = {}
    for tier in mc.TIER_ORDER:
        for workload in mc.tier_workloads(overlay, tier):
            referenced.setdefault(workload, []).append(tier)

    problems: List[str] = []
    for name in sorted(set(referenced) - set(defined)):
        problems.append(f"  {name}: named by {', '.join(referenced[name])} "
                        f"but no TEST_CASE in {test_dir.name} defines it")
    for name, src in sorted(defined.items()):
        if src in MATRIX_WORKLOAD_SOURCES and name not in referenced:
            problems.append(f"  {name}: defined in {src} but no tier names it, "
                            "so nothing it calls is counted")

    if problems:
        print("ERROR: api_matrix.yaml and the Catch2 workloads disagree.\n"
              "Add the workload to a tier, or correct the name it is listed\n"
              "under, then regenerate the header.\n", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1

    matrix_defined = sum(1 for s in defined.values()
                         if s in MATRIX_WORKLOAD_SOURCES)
    print(f"workloads OK: {len(referenced)} named by tiers, "
          f"{matrix_defined} defined in the matrix workload sources, "
          f"{len(defined)} direct workloads in total")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=mc.DEFAULT_MANIFEST)
    ap.add_argument("--overlay", type=Path, default=mc.DEFAULT_OVERLAY)
    ap.add_argument("--emit-cxx", type=Path, nargs="?", const=DEFAULT_HEADER,
                    default=None,
                    help="write the C++ expectations header to this path "
                         f"(bare flag writes {DEFAULT_HEADER.name} in "
                         "tests/integration)")
    ap.add_argument("--summary", action="store_true",
                    help="print the tier table and exit")
    ap.add_argument("--check-workloads", action="store_true",
                    help="cross-check the tier workload lists against the "
                         "Catch2 sources and exit; needs no GPU and no "
                         "derived manifest")
    ap.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR,
                    help=f"Catch2 sources for --check-workloads "
                         f"(default: {DEFAULT_TEST_DIR})")
    ap.add_argument("--results", type=Path, default=None,
                    help="directory of per-tier observation JSON from a run")
    ap.add_argument("--report-dir", type=Path, default=None,
                    help="where to write matrix_report.{json,md} "
                         "(default: alongside --results)")
    args = ap.parse_args()

    # Reads only the overlay and the test sources, so it still runs in a fresh
    # clone where api_classes.json has not been derived yet.
    if args.check_workloads:
        try:
            return check_workloads(mc.load_overlay(args.overlay),
                                   args.test_dir)
        except mc.MatrixError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    try:
        rows = mc.load_rows(args.manifest, args.overlay)
    except mc.MatrixError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    overlay = mc.load_overlay(args.overlay)

    did_something = False

    if args.emit_cxx:
        emit_cxx(rows, overlay, args.emit_cxx)
        print(f"wrote {args.emit_cxx} ({len(rows)} APIs)")
        did_something = True

    if args.summary:
        print_summary(rows, overlay)
        did_something = True

    if args.results is None:
        if not did_something:
            ap.print_help()
            return 2
        return 0

    observations = load_observations(args.results)
    report = build_report(rows, overlay, observations)

    report_dir = args.report_dir or args.results
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "matrix_report.json"
    md_path = report_dir / "matrix_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(render_markdown(report, overlay))

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print()
    for tier, bucket in report["per_tier"].items():
        cov = report["coverage"].get(tier, {})
        print(f"  {tier}  pass={bucket[VERDICT_PASS]:4d} "
              f"xfail={bucket[VERDICT_XFAIL]:3d} "
              f"fail={bucket[VERDICT_FAIL]:3d} "
              f"not-exercised={bucket[VERDICT_NOT_EXERCISED]:4d} "
              f"covered={cov.get('covered', 0)}/{cov.get('min_covered', 0)}")

    if report["failures"]:
        print(f"\n{len(report['failures'])} API(s) diverged from the matrix:",
              file=sys.stderr)
        for r in report["failures"]:
            print(f"  {r['api']:52s} expected {r['expect']:11s} "
                  f"observed {r['observed']}", file=sys.stderr)
    for problem in report["coverage_problems"]:
        print(f"coverage: {problem}", file=sys.stderr)

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
