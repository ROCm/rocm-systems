#!/usr/bin/env python3
"""Score a rocjitsu timing model against a rocm-meter run recorded on hardware.

Two quantities are extracted per case and they are not interchangeable.

``kernel_us`` is the sum of ``self_device_time_us`` over the real device
kernels in the torch profiler block, divided by the profiler's iteration
count.  It is the device work the case's callable launched, and it is the
quantity a timing model predicts.  Summing is deliberate: 199 of the 304
timed cases in the reference run launch two or more kernels, and reporting
only the largest understates them by a median factor of two.  The filter is
deliberate too: torch attributes the same device time to the ``aten::``
operator and to the kernel it launched, so an unfiltered sum double counts,
and ``torch.compile`` adds ``## Call CompiledFxGraph`` and
``Torch-Compiled Region`` pseudo-operators that carry a large device time of
their own and are not kernels.

``event_us`` is ``device_timing.median_us``, a ``torch.cuda.Event`` bracket.
On hardware it also contains the host's enqueue gap -- the meter records its
event pairs back to back with no host synchronisation, so when the device
drains faster than the host enqueues, the idle stretch lands inside the
measured interval.  In the reference run that is a ~9 microsecond constant
independent of the work.  Under emulation the modelled clock does not
advance while the device is idle, so the emulated bracket contains the
modelled kernels and nothing else.

The two facts above decide the scoring rule:

    predicted = emulated event_us       (the model's opinion)
    measured  = real kernel_us          (the device work that actually happened)

Comparing the emulated bracket against the *real* bracket instead would
score the model on the host's python enqueue latency, which is a property
of the measurement harness and not of the modelled part.  Both views are
reported; only the first is the model's accuracy.

Scoring is on the ratio, per case, never on a mean of signed errors.  A mean
over both directions cancels, so a model that is 2x fast on half the corpus
and 2x slow on the other half reads as perfect.  The headline figures are
the median absolute log-ratio and the fraction of cases inside a stated
tolerance band.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from typing import Any, Iterable

# Profiler rows that are not device kernels.  ``aten::`` and ``hip`` are the
# operator and API wrappers; ``[`` is ``[memory]``; the remaining four are
# torch.compile's pseudo-operators, which carry a device time equal to the
# whole compiled region and would otherwise be counted on top of the kernels
# they contain.
WRAPPER_PREFIXES = (
    "aten::",
    "hip",
    "[",
    "## Call",
    "Torch-Compiled",
    "TorchDynamo",
    "AOTDispatcher",
    "Pregraph",
    "cudaLaunch",
    "Activity Buffer",
)


def is_device_kernel(name: str) -> bool:
    """Return whether a profiler operator row names a real device kernel."""
    return not name.startswith(WRAPPER_PREFIXES)


def device_kernel_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The profiler rows that name a real device kernel, deduplicated.

    ``key_averages(group_by_input_shape=True)`` keys a row on the operator name
    *and* its recorded input shapes, and a Triton kernel launched through a
    compiled graph is reported twice: once with shapes attached and once with an
    empty shape list, carrying the same call count and the same device time.
    Summing both double counts the kernel.  It is not a small effect -- 114 of
    the 308 cases in the reference run are affected, all of them compiled ones,
    and every one of them reads exactly twice its real device time.

    Two rows are treated as one launch when the name, the call count and the
    device time all match.  Rows that differ in any of those are kept: a case
    that genuinely launches the same kernel twice with different durations still
    counts twice, and a case that launches two differently named kernels is
    untouched.  The cross-check is the compiled-region wrapper, whose own device
    time is the total for the region: before deduplication the kernel sum
    exceeds it, which is impossible, and after deduplication it matches.
    """
    profiler = record.get("profiler")
    if not profiler:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, float]] = set()
    for op in profiler.get("operators", ()):
        name = op.get("name", "")
        if not is_device_kernel(name):
            continue
        device_us = op.get("self_device_time_us") or 0.0
        if device_us <= 0.0:
            continue
        key = (name, op.get("calls") or 0, device_us)
        if key in seen:
            continue
        seen.add(key)
        rows.append(op)
    return rows


def kernel_us(record: dict[str, Any]) -> float | None:
    """Device work per call, summed over the real kernels of one case."""
    profiler = record.get("profiler")
    if not profiler:
        return None
    iterations = profiler.get("iterations") or 0
    if iterations <= 0:
        return None
    rows = device_kernel_rows(record)
    if not rows:
        return None
    return sum(row["self_device_time_us"] for row in rows) / iterations


def kernel_launches(record: dict[str, Any]) -> int:
    """Number of distinct device kernels the case launched per call."""
    profiler = record.get("profiler")
    if not profiler:
        return 0
    iterations = max(1, profiler.get("iterations") or 1)
    launches = 0
    for op in device_kernel_rows(record):
        launches += max(1, round((op.get("calls") or 1) / iterations))
    return launches


def event_us(record: dict[str, Any]) -> float | None:
    """The ``torch.cuda.Event`` bracket median, in microseconds."""
    timing = record.get("device_timing")
    if not timing:
        return None
    value = timing.get("median_us")
    return value if isinstance(value, (int, float)) and value > 0 else None


def passed_results(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a report's passing results by case id."""
    return {
        r["case_id"]: r
        for r in report.get("results", ())
        if r.get("status") == "passed"
    }


def load(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def merge(paths: Iterable[str]) -> dict[str, Any]:
    """Combine shard reports into one.  Later shards win on a duplicate id."""
    merged: dict[str, Any] | None = None
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        report = load(path)
        if merged is None:
            merged = report
        for result in report.get("results", ()):
            by_id[result["case_id"]] = result
    if merged is None:
        raise SystemExit("no reports given")
    merged["results"] = [by_id[k] for k in sorted(by_id)]
    return merged


class Comparison:
    """One case scored in both views."""

    __slots__ = (
        "case_id",
        "category",
        "measured",
        "predicted",
        "ratio",
        "launches",
        "real_event",
        "emul_event",
    )

    def __init__(
        self, case_id, category, measured, predicted, launches, real_event, emul_event
    ):
        self.case_id = case_id
        self.category = category
        self.measured = measured
        self.predicted = predicted
        self.ratio = predicted / measured
        self.launches = launches
        self.real_event = real_event
        self.emul_event = emul_event

    @property
    def log2_ratio(self) -> float:
        return math.log2(self.ratio)

    @property
    def abs_pct(self) -> float:
        """Absolute error as a percentage of the measured value."""
        return abs(self.predicted - self.measured) / self.measured * 100.0


def compare(
    real: dict[str, Any], emulated: dict[str, Any]
) -> tuple[list[Comparison], list[str], list[str]]:
    """Score every case both reports agree on and passed in both."""
    real_by_id = passed_results(real)
    emul_by_id = passed_results(emulated)
    scored: list[Comparison] = []
    unscorable: list[str] = []
    for case_id in sorted(set(real_by_id) & set(emul_by_id)):
        measured = kernel_us(real_by_id[case_id])
        predicted = event_us(emul_by_id[case_id])
        if measured is None or predicted is None:
            unscorable.append(case_id)
            continue
        scored.append(
            Comparison(
                case_id,
                real_by_id[case_id].get("category", "?"),
                measured,
                predicted,
                kernel_launches(real_by_id[case_id]),
                event_us(real_by_id[case_id]),
                event_us(emul_by_id[case_id]),
            )
        )
    missing = sorted(set(real_by_id) - set(emul_by_id))
    return scored, unscorable, missing


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(scored: list[Comparison], tolerance_pct: float) -> dict[str, Any]:
    ratios = [c.ratio for c in scored]
    logs = [abs(c.log2_ratio) for c in scored]
    pcts = [c.abs_pct for c in scored]
    within = [c for c in scored if c.abs_pct <= tolerance_pct]
    return {
        "cases": len(scored),
        "median_ratio": statistics.median(ratios) if ratios else float("nan"),
        "median_abs_pct": statistics.median(pcts) if pcts else float("nan"),
        "p90_abs_pct": quantile(pcts, 0.90),
        "max_abs_pct": max(pcts) if pcts else float("nan"),
        "median_abs_log2": statistics.median(logs) if logs else float("nan"),
        "p90_abs_log2": quantile(logs, 0.90),
        "within_tolerance": len(within),
        "within_2x": sum(1 for c in scored if 0.5 <= c.ratio <= 2.0),
        "min_ratio": min(ratios) if ratios else float("nan"),
        "max_ratio": max(ratios) if ratios else float("nan"),
    }


def report_text(scored, unscorable, missing, tolerance_pct, worst, show_cases) -> str:
    out: list[str] = []
    overall = summarize(scored, tolerance_pct)
    out.append("timing model accuracy: emulated event bracket vs real kernel time")
    out.append("")
    out.append(f"cases scored          {overall['cases']}")
    if overall["cases"] == 0:
        out.append("nothing to score")
        return "\n".join(out)
    out.append(f"median ratio          {overall['median_ratio']:.3f}x")
    out.append(f"median |error|        {overall['median_abs_pct']:.1f}%")
    out.append(f"p90 |error|           {overall['p90_abs_pct']:.1f}%")
    out.append(f"max |error|           {overall['max_abs_pct']:.1f}%")
    out.append(f"median |log2 ratio|   {overall['median_abs_log2']:.3f}")
    out.append(f"p90 |log2 ratio|      {overall['p90_abs_log2']:.3f}")
    out.append(
        f"within {tolerance_pct:.0f}%           "
        f"{overall['within_tolerance']}/{overall['cases']} "
        f"({100.0 * overall['within_tolerance'] / overall['cases']:.1f}%)"
    )
    out.append(
        f"within 2x             {overall['within_2x']}/{overall['cases']} "
        f"({100.0 * overall['within_2x'] / overall['cases']:.1f}%)"
    )
    out.append(
        f"ratio range           {overall['min_ratio']:.3f}x .. "
        f"{overall['max_ratio']:.3f}x"
    )
    out.append("")

    by_category: dict[str, list[Comparison]] = {}
    for case in scored:
        by_category.setdefault(case.category, []).append(case)
    out.append(
        f"{'category':<20} {'n':>4} {'med ratio':>10} {'med |err|':>10} "
        f"{'p90 |err|':>10} {'within':>10}"
    )
    for category in sorted(by_category):
        cases = by_category[category]
        stats = summarize(cases, tolerance_pct)
        out.append(
            f"{category:<20} {stats['cases']:>4} "
            f"{stats['median_ratio']:>9.3f}x {stats['median_abs_pct']:>9.1f}% "
            f"{stats['p90_abs_pct']:>9.1f}% "
            f"{stats['within_tolerance']:>4}/{stats['cases']:<5}"
        )
    out.append("")

    if worst:
        out.append(f"worst {worst} cases by |error|")
        out.append(
            f"{'case_id':<58} {'real us':>9} {'model us':>9} "
            f"{'ratio':>8} {'launch':>7}"
        )
        for case in sorted(scored, key=lambda c: -c.abs_pct)[:worst]:
            out.append(
                f"{case.case_id[:58]:<58} {case.measured:>9.2f} "
                f"{case.predicted:>9.2f} {case.ratio:>7.2f}x "
                f"{case.launches:>7}"
            )
        out.append("")

    if show_cases:
        out.append(f"{'case_id':<58} {'real us':>9} {'model us':>9} {'ratio':>8}")
        for case in sorted(scored, key=lambda c: c.case_id):
            out.append(
                f"{case.case_id[:58]:<58} {case.measured:>9.2f} "
                f"{case.predicted:>9.2f} {case.ratio:>7.2f}x"
            )
        out.append("")

    if unscorable:
        out.append(f"unscorable (no kernel time on one side): {len(unscorable)}")
        for case_id in unscorable[:8]:
            out.append(f"  {case_id}")
        if len(unscorable) > 8:
            out.append(f"  ... and {len(unscorable) - 8} more")
        out.append("")
    if missing:
        out.append(f"in the reference run but not the emulated one: {len(missing)}")
        for case_id in missing[:8]:
            out.append(f"  {case_id}")
        if len(missing) > 8:
            out.append(f"  ... and {len(missing) - 8} more")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--real", required=True, help="rocm-meter report recorded on hardware"
    )
    parser.add_argument(
        "--emulated",
        required=True,
        nargs="+",
        help="rocm-meter report(s) recorded under the emulator",
    )
    parser.add_argument("--tolerance-pct", type=float, default=20.0)
    parser.add_argument(
        "--worst", type=int, default=15, help="how many worst cases to list; 0 for none"
    )
    parser.add_argument("--cases", action="store_true", help="list every scored case")
    parser.add_argument(
        "--json", dest="json_out", help="write the scored comparison as JSON"
    )
    parser.add_argument(
        "--fail-over-pct",
        type=float,
        help="exit non-zero when the median absolute error " "exceeds this",
    )
    args = parser.parse_args(argv)

    real = load(args.real)
    emulated = merge(args.emulated)
    scored, unscorable, missing = compare(real, emulated)

    print(
        report_text(
            scored, unscorable, missing, args.tolerance_pct, args.worst, args.cases
        )
    )

    if args.json_out:
        payload = {
            "summary": summarize(scored, args.tolerance_pct),
            "tolerance_pct": args.tolerance_pct,
            "cases": [
                {
                    "case_id": c.case_id,
                    "category": c.category,
                    "real_kernel_us": c.measured,
                    "model_us": c.predicted,
                    "ratio": c.ratio,
                    "abs_pct": c.abs_pct,
                    "launches": c.launches,
                    "real_event_us": c.real_event,
                    "model_event_us": c.emul_event,
                }
                for c in sorted(scored, key=lambda c: c.case_id)
            ],
            "unscorable": unscorable,
            "missing": missing,
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    if args.fail_over_pct is not None and scored:
        median = summarize(scored, args.tolerance_pct)["median_abs_pct"]
        if median > args.fail_over_pct:
            print(
                f"\nmedian absolute error {median:.1f}% exceeds "
                f"{args.fail_over_pct:.1f}%",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
