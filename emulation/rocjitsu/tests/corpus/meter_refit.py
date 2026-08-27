#!/usr/bin/env python3
"""Re-score a recorded corpus under a different timing tuning, offline.

A corpus pass under the emulator costs about ten minutes, which is far too slow
to search a parameter space with.  Nearly every tuning question, though, only
changes how a dispatch's *already recorded* terms are recombined: what the
compute units did, how many bytes reached each level, and how deep the longest
dependence chain ran do not move when a rate or a composition does.

So the emulator writes those terms per dispatch (ROCJITSU_TIMING_TRACE), the
meter writes the device-clock window of each case (ROCM_METER_DEVICE_WINDOW),
and this joins the two: every dispatch lands in the case whose window contains
it, and the case's modelled duration is recomputed from the terms.  A candidate
tuning is then a function from terms to cycles, evaluated in about a second
against the whole corpus.

The recomputation is exact for anything the composition does with the recorded
terms -- rates, additive structure, launch costs, per-unit issue weights -- and
is *not* valid for anything that changes what a compute unit records in the
first place, such as cache geometry or the memory hierarchy's routing.  Those
still need a real run, and --verify reports how far the untouched
recomputation drifts from the emulator's own answer so that a mistake here
cannot be read as an improvement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import meter_score  # noqa: E402

UNITS = (
    "none",
    "vector_alu",
    "scalar_alu",
    "transcendental",
    "matrix_multiply",
    "local_data_share",
    "vector_memory",
    "scalar_memory",
    "branch",
    "export",
)


def load_config(path: str) -> dict[str, float]:
    """The shipped tuning, flattened, as the baseline the knobs multiply.

    Read from the same file the emulator reads so that the offline model starts
    where the emulator is rather than at one-of-everything.  Getting this wrong
    is silent and total: the port counts alone are the difference between an
    issue term of twenty-one thousand cycles and one of fifty-eight thousand.
    """
    text = []
    for line in open(path):
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        text.append(line)
    document = json.loads("".join(text))
    flat: dict[str, float] = {}

    def walk(node: Any, prefix: str) -> None:
        for key, value in node.items():
            if key.startswith("//"):
                continue
            if isinstance(value, dict):
                walk(value, prefix + key + ".")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                flat[prefix + key] = float(value)

    walk(document.get("timing", {}), "")
    return flat


def config_defaults(config: dict[str, float]) -> dict[str, float]:
    """Turn the shipped tuning into the knob dictionary that reproduces it."""

    def lookup(suffix: str, fallback: float) -> float:
        # The file nests its blocks and also allows dotted keys inside them, so
        # a key is identified by its tail rather than by a fixed path.
        for key, value in config.items():
            if key == suffix or key.endswith("." + suffix):
                return value
        return fallback

    tuning: dict[str, float] = {}
    for unit in UNITS:
        tuning[f"ports.{unit}"] = lookup(f"{unit}.ports", 1.0)
    # The front end's queue is recorded in sixteenths of a cycle, so its
    # divisor carries the same factor.
    tuning["ports.none"] *= FRONT_END_SCALE
    tuning["stall_overlap"] = lookup("stall_overlap_wavefronts", 1.0)
    shared = lookup("front_end.issue_cycles", 1.0)
    for name in CLASSES:
        tuning[f"fe_base.{name}"] = (
            lookup(f"front_end.{name}.issue_cycles", shared) * FRONT_END_SCALE
        )
    tuning["straggler_base"] = lookup("straggler_cycles", 0.0)
    tuning["gamma_base"] = lookup("issue_occupancy_exponent", 0.0)
    tuning["barrier_base"] = lookup("barrier_cycles", 0.0)
    tuning["barrier_lockstep"] = lookup("barrier_lockstep", 0.0)
    tuning["issue_occupancy_exponent"] = tuning["gamma_base"]
    tuning["barrier_cycles"] = tuning["barrier_base"]
    return tuning


def load_trace(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A run killed by the shard timeout leaves a partial last line.
                continue
    # A torch run with a compiler in it forks, and every child gets its own
    # emulator plane, its own clock starting again at zero, and appends to this
    # same file.  Only the process that ran the benchmark has a timeline the
    # meter's events can be positioned on; the others are separate timelines
    # that happen to share a file.  The parent is the one whose first dispatch
    # is the file's first dispatch.
    if records and "pid" in records[0]:
        parent = records[0]["pid"]
        records = [record for record in records if record["pid"] == parent]
    records.sort(key=lambda record: record["device_cycles"])
    return records


def join(shard_json: str, trace_path: str, clock_mhz: float) -> list[dict[str, Any]]:
    """Attach each case's dispatches, by device-clock window.

    The meter's window is measured from an epoch event it records itself, and
    the emulator's trace counts from the first dispatch of the process, so the
    two axes are the same axis shifted by however much simulated time had
    already accumulated when the epoch was taken.  That shift is recovered
    rather than assumed: the first case's window has to start on some dispatch
    boundary, so every boundary is tried and the one that best reproduces the
    emulator's own per-case durations wins.  A join that is wrong shows up as a
    large residual, which is reported, rather than as a plausible wrong answer.
    """
    report = json.load(open(shard_json))
    trace = load_trace(trace_path)
    if not trace:
        return []
    windows = [
        (result, result["device_window"])
        for result in report.get("results", [])
        if result.get("status") == "passed" and result.get("device_window")
    ]
    if not windows:
        return []

    cycles = [record["device_cycles"] for record in trace]
    # Zero is where the meter's epoch should put it -- the epoch is recorded
    # before any work reaches the device -- and the search is kept as the check
    # on that, not as a substitute for it.
    best: tuple[float, int] | None = None
    for candidate in [0, *cycles]:
        residual = _residual(trace, cycles, windows, clock_mhz, candidate)
        if residual is not None and (best is None or residual < best[0]):
            best = (residual, candidate)
    if best is None:
        return []
    residual, origin = best

    cases: list[dict[str, Any]] = []
    for result, window in windows:
        lo = _bisect(cycles, origin + window["start_us"] * clock_mhz)
        hi = _bisect(cycles, origin + window["end_us"] * clock_mhz)
        if hi <= lo:
            continue
        cases.append(
            {
                "case_id": result["case_id"],
                "category": result.get("category", "?"),
                "samples": max(1, int(result["device_timing"]["count"])),
                "emulator_us": result["device_timing"]["median_us"],
                "dispatches": trace[lo:hi],
                "join_residual": residual,
            }
        )
    return cases


def _residual(trace, cycles, windows, clock_mhz, origin) -> float | None:
    """How far a candidate origin puts the join from the emulator's own answer."""
    total = 0.0
    counted = 0
    for result, window in windows:
        lo = _bisect(cycles, origin + window["start_us"] * clock_mhz)
        hi = _bisect(cycles, origin + window["end_us"] * clock_mhz)
        samples = max(1, int(result["device_timing"]["count"]))
        rebuilt = sum(record["cycles"] for record in trace[lo:hi]) / clock_mhz / samples
        expected = result["device_timing"]["median_us"]
        if expected <= 0.0:
            continue
        total += abs(rebuilt - expected) / expected
        counted += 1
    return total / counted if counted else None


def _bisect(ordered: list[float], value: float) -> int:
    low, high = 0, len(ordered)
    while low < high:
        mid = (low + high) // 2
        if ordered[mid] < value:
            low = mid + 1
        else:
            high = mid
    return low


def baseline_cycles(dispatch: dict[str, Any], _tuning: dict[str, float]) -> float:
    """What the emulator itself composed.  The identity, for --verify."""
    return float(dispatch["cycles"])


FRONT_END_SCALE = 16.0

CLASSES = (
    "unknown",
    "vector_alu",
    "scalar_alu",
    "transcendental",
    "matrix_multiply",
    "lds_read",
    "lds_write",
    "vector_memory_read",
    "vector_memory_write",
    "vector_memory_atomic",
    "scalar_memory",
    "tensor_memory",
    "export",
    "branch",
    "wait_counter",
    "delay_alu",
    "barrier",
    "message",
    "nop",
    "terminate",
)


def _front_end(dispatch, share: float, tuning: dict[str, float]) -> float | None:
    """Front-end occupancy from per-class weights, when any are set.

    The front end is the slot every instruction takes before it reaches any
    execution unit, and the model charges one number for all of them.  That
    number is the whole median-versus-maximum trade: instruction-dense matrix
    kernels need about two cycles an instruction to match measurement and
    memory-bound elementwise kernels about one, and no single value is both.
    Splitting it by class is the hypothesis that the difference is a property
    of the instructions rather than of the kernels.
    """
    if not any(f"fe.{name}" in tuning for name in CLASSES):
        return None
    # Relative to what the config already ships, so that a knob of one
    # reproduces the recorded run exactly.
    total = 0.0
    for name in CLASSES:
        count = dispatch["class_counts"].get(name, 0)
        if count:
            total += (
                count
                * tuning.get(f"fe_base.{name}", FRONT_END_SCALE)
                * tuning.get(f"fe.{name}", 1.0)
            )
    return total * share


def _issue(
    unit_cycles: dict[str, int],
    tuning: dict[str, float],
    front_end: float | None = None,
) -> float:
    """The busiest port's queue, under candidate per-unit weights."""
    busiest = 0.0
    for unit in UNITS:
        cycles = unit_cycles[unit]
        if unit == "none" and front_end is not None:
            cycles = front_end
        weight = tuning.get(f"unit.{unit}", 1.0)
        ports = tuning.get(f"ports.{unit}", 1.0)
        busiest = max(busiest, cycles * weight / ports)
    return busiest


def recompose(dispatch: dict[str, Any], tuning: dict[str, float]) -> float:
    """The shipped composition, with every constant exposed as a knob.

    Mirrors DispatchDes::end() statement for statement.  It runs on the two
    compute units the emulator's own maxima selected -- the busiest for the
    issue term, the most exposed for the latency term -- rather than on device
    sums, because a sum over compute units cannot reproduce a maximum over
    them.  The moment the two drift this stops predicting anything about the
    emulator, which is what --verify is for.
    """
    fixed = tuning.get("fixed_scale", 1.0) * dispatch["fixed"] + tuning.get(
        "fixed_add", 0.0
    )
    if dispatch["waves"] == 0:
        return fixed

    waves_total = max(1.0, float(dispatch["waves"]))
    issue = _issue(
        dispatch["unit_cycles"],
        tuning,
        _front_end(dispatch, float(dispatch["unit_waves"]) / waves_total, tuning),
    )

    # The latency term is composed on its own compute unit, whose issue work
    # is what hides its wavefronts' stalls.
    scaled = _issue(
        dispatch["latency_unit_cycles"],
        tuning,
        _front_end(dispatch, float(dispatch["latency_waves"]) / waves_total, tuning),
    )
    # A unit weight changes how long a wavefront's own chain of issue slots is,
    # not only how long the unit is busy, so the recorded chain is rescaled by
    # the same factor -- but only its issue half.  The stall half belongs to the
    # memory system and does not move when an execution unit's rate does.
    stall = float(dispatch["latency_worst_stall"]) * tuning.get("stall_scale", 1.0)
    chain = max(
        0.0,
        float(dispatch["latency_critical"]) - float(dispatch["latency_worst_stall"]),
    )
    # The chain itself is *not* rescaled by the unit weights. Those weights are
    # implemented as fractional port counts, and a port count divides a unit's
    # aggregate queue without changing how long any one wavefront waits for its
    # own instruction. Only the throughput term sees them.
    critical = chain + stall

    waves = max(1.0, float(dispatch["latency_waves"]))
    resident = max(1.0, min(waves, float(dispatch["resident"])))
    rounds = math.ceil(waves / resident)
    hidden = (resident - 1.0) * (scaled / waves)
    # Hiding assumes the other resident wavefronts always have independent work
    # ready the instant this one stalls. A barrier is exactly the instruction
    # that makes that false: it puts the whole group in lockstep, so nothing on
    # the far side of one can cover a stall on the near side. A kernel with
    # hundreds of barriers per wavefront -- any software-pipelined matrix
    # multiply staging tiles through the local data share -- has almost no
    # hiding available, and the model was crediting it with all of it.
    lockstep = tuning.get("barrier_lockstep", 0.0)
    if lockstep > 0.0:
        share = float(dispatch["latency_waves"]) / max(1.0, float(dispatch["waves"]))
        per_wave_barriers = (
            dispatch["class_counts"].get("barrier", 0)
            * share
            / max(1.0, float(dispatch["latency_waves"]))
        )
        hidden /= 1.0 + lockstep * per_wave_barriers
    exposed = max(0.0, critical - hidden)
    overlap = max(1.0, min(resident, tuning.get("stall_overlap", 1.0)))
    exposed = max(exposed, stall / overlap)
    latency = exposed * rounds * tuning.get("latency_scale", 1.0)

    bandwidth = 0.0
    for level in ("l1", "l2", "fabric", "mall", "dram"):
        rate = tuning.get("bandwidth_scale", 1.0) * tuning.get(f"level.{level}", 1.0)
        bandwidth = max(bandwidth, dispatch[f"{level}_cycles"] / rate)

    # One handle on everything the dispatch's *work* costs, as against what its
    # launch costs. The two are separately observable -- a kernel at the launch
    # floor reads over and a kernel with real work in it reads under -- and no
    # single existing knob moves them apart.
    work = tuning.get("work_scale", 1.0)
    issue *= work
    bandwidth *= work
    placement = float(dispatch["placement"]) * tuning.get("placement_scale", 1.0)
    # Per-instruction issue cost is not independent of how many wavefronts are
    # resident. The model charges a unit's queue linearly in the work on it,
    # and the same kernel measured at four wavefronts per compute unit and at
    # sixteen does not scale that way: the low-occupancy case reaches a higher
    # instruction rate, because fewer wavefronts contend for the same issue
    # slot. An exponent on the resident count is the smallest form of that.
    # Relative to the exponent the config already applies, because the recorded
    # issue term was produced with it. Adding it again would double it.
    gamma = tuning.get("issue_occupancy_exponent", 0.0) - tuning.get("gamma_base", 0.0)
    if gamma != 0.0:
        issue *= (max(1.0, float(dispatch["resident"])) / 8.0) ** gamma

    throughput = max(issue, bandwidth, placement)
    latency = max(latency, dispatch["fill"] * tuning.get("fill_scale", 1.0))

    # A dispatch ends when its slowest wavefront ends, not when the average one
    # does.  Every compute unit in this model runs the same instruction stream
    # and so finishes at the same modelled cycle, which makes the modelled
    # maximum equal to the modelled mean and drops the straggler entirely.  The
    # expected maximum of many noisy durations grows with the logarithm of how
    # many there are, which is what the measurements show: a launch-dominated
    # kernel costs about the same extra per doubling of its wavefront count,
    # from four wavefronts up to a thousand.
    # The straggler is the spread between *independent* scheduling units. The
    # emulator counts wavefronts; wavefronts inside one workgroup are not
    # independent -- they share a compute unit and a barrier -- so a
    # single-workgroup kernel should carry no straggler at all and the model
    # was charging it log2 of its wavefront count.
    if tuning.get("straggler_over_workgroups", 0.0) > 0.0:
        base = tuning.get("straggler_base", 0.0) * tuning.get("straggler_scale", 1.0)
        groups = max(1.0, float(dispatch["workgroups"]))
        latency += base * math.log2(groups) if groups > 1 else 0.0
    else:
        latency += dispatch.get("straggler", 0.0) * tuning.get("straggler_scale", 1.0)

    # A workgroup barrier costs the spread between its wavefronts, and this
    # model gives every wavefront the same instruction stream, so the modelled
    # spread is near zero and a barrier is nearly free. On hardware it is not:
    # the wavefronts arrive skewed by whatever their memory accesses did, and a
    # kernel that barriers thousands of times per wavefront pays for it. Charged
    # per barrier on the wavefront's own path.
    # Likewise relative: the wavefront chains in the trace were already charged
    # whatever the config says a barrier costs.
    per_barrier = tuning.get("barrier_cycles", 0.0) - tuning.get("barrier_base", 0.0)
    if per_barrier != 0.0:
        waves_here = max(1.0, float(dispatch["waves"]))
        share = float(dispatch["latency_waves"]) / waves_here
        barriers = dispatch["class_counts"].get("barrier", 0) * share
        per_wave = barriers / max(1.0, float(dispatch["latency_waves"]))
        latency += per_barrier * per_wave

    body = throughput + latency + dispatch["filling"] * tuning.get("filling_scale", 1.0)
    return fixed + body


def case_us(
    case: dict[str, Any], model: Callable, tuning: dict, clock_mhz: float
) -> float:
    total = sum(model(dispatch, tuning) for dispatch in case["dispatches"])
    return total / clock_mhz / case["samples"]


def score(cases, real, model, tuning, clock_mhz) -> dict[str, Any]:
    real_by_id = meter_score.passed_results(real)
    rows = []
    for case in cases:
        measured = meter_score.kernel_us(real_by_id.get(case["case_id"], {}))
        if not measured:
            continue
        predicted = case_us(case, model, tuning, clock_mhz)
        rows.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "measured": measured,
                "predicted": predicted,
                "ratio": predicted / measured,
                "emulator_us": case["emulator_us"],
            }
        )
    ratios = [row["ratio"] for row in rows]
    errors = [abs(row["ratio"] - 1.0) * 100.0 for row in rows]
    return {
        "rows": rows,
        "n": len(rows),
        "median_ratio": statistics.median(ratios) if ratios else float("nan"),
        "median_pct": statistics.median(errors) if errors else float("nan"),
        "max_pct": max(errors) if errors else float("nan"),
        "p90_pct": (
            sorted(errors)[int(0.9 * (len(errors) - 1))] if errors else float("nan")
        ),
        "within": sum(1 for error in errors if error <= 17.0),
        "spread": (max(ratios) / min(ratios)) if ratios else float("nan"),
        "lo": min(ratios) if ratios else float("nan"),
        "hi": max(ratios) if ratios else float("nan"),
    }


def load_cases(
    shard_dir: str, trace_dir: str, clock_mhz: float
) -> list[dict[str, Any]]:
    # Shard filters are allowed to overlap and the scorer drops the duplicates,
    # so this has to as well: a case counted twice would be weighted twice by
    # anything fitted against this set, and the two counts would not even be
    # equal, because the two shards ran it on differently warmed caches.
    seen: dict[str, dict[str, Any]] = {}
    for name in sorted(os.listdir(shard_dir)):
        if not name.endswith(".json"):
            continue
        trace = os.path.join(trace_dir, name[:-5] + ".jsonl")
        if not os.path.exists(trace):
            continue
        for case in join(os.path.join(shard_dir, name), trace, clock_mhz):
            seen.setdefault(case["case_id"], case)
    return list(seen.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shards", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument("--clock-mhz", type=float, default=2100.0)
    parser.add_argument("--config", default="configs/gfx950_mi355x_kmd.json")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="score the recorded totals and the recomposition side by side",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--cases", action="store_true")
    args = parser.parse_args(argv)

    tuning = config_defaults(load_config(args.config))
    for assignment in args.set:
        key, _, value = assignment.partition("=")
        tuning[key] = float(value)

    cases = load_cases(args.shards, args.traces, args.clock_mhz)
    real = meter_score.load(args.real)
    residuals = [case["join_residual"] for case in cases]
    print(
        f"joined {len(cases)} cases"
        + (
            f", worst shard join residual {max(residuals) * 100:.1f}%"
            if residuals
            else ""
        )
    )

    if args.verify:
        recorded = score(cases, real, baseline_cycles, tuning, args.clock_mhz)
        rebuilt = score(cases, real, recompose, tuning, args.clock_mhz)
        drift = [
            abs(a["predicted"] - b["predicted"]) / b["predicted"] * 100.0
            for a, b in zip(recorded["rows"], rebuilt["rows"])
        ]
        print(
            f"  recorded    median {recorded['median_pct']:.1f}%  max {recorded['max_pct']:.1f}%"
        )
        print(
            f"  recomposed  median {rebuilt['median_pct']:.1f}%  max {rebuilt['max_pct']:.1f}%"
        )
        print(
            f"  recomposition drift: max {max(drift):.3f}%  median {statistics.median(drift):.3f}%"
        )
        return 0

    result = score(cases, real, recompose, tuning, args.clock_mhz)
    print(f"  n {result['n']}  median ratio {result['median_ratio']:.3f}")
    print(
        f"  median |e| {result['median_pct']:.1f}%   p90 {result['p90_pct']:.1f}%"
        f"   max {result['max_pct']:.1f}%"
    )
    print(
        f"  within 17% {result['within']}/{result['n']}"
        f"   range {result['lo']:.3f}..{result['hi']:.3f}  spread {result['spread']:.2f}x"
    )
    if args.cases:
        for row in sorted(
            result["rows"], key=lambda row: -abs(math.log2(row["ratio"]))
        )[:25]:
            print(f"    {row['ratio']:6.3f}  {row['category']:18s} {row['case_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
