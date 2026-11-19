#!/usr/bin/env python3
"""Attribute SPM counter activity to source lines with a least-squares fit.

Idea
----
A hardware counter such as SQC_LDS_BANK_CONFLICT only says *how many* events
happened, not *which instruction* caused them. SPM (the streaming performance
monitor) samples counters many times during a kernel. If the run is split into
time windows, then for each window w:

    Y[w] = sum over sites s of  A[w][s] * X[s]

where
    Y[w]    = counter delta observed in window w (from SPM),
    A[w][s] = how many times site s executed in window w (from the trace),
    X[s]    = average counter events per execution of site s (the unknown).

There are many more windows than sites, so the system is overdetermined and X
can be recovered with ordinary least squares, as long as the sites do not
always run in the same ratio (otherwise the columns of A are collinear).

Any counter can be attributed this way, as long as instructions at the traced
sites cause its events (LDS conflicts, LDS instructions, cache accesses for
global loads, ...). Pass one or more --counter names; each gets its own fit
over the same A.

Where the data comes from
-------------------------
* A comes from the SQTT thread trace. The kernel is built with the
  sqtt-marker pass and SQTT_TRACE_ADDRESSES=lds (and/or memory), so every
  traced instruction emits a marker carrying its source line, a timestamp and
  its per-lane addresses.
* Y comes from SPM samples captured in the same rocprofv3 run (--spm with
  --output-format json). Each SPM sample carries a realtime-clock timestamp.
* Marker timestamps are in shader-clock ticks. The REALTIME records in the
  trace pair shader clock with realtime, so the two time bases can be lined up.

Trace scope vs counter scope
----------------------------
SPM counts the whole GPU, so the trace should too: build with markers on every
CU and trace every shader engine. Then A and Y cover the same hardware and X is
events per execution directly.

If only part of the GPU is traced (sqtt-marker limits markers to CU 0-1 by
default, and ATT traces shader engine 0 by default), then without correction X is
"GPU-wide events per *traced* execution". With --normalize-counter (a counter
that counts the traced instructions themselves, e.g. SQ_INSTS_LDS for LDS
sites), each window's trace counts are scaled up to GPU-wide counts, so X
becomes events per execution. This assumes the traced CUs see the same mix of
sites as the rest of the GPU in each window.

Without --spm-json the script runs a self-check: Y is computed from the traced
LDS addresses with a bank model, so the fit can be compared with exact values.

Usage
-----
    # Capture trace (all shader engines) and SPM together. gfx1201 needs a
    # rocprofiler-sdk whose aqlprofile enables SPM for gfx12 (develop does).
    # Build the kernel with SQTT_SCOPE_CU=-1 SQTT_SCOPE_SIMD=-1 so every CU
    # emits markers; see lds_conflict.hip.
    rocprofv3 --att --att-shader-engine-mask 0xF --kernel-trace --output-format json \
        --spm-beta-enabled=on --spm SQC_LDS_BANK_CONFLICT SQ_INSTS_LDS \
        --spm-sample-interval-unit sclk_cycles --spm-sample-interval 1024 \
        -d captures -o run -- ./lds_conflict 64 64 64

    PYTHONPATH=python:samples python3 samples/markers/counter_fit.py \
        captures/*.att captures/*_code_object_id_*.out \
        --spm-json captures/run_results.json \
        --counter SQC_LDS_BANK_CONFLICT --normalize-counter SQ_INSTS_LDS

The companion kernel lds_conflict.hip has one conflict-free and one fully
conflicting LDS load, run in a ratio that changes with the block index.
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict

# Reuse the existing marker samples: the shared loader and the address-trace
# block decoder (header + EXEC mask + one payload word per lane).
from address_trace import BUFFER_KINDS, MEMORY_KINDS, decode_address_traces
from marker_common import add_marker_args, load_marker_traces, marker_selected, marker_time
from rocprof_trace_decoder import RecordType

# LDS data accesses. ds_permute and ds_bpermute go through the LDS crossbar but
# their payload is a lane index, not an LDS address, so they are left out.
LDS_KINDS = {"lds_load", "lds_store", "lds_atomic"}
# Name prefix of the payload-free markers emitted by SQTT_TRACE_SITES.
SITE_PREFIX = "site_"
# Global memory accesses (flat/global and buffer instructions).
GLOBAL_KINDS = MEMORY_KINDS | BUFFER_KINDS
# --sites choices -> which address-trace kinds become columns of A.
SITE_KINDS = {"lds": LDS_KINDS, "memory": GLOBAL_KINDS, "all": LDS_KINDS | GLOBAL_KINDS}


def conflict_cycles(addresses: list[int], banks: int, bank_bytes: int) -> int:
    """Extra cycles one wave instruction spends on LDS bank conflicts.

    Each address maps to a bank by its dword index modulo the number of banks.
    Lanes reading the *same* dword are broadcast and do not conflict, so we
    count distinct dwords per bank. The busiest bank sets how many passes the
    LDS needs; every pass beyond the first is a conflict cycle.
    """
    # bank -> set of distinct dword indices that land in that bank
    per_bank: dict[int, set[int]] = defaultdict(set)
    for address in addresses:
        dword = address // bank_bytes
        per_bank[dword % banks].add(dword)
    # No active lanes means no access, hence no conflict (max of empty -> 1).
    return max((len(dwords) for dwords in per_bank.values()), default=1) - 1


def fit_clock_map(pairs: list[tuple[int, int]]) -> tuple[float, float, float]:
    """Least-squares line realtime = slope * shader_clock + offset.

    Returns (slope, offset, worst residual in realtime ticks). The residual is
    a sanity check: a large value means the clocks drifted or wrapped.
    """
    n = len(pairs)
    sx = sum(x for x, _ in pairs)
    sy = sum(y for _, y in pairs)
    sxx = sum(x * x for x, _ in pairs)
    sxy = sum(x * y for x, y in pairs)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    offset = (sy - slope * sx) / n
    worst = max(abs(y - (slope * x + offset)) for x, y in pairs)
    return slope, offset, worst


def solve_least_squares(rows: list[list[float]], y: list[float]) -> list[float]:
    """Solve min |A.x - y| through the normal equations (A^T A) x = A^T y.

    The number of unknowns is the number of sites, which is small, so a dense
    Gauss-Jordan elimination is plenty and avoids a numpy dependency.
    """
    k = len(rows[0])
    # Augmented matrix [A^T A | A^T y], built one window (row) at a time.
    m = [[0.0] * (k + 1) for _ in range(k)]
    for row, value in zip(rows, y):
        for i in range(k):
            for j in range(k):
                m[i][j] += row[i] * row[j]
            m[i][k] += row[i] * value
    for c in range(k):
        # Partial pivoting keeps the elimination numerically stable.
        pivot = max(range(c, k), key=lambda i: abs(m[i][c]))
        if abs(m[pivot][c]) < 1e-12:
            raise SystemExit(
                "A is rank deficient: some sites always execute in the same ratio, "
                "so their counter activity cannot be told apart. Use more windows "
                "or a workload whose mix changes over time."
            )
        m[c], m[pivot] = m[pivot], m[c]
        for i in range(k):
            if i != c:
                factor = m[i][c] / m[c][c]
                for j in range(c, k + 1):
                    m[i][j] -= factor * m[c][j]
    return [m[i][k] / m[i][i] for i in range(k)]


def relative_residual(rows: list[list[float]], y: list[float], x: list[float]) -> float:
    """|A.x - y| / |y|: how much of the counter the fit leaves unexplained."""
    err = sum((sum(a * b for a, b in zip(row, x)) - v) ** 2 for row, v in zip(rows, y))
    norm = sum(v * v for v in y)
    return (err / norm) ** 0.5 if norm else 0.0


def load_spm_samples(path: str, names: list[str]) -> dict[str, dict[int, float]]:
    """Read rocprofv3 JSON output: counter name -> {realtime timestamp: value}.

    SPM reports one value per counter instance (e.g. per SQ) per sample; the
    values are deltas since the previous sample, so instances are summed to
    get the GPU-wide delta for that sample.
    """
    return {name: summed for name, (summed, _) in load_spm_streams(path, names).items()}


def load_spm_streams(path: str, names: list[str]):
    """Like load_spm_samples, plus each instance's own series:
    counter name -> ({timestamp: summed value}, {instance: [(timestamp, value)]})."""
    with open(path) as f:
        tool = json.load(f)["rocprofiler-sdk-tool"][0]
    ids = {c["id"]["handle"]: c["name"] for c in tool["counters"]}
    wanted = set(names)
    samples: dict[str, dict[int, float]] = {name: defaultdict(float) for name in names}
    streams: dict[str, dict[int, list]] = {name: defaultdict(list) for name in names}
    for batch in tool["callback_records"].get("spm_counter_collection", []):
        for record in batch["records"]:
            name = ids.get(record["counter_id"]["handle"])
            if name in wanted:
                samples[name][record["timestamp"]] += record["value"]
                streams[name][record["instance_id"]].append((record["timestamp"], record["value"]))
    missing = [name for name in names if not samples[name]]
    if missing:
        raise SystemExit(
            f"No SPM samples for {', '.join(missing)} in {path}. Collect them with "
            "rocprofv3 --spm ... --output-format json in the same run as --att."
        )
    return {name: (samples[name], streams[name]) for name in names}


def print_table(sites, executions_per_site, x, model, unit):
    """One row per site: traced executions, model value (if any), fitted value."""
    name_width = max(len("site"), *(len(site) for site in sites))
    print(f"{'site':{name_width}s} {'execs':>8s} {'model/exec':>11s} {unit:>14s}")
    for site, value in zip(sites, x):
        count = executions_per_site[site]
        reference = f"{model[site] / count:11.3f}" if site in model else f"{'-':>11s}"
        print(f"{site:{name_width}s} {count:8d} {reference} {value:14.3f}")


def group_collinear(rows: list[list[float]], threshold: float) -> list[list[int]]:
    """Group sites whose per-window execution counts are proportional.

    Sites in the same loop body run the same number of times in every window,
    so their columns of A are (nearly) parallel and the fit cannot tell them
    apart: it only sees their sum, and individual values swing wildly. Sites
    whose columns have cosine similarity >= threshold are merged, and each
    group is fitted as one column. Largest sites seed groups first.
    """
    cols = [list(col) for col in zip(*rows)]
    norms = [sum(v * v for v in col) ** 0.5 for col in cols]
    groups: list[list[int]] = []
    for i in sorted(range(len(cols)), key=lambda c: -norms[c]):
        for group in groups:
            seed = group[0]
            if norms[i] and norms[seed]:
                cosine = sum(a * b for a, b in zip(cols[i], cols[seed])) / (norms[i] * norms[seed])
                if cosine >= threshold:
                    group.append(i)
                    break
        else:
            groups.append([i])
    return groups


def group_label(members: list[str]) -> str:
    """Short name for a group: the site, or its distinct locations."""
    if len(members) == 1:
        return members[0]
    locations = sorted({m.split("@", 1)[1] for m in members})
    shown = ", ".join(locations[:3]) + (
        f", +{len(locations) - 3} more" if len(locations) > 3 else ""
    )
    return f"{len(members)} sites: {shown}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit per-source-line counter activity from SQTT markers and SPM samples."
    )
    add_marker_args(parser)
    parser.add_argument(
        "--spm-json",
        help="rocprofv3 JSON results from the same run (--spm ... --output-format json). "
        "Without it, Y is synthesised from LDS addresses as a self-check.",
    )
    parser.add_argument(
        "--counter",
        action="append",
        help="SPM counter to attribute; repeat for several. Default: SQC_LDS_BANK_CONFLICT.",
    )
    parser.add_argument(
        "--normalize-counter",
        help="SPM counter that counts the traced instructions themselves (e.g. SQ_INSTS_LDS), "
        "used to scale trace counts from the traced CUs up to the whole GPU.",
    )
    parser.add_argument(
        "--samples-per-window",
        type=int,
        default=16,
        help="Consecutive SPM samples merged into one fit window (more = less noise).",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=64,
        help="Number of equal time windows in self-check mode (no --spm-json).",
    )
    parser.add_argument(
        "--sites",
        choices=sorted(SITE_KINDS),
        default="lds",
        help="Which traced instructions become sites (columns of A).",
    )
    parser.add_argument(
        "--group-threshold",
        type=float,
        default=0.97,
        help="Merge sites whose per-window counts have at least this cosine similarity "
        "(they cannot be told apart); 1 disables merging.",
    )
    parser.add_argument("--banks", type=int, default=64, help="LDS banks in the bank model.")
    parser.add_argument("--bank-bytes", type=int, default=4, help="Bytes per bank word.")
    parser.add_argument(
        "--loads-only",
        action="store_true",
        help="Ignore stores, e.g. a one-off initialisation loop. Do not combine with a "
        "normalising counter that also counts stores.",
    )
    args = parser.parse_args()
    counters = args.counter or ["SQC_LDS_BANK_CONFLICT"]
    kinds = SITE_KINDS[args.sites]

    # Ask the decoder for REALTIME records as well as markers, so the shader
    # clock used by marker timestamps can be tied to the realtime clock.
    # RT_FREQUENCY carries the realtime clock rate (ticks per second).
    traces = load_marker_traces(
        args, extra_record_types=(RecordType.REALTIME, RecordType.RT_FREQUENCY)
    )

    # One entry per traced instruction:
    #   (shader-clock time, realtime or None, site, model cycles).
    # The model value is the LDS bank-conflict estimate and only exists for LDS.
    executions: list[tuple[int, float | None, str, int | None]] = []
    frequency = None
    worst_residual = 0.0
    unmapped = 0
    for decoded in traces:
        # Each .att file is one shader engine, and each shader engine has its own
        # shader clock, so the shader clock -> realtime map is fitted per file.
        pairs = [(r.shader_clock, r.realtime_clock) for r in decoded.records.realtime]
        clock = None
        if len(pairs) >= 2:
            slope, offset, worst = fit_clock_map(pairs)
            clock = (slope, offset)
            worst_residual = max(worst_residual, worst)
        else:
            unmapped += 1
        frequency = decoded.records.realtime_frequency or frequency
        for trace in decode_address_traces(decoded.records.markers, args):
            if trace.kind not in kinds:
                continue
            if args.loads_only and "store" in trace.kind:
                continue
            # The site is the source line when built with -g; fall back to the
            # marker id so unannotated builds still produce separate columns.
            site = f"{trace.kind}@{trace.source_location or f'marker#{trace.marker_id}'}"
            model = (
                conflict_cycles(trace.addresses, args.banks, args.bank_bytes)
                if trace.kind in LDS_KINDS
                else None
            )
            realtime = clock[0] * trace.time + clock[1] if clock else None
            executions.append((trace.time, realtime, site, model))
        # Light site markers (SQTT_TRACE_SITES=lds|memory): one header per executed
        # instruction, named site_<kind>, with no addresses. They are enough for A;
        # only the bank-model column needs addresses.
        for marker in decoded.records.markers:
            name = marker.name or ""
            if not name.startswith(SITE_PREFIX) or not marker_selected(marker, args):
                continue
            kind = name[len(SITE_PREFIX) :]
            if kind not in kinds or (args.loads_only and "store" in kind):
                continue
            site = f"{kind}@{marker.source_location or f'marker#{marker.marker_id}'}"
            time = marker_time(marker)
            realtime = clock[0] * time + clock[1] if clock else None
            executions.append((time, realtime, site, None))

    if not executions:
        raise SystemExit(
            "No markers found for the selected sites. Build the kernel with the "
            "sqtt-marker pass and SQTT_TRACE_SITES=lds (cheap) or SQTT_TRACE_ADDRESSES=lds."
        )

    # Give every distinct site a column in A, and keep per-site totals.
    sites = sorted({site for _, _, site, _ in executions})
    column = {site: i for i, site in enumerate(sites)}
    executions_per_site: dict[str, int] = defaultdict(int)
    model: dict[str, int] = defaultdict(int)
    for _, _, site, cycles in executions:
        executions_per_site[site] += 1
        if cycles is not None:
            model[site] += cycles

    # The residual says how well each shader engine's clock map fits; a few tens
    # of ticks is normal jitter, much more means the clocks drifted or wrapped.
    rate = f", realtime clock {frequency} Hz" if frequency else ""
    print(
        f"clock maps: {len(traces) - unmapped}/{len(traces)} trace files mapped to realtime "
        f"(worst residual {worst_residual:.1f} ticks{rate})"
    )

    if not args.spm_json:
        return self_check(args, executions, sites, column, executions_per_site, model)

    if unmapped:
        raise SystemExit(
            "Some trace files have no REALTIME records, so their markers cannot be "
            "aligned with SPM samples."
        )

    # dict.fromkeys de-duplicates while keeping order, so a counter that is both
    # attributed and used for normalising is only read (and summed) once.
    names = list(dict.fromkeys(counters + [args.normalize_counter or counters[0]]))
    spm_streams = load_spm_streams(args.spm_json, names)
    spm = {name: summed for name, (summed, _) in spm_streams.items()}
    # On multi-die GPUs (MI300: one SPM stream per XCC) the instances sample on
    # different timelines, so a sample's delta covers the time since *its own
    # instance's* previous sample, not since the previous timestamp of any
    # instance. Then windows are a regular grid and each delta is spread over
    # the windows its own interval overlaps.
    stamp_sets = {
        frozenset(t for t, _ in series)
        for name in names
        for series in spm_streams[name][1].values()
    }
    own_timelines = len(stamp_sets) > 1

    # Window boundaries are SPM sample timestamps. A sample's value counts the
    # events since the previous sample, so window k covers the realtime interval
    # (edges[k], edges[k + 1]] and owns every sample stamped inside it.
    stamps = sorted(set().union(*(spm[name].keys() for name in names)))
    if own_timelines:
        gaps = []
        for name in names:
            for series in spm_streams[name][1].values():
                ts = sorted({t for t, _ in series})
                gaps += [b - a for a, b in zip(ts, ts[1:])]
        gaps.sort()
        period = gaps[len(gaps) // 2]  # median sampling period of one instance
        width = period * args.samples_per_window
        edges = list(range(stamps[0], stamps[-1] + width, width))
        print(
            f"SPM instances sample on {len(stamp_sets)} separate timelines (e.g. one per XCC): "
            f"windows of {width} ticks, each sample spread over its own interval"
        )
    else:
        edges = stamps[:: args.samples_per_window]
        if edges[-1] != stamps[-1]:
            edges.append(stamps[-1])
    windows = len(edges) - 1
    if windows < len(sites):
        raise SystemExit(
            f"Only {windows} windows for {len(sites)} sites; lower --samples-per-window "
            "or the SPM sample interval."
        )

    def window_of(realtime: float) -> int | None:
        # bisect_left finds the first edge >= t; the window ends at that edge.
        k = bisect.bisect_left(edges, realtime) - 1
        return k if 0 <= k < windows else None

    # Y per counter: sum the SPM deltas that fall into each window.
    Y: dict[str, list[float]] = {name: [0.0] * windows for name in names}
    for name in names:
        if not own_timelines:
            for stamp, value in spm[name].items():
                k = window_of(stamp)
                if k is not None:
                    Y[name][k] += value
            continue
        for series in spm_streams[name][1].values():
            series = sorted(series)
            # The first sample's interval start is unknown, so it is left out.
            for (prev, _), (stamp, value) in zip(series, series[1:]):
                if stamp <= prev:
                    continue
                k = max(0, bisect.bisect_right(edges, prev) - 1)
                while k < windows and edges[k] < stamp:
                    overlap = min(stamp, edges[k + 1]) - max(prev, edges[k])
                    if overlap > 0:
                        Y[name][k] += value * overlap / (stamp - prev)
                    k += 1

    # A: traced executions per site per window, after mapping marker shader-clock
    # times onto the realtime axis of the SPM samples.
    A = [[0.0] * len(sites) for _ in range(windows)]
    outside = 0
    for _, realtime, site, _ in executions:
        k = window_of(realtime)
        if k is None:
            outside += 1
            continue
        A[k][column[site]] += 1

    # Optional scale-up from traced CUs to the whole GPU: in each window, the
    # normalising counter gives the GPU-wide number of traced-type instructions,
    # and the trace gives how they split between sites.
    # Without it, fit/exec is events per *traced* execution, which equals events
    # per execution when the trace covers the whole GPU.
    scope = "fit/exec is per traced execution"
    if args.normalize_counter:
        scope = f"fit/exec is per execution, scaled by {args.normalize_counter}"
        for k, row in enumerate(A):
            traced = sum(row)
            scale = Y[args.normalize_counter][k] / traced if traced else 0.0
            A[k] = [value * scale for value in row]

    # Only windows with traced activity can constrain the fit.
    used = [k for k in range(windows) if any(A[k])]
    print(
        f"{len(executions)} traced executions ({outside} outside the SPM span), "
        f"{len(sites)} sites, {len(used)}/{windows} windows of "
        f"{args.samples_per_window} SPM samples; {scope}"
    )
    rows = [A[k] for k in used]

    # Merge sites that always run together; the fit then reports one value per
    # group, in events per execution of one of its sites.
    columns, labels = sites, sites
    counts, references = dict(executions_per_site), dict(model)
    if args.group_threshold < 1:
        groups = group_collinear(rows, args.group_threshold)
        if len(groups) < len(sites):
            labels = [group_label([sites[i] for i in group]) for group in groups]
            rows = [[sum(row[i] for i in group) for group in groups] for row in rows]
            counts = {
                label: sum(executions_per_site[sites[i]] for i in group)
                for label, group in zip(labels, groups)
            }
            references = {
                label: sum(model[sites[i]] for i in group)
                for label, group in zip(labels, groups)
                if all(sites[i] in model for i in group)
            }
            columns = labels
            print(
                f"{len(sites)} sites merged into {len(groups)} groups that run together "
                f"(cosine >= {args.group_threshold})"
            )

    for name in counters:
        y = [Y[name][k] for k in used]
        x = solve_least_squares(rows, y)
        print(
            f"\n{name}: {sum(Y[name]):.0f} total, fit leaves "
            f"{100 * relative_residual(rows, y, x):.1f}% unexplained"
        )
        print_table(columns, counts, x, references, "fit/exec")
    return 0


def self_check(args, executions, sites, column, executions_per_site, model) -> int:
    """No SPM: synthesise Y from the bank model and check the fit recovers it."""
    if not model:
        raise SystemExit("Self-check mode needs LDS sites; pass --spm-json for other sites.")
    # Windows are cut on the shader clock. Y is built from the same executions,
    # so per-shader-engine clock phase differences do not matter here.
    start = min(time for time, _, _, _ in executions)
    end = max(time for time, _, _, _ in executions) + 1
    width = (end - start) / args.windows
    A = [[0.0] * len(sites) for _ in range(args.windows)]
    Y = [0.0] * args.windows
    for time, _, site, cycles in executions:
        # min() keeps the last execution inside the final window.
        w = min(int((time - start) / width), args.windows - 1)
        A[w][column[site]] += 1
        Y[w] += cycles or 0
    used = [k for k in range(args.windows) if any(A[k])]
    print(
        f"self-check (no --spm-json): {len(executions)} executions, {len(sites)} sites, "
        f"{len(used)} non-empty windows of {width:.0f} shader-clock ticks"
    )
    x = solve_least_squares([A[k] for k in used], [Y[k] for k in used])
    print_table(sites, executions_per_site, x, model, "fit/exec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
