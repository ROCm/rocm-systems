#!/usr/bin/env python3
"""Attribute LDS bank conflicts to source lines with a least-squares fit.

Idea
----
A hardware counter such as SQC_LDS_BANK_CONFLICT only says *how many* conflict
cycles happened, not *which instruction* caused them. If we split the run into
many short time windows, then for each window w:

    Y[w] = sum over LDS sites s of  A[w][s] * X[s]

where
    Y[w]    = conflict cycles observed in window w (the counter delta),
    A[w][s] = how many times LDS site s executed in window w,
    X[s]    = average conflict cycles per execution of site s (the unknown).

There are many more windows than LDS sites, so the system is overdetermined
and X can be recovered with ordinary least squares, as long as the sites do
not always run in the same ratio (otherwise the columns of A are collinear).

Where the data comes from
-------------------------
* A comes from the SQTT thread trace. The kernel is built with the
  sqtt-marker pass and SQTT_TRACE_ADDRESSES=lds, so every LDS instruction
  emits a marker carrying its source line, a timestamp and the per-lane
  addresses.
* Y should come from SPM (streaming perf monitor) samples taken during the
  same run. SPM and ATT cannot run together yet, so for now Y is synthesised
  from the traced lane addresses with a simple bank model. That validates the
  windowing and the fit; swapping in real SPM deltas is the next step.
* SQTT timestamps are in shader-clock ticks. The REALTIME records pair shader
  clock with the fixed-frequency realtime clock; the fitted mapping is what
  an SPM sample timestamp would be aligned against.

Usage
-----
    PYTHONPATH=python:samples python3 samples/markers/lds_conflict_fit.py \
        captures/*.att captures/*_code_object_id_*.out --loads-only

The companion kernel lds_conflict.hip has one conflict-free and one fully
conflicting LDS load, run in a ratio that changes with the block index.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

# Reuse the existing marker samples: the shared loader and the address-trace
# block decoder (header + EXEC mask + one payload word per lane).
from address_trace import decode_address_traces
from marker_common import add_marker_args, load_marker_traces
from rocprof_trace_decoder import RecordType

# Only LDS data accesses are relevant for bank conflicts. ds_permute and
# ds_bpermute go through the LDS crossbar but their payload is a lane index,
# not an LDS address, so they are left out.
LDS_KINDS = {"lds_load", "lds_store", "lds_atomic"}


def conflict_cycles(addresses: list[int], banks: int, bank_bytes: int) -> int:
    """Extra cycles one wave instruction spends on bank conflicts.

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


def solve_least_squares(rows: list[list[int]], y: list[float]) -> list[float]:
    """Solve min |A.x - y| through the normal equations (A^T A) x = A^T y.

    The number of unknowns is the number of LDS sites, which is small, so a
    dense Gauss-Jordan elimination is plenty and avoids a numpy dependency.
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
                "A is rank deficient: some LDS sites always execute in the same "
                "ratio, so their conflicts cannot be told apart. Use more windows "
                "or a workload whose mix changes over time."
            )
        m[c], m[pivot] = m[pivot], m[c]
        for i in range(k):
            if i != c:
                factor = m[i][c] / m[c][c]
                for j in range(c, k + 1):
                    m[i][j] -= factor * m[c][j]
    return [m[i][k] / m[i][i] for i in range(k)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit per-source-line LDS bank conflicts from SQTT address markers."
    )
    add_marker_args(parser)
    parser.add_argument(
        "--windows",
        type=int,
        default=64,
        help="Number of equal time windows; stands in for SPM sample intervals.",
    )
    parser.add_argument("--banks", type=int, default=64, help="LDS banks in the bank model.")
    parser.add_argument("--bank-bytes", type=int, default=4, help="Bytes per bank word.")
    parser.add_argument(
        "--loads-only",
        action="store_true",
        help="Ignore LDS stores, e.g. a one-off initialisation loop.",
    )
    args = parser.parse_args()

    # Ask the decoder for REALTIME records as well as markers, so the shader
    # clock used by marker timestamps can be tied to the realtime clock.
    # RT_FREQUENCY carries the realtime clock rate (ticks per second).
    traces = load_marker_traces(
        args, extra_record_types=(RecordType.REALTIME, RecordType.RT_FREQUENCY)
    )

    # One entry per executed LDS instruction: (shader-clock time, site, cycles).
    executions: list[tuple[int, str, int]] = []
    clock_pairs: list[tuple[int, int]] = []
    frequency = None
    for decoded in traces:
        for trace in decode_address_traces(decoded.records.markers, args):
            if trace.kind not in LDS_KINDS:
                continue
            if args.loads_only and trace.kind != "lds_load":
                continue
            # The site is the source line when built with -g; fall back to the
            # marker id so unannotated builds still produce separate columns.
            site = f"{trace.kind}@{trace.source_location or f'marker#{trace.marker_id}'}"
            cycles = conflict_cycles(trace.addresses, args.banks, args.bank_bytes)
            executions.append((trace.time, site, cycles))
        clock_pairs.extend((r.shader_clock, r.realtime_clock) for r in decoded.records.realtime)
        frequency = decoded.records.realtime_frequency or frequency

    if not executions:
        raise SystemExit(
            "No LDS address markers found. Build the kernel with the sqtt-marker "
            "pass and SQTT_TRACE_ADDRESSES=lds."
        )

    # Shader clock -> realtime. With SPM in the same run, each SPM sample
    # timestamp would be converted into shader-clock time with this mapping
    # and used as a window boundary instead of the equal windows below.
    if len(clock_pairs) >= 2:
        slope, offset, worst = fit_clock_map(clock_pairs)
        rate = f", realtime clock {frequency} Hz" if frequency else ""
        print(
            f"clock map: realtime = {slope:.6f} * shader_clock + {offset:.0f} "
            f"({len(clock_pairs)} pairs, worst residual {worst:.1f} ticks{rate})"
        )
    else:
        print("clock map: no REALTIME records in the trace")

    # Give every distinct site a column in A.
    sites = sorted({site for _, site, _ in executions})
    column = {site: i for i, site in enumerate(sites)}

    # Ground truth per site, known here only because the addresses are traced.
    # In a real SPM run this is what the fit is meant to recover.
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for _, site, cycles in executions:
        totals[site][0] += 1
        totals[site][1] += cycles

    # Split the traced span into equal windows and fill A and Y.
    start = min(time for time, _, _ in executions)
    end = max(time for time, _, _ in executions) + 1
    width = (end - start) / args.windows
    A = [[0] * len(sites) for _ in range(args.windows)]
    Y = [0] * args.windows
    for time, site, cycles in executions:
        # min() keeps the last execution inside the final window.
        w = min(int((time - start) / width), args.windows - 1)
        A[w][column[site]] += 1
        # Synthetic counter delta; replace with the SPM delta for window w.
        Y[w] += cycles

    # Windows with no LDS activity add nothing to the fit.
    used = [(row, y) for row, y in zip(A, Y) if any(row)]
    print(
        f"{len(executions)} LDS executions, {len(sites)} sites, "
        f"{len(used)} non-empty windows of {width:.0f} shader-clock ticks"
    )
    X = solve_least_squares([row for row, _ in used], [y for _, y in used])

    # exact/exec comes from the traced addresses; fit/exec from Y = A.X alone.
    # They agree here because Y is synthetic; with SPM, the gap measures how
    # well the counter matches the bank model.
    # Size the first column to the longest site name (source paths can be long).
    name_width = max(len("site"), *(len(site) for site in sites))
    print(f"{'site':{name_width}s} {'execs':>8s} {'exact/exec':>11s} {'fit/exec':>9s}")
    for site, x in zip(sites, X):
        count, cycles = totals[site]
        print(f"{site:{name_width}s} {count:8d} {cycles / count:11.3f} {x:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
