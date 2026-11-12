#!/usr/bin/env python3
"""Audit kept ATT traces and the merged RCV views of one campaign.

1. Same work: for every (SE, SIMD), the traced wave has the same identity (A1 uid) and the same executed
   instruction sequence (code-line indices) in every sweep.
2. Merged views: within a merged directory, the four SIMD waves of a workgroup release from each s_barrier_wait at
   the same time; reports the per-barrier spread (max - min over the four SIMDs) in shader clocks. Also checks
   instruction timestamps are non-decreasing and lie within each wave's [begin, end].
3. Kernel time: per replay and SE, first dispatch to last wave end on CU 0, converted to microseconds through the
   realtime counter pairs (100 MHz realtime clock assumed).
4. Start-up: per-instruction durations of the first instructions of every traced wave.
Usage: audit.py RUN_DIR OUT.json   (RUN_DIR holds table-traces/ and rcv-merged-sweep-*/)
Same work and merged checks are per wave identity: CU 0 holds two 8-wave workgroups, so 4 waves per SIMD.
"""

import argparse
import hashlib
import os
import align
import collections
import glob
import json
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

A1 = 0xA1
REALTIME_HZ = 100e6


def load_ui(ui):
    ui = Path(ui)
    names = json.loads((ui / "filenames.json").read_text())
    code = json.loads((ui / "code.json").read_text())["code"]
    return ui, names, code


def wave_files(names):
    for se_key, per_simd in names["wave_filenames"].items():
        for simd, per_slot in per_simd.items():
            for slot, waves in per_slot.items():
                for index, (fname, _, _) in waves.items():
                    yield int(se_key), int(simd), int(slot), fname


def cell_summary(cell):
    cell = Path(cell)
    ui, names, code = load_ui(glob.glob(str(cell / "ui_output_agent_*_dispatch_1"))[0])
    text = {row[2]: row[0] for row in code}
    occupancy = json.loads((ui / "occupancy.json").read_text())
    realtime = json.loads((ui / "realtime.json").read_text())
    markers = collections.defaultdict(list)
    for se_key, files in names["shaderdata_filenames"].items():
        for name in files:
            markers[int(se_key)] += json.loads((ui / name[0]).read_text())["records"]
    lo = int(os.environ.get("REMAPPER_MERGE_PROLOGUE_VADDR", str(1 << 64)))
    hi = lo + int(os.environ.get("REMAPPER_MERGE_PROLOGUE_BYTES", "0"))
    addresses = {row[2]: row[5] for row in code if not row[0].startswith(";")}
    # Sparse launches: only shader engines whose traced WGP held a real workgroup are audited.
    engines = os.environ.get("REMAPPER_MERGE_ENGINES")
    keep = None if engines is None else {int(s) for s in engines.split(",") if s}
    waves = {}
    for se, simd, slot, fname in wave_files(names):
        if keep is not None and se not in keep:
            continue
        w = json.loads((ui / fname).read_text())["wave"]
        uids = [
            v & 0xFFFFFF
            for t, v, cu, sm, sl, fl in markers[se]
            if v >> 24 == A1
            and cu == w["cu"]
            and sm == simd
            and sl == slot
            and w["begin"] <= t <= w["end"]
        ]
        insts = w["instructions"]
        all_lines = [i[4] for i in insts]
        lines = [i[4] for i in insts if not lo <= addresses.get(i[4], lo) < hi]
        if not lines and os.environ.get("REMAPPER_MERGE_DROP_FAKES"):
            continue  # admission launch: a workgroup parked in the prologue never reaches the body
        waves[(se, simd, uids[0] if len(uids) == 1 else ("unidentified", slot))] = dict(
            uid=uids[0] if len(uids) == 1 else None,
            n=len(lines),
            full_lines_hash=hashlib.sha256(json.dumps(all_lines).encode()).hexdigest(),
            lines_hash=hashlib.sha256(json.dumps(lines).encode()).hexdigest(),
            begin=w["begin"],
            end=w["end"],
            head=[(text.get(i[4], "?"), i[3], i[0] - w["begin"]) for i in insts[:40]],
        )
    kernel = {}
    for se_key, rows in occupancy.items():
        if not se_key.isdigit():
            continue
        dispatch = [
            e["time"]
            for e in occupancy["events"][se_key]
            if e.get("kind") == "dispatch"
        ]
        ends = [r[0] for r in rows if not r[4]]
        rt = np.array(realtime.get(f"SE{se_key}", []), dtype=np.float64)
        mhz = None
        if len(rt) >= 2:
            mhz = (rt[-1, 0] - rt[0, 0]) / ((rt[-1, 1] - rt[0, 1]) / REALTIME_HZ) / 1e6
        kernel[int(se_key)] = dict(
            cycles=(max(ends) - min(dispatch)) if dispatch and ends else None, mhz=mhz
        )
    m = re.search(r"(sweep-\d+)/cu-\d+-simd-(\d)", str(cell))
    return m[1], int(m[2]), waves, kernel


def merged_summary(ui):
    """Barrier release spread over all waves of each traced workgroup (8 waves on 4 SIMDs from 4 replays)."""
    ui, names, code = load_ui(ui)
    barrier_lines = {row[2] for row in code if row[0].startswith("s_barrier_wait")}
    uid_of = json.loads((ui / "merge_provenance.json").read_text())["wave_uids"]
    per_wg = collections.defaultdict(dict)
    order_problems = unidentified = 0
    for se, simd, slot, fname in wave_files(names):
        w = json.loads((ui / fname).read_text())["wave"]
        if w["cu"] != align.TARGET_CU:
            continue
        t = np.array([i[0] for i in w["instructions"]], dtype=np.int64)
        if len(t) and (
            np.any(np.diff(t) < 0)
            or t[0] < w["begin"]
            or t[-1] > w["end"]
            or any(i[0] + i[3] > w["end"] for i in w["instructions"])
        ):
            order_problems += 1
        uid = uid_of.get(fname)
        if uid is None:
            unidentified += 1
            continue
        per_wg[(se, uid // align.WAVES_PER_WORKGROUP)][uid] = [
            i[0] + i[3] for i in w["instructions"] if i[4] in barrier_lines
        ]
    spreads, counts = [], collections.Counter()
    for key, waves in per_wg.items():
        counts[(len(waves), tuple(sorted({len(v) for v in waves.values()})))] += 1
        if (
            len(waves) == align.WAVES_PER_WORKGROUP
            and len({len(v) for v in waves.values()}) == 1
        ):
            arr = np.array([waves[u] for u in sorted(waves)])
            spreads += (arr.max(0) - arr.min(0)).tolist()
    s = np.array(spreads)
    return dict(
        dir=str(ui),
        workgroups=len(per_wg),
        waves_and_barrier_counts=str(dict(counts)),
        unidentified_waves=unidentified,
        waves_with_timestamp_problems=order_problems,
        barrier_release_spread_clocks=(
            dict(
                n=int(len(s)),
                median=float(np.median(s)),
                p90=float(np.percentile(s, 90)),
                p99=float(np.percentile(s, 99)),
                max=float(s.max()),
            )
            if len(s)
            else None
        ),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--extract", type=Path, required=True)
    ap.add_argument("--merged-glob", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run, out = args.traces.parent, args.out
    cells = sorted(
        p for p in args.traces.glob("sweep-*/cu-*-simd-*-attempt-*") if p.is_dir()
    )
    with ProcessPoolExecutor(16) as pool:
        cells_out = list(pool.map(cell_summary, map(str, cells)))
        merged = list(pool.map(merged_summary, sorted(glob.glob(args.merged_glob))))
    result = dict(run=str(run), cells=len(cells_out))

    # Per (SE, traced SIMD): the set of wave identities must be the same in every sweep; per (SE, uid): the executed
    # instruction sequence must be identical in every sweep. Identities come from extract.py's marker matching.
    uids = collections.defaultdict(dict)
    for p in sorted(args.extract.glob("*.pkl")):
        r = pickle.loads(p.read_bytes())
        for se, e in r["engines"].items():
            uids[(se, r["traced_simd"])][r["sweep"]] = sorted(
                (w["uid"] for w in e["detailed"]), key=str
            )
    nsweeps = len({s for v in uids.values() for s in v})
    same_set = sum(
        len({tuple(map(str, v)) for v in per.values()}) == 1 and len(per) == nsweeps
        for per in uids.values()
    )
    unidentified = sum(
        u is None for per in uids.values() for v in per.values() for u in v
    )
    seqs = collections.defaultdict(dict)
    full_seqs = collections.defaultdict(dict)
    for sweep, simd, waves, _ in cells_out:
        for (se, wsimd, uid), w in waves.items():
            seqs[(se, uid)][sweep] = (w["n"], w["lines_hash"])
            full_seqs[(se, uid)][sweep] = w["full_lines_hash"]
    same_seq = sum(
        len(set(v.values())) == 1 and len(v) == nsweeps for v in seqs.values()
    )
    insts = sorted({n for v in seqs.values() for n, _ in v.values()})
    result["same_work"] = dict(
        se_simd_pairs=len(uids),
        sweeps=nsweeps,
        traced_waves_per_se_simd=sorted(
            {len(v) for per in uids.values() for v in per.values()}
        ),
        same_wave_identities_every_sweep=same_set,
        unidentified_waves=unidentified,
        waves=len(seqs),
        identical_instruction_sequence_every_sweep=same_seq,
        instructions_per_wave_min_max=[insts[0], insts[-1]] if insts else None,
        sequence_scope="original kernel body",
        identical_full_sequence_every_sweep=sum(
            len(set(v.values())) == 1 and len(v) == nsweeps for v in full_seqs.values()
        ),
        prologue_note="Hardware leader role can change when logical cluster rank is renamed; body sequences must remain identical.",
    )

    cyc = collections.defaultdict(list)
    mhz = []
    for sweep, simd, _, kernel in cells_out:
        for se, k in kernel.items():
            if k["cycles"] is not None and k["mhz"]:
                cyc[(sweep, simd)].append(k["cycles"] / k["mhz"])
                mhz.append(k["mhz"])
    per_replay = [max(v) for v in cyc.values()]
    result["kernel_time_us"] = dict(
        replays=len(per_replay),
        median=float(np.median(per_replay)),
        min=float(min(per_replay)),
        max=float(max(per_replay)),
        shader_clock_mhz_median=float(np.median(mhz)),
    )

    heads = collections.defaultdict(list)
    for _, _, waves, _ in cells_out:
        for w in waves.values():
            for i, (name, dur, off) in enumerate(w["head"]):
                heads[(i, name)].append((dur, off))
    rows = []
    for (i, name), v in sorted(heads.items()):
        d = np.array([x[0] for x in v])
        o = np.array([x[1] for x in v])
        rows.append(
            dict(
                index=i,
                inst=name,
                n=len(v),
                dur_median=float(np.median(d)),
                dur_max=float(d.max()),
                start_offset_median=float(np.median(o)),
            )
        )
    result["startup"] = rows
    result["merged"] = merged
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "startup"}, indent=2))
    for r in rows:
        if r["dur_median"] > 200 or r["index"] < 24:
            print(
                f'{r["index"]:3d} {r["inst"][:60]:60s} dur med {r["dur_median"]:9.0f} max {r["dur_max"]:9.0f} '
                f'start {r["start_offset_median"]:9.0f}'
            )


if __name__ == "__main__":
    main()
