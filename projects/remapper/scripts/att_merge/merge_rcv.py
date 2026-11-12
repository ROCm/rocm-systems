#!/usr/bin/env python3
"""Write one RCV-loadable rocprofv3 UI directory holding all four SIMDs of CU 0 on every shader engine.

The reference replay (one sweep's SIMD-REF cell) is copied as-is: its occupancy, markers, realtime, partner-SIMD
stream and its own detailed waves are same-run data. The other three SIMDs' detailed waves come from the other
replays of that sweep, with every timestamp mapped onto the reference replay's per-SE shader clock by a
piecewise-linear warp through shared wave begin/end and barrier-release events (see validate_other_simd.py).
code.json Hit/Latency/Stall/Idle are summed over the four replays. The result is a composite of four executions.
Usage: merge_rcv.py TRACES_DIR EXTRACT_DIR SWEEP OUT_DIR [--reference-simd 0]
"""

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np

import align

STAT_COLUMNS = slice(6, 10)  # Hit, Latency, Stall, Idle


def ui_dir(traces, sweep, simd):
    return Path(
        glob.glob(
            str(
                traces
                / sweep
                / f"cu-{align.TARGET_CU:02d}-simd-{simd}-attempt-0"
                / "ui_output_agent_*_dispatch_*"
            )
        )[0]
    )


def warp_to(a, b):
    occ = set(a["occ"]) & set(b["occ"])
    bar = set(a["bar"]) & set(b["bar"])
    # Kernels without barriers align on the traced WGP's shared wave begin/end events alone.
    if len(bar) < 2 and len(occ) < 2:
        raise RuntimeError(
            "too few shared barrier or wave begin/end events to align this pair"
        )
    both_a = {
        **{("o",) + k: v for k, v in a["occ"].items()},
        **{("b",) + k: v for k, v in a["bar"].items()},
    }
    both_b = {
        **{("o",) + k: v for k, v in b["occ"].items()},
        **{("b",) + k: v for k, v in b["bar"].items()},
    }
    return (
        align.warp(
            [("o",) + k for k in occ] + [("b",) + k for k in bar], both_a, both_b
        ),
        len(occ),
        len(bar),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", type=Path)
    ap.add_argument("extract", type=Path)
    ap.add_argument("sweep")
    ap.add_argument("out", type=Path)
    ap.add_argument("--reference-simd", type=int, default=0)
    ap.add_argument(
        "--waves-per-workgroup", type=int, default=align.WAVES_PER_WORKGROUP
    )
    ap.add_argument("--target-cu", type=int, default=align.TARGET_CU)
    args = ap.parse_args()
    align.WAVES_PER_WORKGROUP, align.TARGET_CU = (
        args.waves_per_workgroup,
        args.target_cu,
    )
    ref = args.reference_simd
    src = ui_dir(args.traces, args.sweep, ref)
    out = args.out / src.name
    if out.exists():
        ap.error(f"output exists: {out}")
    shutil.copytree(src, out)
    anchors = align.load(args.extract)[args.sweep]
    names = json.loads((out / "filenames.json").read_text())
    code = json.loads((out / "code.json").read_text())
    provenance = dict(
        composite_of_replays=True,
        sweep=args.sweep,
        reference_simd=ref,
        reference_dir=str(src),
        alignment="piecewise-linear through shared CU-0 wave begin/end and barrier releases",
        validation="att_align/validate_other_simd.py (results in the run's table-other-simd.json)",
        sources={},
        engines={},
    )
    # Sparse launches: engines extract.py excluded (traced WGP held only prologue-ended workgroups) are dropped.
    excluded = sorted(
        k for k in names["wave_filenames"] if int(k) not in anchors[ref]["engines"]
    )
    for k in excluded:
        del names["wave_filenames"][k]
    if excluded:
        provenance["excluded_engines"] = excluded
    # Admission launches: waves of workgroups parked in the prologue are not part of the traced work.
    fakes = {
        simd: {f for e in cell["engines"].values() for f in e.get("fake_files", ())}
        for simd, cell in anchors.items()
    }
    for se_key, per_simd in names["wave_filenames"].items():
        for slot, waves in per_simd.get(str(ref), {}).items():
            for index in [i for i, (f, _, _) in waves.items() if f in fakes[ref]]:
                del waves[index]
    provenance["parked_waves_dropped"] = sum(len(v) for v in fakes.values())
    for simd in range(4):
        if simd == ref:
            continue
        other = ui_dir(args.traces, args.sweep, simd)
        provenance["sources"][simd] = str(other)
        other_code = json.loads((other / "code.json").read_text())["code"]
        for row, extra in zip(code["code"], other_code):
            if row[:6] != extra[:6]:
                raise RuntimeError("ISA rows differ between replays")
            row[STAT_COLUMNS] = [
                x + y for x, y in zip(row[STAT_COLUMNS], extra[STAT_COLUMNS])
            ]
        other_names = json.loads((other / "filenames.json").read_text())
        for se_key, per_simd in other_names["wave_filenames"].items():
            if se_key in excluded:
                continue
            se = int(se_key)
            fn, n_occ, n_bar = warp_to(
                align.events(anchors[simd], se), align.events(anchors[ref], se)
            )
            provenance["engines"].setdefault(se_key, {})[simd] = dict(
                occupancy_anchors=n_occ, barrier_anchors=n_bar
            )
            entries = {}
            for slot, waves in per_simd[str(simd)].items():
                entries[slot] = {}
                for index, (fname, _, _) in waves.items():
                    if fname in fakes[simd]:
                        continue
                    doc = json.loads((other / fname).read_text())
                    w = doc["wave"]
                    # Map each instruction's start and end separately: the anchors are barrier completions
                    # (start + duration), and the offset can change during a long wait.
                    insts = w["instructions"]
                    starts = np.array([i[0] for i in insts], dtype=np.float64)
                    ends = starts + np.array([i[3] for i in insts], dtype=np.float64)
                    mapped = np.maximum.accumulate(np.rint(fn(starts))).astype(np.int64)
                    mapped_end = np.maximum(np.rint(fn(ends)).astype(np.int64), mapped)
                    for inst, t, e in zip(insts, mapped.tolist(), mapped_end.tolist()):
                        inst[0], inst[3] = t, e - t
                        inst[2] = min(inst[2], inst[3])
                    # The state timeline is a run of (state, duration) from begin; warp its boundaries too.
                    bounds = w["begin"] + np.concatenate(
                        [[0], np.cumsum([d for _, d in w["timeline"]])]
                    )
                    mb = np.maximum.accumulate(
                        np.rint(fn(bounds.astype(np.float64)))
                    ).astype(np.int64)
                    if len(mapped):
                        mb[0] = min(mb[0], int(mapped[0]))
                        mb[-1] = max(mb[-1], int(mapped_end.max()))
                    for seg, d in zip(w["timeline"], np.diff(mb).tolist()):
                        seg[1] = d
                    w["begin"], w["end"] = int(mb[0]), int(mb[-1])
                    doc["duration"] = w["end"] - w["begin"]
                    (out / fname).write_text(json.dumps(doc))
                    entries[slot][index] = [fname, w["begin"], w["end"]]
            names["wave_filenames"].setdefault(se_key, {})[str(simd)] = entries
    # Wave identity per merged wave file (extract.py's marker matching), for audit.py.
    provenance["wave_uids"] = {
        w["file"]: w["uid"]
        for simd, cell in anchors.items()
        for e in cell["engines"].values()
        for w in e["detailed"]
    }
    (out / "filenames.json").write_text(json.dumps(names))
    (out / "code.json").write_text(json.dumps(code))
    (out / "merge_provenance.json").write_text(json.dumps(provenance, indent=2))
    missing = [
        f
        for se in names["wave_filenames"].values()
        for s in se.values()
        for sl in s.values()
        for f, _, _ in sl.values()
        if not (out / f).is_file()
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} referenced wave files missing")
    print(
        json.dumps(
            dict(
                output=str(out),
                simds_per_engine={
                    k: sorted(v) for k, v in names["wave_filenames"].items()
                }["0"],
                wave_files=sum(
                    len(sl)
                    for se in names["wave_filenames"].values()
                    for s in se.values()
                    for sl in s.values()
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
