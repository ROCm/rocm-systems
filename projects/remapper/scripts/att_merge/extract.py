#!/usr/bin/env python3
"""Reduce kept ATT traces to compact per-cell event files for alignment.

Per cell (one replay tracing one SIMD of CU 0 on every shader engine), per shader engine:
  residencies: every wave's (cu, simd, slot, begin, end, uid) from occupancy, identified by its A1 marker;
  detailed:    traced waves with uid, begin/end and s_barrier_wait completion times (time + duration);
  dispatch:    the SE's dispatch event time; realtime: (shader clock, realtime counter) pairs.
All times are that SE's shader clock in that replay. Usage: extract.py TRACES_DIR OUT_DIR
Sparse launches (REMAPPER_MERGE_ENGINES=comma list): waves with an A2 marker ended in the remap prologue and are flagged
fake rather than unidentified; shader engines outside the list (their traced WGP held only fakes) are moved to
excluded_engines, with a problem recorded if any of their detailed waves was not fake. Admission launches
(REMAPPER_MERGE_DROP_FAKES=1): fake detailed waves are removed and their files listed in each engine's fake_files.
"""

import collections
import glob
import json
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

A1 = 0xA1
A2 = 0xA2
FAKE_MARKER_LAG = 512
ENGINES = os.environ.get("REMAPPER_MERGE_ENGINES")
# Admission launches: workgroups parked in the prologue (A2) share the traced WGPs with real ones; drop their waves.
DROP_FAKES = bool(os.environ.get("REMAPPER_MERGE_DROP_FAKES"))


def residencies(rows):
    """Pair begin/end rows per (cu, simd, slot) in time order."""
    open_, out = {}, []
    for time, cu, simd, slot, start, *_ in sorted(rows, key=lambda r: (r[0], -r[4])):
        key = (cu, simd, slot)
        if start:
            if key in open_:
                raise ValueError(f"nested residency at {key}")
            open_[key] = time
        else:
            out.append(
                dict(cu=cu, simd=simd, slot=slot, begin=open_.pop(key), end=time)
            )
    if open_:
        raise ValueError(f"{len(open_)} unterminated residencies")
    return out


def identify(waves, markers):
    """One A1 marker strictly inside each residency; a uid may own only one residency."""
    by_slot = collections.defaultdict(list)
    for i, w in enumerate(waves):
        by_slot[(w["cu"], w["simd"], w["slot"])].append(i)
    hits = collections.defaultdict(list)
    fakes = set()
    pending = []
    problems = collections.Counter()
    for time, value, cu, simd, slot, flags in markers:
        if value >> 24 not in (A1, A2):
            continue
        slot_waves = by_slot.get((cu, simd, slot), ())
        owners = [i for i in slot_waves if waves[i]["begin"] <= time <= waves[i]["end"]]
        if value >> 24 == A2:
            # A2 is emitted just before s_endpgm and can be reported up to ~100 clocks after the wave's end: take the
            # one residency of the slot that started before it and ended at most FAKE_MARKER_LAG earlier.
            ended = [
                i
                for i in slot_waves
                if waves[i]["begin"] < time <= waves[i]["end"] + FAKE_MARKER_LAG
            ]
            if not flags and len(ended) == 1:
                fakes.add(ended[0])
                continue
            if DROP_FAKES and not flags and ended:
                pending.append(
                    ended
                )  # the slot was reused within the lag; resolved after A1 markers below
                continue
            problems["ambiguous_or_boundary_marker"] += 1
            continue
        if (
            len(owners) != 1
            or flags
            or time in (waves[owners[0]]["begin"], waves[owners[0]]["end"])
        ):
            problems["ambiguous_or_boundary_marker"] += 1
            continue
        hits[owners[0]].append(value & 0xFFFFFF)
    # An A2 marker belongs to a residency without an A1 marker that no other A2 marker already claimed.
    while pending:
        unresolved = []
        for candidates in pending:
            left = [i for i in candidates if i not in hits and i not in fakes]
            if len(left) == 1:
                fakes.add(left[0])
            else:
                unresolved.append(candidates)
        if len(unresolved) == len(pending):
            problems["ambiguous_or_boundary_marker"] += len(unresolved)
            break
        pending = unresolved
    counts = collections.Counter(u for us in hits.values() for u in us)
    for i, w in enumerate(waves):
        us = hits.get(i, [])
        w["uid"] = us[0] if len(us) == 1 and counts[us[0]] == 1 else None
        if i in fakes:
            w["fake"] = True
            if us:
                problems["fake_and_identified_residency"] += 1
        elif w["uid"] is None:
            problems["unidentified_residency"] += 1
    return problems


def cell(directory):
    d = Path(directory)
    ui = Path(glob.glob(str(d / "ui_output_agent_*_dispatch_1"))[0])
    simd = int(re.search(r"simd-(\d)", d.name)[1])
    names = json.loads((ui / "filenames.json").read_text())
    occupancy = json.loads((ui / "occupancy.json").read_text())
    realtime = json.loads((ui / "realtime.json").read_text())
    code = json.loads((ui / "code.json").read_text())["code"]
    text = {row[2]: row[0] for row in code}
    barrier_lines = {line for line, t in text.items() if t.startswith("s_barrier_wait")}
    result = dict(cell=d.name, sweep=d.parent.name, traced_simd=simd, engines={})
    for se_key, rows in occupancy.items():
        if not se_key.isdigit():
            continue
        se = int(se_key)
        waves = residencies(rows)
        markers = []
        for name in names["shaderdata_filenames"].get(se_key, []):
            markers += json.loads((ui / name[0]).read_text())["records"]
        problems = identify(waves, markers)
        dispatch = [
            e["time"]
            for e in occupancy["events"][se_key]
            if e.get("kind") == "dispatch"
        ]
        detailed = []
        for per_simd in names["wave_filenames"].get(se_key, {}).values():
            for per_slot in per_simd.values():
                for fname, _, _ in per_slot.values():
                    w = json.loads((ui / fname).read_text())["wave"]
                    # The wave's "id" is its sequence number; the hardware slot is the file name's sl field.
                    _, file_simd, slot, _ = map(
                        int,
                        re.match(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)", fname).groups(),
                    )
                    if file_simd != simd:
                        raise ValueError(f"{fname} is not the traced SIMD {simd}")
                    w["id"] = slot
                    # The decoder can report a detailed wave's end thousands of clocks late; a slot's residencies
                    # never share a start, so match on it.
                    owners = [
                        r
                        for r in waves
                        if r["cu"] == w["cu"]
                        and r["simd"] == simd
                        and r["slot"] == w["id"]
                        and abs(r["begin"] - w["begin"]) <= 64
                        and r["end"] >= w["begin"]
                    ]
                    insts = w["instructions"]
                    detailed.append(
                        dict(
                            file=fname,
                            cu=w["cu"],
                            slot=w["id"],
                            begin=w["begin"],
                            end=w["end"],
                            uid=owners[0]["uid"] if len(owners) == 1 else None,
                            owners=len(owners),
                            fake=len(owners) == 1 and owners[0].get("fake", False),
                            num_insts=len(insts),
                            barriers=[
                                t + dur
                                for t, _, _, dur, line in insts
                                if line in barrier_lines
                            ],
                            first_inst=insts[0][0] if insts else None,
                        )
                    )
        result["engines"][se] = dict(
            residencies=waves,
            detailed=detailed,
            problems=dict(problems),
            dispatch=dispatch,
            realtime=realtime.get(f"SE{se}", []),
        )
    if ENGINES is not None:
        keep = {int(s) for s in ENGINES.split(",") if s}
        result["excluded_engines"] = {}
        for se in sorted(set(result["engines"]) - keep):
            e = result["engines"].pop(se)
            result["excluded_engines"][se] = dict(
                detailed=len(e["detailed"]),
                not_fake=sum(not w["fake"] for w in e["detailed"]),
            )
    if DROP_FAKES:
        for e in result["engines"].values():
            e["fake_files"] = [w["file"] for w in e["detailed"] if w["fake"]]
            e["detailed"] = [w for w in e["detailed"] if not w["fake"]]
    return result


def main():
    traces, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cells = sorted(
        p for p in traces.glob("sweep-*/cu-*-simd-*-attempt-*") if p.is_dir()
    )
    with ProcessPoolExecutor(min(len(cells), 16)) as pool:
        for c, r in zip(cells, pool.map(cell, map(str, cells))):
            (out / f"{c.parent.name}-{c.name}.pkl").write_bytes(pickle.dumps(r))
            det = [w for e in r["engines"].values() for w in e["detailed"]]
            probs = collections.Counter()
            for e in r["engines"].values():
                probs.update(e["problems"])
            print(
                c.parent.name,
                c.name,
                "residencies",
                sum(len(e["residencies"]) for e in r["engines"].values()),
                "detailed",
                len(det),
                "detailed_identified",
                sum(w["uid"] is not None for w in det),
                "barriers/wave",
                sorted({len(w["barriers"]) for w in det})[:6],
                dict(probs),
                flush=True,
            )


if __name__ == "__main__":
    main()
