#!/usr/bin/env python3
"""Perturbation by phase from kept ATT traces, split by instruction address (the body keeps its addresses).

Per detailed wave:
  remap prologue  wave begin -> first instruction outside the added prologue (0 for the original kernel)
  body prologue   first body instruction -> first body s_barrier_wait completion
  K loop          first -> last body barrier, and every barrier interval
  epilogue        last body barrier -> wave end
Usage: phase.py SIDECAR OUT.json MODE=TRACES_DIR...   (modes such as original, natural, table)
"""

import json
import statistics
import sys
from pathlib import Path


def waves(traces, prologue):
    lo, hi = prologue
    for cell in sorted(Path(traces).glob("sweep-*/cu-*-simd-*-attempt-*")):
        ui = next(cell.glob("ui_output_agent_*_dispatch_*"))
        rows = json.loads((ui / "code.json").read_text())["code"]
        names = json.loads((ui / "filenames.json").read_text())
        if not rows or not names.get("wave_filenames"):
            continue  # a small unmodified launch put no workgroup on the traced WGP in this replay
        # Label rows (";" text) share the line number of the instruction before them.
        line = {r[2]: (r[0], r[5]) for r in rows if not r[0].startswith(";")}
        for per_se in names["wave_filenames"].values():
            for per_simd in per_se.values():
                for per_slot in per_simd.values():
                    for entry in per_slot.values():
                        w = json.loads((ui / entry[0]).read_text())["wave"]
                        insts = [
                            (t, dur, *line[ln])
                            for t, _, _, dur, ln in w["instructions"]
                            if ln in line
                        ]
                        body = [i for i in insts if not lo <= i[3] < hi]
                        if not body:
                            continue
                        bars = [
                            t + dur
                            for t, dur, text, _ in body
                            if text.startswith("s_barrier_wait")
                        ]
                        yield dict(
                            sweep=cell.parent.name,
                            begin=w["begin"],
                            end=w["end"],
                            body_start=body[0][0],
                            barriers=bars,
                            body_insts=len(body),
                            prologue_insts=len(insts) - len(body),
                        )


def stats(v):
    v = sorted(v)
    if not v:
        return None
    return dict(
        n=len(v),
        median=statistics.median(v),
        mean=round(statistics.mean(v), 1),
        p10=v[int(0.1 * (len(v) - 1))],
        p90=v[int(0.9 * (len(v) - 1))],
    )


def summarize(ws):
    if not ws:
        raise ValueError("no detailed waves for phase analysis")
    nb = statistics.mode([len(w["barriers"]) for w in ws])
    full = [w for w in ws if len(w["barriers"]) == nb and nb >= 2]
    intervals = [
        stats([w["barriers"][i + 1] - w["barriers"][i] for w in full])
        for i in range(nb - 1)
    ]
    return dict(
        waves=len(ws),
        waves_with_modal_barriers=len(full),
        body_barriers=nb,
        body_instructions=sorted({w["body_insts"] for w in ws})[:4],
        remap_prologue=stats([w["body_start"] - w["begin"] for w in ws]),
        body_prologue=stats([w["barriers"][0] - w["body_start"] for w in full]),
        k_loop=stats([w["barriers"][-1] - w["barriers"][0] for w in full]),
        epilogue=stats([w["end"] - w["barriers"][-1] for w in full]),
        wave=stats([w["end"] - w["begin"] for w in ws]),
        k_loop_by_sweep={
            s: statistics.median(
                w["barriers"][-1] - w["barriers"][0] for w in full if w["sweep"] == s
            )
            for s in sorted({w["sweep"] for w in full})
        },
        barrier_interval_medians=[i["median"] for i in intervals],
    )


def add_comparisons(res):
    """Median differences of every mode against each baseline present (the unmodified kernel and natural placement)."""
    phases = ("remap_prologue", "body_prologue", "k_loop", "epilogue", "wave")
    for name in ("original", "natural"):
        base = res.get(name)
        if base is None:
            continue
        for r in res.values():
            if r is base:
                continue
            r["vs_" + name] = {
                k: round(r[k]["median"] - base[k]["median"], 1)
                for k in phases
                if r[k] and base[k]
            }
            r["percent_vs_" + name] = {
                k: round(100 * (r[k]["median"] / base[k]["median"] - 1), 3)
                for k in phases
                if r[k] and base[k] and base[k]["median"]
            }
    return res


def main():
    sidecar, out = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
    res = {}
    rw = sidecar["rewrite"]
    prologue = (rw["prologue_vaddr"], rw["prologue_vaddr"] + rw["prologue_bytes"])
    for arg in sys.argv[3:]:
        mode, traces = arg.split("=", 1)
        ws = list(waves(traces, prologue))
        if not ws and mode == "original" and "launch" in sidecar:
            continue  # sparse launch: the caller reports that no unmodified workgroup reached a traced WGP
        res[mode] = summarize(ws)
    out.write_text(json.dumps(add_comparisons(res), indent=1))
    for mode, r in res.items():
        print(
            mode,
            json.dumps(
                {
                    k: (v["median"] if isinstance(v, dict) and "median" in v else v)
                    for k, v in r.items()
                    if k != "barrier_interval_medians"
                }
            ),
        )


if __name__ == "__main__":
    main()
