#!/usr/bin/env python3
"""Validate the merged timeline between barriers against ATT's same-run partner-SIMD stream.

Replay j traces SIMD j in detail and also records the issue times of its partner SIMD p = j ^ 2 on the same CU
(OTHER_SIMD_INSTRUCTIONS: every LDS instruction, category 5, and vector-memory instructions, category 3, except
tensor_load_to_lds/tensor_store_from_lds). The merged view predicts those events from the replay that traced
SIMD p in detail, mapped onto replay j's clock. Per category, observed and predicted counts must match exactly;
the streams are then compared in issue order, so the i-th observed event is the i-th predicted instruction.
Usage: validate_other_simd.py TRACES_DIR EXTRACT_DIR OUT.json
"""

import collections
import glob
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import align

EXCLUDED = ("tensor_load_to_lds", "tensor_store_from_lds")


def cell_streams(directory):
    """Per SE: detailed partner-comparable instruction times of the traced SIMD, and the other-SIMD stream."""
    d = Path(directory)
    ui = Path(glob.glob(str(d / "ui_output_agent_*_dispatch_1"))[0])
    simd = int(re.search(r"simd-(\d)", d.name)[1])
    names = json.loads((ui / "filenames.json").read_text())
    text = {
        row[2]: row[0] for row in json.loads((ui / "code.json").read_text())["code"]
    }
    out = {}
    for se_key, files in names["other_simd_filenames"].items():
        se = int(se_key)
        other = collections.defaultdict(list)
        for f in files:
            for t, _, cat in json.loads((ui / f[0]).read_text())["instructions"]:
                other[cat].append(t)
        detailed = collections.defaultdict(list)
        for fname in glob.glob(str(ui / f"se{se}_sm{simd}_sl*_wv*.json")):
            w = json.loads(Path(fname).read_text())["wave"]
            if w["cu"] != align.TARGET_CU:
                continue
            for t, cat, _, _, line in w["instructions"]:
                if cat == 5 or (cat == 3 and not text[line].startswith(EXCLUDED)):
                    detailed[cat].append(t)
        out[se] = dict(
            other={c: np.sort(v) for c, v in other.items()},
            detailed={c: np.sort(v) for c, v in detailed.items()},
        )
    return d.parent.name, simd, out


def step(anchors, a, b):
    """Offset of the latest anchor at or before t: work after a release keeps that release's offset."""
    _, x, d = align.offsets(anchors, a, b)
    return (
        lambda t: t + d[np.clip(np.searchsorted(x, t, side="right") - 1, 0, len(x) - 1)]
    )


def methods(a, b):
    """Maps from replay a's clock to replay b's clock (a, b are align.events dicts)."""
    occ = set(a["occ"]) & set(b["occ"])
    bar = set(a["bar"]) & set(b["bar"])
    both_a = {
        **{("o",) + k: v for k, v in a["occ"].items()},
        **{("b",) + k: v for k, v in a["bar"].items()},
    }
    both_b = {
        **{("o",) + k: v for k, v in b["occ"].items()},
        **{("b",) + k: v for k, v in b["bar"].items()},
    }
    fns = {
        "dispatch timestamp": lambda t: t + b["dispatch"] - a["dispatch"],
        "first wave on WGP": lambda t: t + b["first"] - a["first"],
        "occupancy, traced WGP": align.warp(occ, a["occ"], b["occ"]),
        "occupancy + barriers": align.warp(
            [("o",) + k for k in occ] + [("b",) + k for k in bar], both_a, both_b
        ),
    }
    every32 = [k for k in bar if k[1] % 32 == 0]
    if bar:  # kernels without barriers align on occupancy alone
        fns["barriers, all"] = align.warp(bar, a["bar"], b["bar"])
        fns["barriers, previous release only"] = step(bar, a["bar"], b["bar"])
    if every32:
        fns["barriers, every 32"] = align.warp(every32, a["bar"], b["bar"])
    return fns


def summarize(xs):
    xs = np.asarray(xs, dtype=np.float64)
    a = np.abs(xs)
    centered = np.abs(xs - np.median(xs))
    return dict(
        n=int(xs.size),
        abs_median=float(np.median(a)),
        abs_p90=float(np.percentile(a, 90)),
        abs_p99=float(np.percentile(a, 99)),
        signed_median=float(np.median(xs)),
        abs_median_after_removing_bias=float(np.median(centered)),
        abs_p90_after_removing_bias=float(np.percentile(centered, 90)),
    )


def main():
    traces, extract, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    cells = sorted(
        p for p in traces.glob("sweep-*/cu-*-simd-*-attempt-*") if p.is_dir()
    )
    with ProcessPoolExecutor(16) as pool:
        streams = {
            (s, simd): data for s, simd, data in pool.map(cell_streams, map(str, cells))
        }
    anchors = align.load(extract)
    errors = collections.defaultdict(lambda: collections.defaultdict(list))
    checks = collections.Counter()
    gaps = []
    position = collections.defaultdict(lambda: collections.defaultdict(list))
    for (sweep, j), data in sorted(streams.items()):
        p = j ^ 2
        for se, s in data.items():
            if se not in anchors[sweep][j]["engines"]:  # sparse launch: excluded engine
                continue
            ej = align.events(anchors[sweep][j], se)
            for source_sweep in sorted(anchors):
                if (source_sweep, p) not in streams:
                    continue
                ep = align.events(anchors[source_sweep][p], se)
                fns = methods(ep, ej)
                label = "same sweep" if source_sweep == sweep else "other sweep"
                for cat, obs in s["other"].items():
                    pred = streams[(source_sweep, p)][se]["detailed"].get(
                        cat, np.array([])
                    )
                    if len(pred) != len(obs):
                        checks[f"count_mismatch_cat{cat}"] += 1
                        continue
                    checks[f"matched_streams_cat{cat}"] += 1
                    if source_sweep == sweep:
                        gaps += list(np.diff(obs))
                    for name, fn in fns.items():
                        err = fn(pred) - obs
                        errors[(label, name)][cat] += list(err)
                        if (
                            name == "barriers, previous release only"
                            and label == "same sweep"
                        ):
                            # Where in the barrier interval does the error occur?
                            bars = np.sort(
                                np.array(list(ej["bar"].values()), dtype=np.float64)
                            )
                            i = np.searchsorted(bars, obs)
                            ok = (i > 0) & (i < len(bars))
                            frac = (obs[ok] - bars[i[ok] - 1]) / (
                                bars[i[ok]] - bars[i[ok] - 1]
                            )
                            for f, e in zip(
                                np.minimum((frac * 4).astype(int), 3), np.abs(err[ok])
                            ):
                                position[cat][int(f)].append(e)
    result = dict(
        checks=dict(checks),
        observed_event_gap=summarize(gaps),
        step_error_by_quarter_of_barrier_interval={
            f"category {c}": {
                f"quarter {q + 1}": float(np.median(v)) for q, v in sorted(qs.items())
            }
            for c, qs in position.items()
        },
        methods={
            f"{label} | {name}": {
                f"category {c}": summarize(v) for c, v in sorted(cats.items())
            }
            for (label, name), cats in errors.items()
        },
    )
    out.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(result["checks"]),
        "observed gap median",
        result["observed_event_gap"]["abs_median"],
    )
    print(json.dumps(result["step_error_by_quarter_of_barrier_interval"]))
    print(
        f"{'source | alignment':42} {'LDS med':>8} {'p90':>7} {'bias':>6} {'-bias med':>9} {'p90':>6}"
        f" {'VMEM med':>9} {'p90':>7} {'bias':>6} {'-bias med':>9}"
    )
    for key, cats in result["methods"].items():
        l, v = cats.get("category 5"), cats.get("category 3")
        if l is None or v is None:
            continue
        print(
            f"{key:42} {l['abs_median']:8.0f} {l['abs_p90']:7.0f} {l['signed_median']:6.0f} "
            f"{l['abs_median_after_removing_bias']:9.0f} {l['abs_p90_after_removing_bias']:6.0f} "
            f"{v['abs_median']:9.0f} {v['abs_p90']:7.0f} {v['signed_median']:6.0f} "
            f"{v['abs_median_after_removing_bias']:9.0f}"
        )


if __name__ == "__main__":
    main()
