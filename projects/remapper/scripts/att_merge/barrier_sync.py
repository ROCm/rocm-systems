#!/usr/bin/env python3
"""Same-role barrier completion spread in shader clocks, across the four replay SIMDs."""

import argparse
import collections
import json
from pathlib import Path
import numpy as np


def releases(ui, waves_per_workgroup=8):
    code = {
        r[2]: r[0]
        for r in json.loads((ui / "code.json").read_text())["code"]
        if not r[0].startswith(";")
    }
    uids = json.loads((ui / "merge_provenance.json").read_text())["wave_uids"]
    rel = collections.defaultdict(list)
    for f in sorted(ui.glob("se*_sm*_sl*_wv*.json")):
        uid = uids.get(f.name)
        if uid is None:
            continue
        w = json.loads(f.read_text())["wave"]
        n = cluster_n = 0
        for x in w["instructions"]:
            text = code.get(x[4], "")
            if not text.startswith("s_barrier_wait"):
                continue
            kind = "cluster" if "0xfffd" in text or " -3" in text else "workgroup"
            phase = "prologue" if kind == "cluster" and cluster_n < 2 else "loop"
            key = (
                f.name.split("_")[0],
                uid // waves_per_workgroup,
                (uid % waves_per_workgroup) // 4,
                n,
                kind,
                phase,
            )
            rel[key].append((x[0] + x[3], w["simd"]))
            n += 1
            cluster_n += kind == "cluster"
    return rel


def summarize(dirs, waves_per_workgroup=8):
    pooled = collections.defaultdict(list)
    incomplete = 0
    for ui in dirs:
        for (_, _, role, _, kind, phase), values in releases(
            Path(ui), waves_per_workgroup
        ).items():
            if len(values) != 4 or len({s for _, s in values}) != 4:
                incomplete += 1
                continue
            label = f"{phase}_{kind}_role_{role}"
            ends = [t for t, _ in values]
            pooled[label].append(max(ends) - min(ends))
    stats = {
        k: dict(
            n=len(v),
            median=float(np.median(v)),
            p90=float(np.percentile(v, 90)),
            p99=float(np.percentile(v, 99)),
            max=float(max(v)),
        )
        for k, v in pooled.items()
    }
    return dict(groups=stats, incomplete_groups=incomplete)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--waves-per-workgroup", type=int, default=8)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    result = summarize(args.dirs, args.waves_per_workgroup)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
