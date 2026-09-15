#!/usr/bin/env python3
"""Pick N probe-OK hosts that share a fingerprint (8 GPUs + ionic_0..7).

Reads crusoe_node_probe.sh output files in PROBE_DIR (*.out). Prints
host1,host2,... on stdout, or exits 1 if no group is large enough.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: crusoe_pick_symmetric.py NEED PROBE_DIR", file=sys.stderr)
        return 2
    need = int(sys.argv[1])
    probe_dir = sys.argv[2]
    ok = []
    try:
        names = sorted(os.listdir(probe_dir))
    except OSError as e:
        print(f"cannot list {probe_dir}: {e}", file=sys.stderr)
        return 1
    for name in names:
        if not name.endswith(".out"):
            continue
        path = os.path.join(probe_dir, name)
        try:
            lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.startswith("OK "):
                continue
            kv = {}
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    kv[k] = v
            if kv.get("host") and kv.get("fp"):
                ok.append(kv)
    groups: dict[str, list[str]] = defaultdict(list)
    for row in ok:
        if row["host"] not in groups[row["fp"]]:
            groups[row["fp"]].append(row["host"])
    for fp, hosts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(
            f"symmetric group fp={fp[:12]} n={len(hosts)} hosts={','.join(hosts)}",
            file=sys.stderr,
        )
        if len(hosts) >= need:
            print(",".join(hosts[:need]))
            return 0
    print("no symmetric pair with 8 GPUs and ionic_0..7", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
