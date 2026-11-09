#!/usr/bin/env python3
"""Compare a probe trace with rocprofv3 --hip-trace, both in hip_api_trace.csv format.

    compare_rocprofv3.py <rocprofv3.csv> <probes.csv> [--allow-extra=api,api] [--slack-ns=N]

Both traces come from the same run (rocprofv3 and the LTTng session record the same process),
so every call should appear on both sides. Per (process, thread), the calls are aligned by
function name in start order, so a call one side lacks shows up as a gap and does not shift the
rest. For each aligned pair:

  domain      must be equal
  nesting     rocprofv3's wrapper is outermost, so its interval should contain the probe's:
              rocprofv3 start <= probe start and probe end <= rocprofv3 end (within --slack-ns)
  timing      start delta (probe - rocprofv3), end delta (rocprofv3 - probe), duration delta

Exit status 0 when every call is matched (apart from --allow-extra APIs) with equal domains and
correct nesting; 1 otherwise.
"""
import csv
import difflib
import statistics
import sys
from collections import defaultdict


def option(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


paths = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(paths) != 2:
    sys.exit(__doc__)
allow_extra = set(filter(None, option("allow-extra", "").split(",")))
slack = int(option("slack-ns", "0"))


def load(path):
    by_thread = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by_thread[(int(r["Process_Id"]), int(r["Thread_Id"]))].append(
            dict(domain=r["Domain"], name=r["Function"], id=int(r["Correlation_Id"]),
                 start=int(r["Start_Timestamp"]), end=int(r["End_Timestamp"])))
    for calls in by_thread.values():
        calls.sort(key=lambda c: (c["start"], -c["end"]))
    return by_thread


ref, probe = load(paths[0]), load(paths[1])

pairs = []  # (thread, rocprofv3 call, probe call)
only_ref, only_probe = [], []
for thread in sorted(set(ref) | set(probe)):
    a, b = ref.get(thread, []), probe.get(thread, [])
    sm = difflib.SequenceMatcher(a=[c["name"] for c in a], b=[c["name"] for c in b],
                                 autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pairs += [(thread, a[i], b[j]) for i, j in zip(range(i1, i2), range(j1, j2))]
        else:
            only_ref += [(thread, c) for c in a[i1:i2]]
            only_probe += [(thread, c) for c in b[j1:j2]]

domain_diff = [(t, r, p) for t, r, p in pairs if r["domain"] != p["domain"]]
not_nested = [(t, r, p) for t, r, p in pairs
              if p["start"] + slack < r["start"] or r["end"] + slack < p["end"]]
unexpected_probe = [(t, c) for t, c in only_probe if c["name"] not in allow_extra]
allowed_probe = [(t, c) for t, c in only_probe if c["name"] in allow_extra]


def stats(xs):
    xs = sorted(xs)
    return (f"min {xs[0]:>9,}  median {statistics.median(xs):>11,.0f}  p95 "
            f"{xs[int(0.95 * (len(xs) - 1))]:>9,}  max {xs[-1]:>9,}") if xs else "-"


ids_equal = sum(1 for _, r, p in pairs if r["id"] == p["id"])
print(f"threads: rocprofv3 {len(ref)}, probes {len(probe)}")
print(f"calls:   rocprofv3 {sum(map(len, ref.values()))}, probes {sum(map(len, probe.values()))}, "
      f"matched {len(pairs)}")
print(f"only in rocprofv3: {len(only_ref)}   only in probes: {len(only_probe)} "
      f"({len(allowed_probe)} allowed: {', '.join(sorted(allow_extra)) or '-'})")
print(f"domain differs: {len(domain_diff)}   not nested in rocprofv3's interval: {len(not_nested)}"
      f"   correlation ids equal: {ids_equal} of {len(pairs)}")
print("timing (ns), over matched calls:")
print(f"  start delta (probe - rocprofv3)   {stats([p['start'] - r['start'] for _, r, p in pairs])}")
print(f"  end delta   (rocprofv3 - probe)   {stats([r['end'] - p['end'] for _, r, p in pairs])}")
print(f"  duration    (rocprofv3 - probe)   "
      f"{stats([(r['end'] - r['start']) - (p['end'] - p['start']) for _, r, p in pairs])}")

per_api = defaultdict(list)
for _, r, p in pairs:
    per_api[r["name"]].append((r["end"] - r["start"]) - (p["end"] - p["start"]))
print("duration difference per API (ns, median), most frequent first:")
for name, xs in sorted(per_api.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:15]:
    print(f"  {len(xs):6d} x {name:40s} {statistics.median(xs):>10,.0f}")


def show(title, rows, fmt):
    if rows:
        print(f"{title} (first 20 of {len(rows)}):")
        for row in rows[:20]:
            print("  " + fmt(row))


show("only in rocprofv3", only_ref, lambda x: f"tid {x[0][1]} {x[1]['name']} @ {x[1]['start']}")
show("only in probes", only_probe, lambda x: f"tid {x[0][1]} {x[1]['name']} @ {x[1]['start']}")
show("domain differs", domain_diff,
     lambda x: f"{x[1]['name']}: rocprofv3 {x[1]['domain']}, probes {x[2]['domain']}")
show("not nested", not_nested,
     lambda x: f"tid {x[0][1]} {x[1]['name']}: rocprofv3 [{x[1]['start']}, {x[1]['end']}] "
               f"probe [{x[2]['start']}, {x[2]['end']}]")

ok = not only_ref and not unexpected_probe and not domain_diff and not not_nested
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
