#!/usr/bin/env python3
"""Compare a probe trace with rocprofv3 --hip-trace, both in hip_api_trace.csv format.

    compare_rocprofv3.py <rocprofv3.csv> <probes.csv> [--allow-extra=api,api] [--slack-ns=N]

Both traces come from the same run (rocprofv3 and the LTTng session record the same process,
with the same clock), so every call should appear on both sides. rocprofv3's wrapper is the
outermost dispatch-table entry and calls the probe, so a rocprofv3 call's interval contains its
probe call's. Per (process, thread), each rocprofv3 call is matched to the earliest unmatched
probe call with the same function whose interval lies inside it (within --slack-ns). Then:

  domain           must be equal
  timing           start delta (probe - rocprofv3), end delta (rocprofv3 - probe), duration delta
  inside rocprofv3 a probe call inside a rocprofv3 call's interval that is not its match was made
                   by rocprofv3 itself through the HIP table (it does not trace its own calls);
                   reported per enclosing API, not a failure
  before rocprofv3 a probe call that starts before rocprofv3's first call in its domain: other
                   threads can enter the table while the first call is still initializing HIP,
                   before rocprofv3 wraps it; reported, not a failure
  after rocprofv3  a probe call that starts after rocprofv3's last call in the process: library
                   teardown at exit, after rocprofv3 has finalized; reported, not a failure
  only in one side everything else

Exit status 0 when every rocprofv3 call is matched and every other probe call is either inside a
rocprofv3 call or an --allow-extra API, with equal domains; 1 otherwise.
"""
import bisect
import csv
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
only_ref, only_probe, inside_ref = [], [], []
for thread in sorted(set(ref) | set(probe)):
    a, b = ref.get(thread, []), probe.get(thread, [])
    by_name = defaultdict(list)  # name -> probe calls in start order
    for c in b:
        by_name[c["name"]].append(c)
    starts = {n: [c["start"] for c in cs] for n, cs in by_name.items()}
    matched = set()
    for r in a:
        cs = by_name.get(r["name"], [])
        i = bisect.bisect_left(starts.get(r["name"], []), r["start"] - slack)
        while i < len(cs) and (id(cs[i]) in matched or cs[i]["end"] > r["end"] + slack):
            if cs[i]["start"] > r["end"] + slack:
                break
            i += 1
        if i < len(cs) and id(cs[i]) not in matched and cs[i]["start"] <= r["end"] + slack \
                and cs[i]["end"] <= r["end"] + slack:
            matched.add(id(cs[i]))
            pairs.append((thread, r, cs[i]))
        else:
            only_ref.append((thread, r))
    ref_starts = [r["start"] for r in a]
    for c in b:
        if id(c) in matched:
            continue
        j = bisect.bisect_right(ref_starts, c["start"]) - 1
        if j >= 0 and c["end"] <= a[j]["end"]:
            inside_ref.append((thread, c, a[j]))
        else:
            only_probe.append((thread, c))

# Calls that started before rocprofv3 recorded anything in the process: other threads can enter
# the HIP table while the first call is still initializing HIP, before rocprofv3 has wrapped it.
# rocprofv3 wraps each dispatch table (domain) when the runtime hands it over, so the cut-off is
# per process and domain.
first_ref = {}
for (pid, _), calls in ref.items():
    for c in calls:
        key = (pid, c["domain"])
        first_ref[key] = min(first_ref.get(key, c["start"]), c["start"])


def before_first(t, c):
    return c["start"] < first_ref.get((t[0], c["domain"]), c["start"] + 1)


before_ref = [(t, c) for t, c in only_probe if before_first(t, c)]
only_probe = [(t, c) for t, c in only_probe if not before_first(t, c)]

# Calls that started after rocprofv3's last recorded call in the process: static destructors and
# library teardown at exit, after rocprofv3 has finalized.
last_ref = {}
for (pid, _), calls in ref.items():
    for c in calls:
        last_ref[pid] = max(last_ref.get(pid, c["end"]), c["end"])
after_ref = [(t, c) for t, c in only_probe if c["start"] > last_ref.get(t[0], c["start"])]
only_probe = [(t, c) for t, c in only_probe if c["start"] <= last_ref.get(t[0], c["start"])]

domain_diff = [(t, r, p) for t, r, p in pairs if r["domain"] != p["domain"]]
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
inside_by_api = defaultdict(int)
for _, c, r in inside_ref:
    inside_by_api[f"{c['name']} inside {r['name']}"] += 1
print(f"probe calls before rocprofv3's first call (it had not wrapped the table yet): "
      f"{len(before_ref)}" + "".join(f" [tid {t[1]} {c['name']}]" for t, c in before_ref[:10]))
after_by_api = defaultdict(int)
for _, c in after_ref:
    after_by_api[c["name"]] += 1
print(f"probe calls after rocprofv3's last call (exit, after it finalized): {len(after_ref)}"
      + "".join(f"\n  {n:6d} x {k}" for k, n in sorted(after_by_api.items())))
print(f"probe calls made by rocprofv3 itself (inside its calls): {len(inside_ref)}"
      + "".join(f"\n  {n:6d} x {k}" for k, n in sorted(inside_by_api.items())))
print(f"domain differs: {len(domain_diff)}   correlation ids equal: {ids_equal} of {len(pairs)}")
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
ok = not only_ref and not unexpected_probe and not domain_diff
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
