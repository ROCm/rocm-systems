#!/usr/bin/env python3
"""Compare HIP API arguments: rocprofv3's rocpd database against the probes' calls.

    compare_args.py <rocpd.db> <calls.json> <hip_include_dir>

Both come from the same run (run_both.sh): rocprofv3 --output-format rocpd, and join.py
--calls-json. Calls are paired as in compare_rocprofv3.py: same thread and function, the probe's
interval inside rocprofv3's. Arguments are paired by name (a parameter named like a reserved
probe field gets the probe's arg_ prefix) and compared after normalizing rocprofv3's text:

  pointer   "0x..." or "(null)"            -> integer
  integer   decimal                        -> integer
  enum      enumerator without its prefix  -> value, from the HIP headers
  dim3      "{z=1, y=1, x=4096}"           -> <name>_x, <name>_y, <name>_z
  string    as is; a probe string at the cap (256 bytes) matches a longer rocprofv3 string it
            starts

rocprofv3 records the arguments at entry only; the probes' exit fields (result, outputs,
exception) have no rocprofv3 counterpart and are not compared.
"""
import bisect
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

STRING_CAP = 256
RESERVED = {"result", "tid", "timestamp", "exception"}

if len(sys.argv) != 4:
    sys.exit(__doc__)
db_path, calls_path, include_dir = sys.argv[1:]

# ------------------------------------------------------------------ enums from the HIP headers
enums = defaultdict(dict)  # enum type -> enumerator name -> value
for header in Path(include_dir).rglob("*.h"):
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", header.read_text(errors="ignore"), flags=re.S)
    for body, name in re.findall(r"typedef\s+enum\s*\w*\s*\{([^}]*)\}\s*(\w+)\s*;", text):
        value = -1
        for item in body.split(","):
            item = item.strip()
            if not item or item.startswith("#"):
                continue
            m = re.match(r"(\w+)\s*(?:=\s*(.+))?$", item, flags=re.S)
            if not m:
                continue
            if m.group(2) is not None:
                try:
                    value = int(eval(m.group(2).strip().rstrip("uUlL"), {}, {}))
                except Exception:
                    value += 1
            else:
                value += 1
            enums[name][m.group(1)] = value


def enum_value(type_name, text):
    names = enums.get(type_name.split()[-1], {})
    for enumerator, value in names.items():
        if enumerator == text or enumerator.endswith(text):
            return value
    return None


# ------------------------------------------------------------------ rocprofv3 (rocpd)
db = sqlite3.connect(db_path)
regions = defaultdict(list)  # (tid, name) -> [(start, end, [(pos, type, name, value)])]
args_by_event = defaultdict(list)
for event_id, pos, typ, name, value in db.execute(
        "select event_id, position, type, name, value from rocpd_arg order by event_id, position"):
    args_by_event[event_id].append((pos, typ, name, value))
for tid, start, end, name, event_id in db.execute(
        'select r.tid, r.start, r."end", s.string, r.event_id from rocpd_region r '
        "join rocpd_string s on s.id = r.name_id"):
    regions[(tid, name)].append((start, end, args_by_event.get(event_id, [])))
region_starts = {}
for key, lst in regions.items():
    lst.sort(key=lambda r: r[0])
    region_starts[key] = [r[0] for r in lst]

# ------------------------------------------------------------------ probes
calls = json.load(open(calls_path))


def normalize(typ, text, name):
    """rocprofv3's value text as the probe would record it: {field: value}."""
    m = re.fullmatch(r"\{(.*)\}", text)
    if m:
        out = {}
        for part in m.group(1).split(","):
            k, _, v = part.strip().partition("=")
            out[f"{name}_{k.strip()}"] = int(v)
        return out
    if text == "(null)":
        return {name: 0}
    if re.fullmatch(r"0x[0-9a-fA-F]+", text):
        return {name: int(text, 16)}
    if re.fullmatch(r"-?\d+", text):
        return {name: int(text)}
    if re.fullmatch(r"-?\d+\.\d*(e[-+]?\d+)?", text):
        return {name: float(text)}
    ev = enum_value(typ, text)
    if ev is not None:
        return {name: ev}
    return {name: text}


compared = Counter()   # api -> argument values compared
equal = Counter()
differ = defaultdict(Counter)  # (api, arg) -> kind -> count
examples = {}
renamed = Counter()  # (api, probe name, rocprofv3 name) -> count
unpaired = 0
for c in calls:
    lst = regions.get((c["tid"], c["name"]), [])
    i = bisect.bisect_right(region_starts.get((c["tid"], c["name"]), []), c["start"]) - 1
    if i < 0 or lst[i][1] < c["end"]:
        unpaired += 1
        continue
    probe_keys = list(c["args"])
    for pos, typ, name, text in lst[i][2]:
        probe_name = f"arg_{name}" if name in RESERVED else name
        values = normalize(typ, text, probe_name)
        if probe_name not in c["args"] and len(values) == 1 and \
                len(probe_keys) == len(lst[i][2]) and pos < len(probe_keys):
            # Not flattened on either side, so positions agree: compare the value under the
            # probe's name, and record the naming difference.
            renamed[(c["name"], probe_keys[pos], name)] += 1
            values = {probe_keys[pos]: next(iter(values.values()))}
        for key, ref_value in values.items():
            api = c["name"]
            compared[api] += 1
            if key not in c["args"]:
                kind = "argument name not in probe"
            else:
                got = c["args"][key]
                if got == ref_value:
                    equal[api] += 1
                    continue
                if isinstance(got, str) and isinstance(ref_value, str) and \
                        len(got.encode()) >= STRING_CAP - 1 and ref_value.startswith(got):
                    equal[api] += 1
                    continue
                if isinstance(got, int) and isinstance(ref_value, str):
                    kind = "rocprofv3 dereferences, probe records the address"
                elif isinstance(got, int) and isinstance(ref_value, int):
                    kind = "different integer"
                else:
                    kind = "different value"
            differ[(api, key)][kind] += 1
            examples.setdefault((api, key, kind),
                                f"{typ} {name}: rocprofv3 {text!r}, probe {c['args'].get(key)!r}")

total, same = sum(compared.values()), sum(equal.values())
print(f"calls: probes {len(calls)}, paired with a rocprofv3 region {len(calls) - unpaired}")
print(f"argument values compared: {total}, equal: {same} ({100 * same / max(total, 1):.2f} %)")
print(f"APIs: {len(compared)}, every argument equal: "
      f"{sum(1 for a in compared if equal[a] == compared[a])}")
if renamed:
    print("named differently (compared by position):")
    for (api, probe_name, ref_name), n in sorted(renamed.items()):
        print(f"  {api}: probe {probe_name!r}, rocprofv3 {ref_name!r} x {n}")
if differ:
    print("differences (API, argument: kind x count):")
    for (api, key), kinds in sorted(differ.items()):
        for kind, n in kinds.items():
            print(f"  {api}.{key}: {kind} x {n}")
            print(f"      e.g. {examples[(api, key, kind)]}")
