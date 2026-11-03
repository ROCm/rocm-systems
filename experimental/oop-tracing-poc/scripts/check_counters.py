import csv, glob, statistics, sys
from collections import defaultdict
base = sys.argv[1]
for rounds in (500, 2000, 10000):
    out = []
    for cfg in ("poc", "rocprofv3"):
        vals = defaultdict(list)
        for d in sorted(glob.glob(f"{base}/{cfg}-r{rounds}-*")):
            if cfg == "poc":
                for r in csv.DictReader(open(d + "/poc.csv")):
                    if "spin_fixed" in r["kernel"]:
                        vals["SQ_WAVES"].append(float(r["SQ_WAVES"]))
                        vals["GRBM_COUNT"].append(float(r["GRBM_COUNT"]))
            else:
                f = glob.glob(d + "/rp/**/*counter_collection.csv", recursive=True)
                for r in (csv.DictReader(open(f[0])) if f else []):
                    if "spin_fixed" in r["Kernel_Name"]:
                        vals[r["Counter_Name"]].append(float(r["Counter_Value"]))
        sq = sorted(set(vals["SQ_WAVES"]))
        g = vals["GRBM_COUNT"]
        out.append(f"{cfg}: {len(vals['SQ_WAVES'])} rows, SQ_WAVES distinct {sq[:4]}, "
                   f"GRBM_COUNT median {statistics.median(g) if g else 0:.0f}")
    print(f"rounds {rounds}:  " + "  |  ".join(out))
