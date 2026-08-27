"""What maximum error would a *perfect* model show against this reference?

Every case in the reference is one measurement, not a converged mean, and the
report records how much that measurement moved between samples.  Scoring a
model against a single draw therefore charges it for the draw's own scatter as
well as for its error, and a *maximum* over 286 such comparisons is dominated
by the worst draw rather than by the worst prediction.

This puts a number on that: it draws the reference's own scatter and asks what
the scoring script would have reported for a model that predicted every case's
true mean exactly.
"""

import json, math, os, random, statistics, sys

sys.path.insert(
    0, os.path.join(os.environ["HOME"], "rocm-systems/emulation/rocjitsu/tests/corpus")
)
import meter_score

root = os.path.join(os.environ["HOME"], "rocm-systems/emulation/rocjitsu")
real = meter_score.passed_results(
    meter_score.load(
        sys.argv[2]
        if len(sys.argv) > 2
        else root + "/Testing/results/meter-real-gfx950.json"
    )
)
if len(sys.argv) < 2:
    raise SystemExit("usage: meter_noise_floor.py SCORED_JSON [REAL_JSON]")
scored = json.load(open(sys.argv[1]))

covs = []
for case in scored["cases"]:
    record = real.get(case["case_id"], {})
    cov = record.get("device_timing", {}).get("coefficient_of_variation_pct")
    if cov is not None:
        covs.append(cov / 100.0)
print(f"{len(covs)} scored cases carry a run-to-run spread in the reference")
print(
    f"  median {statistics.median(covs)*100:.1f}%   "
    f"p90 {sorted(covs)[int(0.9*(len(covs)-1))]*100:.1f}%   max {max(covs)*100:.1f}%"
)

random.seed(11)
maxima, within = [], []
for _ in range(2000):
    errs = [abs(random.gauss(1.0, c) - 1.0) for c in covs]
    maxima.append(max(errs) * 100.0)
    within.append(sum(1 for e in errs if e <= 0.17) / len(errs) * 100.0)
maxima.sort()
within.sort()
print("\na model that predicted every true mean exactly would still score:")
print(
    f"  max |error|   median {statistics.median(maxima):.1f}%   "
    f"10th pct {maxima[200]:.1f}%   90th pct {maxima[1800]:.1f}%"
)
print(
    f"  within 17%    median {statistics.median(within):.1f}%   "
    f"10th pct {within[200]:.1f}%   90th pct {within[1800]:.1f}%"
)
print(f"  P(max <= 17%) = {sum(1 for m in maxima if m <= 17.0)/len(maxima)*100:.2f}%")

# And where the shipped model sits, per case, in units of the reference's own scatter.
zs = []
for case in scored["cases"]:
    record = real.get(case["case_id"], {})
    cov = record.get("device_timing", {}).get("coefficient_of_variation_pct")
    if not cov:
        continue
    zs.append((abs(case["ratio"] - 1.0) / (cov / 100.0), case))
zs.sort(key=lambda x: -x[0])
inside = sum(1 for z, _ in zs if z <= 2.0)
print(f"\nshipped model in units of the reference's own scatter:")
print(
    f"  median {statistics.median([z for z, _ in zs]):.2f} sigma   "
    f"within 2 sigma {inside}/{len(zs)} ({inside/len(zs)*100:.0f}%)   max {zs[0][0]:.2f} sigma"
)
for z, case in zs[:5]:
    print(f"    {z:5.2f} sigma  ratio {case['ratio']:.3f}  {case['case_id'][:52]}")
