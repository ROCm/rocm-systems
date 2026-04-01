#!/usr/bin/env python3
"""
Step 4b: Determine which formula is correct under NaN mismatch.

Approach:
  1. Use synthetic_clean (no NaN) to establish
        GROUND TRUTH = SUM(A)/SUM(B) over all rows
  2. Introduce NaN patterns (aligned, mismatched, random)
  3. Compute 4 formulas on each NaN variant:
     - AVG(A/B)              : current behavior
     - SUM(A)/SUM(B)         : nansum — sums over different subsets if NaN mismatch
     - AVG(A)/AVG(B)         : nanmean — averages over different subsets if NaN mismatch
     - PAIRWISE SUM(A)/SUM(B): only include rows where BOTH A and B are non-NaN
  4. Compare each formula's error vs ground truth
"""

import numpy as np
import pandas as pd

np.random.seed(42)


# ============================================================================
# Synthetic data (same as step4 but reproduced here for clarity)
# ============================================================================


def make_base_data():
    """10 dispatches with intentional large variation between dispatches."""
    return pd.DataFrame({
        "TCC_HIT_sum": [50, 200, 10, 800, 30, 500, 15, 900, 40, 300],
        "TCC_MISS_sum": [950, 100, 90, 200, 970, 50, 85, 100, 960, 200],
        "TCC_EA0_RDREQ_LEVEL_sum": [
            5000,
            12000,
            3000,
            80000,
            4000,
            25000,
            2000,
            90000,
            5000,
            15000,
        ],
        "TCC_EA0_RDREQ_sum": [100, 200, 50, 1000, 80, 500, 40, 1200, 90, 300],
        "SQ_INSTS": [1000, 5000, 500, 30000, 800, 15000, 400, 35000, 900, 8000],
        "SQ_BUSY_CU_CYCLES": [
            2000,
            4000,
            1000,
            25000,
            1500,
            12000,
            800,
            28000,
            1800,
            6000,
        ],
        "SQ_WAVES": [10, 50, 5, 300, 8, 150, 4, 350, 9, 80],
        "TCP_TOTAL_CACHE_ACCESSES_sum": [
            1000,
            3000,
            500,
            20000,
            800,
            10000,
            400,
            25000,
            900,
            5000,
        ],
        "TCP_TCC_READ_REQ_sum": [200, 100, 300, 500, 600, 200, 350, 400, 550, 150],
    })


# Metric definitions: (name, numerator_expr, denominator_expr, scale)
METRICS = [
    ("L2 Hit Rate", "TCC_HIT_sum", lambda d: d["TCC_HIT_sum"] + d["TCC_MISS_sum"], 100),
    ("L2-Fabric Rd Lat", "TCC_EA0_RDREQ_LEVEL_sum", "TCC_EA0_RDREQ_sum", 1),
    ("IPC", "SQ_INSTS", "SQ_BUSY_CU_CYCLES", 1),
    ("Insts/Wave", "SQ_INSTS", "SQ_WAVES", 1),
    (
        "vL1D Hit Rate",
        lambda d: d["TCP_TOTAL_CACHE_ACCESSES_sum"] - d["TCP_TCC_READ_REQ_sum"],
        "TCP_TOTAL_CACHE_ACCESSES_sum",
        100,
    ),
]


def get_series(df, spec):
    """Resolve a column name or lambda to a Series."""
    if callable(spec):
        return spec(df).astype(float)
    return df[spec].astype(float)


def compute_4_formulas(num, denom, scale):
    """Compute all 4 formula variants on num/denom Series."""
    # 1. AVG(A/B)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_dispatch = scale * num / denom
    avg_of_ratio = np.nanmean(per_dispatch)

    # 2. SUM(A)/SUM(B) — nansum
    s_num, s_denom = np.nansum(num), np.nansum(denom)
    sum_over_sum = scale * s_num / s_denom if s_denom != 0 else np.nan

    # 3. AVG(A)/AVG(B) — nanmean
    a_num, a_denom = np.nanmean(num), np.nanmean(denom)
    avg_over_avg = scale * a_num / a_denom if a_denom != 0 else np.nan

    # 4. PAIRWISE-COMPLETE SUM(A)/SUM(B)
    valid = num.notna() & denom.notna()
    if valid.sum() > 0:
        pc_num = num[valid].sum()
        pc_denom = denom[valid].sum()
        pairwise = scale * pc_num / pc_denom if pc_denom != 0 else np.nan
    else:
        pairwise = np.nan

    n_valid_both = int(valid.sum())
    n_valid_num = int(num.notna().sum())
    n_valid_denom = int(denom.notna().sum())

    return {
        "AVG(A/B)": avg_of_ratio,
        "SUM/SUM": sum_over_sum,
        "AVG/AVG": avg_over_avg,
        "PAIRWISE": pairwise,
        "n_valid_num": n_valid_num,
        "n_valid_denom": n_valid_denom,
        "n_valid_both": n_valid_both,
    }


# ============================================================================
# NaN injection patterns
# ============================================================================


def inject_nan(df, pattern_name):
    """Return a copy of df with NaN injected according to the named pattern."""
    out = df.copy()
    num_cols = [
        "TCC_HIT_sum",
        "TCC_EA0_RDREQ_LEVEL_sum",
        "SQ_INSTS",
        # TCP_TOTAL and TCP_TCC_READ are both used in vL1D hit rate — treat as num
    ]
    denom_cols = [
        "TCC_MISS_sum",
        "TCC_EA0_RDREQ_sum",
        "SQ_BUSY_CU_CYCLES",
        "SQ_WAVES",
        "TCP_TOTAL_CACHE_ACCESSES_sum",
    ]

    if pattern_name == "aligned":
        # Same rows NaN for both num and denom
        for c in num_cols + denom_cols + ["TCP_TCC_READ_REQ_sum"]:
            if c in out.columns:
                out.loc[[8, 9], c] = np.nan

    elif pattern_name == "mismatch_trailing":
        # Num NaN in last 3 rows, denom NaN in rows 5-6
        for c in num_cols:
            if c in out.columns:
                out.loc[[7, 8, 9], c] = np.nan
        out.loc[[7, 8, 9], "TCP_TCC_READ_REQ_sum"] = np.nan
        for c in denom_cols:
            if c in out.columns:
                out.loc[[5, 6], c] = np.nan

    elif pattern_name == "mismatch_interleaved":
        # Odd rows NaN for num, even rows NaN for denom (extreme case)
        for c in num_cols:
            if c in out.columns:
                out.loc[[1, 3, 5, 7, 9], c] = np.nan
        out.loc[[1, 3, 5, 7, 9], "TCP_TCC_READ_REQ_sum"] = np.nan
        for c in denom_cols:
            if c in out.columns:
                out.loc[[0, 2, 4, 6, 8], c] = np.nan

    elif pattern_name == "mismatch_random":
        # Random NaN: 30% chance per cell, independently for num and denom
        rng = np.random.RandomState(123)
        for c in num_cols:
            if c in out.columns:
                mask = rng.random(len(out)) < 0.3
                out.loc[mask, c] = np.nan
        mask2 = rng.random(len(out)) < 0.3
        out.loc[mask2, "TCP_TCC_READ_REQ_sum"] = np.nan
        for c in denom_cols:
            if c in out.columns:
                mask = rng.random(len(out)) < 0.3
                out.loc[mask, c] = np.nan

    elif pattern_name == "heavy_nan_mismatch":
        # 60% NaN, completely non-overlapping for num vs denom
        for c in num_cols:
            if c in out.columns:
                out.loc[[0, 1, 2, 3, 4, 5], c] = np.nan
        out.loc[[0, 1, 2, 3, 4, 5], "TCP_TCC_READ_REQ_sum"] = np.nan
        for c in denom_cols:
            if c in out.columns:
                out.loc[[6, 7, 8, 9], c] = np.nan

    return out


# ============================================================================
# Main analysis
# ============================================================================


def main():
    base = make_base_data()

    # Ground truth: SUM(A)/SUM(B) on the full clean dataset
    print("=" * 130)
    print("GROUND TRUTH (full clean data, SUM(A)/SUM(B))")
    print("=" * 130)
    ground_truth = {}
    for name, num_spec, denom_spec, scale in METRICS:
        num = get_series(base, num_spec)
        denom = get_series(base, denom_spec)
        gt = scale * num.sum() / denom.sum()
        ground_truth[name] = gt
        print(f"  {name:<25} = {gt:.6f}")

    # NaN patterns to test
    patterns = [
        "aligned",
        "mismatch_trailing",
        "mismatch_interleaved",
        "mismatch_random",
        "heavy_nan_mismatch",
    ]

    all_results = []

    for pattern in patterns:
        df_nan = inject_nan(base, pattern)
        nan_count = df_nan.isna().sum().sum()
        print(f"\n{'=' * 130}")
        print(f"PATTERN: {pattern} ({nan_count} NaN cells)")
        print(f"{'=' * 130}")
        header = (
            f"{'Metric':<25} "
            f"{'TRUTH':>10} "
            f"{'AVG(A/B)':>10} {'err%':>7} "
            f"{'SUM/SUM':>10} {'err%':>7} "
            f"{'AVG/AVG':>10} {'err%':>7} "
            f"{'PAIRWISE':>10} {'err%':>7} "
            f"{'N_num':>5} {'N_den':>5} {'N_both':>6}"
        )
        print(header)
        print("-" * len(header))

        for name, num_spec, denom_spec, scale in METRICS:
            num = get_series(df_nan, num_spec)
            denom = get_series(df_nan, denom_spec)
            results = compute_4_formulas(num, denom, scale)
            gt = ground_truth[name]

            def err_pct(val):
                if np.isnan(val) or np.isnan(gt) or gt == 0:
                    return np.nan
                return 100 * (val - gt) / abs(gt)

            row = {
                "pattern": pattern,
                "metric": name,
                "ground_truth": gt,
                **results,
                "err_AVG": err_pct(results["AVG(A/B)"]),
                "err_SUM": err_pct(results["SUM/SUM"]),
                "err_AVGAVG": err_pct(results["AVG/AVG"]),
                "err_PAIRWISE": err_pct(results["PAIRWISE"]),
            }
            all_results.append(row)

            print(
                f"{name:<25} "
                f"{gt:>10.4f} "
                f"{results['AVG(A/B)']:>10.4f} {err_pct(results['AVG(A/B)']):>+6.1f}% "
                f"{results['SUM/SUM']:>10.4f} {err_pct(results['SUM/SUM']):>+6.1f}% "
                f"{results['AVG/AVG']:>10.4f} {err_pct(results['AVG/AVG']):>+6.1f}% "
                f"{results['PAIRWISE']:>10.4f} {err_pct(results['PAIRWISE']):>+6.1f}% "
                f"{results['n_valid_num']:>5} {results['n_valid_denom']:>5} "
                f"{results['n_valid_both']:>6}"
            )

    # Summary: average absolute error across all metrics, by formula
    print(f"\n{'=' * 130}")
    print("SUMMARY: Mean Absolute Error (%) vs Ground Truth, by NaN Pattern")
    print(f"{'=' * 130}")
    rdf = pd.DataFrame(all_results)
    summary_header = (
        f"{'Pattern':<30} {'AVG(A/B)':>10} {'SUM/SUM':>10} {'AVG/AVG':>10} "
        f"{'PAIRWISE':>10}   {'Winner':<12}"
    )
    print(summary_header)
    print("-" * len(summary_header))
    for pattern in patterns:
        sub = rdf[rdf["pattern"] == pattern]
        mae = {
            "AVG(A/B)": sub["err_AVG"].abs().mean(),
            "SUM/SUM": sub["err_SUM"].abs().mean(),
            "AVG/AVG": sub["err_AVGAVG"].abs().mean(),
            "PAIRWISE": sub["err_PAIRWISE"].abs().mean(),
        }
        winner = min(mae, key=mae.get)
        print(
            f"{pattern:<30} "
            f"{mae['AVG(A/B)']:>9.2f}% "
            f"{mae['SUM/SUM']:>9.2f}% "
            f"{mae['AVG/AVG']:>9.2f}% "
            f"{mae['PAIRWISE']:>9.2f}%   "
            f"{winner}"
        )

    # Overall MAE
    print("-" * len(summary_header))
    overall = {
        "AVG(A/B)": rdf["err_AVG"].abs().mean(),
        "SUM/SUM": rdf["err_SUM"].abs().mean(),
        "AVG/AVG": rdf["err_AVGAVG"].abs().mean(),
        "PAIRWISE": rdf["err_PAIRWISE"].abs().mean(),
    }
    winner = min(overall, key=overall.get)
    print(
        f"{'OVERALL':<30} "
        f"{overall['AVG(A/B)']:>9.2f}% "
        f"{overall['SUM/SUM']:>9.2f}% "
        f"{overall['AVG/AVG']:>9.2f}% "
        f"{overall['PAIRWISE']:>9.2f}%   "
        f"{winner}"
    )

    # Save CSV
    rdf.to_csv("step4b_nan_correctness.csv", index=False)
    print("\nDetailed results saved to step4b_nan_correctness.csv")


if __name__ == "__main__":
    main()
