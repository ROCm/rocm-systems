#!/usr/bin/env python3
"""
Step 3a + Step 4: Compare formula variants (AVG(A/B) vs SUM(A)/SUM(B) vs AVG(A)/AVG(B))
across real profiled workloads, synthetic NaN data, and a mixed-kernel simulation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# Step 3a: Create synthetic data with NaN patterns
# ============================================================================


def create_synthetic_data():
    """
    Create a synthetic wide-format DataFrame mimicking cross-pass counter
    multiplexing where numerator and denominator counters have NaN in
    different rows (dispatches).

    Scenario: 10 dispatches. Counters A (numerator) collected in pass 1,
    counters B (denominator) collected in pass 2. After imputation,
    most rows are filled, but trailing rows retain NaN from incomplete
    subgroups.
    """
    np.random.seed(42)
    n = 10

    # Base counter values with significant variation between dispatches
    # (simulating different kernels or different input sizes)
    data = {
        "Dispatch_ID": range(n),
        "Kernel_Name": [f"kernel_{i % 3}" for i in range(n)],
        # Timestamps with varying kernel durations
        "Start_Timestamp": [1000 * i for i in range(n)],
        "End_Timestamp": [
            1000 * i + np.random.choice([100, 500, 2000]) for i in range(n)
        ],
        # L2 hit/miss with wildly varying access patterns
        "TCC_HIT_sum": [50, 200, 10, 800, 30, 500, 15, 900, 40, 300],
        "TCC_MISS_sum": [950, 100, 90, 200, 970, 50, 85, 100, 960, 200],
        # L2-Fabric latency level counters
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
        # VALU utilization counters
        "SQ_ACTIVE_INST_VALU": [
            500,
            8000,
            200,
            50000,
            300,
            20000,
            100,
            60000,
            400,
            10000,
        ],
        "GRBM_GUI_ACTIVE": [
            10000,
            10000,
            10000,
            100000,
            10000,
            50000,
            10000,
            120000,
            10000,
            30000,
        ],
        # IPC counters
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
        # Instructions per wavefront
        "SQ_WAVES": [10, 50, 5, 300, 8, 150, 4, 350, 9, 80],
        # vL1D cache
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
        # Bandwidth counter
        "TCC_REQ_sum": [500, 1500, 250, 10000, 400, 5000, 200, 12000, 450, 2500],
    }
    df_clean = pd.DataFrame(data)

    # Create NaN variants
    # Variant 1: Aligned NaN (same rows NaN for both num and denom) — pass-aligned
    df_aligned = df_clean.copy()
    df_aligned.loc[[8, 9], "TCC_HIT_sum"] = np.nan
    df_aligned.loc[[8, 9], "TCC_MISS_sum"] = np.nan
    df_aligned.loc[[8, 9], "TCC_EA0_RDREQ_LEVEL_sum"] = np.nan
    df_aligned.loc[[8, 9], "TCC_EA0_RDREQ_sum"] = np.nan

    # Variant 2: Mismatched NaN (different rows NaN for num vs denom) — cross-pass
    df_mismatch = df_clean.copy()
    # Numerator counters NaN in rows 7,8,9 (pass 1 incomplete)
    df_mismatch.loc[[7, 8, 9], "TCC_HIT_sum"] = np.nan
    df_mismatch.loc[[7, 8, 9], "TCC_EA0_RDREQ_LEVEL_sum"] = np.nan
    df_mismatch.loc[[7, 8, 9], "SQ_ACTIVE_INST_VALU"] = np.nan
    df_mismatch.loc[[7, 8, 9], "SQ_INSTS"] = np.nan
    # Denominator counters NaN in rows 5,6 (pass 2 incomplete)
    df_mismatch.loc[[5, 6], "TCC_MISS_sum"] = np.nan
    df_mismatch.loc[[5, 6], "TCC_EA0_RDREQ_sum"] = np.nan
    df_mismatch.loc[[5, 6], "GRBM_GUI_ACTIVE"] = np.nan
    df_mismatch.loc[[5, 6], "SQ_BUSY_CU_CYCLES"] = np.nan

    return {
        "synthetic_clean": df_clean,
        "synthetic_aligned_nan": df_aligned,
        "synthetic_mismatch_nan": df_mismatch,
    }


# ============================================================================
# Load real profiled data (rocpd long format -> wide format)
# ============================================================================


def load_rocpd_to_wide(csv_path):
    """Load rocpd-format pmc_perf.csv and pivot to wide format."""
    df = pd.read_csv(csv_path)
    # Pivot: rows = dispatches, columns = counter names, values = counter values
    meta_cols = [
        "Dispatch_ID",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "Grid_Size",
        "Workgroup_Size",
        "GPU_ID",
    ]
    # Keep only cols that exist
    meta_cols = [c for c in meta_cols if c in df.columns]

    # Get metadata for each dispatch (take first row per dispatch)
    meta = df.groupby("Dispatch_ID")[meta_cols].first().reset_index(drop=True)
    if "Dispatch_ID" not in meta.columns:
        meta["Dispatch_ID"] = df.groupby("Dispatch_ID").ngroup().unique()

    # Pivot counter values
    pivot = df.pivot_table(
        index="Dispatch_ID",
        columns="Counter_Name",
        values="Counter_Value",
        aggfunc="first",
    ).reset_index()

    # Merge metadata
    wide = meta.merge(pivot, on="Dispatch_ID", how="left")
    return wide


# ============================================================================
# Define metrics to compare
# ============================================================================

METRICS = [
    {
        "name": "L2 Cache Hit Rate",
        "category": "rate",
        "unit": "%",
        "num": "TCC_HIT_sum",
        "denom": "TCC_HIT_sum + TCC_MISS_sum",
        "scale": 100,
        "num_cols": ["TCC_HIT_sum"],
        "denom_cols": ["TCC_HIT_sum", "TCC_MISS_sum"],
    },
    {
        "name": "L2-Fabric Read Latency",
        "category": "latency",
        "unit": "cycles",
        "num": "TCC_EA0_RDREQ_LEVEL_sum",
        "denom": "TCC_EA0_RDREQ_sum",
        "scale": 1,
        "num_cols": ["TCC_EA0_RDREQ_LEVEL_sum"],
        "denom_cols": ["TCC_EA0_RDREQ_sum"],
    },
    {
        "name": "L2 Cache BW",
        "category": "throughput",
        "unit": "GB/s",
        "num": "TCC_REQ_sum * 128",
        "denom": "End_Timestamp - Start_Timestamp",
        "scale": 1,
        "num_cols": ["TCC_REQ_sum"],
        "denom_cols": ["End_Timestamp", "Start_Timestamp"],
        "num_transform": lambda df: df["TCC_REQ_sum"] * 128,
        "denom_transform": lambda df: df["End_Timestamp"] - df["Start_Timestamp"],
    },
    {
        "name": "IPC",
        "category": "normalized_counter",
        "unit": "insts/cycle",
        "num": "SQ_INSTS",
        "denom": "SQ_BUSY_CU_CYCLES",
        "scale": 1,
        "num_cols": ["SQ_INSTS"],
        "denom_cols": ["SQ_BUSY_CU_CYCLES"],
    },
    {
        "name": "Instructions per Wavefront",
        "category": "normalized_counter",
        "unit": "insts/wave",
        "num": "SQ_INSTS",
        "denom": "SQ_WAVES",
        "scale": 1,
        "num_cols": ["SQ_INSTS"],
        "denom_cols": ["SQ_WAVES"],
    },
    {
        "name": "vL1D Cache Hit Rate",
        "category": "rate",
        "unit": "%",
        "num_cols": ["TCP_TOTAL_CACHE_ACCESSES_sum", "TCP_TCC_READ_REQ_sum"],
        "denom_cols": ["TCP_TOTAL_CACHE_ACCESSES_sum"],
        "scale": 100,
        "num_transform": lambda df: (
            df["TCP_TOTAL_CACHE_ACCESSES_sum"] - df["TCP_TCC_READ_REQ_sum"]
        ),
        "denom_transform": lambda df: df["TCP_TOTAL_CACHE_ACCESSES_sum"],
    },
]


def compute_metric(df, metric):
    """Compute all 3 formula variants for a metric on a DataFrame."""
    # Get numerator and denominator Series
    if "num_transform" in metric:
        num_series = metric["num_transform"](df)
        denom_series = metric["denom_transform"](df)
    else:
        num_series = df[metric["num_cols"][0]].astype(float)
        if len(metric["denom_cols"]) == 1:
            denom_series = df[metric["denom_cols"][0]].astype(float)
        else:
            # Sum of denom columns
            denom_series = sum(df[c].astype(float) for c in metric["denom_cols"])

    scale = metric.get("scale", 1)

    # Per-dispatch ratio (handle zeros)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_dispatch = scale * num_series / denom_series

    # Formula 1: AVG(A/B) — current behavior
    avg_of_ratio = np.nanmean(per_dispatch)

    # Formula 2: SUM(A)/SUM(B) — proposed
    sum_num = np.nansum(num_series)
    sum_denom = np.nansum(denom_series)
    if sum_denom == 0:
        sum_over_sum = np.inf
    else:
        sum_over_sum = scale * sum_num / sum_denom

    # Formula 3: AVG(A)/AVG(B) — alternative
    avg_num = np.nanmean(num_series)
    avg_denom = np.nanmean(denom_series)
    if avg_denom == 0:
        avg_over_avg = np.inf
    else:
        avg_over_avg = scale * avg_num / avg_denom

    # MIN/MAX of per-dispatch ratio
    min_ratio = (
        np.nanmin(per_dispatch) if not np.all(np.isnan(per_dispatch)) else np.nan
    )
    max_ratio = (
        np.nanmax(per_dispatch) if not np.all(np.isnan(per_dispatch)) else np.nan
    )

    # Count NaN mismatches
    num_nan_count = num_series.isna().sum()
    denom_nan_count = denom_series.isna().sum()
    nan_mismatch = num_nan_count != denom_nan_count

    # Differences
    if sum_over_sum != 0 and not np.isinf(sum_over_sum) and not np.isnan(sum_over_sum):
        diff_avg_vs_sum = avg_of_ratio - sum_over_sum
        diff_avg_vs_sum_pct = (
            100 * diff_avg_vs_sum / abs(sum_over_sum) if sum_over_sum != 0 else np.nan
        )
        diff_avgavg_vs_sum = avg_over_avg - sum_over_sum
        diff_avgavg_vs_sum_pct = (
            100 * diff_avgavg_vs_sum / abs(sum_over_sum)
            if sum_over_sum != 0
            else np.nan
        )
    else:
        diff_avg_vs_sum = np.nan
        diff_avg_vs_sum_pct = np.nan
        diff_avgavg_vs_sum = np.nan
        diff_avgavg_vs_sum_pct = np.nan

    return {
        "metric_name": metric["name"],
        "category": metric["category"],
        "unit": metric.get("unit", ""),
        "n_dispatches": len(df),
        "n_valid_ratios": int((~np.isnan(per_dispatch)).sum()),
        "AVG(A/B)": round(avg_of_ratio, 6),
        "SUM(A)/SUM(B)": round(sum_over_sum, 6),
        "AVG(A)/AVG(B)": round(avg_over_avg, 6),
        "MIN(A/B)": round(min_ratio, 6) if not np.isnan(min_ratio) else np.nan,
        "MAX(A/B)": round(max_ratio, 6) if not np.isnan(max_ratio) else np.nan,
        "diff_AVG_vs_SUM": (
            round(diff_avg_vs_sum, 6) if not np.isnan(diff_avg_vs_sum) else np.nan
        ),
        "diff_AVG_vs_SUM_pct": (
            round(diff_avg_vs_sum_pct, 2)
            if not np.isnan(diff_avg_vs_sum_pct)
            else np.nan
        ),
        "diff_AVGAVG_vs_SUM": (
            round(diff_avgavg_vs_sum, 6) if not np.isnan(diff_avgavg_vs_sum) else np.nan
        ),
        "diff_AVGAVG_vs_SUM_pct": (
            round(diff_avgavg_vs_sum_pct, 2)
            if not np.isnan(diff_avgavg_vs_sum_pct)
            else np.nan
        ),
        "num_nan_count": int(num_nan_count),
        "denom_nan_count": int(denom_nan_count),
        "nan_mismatch": nan_mismatch,
    }


# ============================================================================
# Main
# ============================================================================


def main():
    results = []
    base = Path(".")

    # --- Load real workloads ---
    real_workloads = {
        "vcopy_uniform": base / "workloads/vcopy_uniform/MI350/pmc_perf.csv",
        "nbody_multi": base / "workloads/nbody_multi/MI350/pmc_perf.csv",
        "vcopy_small": base / "workloads/vcopy_small/MI350/pmc_perf.csv",
        "memcopy_multi": base / "workloads/memcopy_multi/MI350/pmc_perf.csv",
    }

    wide_dfs = {}
    for name, path in real_workloads.items():
        if path.exists():
            print(f"Loading {name}...")
            wide_dfs[name] = load_rocpd_to_wide(str(path))
            print(
                f"  {len(wide_dfs[name])} dispatches, "
                f"{len(wide_dfs[name].columns)} columns"
            )
        else:
            print(f"  SKIPPED (not found): {path}")

    # --- Create mixed-kernel dataset ---
    # Combine a subset of dispatches from different workloads
    mixed_parts = []
    if "vcopy_uniform" in wide_dfs:
        mixed_parts.append(wide_dfs["vcopy_uniform"].head(10))  # 10 vcopy dispatches
    if "nbody_multi" in wide_dfs:
        mixed_parts.append(wide_dfs["nbody_multi"].head(10))  # 10 nbody dispatches
    if "memcopy_multi" in wide_dfs:
        mixed_parts.append(wide_dfs["memcopy_multi"].head(10))  # 10 memcopy dispatches
    if mixed_parts:
        wide_dfs["mixed_kernels"] = pd.concat(mixed_parts, ignore_index=True)
        print(f"Created mixed_kernels: {len(wide_dfs['mixed_kernels'])} dispatches")

    # --- Create synthetic datasets (Step 3a) ---
    synthetic = create_synthetic_data()
    for name, df in synthetic.items():
        wide_dfs[name] = df
        nan_count = df.isna().sum().sum()
        print(f"Created {name}: {len(df)} dispatches, {nan_count} NaN cells")

    # --- Compute metrics for all datasets ---
    print("\nComputing formula variants...")
    for ds_name, df in wide_dfs.items():
        for metric in METRICS:
            # Check if required columns exist
            required = metric["num_cols"] + metric["denom_cols"]
            # For timestamp-based metrics
            if "Start_Timestamp" in required or "End_Timestamp" in required:
                required = [
                    c for c in required if c not in ("Start_Timestamp", "End_Timestamp")
                ]
            missing = [c for c in required if c not in df.columns]
            if missing:
                continue

            try:
                result = compute_metric(df, metric)
                result["workload"] = ds_name
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {ds_name}/{metric['name']}: {e}")

    # --- Write results CSV ---
    results_df = pd.DataFrame(results)
    col_order = [
        "workload",
        "metric_name",
        "category",
        "unit",
        "n_dispatches",
        "n_valid_ratios",
        "AVG(A/B)",
        "SUM(A)/SUM(B)",
        "AVG(A)/AVG(B)",
        "MIN(A/B)",
        "MAX(A/B)",
        "diff_AVG_vs_SUM",
        "diff_AVG_vs_SUM_pct",
        "diff_AVGAVG_vs_SUM",
        "diff_AVGAVG_vs_SUM_pct",
        "num_nan_count",
        "denom_nan_count",
        "nan_mismatch",
    ]
    results_df = results_df[[c for c in col_order if c in results_df.columns]]
    results_df.to_csv("step4_formula_comparison.csv", index=False)
    print(f"\nResults written to step4_formula_comparison.csv ({len(results_df)} rows)")

    # --- Print summary ---
    print("\n" + "=" * 120)
    print("FORMULA COMPARISON SUMMARY")
    print("=" * 120)

    for ds_name in wide_dfs:
        ds_results = results_df[results_df["workload"] == ds_name]
        if ds_results.empty:
            continue
        print(f"\n--- {ds_name} ({ds_results.iloc[0]['n_dispatches']} dispatches) ---")
        print(
            f"{'Metric':<30} {'AVG(A/B)':>12} "
            f"{'SUM/SUM':>12} {'AVG/AVG':>12} "
            f"{'Diff%':>8} {'AVG/AVG diff%':>14} {'NaN mm':>7}"
        )
        print("-" * 100)
        for _, row in ds_results.iterrows():
            print(
                f"{row['metric_name']:<30} "
                f"{row['AVG(A/B)']:>12.4f} "
                f"{row['SUM(A)/SUM(B)']:>12.4f} "
                f"{row['AVG(A)/AVG(B)']:>12.4f} "
                f"{row['diff_AVG_vs_SUM_pct']:>7.2f}% "
                f"{row['diff_AVGAVG_vs_SUM_pct']:>13.2f}% "
                f"{'YES' if row['nan_mismatch'] else 'no':>7}"
            )


if __name__ == "__main__":
    main()
