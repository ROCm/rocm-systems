"""Profile baseline vs patched eval_metric implementations.

Loads serialized eval_metric inputs from eval_metric_test/data/,
runs five variants under cProfile, and writes pstats .prof files.

Variants:
    baseline            -- original eval_metric from parser.py
    metric_once         -- expression-string caching (evaluate once)
    parallel_metrics    -- expressions evaluated in threads (full data)
    parallel_sharding   -- ThreadPoolExecutor dispatch sharding
    multicpu_sharding   -- ProcessPoolExecutor dispatch sharding

Usage:
    python eval_metric_test/compare_eval_metric.py [--num-shards N]

Outputs:
    eval_metric_test/results/<variant>.prof   (pstats binary)
"""

import argparse
import ast
import copy
import cProfile
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_EXTRA_PATHS = [
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "src" / "rocprof_compute_soc"),
    str(PROJECT_ROOT / "src" / "utils"),
    str(PROJECT_ROOT / "src" / "rocprof_compute_analyze" / "utils"),
]

for _path in _EXTRA_PATHS:
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import pandas as pd
import yaml

from utils.parser import (
    MetricEvaluator,
    calc_builtin_vars,
    clear_noise_clamp_warnings,
    create_empirical_peaks_dict,
    create_sys_vars,
    eval_metric,
    print_noise_clamp_summary,
    to_avg,
    to_max,
    to_median,
    to_min,
    to_quantile,
    to_std,
    to_sum,
    validate_dual_issue_metrics,
)
from utils.utils_common import SUPPORTED_FIELD

DEFAULT_NUM_SHARDS = 256

_AGG_PATTERN = re.compile(
    r"\b(to_avg|to_min|to_max|to_sum|to_median|to_std"
    r"|to_quantile|to_noise_clamp|to_concat)\s*\("
)

_SINGLE_ARG_REDUCTIONS = frozenset({
    "to_avg",
    "to_sum",
    "to_median",
    "to_std",
    "to_min",
    "to_max",
})

_STAT_REDUCTIONS: dict[str, callable] = {
    "to_avg": to_avg,
    "to_min": to_min,
    "to_max": to_max,
    "to_sum": to_sum,
    "to_median": to_median,
    "to_std": to_std,
    "to_quantile": to_quantile,
}


def _strip_outer_aggregation(
    expr: str,
) -> tuple[str | None, str, list]:
    """Separate an outer statistical reduction from its inner expression.

    Returns (agg_name, inner_expr, extra_args) when the outermost AST
    node is a call to a known reduction.  Returns (None, expr, []) for
    expressions that have no strippable outer aggregation.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None, expr, []

    body = tree.body
    if not isinstance(body, ast.Call):
        return None, expr, []
    if not isinstance(body.func, ast.Name):
        return None, expr, []

    func_name = body.func.id

    if func_name in _SINGLE_ARG_REDUCTIONS and len(body.args) == 1:
        inner_expr = ast.unparse(body.args[0])
        return func_name, inner_expr, []

    if func_name == "to_quantile" and len(body.args) == 2:
        inner_expr = ast.unparse(body.args[0])
        quantile_val = ast.literal_eval(body.args[1])
        return func_name, inner_expr, [quantile_val]

    return None, expr, []


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_eval_metric_inputs(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, dict, dict, pd.DataFrame, dict]:
    """Load all serialized eval_metric inputs."""
    raw_pmc_df = pd.read_parquet(data_dir / "raw_pmc_df.parquet")
    sys_info = pd.read_csv(data_dir / "sysinfo.csv").iloc[0]
    dfs = _load_pickle(data_dir / "dfs.pkl")
    dfs_type = _load_pickle(data_dir / "dfs_type.pkl")
    empirical_peaks_df = _load_pickle(
        data_dir / "empirical_peaks_df.pkl",
    )
    with (data_dir / "profiling_config.yaml").open() as fh:
        profiling_config = yaml.safe_load(fh)
    return (
        raw_pmc_df,
        sys_info,
        dfs,
        dfs_type,
        empirical_peaks_df,
        profiling_config,
    )


# -------------------------------------------------------------------
# Shared helpers for patched variants
# -------------------------------------------------------------------


def _collect_expressions(
    dfs: dict,
    dfs_type: dict,
) -> list[tuple[int, str, str, str]]:
    """Collect (df_id, row_id, col, expr) for all metric table cells."""
    collected: list[tuple[int, str, str, str]] = []
    for df_id, df in dfs.items():
        if dfs_type[df_id] != "metric_table":
            continue
        for row_id, row in df.iterrows():
            for col_name in df.columns:
                if col_name not in SUPPORTED_FIELD:
                    continue
                if col_name.lower() == "alias":
                    continue
                if row[col_name]:
                    collected.append((df_id, row_id, col_name, row[col_name]))
                else:
                    row[col_name] = ""
    return collected


def _prepare_eval_context(
    sys_info: pd.Series,
    empirical_peaks_df: pd.DataFrame,
    raw_pmc_df: Union[pd.DataFrame, dict],
    config: dict,
) -> tuple[dict, dict]:
    """Build sys_vars and empirical_peaks for MetricEvaluator."""
    sys_vars = create_sys_vars(sys_info)
    empirical_peaks = create_empirical_peaks_dict(empirical_peaks_df)
    builtin_vars = calc_builtin_vars(raw_pmc_df, config, sys_vars)
    sys_vars.update(builtin_vars)
    return sys_vars, empirical_peaks


def _detect_dominant_aggregation(expr_string: str) -> str:
    match = _AGG_PATTERN.search(str(expr_string))
    if match:
        return match.group(1)
    return "none"


def _build_shard_views(
    raw_pmc_df: Union[pd.DataFrame, dict],
    num_shards: int,
) -> list[dict]:
    pmc_perf = raw_pmc_df.get("pmc_perf") if isinstance(raw_pmc_df, dict) else None
    if pmc_perf is None or len(pmc_perf) <= 1:
        return [raw_pmc_df]

    row_count = len(pmc_perf)
    shard_size = max(1, row_count // num_shards)
    shard_views: list[dict] = []

    for start in range(0, row_count, shard_size):
        shard_df = pmc_perf.iloc[start : start + shard_size]
        shard_pmc = {**raw_pmc_df, "pmc_perf": shard_df}
        shard_views.append(shard_pmc)

    return shard_views


def _combine_shard_results(
    aggregation_type: str,
    shard_results: list,
    shard_sizes: list[int],
) -> object:
    numeric_pairs = [
        (result, size)
        for result, size in zip(shard_results, shard_sizes)
        if isinstance(result, (int, float, np.number))
        and not (np.isscalar(result) and np.isnan(result))
    ]
    if not numeric_pairs:
        return shard_results[0] if shard_results else "N/A"

    results = [pair[0] for pair in numeric_pairs]
    sizes = [pair[1] for pair in numeric_pairs]

    if aggregation_type == "to_min":
        return min(results)
    if aggregation_type == "to_max":
        return max(results)
    if aggregation_type == "to_sum":
        return sum(results)

    total_rows = sum(sizes)
    weighted_sum = sum(size * result for size, result in zip(sizes, results))
    return weighted_sum / total_rows


# -------------------------------------------------------------------
# Variant: metric_once (expression caching)
# -------------------------------------------------------------------


def _apply_aggregation(
    agg_name: str,
    inner_result: object,
    extra_args: list,
) -> object:
    """Apply a statistical reduction to an already-evaluated result."""
    if isinstance(inner_result, str) and inner_result == "N/A":
        return "N/A"
    agg_fn = _STAT_REDUCTIONS[agg_name]
    if extra_args:
        return agg_fn(inner_result, *extra_args)
    return agg_fn(inner_result)


def eval_metric_cached(
    dfs: dict,
    dfs_type: dict,
    sys_info: pd.Series,
    empirical_peaks_df: pd.DataFrame,
    raw_pmc_df: Union[pd.DataFrame, dict],
    debug: bool,
    config: dict,
) -> None:
    """Evaluate each inner expression once; apply statistics separately."""
    sys_vars, empirical_peaks = _prepare_eval_context(
        sys_info,
        empirical_peaks_df,
        raw_pmc_df,
        config,
    )
    clear_noise_clamp_warnings()

    evaluator = MetricEvaluator(
        raw_pmc_df,
        sys_vars,
        empirical_peaks,
    )
    exprs_to_eval = _collect_expressions(dfs, dfs_type)

    inner_cache: dict[str, object] = {}
    full_cache: dict[str, object] = {}

    for df_id, row_id, col, expr in exprs_to_eval:
        agg_name, inner_expr, extra_args = _strip_outer_aggregation(
            expr,
        )

        if agg_name is not None:
            if inner_expr not in inner_cache:
                inner_cache[inner_expr] = evaluator.eval_expression(
                    inner_expr,
                )
            result = _apply_aggregation(
                agg_name,
                inner_cache[inner_expr],
                extra_args,
            )
        else:
            if expr not in full_cache:
                full_cache[expr] = evaluator.eval_expression(expr)
            result = full_cache[expr]

        dfs[df_id].loc[row_id, col] = result

    print_noise_clamp_summary()
    validate_dual_issue_metrics(
        dfs,
        dfs_type,
        sys_info,
        raw_pmc_df,
    )


# -------------------------------------------------------------------
# Variant: parallel_metrics (threaded expression evaluation)
# -------------------------------------------------------------------


def _evaluate_expression_batch(
    evaluator: MetricEvaluator,
    batch: list[tuple[int, str, str, str]],
) -> list[tuple[int, str, str, object]]:
    """Evaluate a batch of expressions; return (df_id, row_id, col, result)."""
    return [
        (df_id, row_id, col, evaluator.eval_expression(expr))
        for df_id, row_id, col, expr in batch
    ]


def eval_metric_parallel_metrics(
    dfs: dict,
    dfs_type: dict,
    sys_info: pd.Series,
    empirical_peaks_df: pd.DataFrame,
    raw_pmc_df: Union[pd.DataFrame, dict],
    debug: bool,
    config: dict,
    *,
    max_workers: int = 8,
) -> None:
    """Evaluate metric expressions in parallel threads over full data."""
    sys_vars, empirical_peaks = _prepare_eval_context(
        sys_info,
        empirical_peaks_df,
        raw_pmc_df,
        config,
    )
    clear_noise_clamp_warnings()

    exprs_to_eval = _collect_expressions(dfs, dfs_type)

    if not exprs_to_eval:
        return

    batch_size = max(1, len(exprs_to_eval) // max_workers)
    batches = [
        exprs_to_eval[i : i + batch_size]
        for i in range(0, len(exprs_to_eval), batch_size)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _evaluate_expression_batch,
                MetricEvaluator(
                    raw_pmc_df,
                    sys_vars,
                    empirical_peaks,
                ),
                batch,
            )
            for batch in batches
        ]
        all_results = [f.result() for f in futures]

    for batch_results in all_results:
        for df_id, row_id, col, result in batch_results:
            dfs[df_id].loc[row_id, col] = result

    print_noise_clamp_summary()
    validate_dual_issue_metrics(
        dfs,
        dfs_type,
        sys_info,
        raw_pmc_df,
    )


# -------------------------------------------------------------------
# Variant: parallel_sharding (ThreadPoolExecutor)
# -------------------------------------------------------------------


def _evaluate_shard_expressions(
    evaluator: MetricEvaluator,
    expressions: list[tuple[int, str, str, str]],
) -> list[tuple[int, str, str, object]]:
    """Evaluate all expressions against one shard; return results."""
    return [
        (df_id, row_id, col, evaluator.eval_expression(expr))
        for df_id, row_id, col, expr in expressions
    ]


def eval_metric_parallel_sharding(
    dfs: dict,
    dfs_type: dict,
    sys_info: pd.Series,
    empirical_peaks_df: pd.DataFrame,
    raw_pmc_df: Union[pd.DataFrame, dict],
    debug: bool,
    config: dict,
    *,
    num_shards: int = DEFAULT_NUM_SHARDS,
) -> None:
    """Shard pmc_perf rows, evaluate per-shard in threads."""
    sys_vars, empirical_peaks = _prepare_eval_context(
        sys_info,
        empirical_peaks_df,
        raw_pmc_df,
        config,
    )
    clear_noise_clamp_warnings()

    shard_views = _build_shard_views(raw_pmc_df, num_shards)
    shard_sizes = [
        len(sv.get("pmc_perf", pd.DataFrame())) if isinstance(sv, dict) else 0
        for sv in shard_views
    ]
    shard_evaluators = [
        MetricEvaluator(sv, sys_vars, empirical_peaks) for sv in shard_views
    ]

    exprs_to_eval = _collect_expressions(dfs, dfs_type)

    if len(shard_views) <= 1:
        for df_id, row_id, col, expr in exprs_to_eval:
            result = shard_evaluators[0].eval_expression(expr)
            dfs[df_id].loc[row_id, col] = result
    else:
        with ThreadPoolExecutor(
            max_workers=len(shard_evaluators),
        ) as executor:
            futures = [
                executor.submit(
                    _evaluate_shard_expressions,
                    evaluator,
                    exprs_to_eval,
                )
                for evaluator in shard_evaluators
            ]
            all_shard_results = [f.result() for f in futures]

        _assign_sharded_results(
            dfs,
            exprs_to_eval,
            all_shard_results,
            shard_sizes,
            shard_views,
        )

    print_noise_clamp_summary()
    validate_dual_issue_metrics(
        dfs,
        dfs_type,
        sys_info,
        raw_pmc_df,
    )


def _assign_sharded_results(
    dfs: dict,
    exprs_to_eval: list[tuple[int, str, str, str]],
    all_shard_results: list[list[tuple]],
    shard_sizes: list[int],
    shard_views: list[dict],
) -> None:
    """Combine per-shard results and assign back into dfs."""
    for expr_idx, (df_id, row_id, col, expr) in enumerate(
        exprs_to_eval,
    ):
        aggregation_type = _detect_dominant_aggregation(expr)
        per_shard = [
            all_shard_results[shard_idx][expr_idx][3]
            for shard_idx in range(len(shard_views))
        ]
        if aggregation_type == "none":
            dfs[df_id].loc[row_id, col] = per_shard[0]
        else:
            combined = _combine_shard_results(
                aggregation_type,
                per_shard,
                shard_sizes,
            )
            dfs[df_id].loc[row_id, col] = combined


# -------------------------------------------------------------------
# Variant: multicpu_sharding (ProcessPoolExecutor)
# -------------------------------------------------------------------


def _evaluate_shard_in_worker(
    shard_pmc: dict,
    sys_vars: dict,
    empirical_peaks: dict,
    expressions: list[tuple[int, str, str, str]],
) -> list[tuple[int, str, str, object]]:
    """Worker function for ProcessPoolExecutor.

    Reconstructs a MetricEvaluator in the child process and evaluates
    all expressions against the shard.
    """
    evaluator = MetricEvaluator(
        shard_pmc,
        sys_vars,
        empirical_peaks,
    )
    return [
        (df_id, row_id, col, evaluator.eval_expression(expr))
        for df_id, row_id, col, expr in expressions
    ]


def eval_metric_multicpu_sharding(
    dfs: dict,
    dfs_type: dict,
    sys_info: pd.Series,
    empirical_peaks_df: pd.DataFrame,
    raw_pmc_df: Union[pd.DataFrame, dict],
    debug: bool,
    config: dict,
    *,
    num_shards: int = DEFAULT_NUM_SHARDS,
) -> None:
    """Shard pmc_perf rows, evaluate per-shard across CPU processes."""
    sys_vars, empirical_peaks = _prepare_eval_context(
        sys_info,
        empirical_peaks_df,
        raw_pmc_df,
        config,
    )
    clear_noise_clamp_warnings()

    shard_views = _build_shard_views(raw_pmc_df, num_shards)
    shard_sizes = [
        len(sv.get("pmc_perf", pd.DataFrame())) if isinstance(sv, dict) else 0
        for sv in shard_views
    ]

    exprs_to_eval = _collect_expressions(dfs, dfs_type)

    if len(shard_views) <= 1:
        evaluator = MetricEvaluator(
            raw_pmc_df,
            sys_vars,
            empirical_peaks,
        )
        for df_id, row_id, col, expr in exprs_to_eval:
            dfs[df_id].loc[row_id, col] = evaluator.eval_expression(expr)
    else:
        with ProcessPoolExecutor(
            max_workers=len(shard_views),
        ) as executor:
            futures = [
                executor.submit(
                    _evaluate_shard_in_worker,
                    shard_view,
                    sys_vars,
                    empirical_peaks,
                    exprs_to_eval,
                )
                for shard_view in shard_views
            ]
            all_shard_results = [f.result() for f in futures]

        _assign_sharded_results(
            dfs,
            exprs_to_eval,
            all_shard_results,
            shard_sizes,
            shard_views,
        )

    print_noise_clamp_summary()
    validate_dual_issue_metrics(
        dfs,
        dfs_type,
        sys_info,
        raw_pmc_df,
    )


# -------------------------------------------------------------------
# Profiling harness
# -------------------------------------------------------------------


_VARIANT_NAMES = [
    "baseline",
    "metric_once",
    "parallel_metrics",
    "parallel_sharding",
    "multicpu_sharding",
]


def _parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        description="Profile eval_metric variants",
    )
    arg_parser.add_argument(
        "--num-shards",
        type=int,
        default=DEFAULT_NUM_SHARDS,
        help=f"Number of dispatch shards (default: {DEFAULT_NUM_SHARDS})",
    )
    arg_parser.add_argument(
        "--variant",
        type=str,
        choices=_VARIANT_NAMES,
        default=None,
        help="Run only the specified variant instead of all five",
    )
    return arg_parser.parse_args()


def _profile_variant(
    variant_name: str,
    eval_fn: callable,
    dfs: dict,
    dfs_type: dict,
    sys_info: pd.Series,
    empirical_peaks_df: pd.DataFrame,
    raw_pmc_df: Union[pd.DataFrame, dict],
    config: dict,
    results_dir: Path,
) -> float:
    """Run one eval_metric variant under cProfile; return wall time."""
    dfs_copy = copy.deepcopy(dfs)
    prof_path = results_dir / f"{variant_name}.prof"

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    eval_fn(
        dfs_copy,
        dfs_type,
        sys_info,
        empirical_peaks_df,
        raw_pmc_df,
        False,
        config,
    )
    profiler.disable()
    elapsed = time.perf_counter() - start

    profiler.dump_stats(str(prof_path))
    return elapsed


def main() -> None:
    args = _parse_args()
    num_shards = args.num_shards

    data_dir = Path(__file__).resolve().parent / "data"
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading serialized eval_metric inputs ...")
    (
        raw_pmc_df,
        sys_info,
        dfs,
        dfs_type,
        empirical_peaks_df,
        profiling_config,
    ) = load_eval_metric_inputs(data_dir)
    print(f"  raw_pmc_df: {raw_pmc_df.shape[0]} rows x {raw_pmc_df.shape[1]} cols")
    metric_table_count = sum(1 for t in dfs_type.values() if t == "metric_table")
    print(f"  dfs: {len(dfs)} tables ({metric_table_count} metric)")

    def _make_sharded_threaded(
        dfs: dict,
        dfs_type: dict,
        sys_info: pd.Series,
        empirical_peaks_df: pd.DataFrame,
        raw_pmc_df: Union[pd.DataFrame, dict],
        debug: bool,
        config: dict,
    ) -> None:
        eval_metric_parallel_sharding(
            dfs,
            dfs_type,
            sys_info,
            empirical_peaks_df,
            raw_pmc_df,
            debug,
            config,
            num_shards=num_shards,
        )

    def _make_sharded_multicpu(
        dfs: dict,
        dfs_type: dict,
        sys_info: pd.Series,
        empirical_peaks_df: pd.DataFrame,
        raw_pmc_df: Union[pd.DataFrame, dict],
        debug: bool,
        config: dict,
    ) -> None:
        eval_metric_multicpu_sharding(
            dfs,
            dfs_type,
            sys_info,
            empirical_peaks_df,
            raw_pmc_df,
            debug,
            config,
            num_shards=num_shards,
        )

    all_variants: list[tuple[str, callable]] = [
        ("baseline", eval_metric),
        ("metric_once", eval_metric_cached),
        ("parallel_metrics", eval_metric_parallel_metrics),
        ("parallel_sharding", _make_sharded_threaded),
        ("multicpu_sharding", _make_sharded_multicpu),
    ]

    if args.variant:
        variants = [(n, fn) for n, fn in all_variants if n == args.variant]
    else:
        variants = all_variants

    timing_rows: list[dict[str, object]] = []

    for variant_name, eval_fn in variants:
        print(f"\n{'=' * 60}")
        print(f"  {variant_name}")
        print(f"{'=' * 60}")

        elapsed = _profile_variant(
            variant_name,
            eval_fn,
            dfs,
            dfs_type,
            sys_info,
            empirical_peaks_df,
            raw_pmc_df,
            profiling_config,
            results_dir,
        )
        timing_rows.append({
            "variant": variant_name,
            "wall_time_seconds": round(elapsed, 6),
        })
        prof_path = results_dir / f"{variant_name}.prof"
        print(f"  time: {elapsed:.4f}s  |  profile: {prof_path}")

    baseline_row = next(
        (r for r in timing_rows if r["variant"] == "baseline"),
        None,
    )
    baseline_time = baseline_row["wall_time_seconds"] if baseline_row else None

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"{'Variant':<25s}{'Time(s)':>10s}{'Speedup':>10s}  {'Profile'}")
    print("-" * 70)
    for row in timing_rows:
        variant = row["variant"]
        wall = row["wall_time_seconds"]
        if baseline_time is not None and wall > 0:
            speedup_str = f"{baseline_time / wall:>9.2f}x"
        else:
            speedup_str = f"{'N/A':>10s}"
        prof = results_dir / f"{variant}.prof"
        print(f"{variant:<25s}{wall:>10.4f}{speedup_str}  {prof}")


if __name__ == "__main__":
    main()
