"""Serialize all eval_metric inputs for eval_metric comparison tests.

One-shot script: run once to produce serialized data files from the
mistral MI300X_A1 workload.  Outputs:

    eval_metric_test/data/raw_pmc_df.parquet
    eval_metric_test/data/sysinfo.csv
    eval_metric_test/data/dfs.pkl
    eval_metric_test/data/dfs_type.pkl
    eval_metric_test/data/empirical_peaks_df.pkl
    eval_metric_test/data/profiling_config.yaml

Usage:
    python eval_metric_test/prepare_data.py [--workload PATH]
"""

import argparse
import pickle
import shutil
import sys
from pathlib import Path

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

import pandas as pd
import yaml

from utils import file_io, parser, schema
from utils.utils_analysis import impute_counters_iteration_multiplex
from utils.utils_common import load_panel_configs

DEFAULT_NORMAL_UNIT = "per_kernel"


def _parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        description="Serialize all eval_metric inputs",
    )
    arg_parser.add_argument(
        "--workload",
        default=str(PROJECT_ROOT / "workloads" / "mistral" / "MI300X_A1"),
        help="Path to the workload directory",
    )
    return arg_parser.parse_args()


def _load_profiling_config(workload_dir: Path) -> dict:
    config_path = workload_dir / "profiling_config.yaml"
    with config_path.open() as fh:
        return yaml.safe_load(fh)


def _build_raw_pmc_df(
    workload_dir: str,
    config_dict: dict,
) -> pd.DataFrame:
    return file_io.create_df_pmc(
        raw_data_root_dir=workload_dir,
        nodes=None,
        spatial_multiplexing=False,
        kernel_verbose=0,
        verbose=0,
        config_dict=config_dict,
    )


def _load_sys_info(workload_dir: Path) -> pd.Series:
    sysinfo_df = pd.read_csv(workload_dir / "sysinfo.csv")
    return sysinfo_df.iloc[0]


def _build_arch_config_dfs(
    gpu_arch: str,
    sys_info: pd.Series,
    profiling_config: dict,
) -> tuple[dict, dict]:
    """Build dfs and dfs_type via the architecture config pipeline."""
    config_dir = str(PROJECT_ROOT / "src" / "rocprof_compute_soc" / "analysis_configs")
    arch_panel_config = [f"{config_dir}/{gpu_arch}"]

    arch_config = schema.ArchConfig()
    arch_config.panel_configs = load_panel_configs(
        arch_panel_config,
    )

    parser.build_dfs(
        arch_configs=arch_config,
        filter_metrics=None,
        sys_info=sys_info,
    )

    parser.build_metric_value_string(
        arch_config.dfs,
        arch_config.dfs_type,
        DEFAULT_NORMAL_UNIT,
        profiling_config,
    )

    return arch_config.dfs, arch_config.dfs_type


def _save_pickle(obj: object, path: Path) -> None:
    with path.open("wb") as fh:
        pickle.dump(obj, fh)
    print(f"Saved {path}")


def main() -> None:
    args = _parse_args()
    workload_dir = Path(args.workload).resolve()
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading workload from {workload_dir}")

    profiling_config = _load_profiling_config(workload_dir)

    raw_pmc_df = _build_raw_pmc_df(
        str(workload_dir),
        profiling_config,
    )
    print(f"Loaded raw_pmc_df: {raw_pmc_df.shape[0]} rows x {raw_pmc_df.shape[1]} cols")

    iteration_policy = profiling_config.get("iteration_multiplexing")
    if iteration_policy is not None:
        raw_pmc_df = impute_counters_iteration_multiplex(
            raw_pmc_df,
            iteration_policy,
        )
        print(
            f"After imputation: {raw_pmc_df.shape[0]} rows x {raw_pmc_df.shape[1]} cols"
        )

    parquet_path = data_dir / "raw_pmc_df.parquet"
    raw_pmc_df.to_parquet(parquet_path)
    print(f"Saved {parquet_path}")

    sysinfo_src = workload_dir / "sysinfo.csv"
    sysinfo_dst = data_dir / "sysinfo.csv"
    shutil.copy2(sysinfo_src, sysinfo_dst)
    print(f"Copied sysinfo.csv to {sysinfo_dst}")

    sys_info = _load_sys_info(workload_dir)
    gpu_arch = sys_info["gpu_arch"]
    print(f"Detected GPU architecture: {gpu_arch}")

    dfs, dfs_type = _build_arch_config_dfs(gpu_arch, sys_info, profiling_config)
    print(
        f"Built {len(dfs)} DataFrames"
        f" ({sum(1 for t in dfs_type.values() if t == 'metric_table')}"
        f" metric tables)"
    )

    _save_pickle(dfs, data_dir / "dfs.pkl")
    _save_pickle(dfs_type, data_dir / "dfs_type.pkl")
    _save_pickle(pd.DataFrame(), data_dir / "empirical_peaks_df.pkl")

    config_src = workload_dir / "profiling_config.yaml"
    config_dst = data_dir / "profiling_config.yaml"
    shutil.copy2(config_src, config_dst)
    print(f"Copied profiling_config.yaml to {config_dst}")


if __name__ == "__main__":
    main()
