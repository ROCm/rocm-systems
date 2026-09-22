# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path

import pandas as pd

from utils import csv_compression, schema, utils_analysis
from utils.logger import console_debug, console_error


def read_rocpd_pmc_csv(workload_dir: Path, verbose: int) -> pd.DataFrame:
    """Read compressed ROCPD result artifacts produced by the current backend."""
    result_files = sorted(
        workload_dir.glob(f"results_*.csv{csv_compression.GZIP_SUFFIX}")
    )
    if not result_files:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for result_file in result_files:
        try:
            frame = pd.read_csv(result_file)
        except pd.errors.EmptyDataError:
            console_error(
                "profiling",
                f"No counter data in {result_file}. Profiling data could be corrupt.",
            )
            continue
        except csv_compression.CORRUPT_CSV_ERRORS as error:
            console_error(
                "profiling",
                f"Could not read {result_file}: {error}\n"
                "The file is truncated or corrupt. Please re-run "
                "'rocprof-compute profile'.",
            )
            continue

        if not {"Counter_Name", "Counter_Value"}.issubset(frame.columns):
            console_error(
                "analysis",
                f"{result_file} is not in the supported ROCPD format. "
                "Please re-profile this workload with a current release.",
            )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    pmc_df = utils_analysis.process_rocpd_csv(pd.concat(frames, ignore_index=True))
    utils_analysis.add_unit_counter(pmc_df)

    if verbose >= 2:
        console_debug(f"pmc_raw_data final_single_df {pmc_df.info}")
    return pmc_df


def export_pmc_data(workload_dir: Path, pmc_df: pd.DataFrame) -> Path:
    """Write the optional debug PMC export without making it an input."""
    output_path = csv_compression.compressed_name(
        workload_dir / f"{schema.PMC_PERF_FILE_PREFIX}.csv"
    )
    pmc_df.to_csv(output_path, index=False, compression="gzip")
    return output_path
