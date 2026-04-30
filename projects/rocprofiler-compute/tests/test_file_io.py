# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.file_io PMC dataframe construction.

Covers the flat single-index `raw_pmc` dataframe behavior in
`create_df_pmc`: per-bucket CSV files are merged horizontally with an
inner join on the row index, duplicate metadata columns are deduplicated,
and the supplemental `SQ_ACCUM_PREV_HIRES` column is renamed to a
collection-level alias (e.g. `SQ_INST_LEVEL_VMEM_ACCUM`) so that analysis
YAML formulas can reference it directly on the flat dataframe.
"""

import pandas as pd
import pytest

from utils import file_io


def _write_csv(path, data):
    pd.DataFrame(data).to_csv(path, index=False)


def _base_metadata(num_rows):
    return {
        "Dispatch_ID": list(range(1, num_rows + 1)),
        "GPU_ID": [0] * num_rows,
        "Kernel_Name": [f"kernel_{i}" for i in range(num_rows)],
        "Grid_Size": [1024] * num_rows,
        "Workgroup_Size": [64] * num_rows,
        "LDS_Per_Workgroup": [32] * num_rows,
        "Scratch_Per_Workitem": [0] * num_rows,
        "Arch_VGPR": [16] * num_rows,
        "Accum_VGPR": [0] * num_rows,
        "SGPR": [32] * num_rows,
        "Start_Timestamp": [1000 * (i + 1) for i in range(num_rows)],
        "End_Timestamp": [1000 * (i + 1) + 500 for i in range(num_rows)],
        "Kernel_ID": list(range(1, num_rows + 1)),
    }


def _make_pmc_workload(workload_dir, base_extra=None, supplemental=None):
    """Materialize a multi-bucket PMC workload directory.

    Args:
        workload_dir: Directory in which to write CSV files.
        base_extra: Optional dict of extra columns to add to ``pmc_perf.csv``.
        supplemental: Mapping ``{coll_level: extra_columns}`` describing the
            supplemental ``pmc_perf_<coll_level>.csv`` files. Each entry's
            ``extra_columns`` is merged with the shared metadata; if the
            extra columns include ``SQ_ACCUM_PREV_HIRES``, the rename
            behavior under test will trigger.
    """
    base = _base_metadata(num_rows=3)
    if base_extra:
        base.update(base_extra)
    _write_csv(workload_dir / "pmc_perf.csv", base)

    for coll_level, extra_columns in (supplemental or {}).items():
        rows = len(next(iter(extra_columns.values())))
        bucket = _base_metadata(num_rows=rows)
        bucket.update(extra_columns)
        _write_csv(workload_dir / f"pmc_perf_{coll_level}.csv", bucket)


@pytest.fixture
def pmc_workload_dir(tmp_path):
    """Materialize a workload directory with one base file and two buckets."""
    _make_pmc_workload(
        tmp_path,
        base_extra={"GRBM_GUI_ACTIVE": [10000, 20000, 30000]},
        supplemental={
            "SQ_INST_LEVEL_VMEM": {
                "SQ_INSTS_VMEM": [11, 22, 33],
                "SQ_ACCUM_PREV_HIRES": [111, 222, 333],
            },
            "SQ_LEVEL_WAVES": {
                "SQ_ACCUM_PREV_HIRES": [444, 555, 666],
            },
        },
    )
    return tmp_path


def test_create_df_pmc_returns_flat_single_index(pmc_workload_dir):
    """Merged dataframe has a flat column Index, not a MultiIndex."""
    merged = file_io.create_df_pmc(
        str(pmc_workload_dir),
        nodes=None,
        spatial_multiplexing=False,
        kernel_verbose=-1,
        verbose=0,
        config_dict={"format_rocprof_output": "csv"},
    )

    assert not isinstance(merged.columns, pd.MultiIndex)
    assert isinstance(merged.columns, pd.Index)


def test_create_df_pmc_renames_supplemental_accum_to_coll_level_alias(
    pmc_workload_dir,
):
    """Supplemental SQ_ACCUM_PREV_HIRES becomes a collection-level alias."""
    merged = file_io.create_df_pmc(
        str(pmc_workload_dir),
        nodes=None,
        spatial_multiplexing=False,
        kernel_verbose=-1,
        verbose=0,
        config_dict={"format_rocprof_output": "csv"},
    )

    assert "SQ_INST_LEVEL_VMEM_ACCUM" in merged.columns
    assert "SQ_LEVEL_WAVES_ACCUM" in merged.columns
    assert "SQ_ACCUM_PREV_HIRES" not in merged.columns

    vmem_values = sorted(merged["SQ_INST_LEVEL_VMEM_ACCUM"].tolist())
    waves_values = sorted(merged["SQ_LEVEL_WAVES_ACCUM"].tolist())
    assert vmem_values == [111, 222, 333]
    assert waves_values == [444, 555, 666]


def test_create_df_pmc_deduplicates_shared_metadata_columns(pmc_workload_dir):
    """Shared metadata columns appear exactly once after the horizontal merge."""
    merged = file_io.create_df_pmc(
        str(pmc_workload_dir),
        nodes=None,
        spatial_multiplexing=False,
        kernel_verbose=-1,
        verbose=0,
        config_dict={"format_rocprof_output": "csv"},
    )

    metadata_cols = ["Dispatch_ID", "Kernel_Name", "GPU_ID", "Start_Timestamp"]
    for col in metadata_cols:
        assert list(merged.columns).count(col) == 1, (
            f"Expected metadata column {col} to appear once, "
            f"got columns: {list(merged.columns)}"
        )


def test_create_df_pmc_preserves_base_counter_columns(pmc_workload_dir):
    """Counter columns from the base pmc_perf.csv survive the merge."""
    merged = file_io.create_df_pmc(
        str(pmc_workload_dir),
        nodes=None,
        spatial_multiplexing=False,
        kernel_verbose=-1,
        verbose=0,
        config_dict={"format_rocprof_output": "csv"},
    )

    assert "GRBM_GUI_ACTIVE" in merged.columns
    assert "SQ_INSTS_VMEM" in merged.columns
    assert sorted(merged["GRBM_GUI_ACTIVE"].tolist()) == [10000, 20000, 30000]


def test_create_df_pmc_inner_joins_mismatched_bucket_lengths(tmp_path):
    """Bucket files with mismatched row counts are inner-joined to the shortest."""
    _make_pmc_workload(
        tmp_path,
        base_extra={"GRBM_GUI_ACTIVE": [10000, 20000, 30000]},
        supplemental={
            "SQ_INST_LEVEL_VMEM": {
                "SQ_INSTS_VMEM": [11, 22],
                "SQ_ACCUM_PREV_HIRES": [111, 222],
            },
        },
    )

    merged = file_io.create_df_pmc(
        str(tmp_path),
        nodes=None,
        spatial_multiplexing=False,
        kernel_verbose=-1,
        verbose=0,
        config_dict={"format_rocprof_output": "csv"},
    )

    assert len(merged) == 2
    assert "SQ_INST_LEVEL_VMEM_ACCUM" in merged.columns
    assert "GRBM_GUI_ACTIVE" in merged.columns
