# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.rocpd_data and the analyze-time loader.

Cover the direct-from-.db analyze path that replaces the intermediate
results_*.csv flow on the rocpd profiling path, plus the per-pass
multi-host SQL merge that produces a single ``<workload>/<fbase>.db``.
"""

import sqlite3
from pathlib import Path

import pytest

from utils import rocpd_data, utils_analysis
from utils.utils_analysis import process_rocpd_csv

COUNTER_COLUMNS = (
    "agent_id",
    "guid",
    "stack_id",
    "dispatch_id",
    "pid",
    "grid_size",
    "workgroup_size",
    "lds_block_size",
    "scratch_size",
    "vgpr_count",
    "accum_vgpr_count",
    "sgpr_count",
    "kernel_name",
    "start",
    "end",
    "kernel_id",
    "counter_name",
    "value",
)

REGION_COLUMNS = (
    "category",
    "extdata",
    "pid",
    "tid",
    "stack_id",
    "guid",
    "start",
    "end",
)


def _create_synthetic_db(path: Path, rows: list[dict]) -> None:
    """Create a minimal rocpd-shaped sqlite database at ``path``."""
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "CREATE TABLE counters_collection ("
            + ", ".join(f"{col}" for col in COUNTER_COLUMNS)
            + ")"
        )
        conn.execute(
            "CREATE TABLE regions ("
            + ", ".join(f"{col}" for col in REGION_COLUMNS)
            + ")"
        )
        placeholders = ", ".join(["?"] * len(COUNTER_COLUMNS))
        conn.executemany(
            f"INSERT INTO counters_collection VALUES ({placeholders})",
            [tuple(row.get(col) for col in COUNTER_COLUMNS) for row in rows],
        )
        conn.commit()


def _row(
    *,
    pid,
    dispatch_id,
    kernel_id,
    kernel_name,
    counter_name,
    value,
    grid=64,
    wg=64,
    lds=0,
    scratch=0,
    vgpr=4,
    accum=0,
    sgpr=4,
    start=100,
    end=200,
    agent=0,
):
    return {
        "agent_id": agent,
        "guid": "abcd-1234",
        "stack_id": 0,
        "dispatch_id": dispatch_id,
        "pid": pid,
        "grid_size": grid,
        "workgroup_size": wg,
        "lds_block_size": lds,
        "scratch_size": scratch,
        "vgpr_count": vgpr,
        "accum_vgpr_count": accum,
        "sgpr_count": sgpr,
        "kernel_name": kernel_name,
        "start": start,
        "end": end,
        "kernel_id": kernel_id,
        "counter_name": counter_name,
        "value": value,
    }


@pytest.fixture
def per_host_dbs(tmp_path: Path) -> list[str]:
    """Two synthetic per-host rocpd .db files for one profiling pass.

    Mimics rocprofv3's per-host output layout under ``out/<pass>/<host>/``
    before ``merge_pass_dbs`` collapses them into one ``<workload>/<fbase>.db``.
    """
    db1 = tmp_path / "out" / "pmc_perf_0" / "host_a" / "1_results.db"
    db2 = tmp_path / "out" / "pmc_perf_0" / "host_b" / "2_results.db"
    db1.parent.mkdir(parents=True)
    db2.parent.mkdir(parents=True)

    db1_rows = [
        _row(
            pid=1,
            dispatch_id=1,
            kernel_id=1,
            kernel_name="kernel_a",
            counter_name="SQ_WAVES",
            value=10,
        ),
        _row(
            pid=1,
            dispatch_id=1,
            kernel_id=1,
            kernel_name="kernel_a",
            counter_name="SQ_INSTS_LDS",
            value=5,
        ),
        _row(
            pid=1,
            dispatch_id=2,
            kernel_id=2,
            kernel_name="kernel_b",
            counter_name="SQ_WAVES",
            value=20,
            grid=128,
            start=300,
            end=400,
        ),
        _row(
            pid=1,
            dispatch_id=2,
            kernel_id=2,
            kernel_name="kernel_b",
            counter_name="SQ_INSTS_LDS",
            value=15,
            grid=128,
            start=300,
            end=400,
        ),
    ]
    db2_rows = [
        _row(
            pid=2,
            dispatch_id=1,
            kernel_id=1,
            kernel_name="kernel_a",
            counter_name="SQ_WAVES",
            value=11,
        ),
        _row(
            pid=2,
            dispatch_id=1,
            kernel_id=1,
            kernel_name="kernel_a",
            counter_name="SQ_INSTS_LDS",
            value=6,
        ),
    ]
    _create_synthetic_db(db1, db1_rows)
    _create_synthetic_db(db2, db2_rows)
    return [str(db1), str(db2)]


def test_merge_pass_dbs_single_source_copy(tmp_path: Path):
    """One source short-circuits to a file copy with all rows preserved."""
    src = tmp_path / "src.db"
    rows = [
        _row(
            pid=1,
            dispatch_id=1,
            kernel_id=1,
            kernel_name="kernel_a",
            counter_name="C0",
            value=42,
        ),
    ]
    _create_synthetic_db(src, rows)

    dst = tmp_path / "merged.db"
    rocpd_data.merge_pass_dbs([str(src)], str(dst))

    assert dst.exists()
    assert rocpd_data.count_counter_rows([str(dst)]) == 1


def test_merge_pass_dbs_multi_source_attach_insert(per_host_dbs, tmp_path: Path):
    """Multiple sources are merged via ATTACH+INSERT preserving all rows."""
    dst = tmp_path / "merged.db"
    rocpd_data.merge_pass_dbs(per_host_dbs, str(dst))

    assert dst.exists()
    assert rocpd_data.count_counter_rows([str(dst)]) == 6

    # Both PIDs are visible after merge - rank rows are not deduplicated.
    with sqlite3.connect(str(dst)) as conn:
        pids = sorted(
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT pid FROM counters_collection"
            ).fetchall()
        )
    assert pids == [1, 2]


def test_build_workload_pmc_db_renumbers_dispatch_ids_per_pid(
    per_host_dbs, tmp_path: Path
):
    """After per-pass merge each rank's local dispatch_id collides; the
    builder renumbers via (PID, original) so process_rocpd_csv groupby
    keeps ranks separate."""
    per_pass_merged = tmp_path / "pmc_perf_0.db"
    rocpd_data.merge_pass_dbs(per_host_dbs, str(per_pass_merged))

    workload_cache = tmp_path / "pmc_perf.db"
    utils_analysis.build_workload_pmc_db([str(per_pass_merged)], str(workload_cache))

    long_df = utils_analysis.load_rocpd_pmc_df(str(tmp_path))
    assert long_df["Dispatch_ID"].nunique() == 3, (
        "Two dispatches from rank A plus one from rank B should produce 3 "
        "unique Dispatch_IDs after PID-aware renumbering"
    )


def test_build_workload_pmc_db_pivots_to_one_row_per_dispatch(
    per_host_dbs, tmp_path: Path
):
    """The wide-format pivot produces one row per dispatch with counter
    columns alongside, matching the pre-PR semantic shape."""
    per_pass_merged = tmp_path / "pmc_perf_0.db"
    rocpd_data.merge_pass_dbs(per_host_dbs, str(per_pass_merged))

    utils_analysis.build_workload_pmc_db(
        [str(per_pass_merged)], str(tmp_path / "pmc_perf.db")
    )
    long_df = utils_analysis.load_rocpd_pmc_df(str(tmp_path))
    wide_df = process_rocpd_csv(long_df)

    assert len(wide_df) == 3
    assert "SQ_WAVES" in wide_df.columns
    assert "SQ_INSTS_LDS" in wide_df.columns
    assert sorted(int(v) for v in wide_df["SQ_WAVES"].tolist()) == [10, 11, 20]


def test_build_workload_pmc_db_empty_source(tmp_path: Path):
    """An empty counters_collection source produces an empty cache table
    and load_rocpd_pmc_df returns an empty DataFrame."""
    src = tmp_path / "empty.db"
    _create_synthetic_db(src, [])

    utils_analysis.build_workload_pmc_db([str(src)], str(tmp_path / "pmc_perf.db"))
    df = utils_analysis.load_rocpd_pmc_df(str(tmp_path))
    assert df.empty


def test_load_rocpd_pmc_df_missing_cache(tmp_path: Path):
    """Without pmc_perf.db at the workload root, load_rocpd_pmc_df
    returns an empty DataFrame rather than failing."""
    df = utils_analysis.load_rocpd_pmc_df(str(tmp_path))
    assert df.empty


def test_build_workload_pmc_db_renumbers_dispatch_ids_across_passes(
    tmp_path: Path,
):
    """Cross-pass merge: Dispatch_IDs from a later pass do not collide
    with those of an earlier pass after build."""
    pass_zero = tmp_path / "pmc_perf_0.db"
    pass_one = tmp_path / "pmc_perf_1.db"
    _create_synthetic_db(
        pass_zero,
        [
            _row(
                pid=1,
                dispatch_id=1,
                kernel_id=1,
                kernel_name="k_a",
                counter_name="SQ_WAVES",
                value=10,
            ),
        ],
    )
    _create_synthetic_db(
        pass_one,
        [
            _row(
                pid=1,
                dispatch_id=1,
                kernel_id=1,
                kernel_name="k_a",
                counter_name="GRBM_GUI_ACTIVE",
                value=20,
            ),
        ],
    )

    utils_analysis.build_workload_pmc_db(
        [str(pass_zero), str(pass_one)],
        str(tmp_path / "pmc_perf.db"),
    )
    long_df = utils_analysis.load_rocpd_pmc_df(str(tmp_path))
    assert sorted(long_df["Dispatch_ID"].unique().tolist()) == [1, 2]


def test_count_counter_rows(per_host_dbs):
    """count_counter_rows sums across all provided dbs."""
    assert rocpd_data.count_counter_rows(per_host_dbs) == 6


def test_find_workload_db_paths_globs_root(tmp_path: Path):
    """find_workload_db_paths returns .db files at workload root, sorted."""
    (tmp_path / "pmc_perf_1.db").touch()
    (tmp_path / "pmc_perf_0.db").touch()
    (tmp_path / "SQ_INST_LEVEL_VMEM.db").touch()
    (tmp_path / "should_not_recurse").mkdir()
    (tmp_path / "should_not_recurse" / "ignored.db").touch()

    paths = rocpd_data.find_workload_db_paths(tmp_path)
    names = [Path(p).name for p in paths]
    assert names == sorted([
        "SQ_INST_LEVEL_VMEM.db",
        "pmc_perf_0.db",
        "pmc_perf_1.db",
    ])
