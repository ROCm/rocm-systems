# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.file_io."""

import gzip
import tempfile

import common
import pandas as pd
import pytest

from utils.file_io import (
    build_node_to_gpu_map,
    create_df_kernel_top_stats,
    create_df_pmc,
    is_single_panel_config,
    load_kernel_short_names,
    load_node_to_gpu_map,
    rank_kernels_by_total_duration,
    validate_kernel_filter_ids,
)

KERNEL_SYMBOLS_COLUMNS = ["Kernel_Name", "Kernel_Short_Name"]

ROCPD_COUNTER_HEADER = (
    "GPU_ID,Dispatch_ID,Grid_Size,Workgroup_Size,LDS_Per_Workgroup,"
    "Scratch_Per_Workitem,Arch_VGPR,Accum_VGPR,SGPR,Kernel_Name,"
    "Start_Timestamp,End_Timestamp,Kernel_ID,Counter_Name,Counter_Value\n"
)
ROCPD_COUNTER_ROW_PREFIX = "0,0,256,64,0,0,8,0,16,kernel_a,10,20,0,"

NATIVE_COUNTERS_HEADER = "dispatch_id,counter_id,counter_name,counter_value\n"
NATIVE_DISPATCH_HEADER = (
    "dispatch_id,gpu_id,kernel_id,grid_size,workgroup_size,lds_per_workgroup,"
    "scratch_per_workitem,start_timestamp,end_timestamp,correlation_id\n"
)
NATIVE_KERNEL_SYMBOLS_CSV = (
    "kernel_id,kernel_name,kernel_short_name,arch_vgpr,accum_vgpr,sgpr\n"
    "7,kernel_a,kernel_a,8,0,16\n"
)


def write_native_process(workload_dir, fbase, pid, counters, dispatch_ids, gpu_id=0):
    """Write one process's native CSVs; counters are (dispatch_id, name, value)."""
    common.write_gzip_csv(
        workload_dir / f"counters_{fbase}_{pid}.csv.gz",
        NATIVE_COUNTERS_HEADER
        + "".join(f"{d},5,{name},{value}\n" for d, name, value in counters),
    )
    common.write_gzip_csv(
        workload_dir / f"dispatch_{fbase}_{pid}.csv.gz",
        NATIVE_DISPATCH_HEADER
        + "".join(
            f"{d},{gpu_id},7,256,64,0,0,{d * 100},{d * 100 + 50},{d + 500}\n"
            for d in dispatch_ids
        ),
    )
    common.write_gzip_csv(
        workload_dir / f"kernel_symbols_{fbase}_{pid}.csv.gz",
        NATIVE_KERNEL_SYMBOLS_CSV,
    )


def _raw_pmc() -> pd.DataFrame:
    """Flat raw_pmc DataFrame for create_df_kernel_top_stats tests."""
    return pd.DataFrame({
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a", "kernel_c"],
        "GPU_ID": [0, 0, 1, 0],
        "Dispatch_ID": [1, 2, 3, 4],
        "Start_Timestamp": [1000, 2000, 3000, 4000],
        "End_Timestamp": [1500, 2800, 3400, 4200],
    })


def make_repeated_dispatch_frame() -> pd.DataFrame:
    """Frame where the longest kernel by total time is not the longest dispatch.

    ``kernel_frequent`` runs three times for 500ns each, ``kernel_long`` once
    for 1000ns.
    """
    return pd.DataFrame({
        "Kernel_Name": [
            "kernel_long",
            "kernel_frequent",
            "kernel_frequent",
            "kernel_frequent",
        ],
        "GPU_ID": [0, 0, 0, 0],
        "Dispatch_ID": [1, 2, 3, 4],
        "Start_Timestamp": [0, 2000, 3000, 4000],
        "End_Timestamp": [1000, 2500, 3500, 4500],
    })


def test_returns_valid_dataframes() -> None:
    """create_df_kernel_top_stats returns valid DFs with correct structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        kernel_top_df, dispatch_info_df = create_df_kernel_top_stats(
            df_in=_raw_pmc(),
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=None,
            time_unit="ns",
        )

        assert isinstance(kernel_top_df, pd.DataFrame)
        assert isinstance(dispatch_info_df, pd.DataFrame)

        expected_columns = [
            "Kernel_Name",
            "Count",
            "Sum(ns)",
            "Mean(ns)",
            "Median(ns)",
            "Percent",
        ]
        for col in expected_columns:
            assert col in kernel_top_df.columns, f"Missing column: {col}"

        assert "Kernel_Name" in dispatch_info_df.columns
        assert "GPU_ID" in dispatch_info_df.columns
        assert "Dispatch_ID" in dispatch_info_df.columns

        assert kernel_top_df.index[0] == 0
        assert kernel_top_df["Percent"].sum() == pytest.approx(100.0, abs=0.01)


def test_grouping_and_aggregation() -> None:
    """Kernel grouping, aggregation functions, and sorting behavior."""
    with tempfile.TemporaryDirectory() as temp_dir:
        kernel_top_df, _ = create_df_kernel_top_stats(
            df_in=_raw_pmc(),
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=None,
            time_unit="ns",
        )

        # kernel_a appears twice in input and must group into one row.
        kernel_a_row = kernel_top_df[kernel_top_df["Kernel_Name"] == "kernel_a"]
        assert len(kernel_a_row) == 1
        assert kernel_a_row["Count"].iloc[0] == 2

        # Sorting by sum is descending.
        sum_values = kernel_top_df["Sum(ns)"].tolist()
        assert sum_values == sorted(sum_values, reverse=True)


def test_ranking_uses_total_duration_not_longest_dispatch() -> None:
    """A kernel that runs often outranks one with a single longer dispatch."""
    assert rank_kernels_by_total_duration(make_repeated_dispatch_frame()) == [
        "kernel_frequent",
        "kernel_long",
    ]


def test_kernel_top_stats_rows_follow_the_ranking() -> None:
    """Top stats rows are ordered by the ranking that -k indexes into."""
    dispatch_frame = make_repeated_dispatch_frame()
    with tempfile.TemporaryDirectory() as temp_dir:
        kernel_top_df, _ = create_df_kernel_top_stats(
            df_in=dispatch_frame,
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=None,
            time_unit="ns",
        )

        assert kernel_top_df["Kernel_Name"].tolist() == (
            rank_kernels_by_total_duration(dispatch_frame)
        )


class TestValidateKernelFilterIds:
    """Tests for utils.file_io.validate_kernel_filter_ids."""

    def test_ids_within_range_are_accepted(self, monkeypatch) -> None:
        mock_error = common.patch_console(monkeypatch, "utils.file_io", "error")[
            "error"
        ]
        validate_kernel_filter_ids([0, 2], kernel_count=3)
        mock_error.assert_not_called()

    @pytest.mark.parametrize("kernel_id", [99, -1])
    def test_id_outside_range_reports_the_valid_range(
        self, monkeypatch, kernel_id
    ) -> None:
        """Ids above the last kernel and negative ids both name no kernel."""
        mock_error = common.patch_console(monkeypatch, "utils.file_io", "error")[
            "error"
        ]

        # The mock records instead of exiting, so validation runs to completion.
        validate_kernel_filter_ids([kernel_id], kernel_count=3)

        assert f"{kernel_id} is an invalid kernel id" in mock_error.call_args.args[1]
        assert "0-2" in mock_error.call_args.args[1]

    def test_no_kernels_is_reported_as_such(self, monkeypatch) -> None:
        """A workload with no kernels reports that, not an empty range."""
        mock_error = common.patch_console(monkeypatch, "utils.file_io", "error")[
            "error"
        ]

        validate_kernel_filter_ids([0], kernel_count=0)

        assert "No kernels found" in mock_error.call_args_list[0].args[1]


def test_filters() -> None:
    """GPU ID, dispatch ID (including '> n' syntax), and empty input handling."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # GPU ID filter: GPU_ID=0 excludes kernel_a at GPU 1 (3 dispatches).
        _, dispatch_df = create_df_kernel_top_stats(
            df_in=_raw_pmc(),
            raw_data_dir=temp_dir,
            filter_gpu_ids="0",
            filter_dispatch_ids=None,
            time_unit="ns",
        )
        assert len(dispatch_df) == 3

        # Dispatch ID filter with "> n" syntax keeps IDs 3 and 4.
        _, dispatch_df = create_df_kernel_top_stats(
            df_in=_raw_pmc(),
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=["> 2"],
            time_unit="ns",
        )
        assert len(dispatch_df) == 2
        assert all(dispatch_df["Dispatch_ID"] > 2)

        # Dispatch ID filter with specific IDs.
        _, dispatch_df = create_df_kernel_top_stats(
            df_in=_raw_pmc(),
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=["1", "2"],
            time_unit="ns",
        )
        assert len(dispatch_df) == 2

        # Empty input yields empty outputs.
        empty_raw_pmc = pd.DataFrame({
            "Kernel_Name": [],
            "GPU_ID": [],
            "Dispatch_ID": [],
            "Start_Timestamp": [],
            "End_Timestamp": [],
        })
        kernel_top_df, dispatch_df = create_df_kernel_top_stats(
            df_in=empty_raw_pmc,
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=None,
            time_unit="ns",
        )
        assert len(kernel_top_df) == 0
        assert len(dispatch_df) == 0


# =============================================================================
# create_df_pmc: long-form result artifacts
# =============================================================================


def test_create_df_pmc_pivots_long_form_without_a_profiling_config(tmp_path) -> None:
    """rocpd counter rows are pivoted into one row per dispatch based on the
    shape of the data, so a workload with no profiling_config.yaml is still read
    correctly instead of being handed to the parser one counter at a time."""
    long_form_csv = (
        ROCPD_COUNTER_HEADER
        + ROCPD_COUNTER_ROW_PREFIX
        + "SQ_WAVES,4\n"
        + ROCPD_COUNTER_ROW_PREFIX
        + "SQ_BUSY_CYCLES,100\n"
    )
    common.write_gzip_csv(tmp_path / "results_pmc_perf_0.csv.gz", long_form_csv)

    df = create_df_pmc(str(tmp_path), verbose=0)

    assert len(df) == 1
    assert df["SQ_WAVES"].iloc[0] == 4
    assert df["SQ_BUSY_CYCLES"].iloc[0] == 100
    assert "Counter_Name" not in df.columns


def test_create_df_pmc_combines_result_files_of_one_workload(tmp_path) -> None:
    """Counters split across passes land on the same dispatch row."""
    common.write_gzip_csv(
        tmp_path / "results_pmc_perf_0.csv.gz",
        ROCPD_COUNTER_HEADER + ROCPD_COUNTER_ROW_PREFIX + "SQ_WAVES,4\n",
    )
    common.write_gzip_csv(
        tmp_path / "results_pmc_perf_1.csv.gz",
        ROCPD_COUNTER_HEADER + ROCPD_COUNTER_ROW_PREFIX + "SQ_BUSY_CYCLES,100\n",
    )

    df = create_df_pmc(str(tmp_path), verbose=0)

    assert len(df) == 1
    assert df["SQ_WAVES"].iloc[0] == 4
    assert df["SQ_BUSY_CYCLES"].iloc[0] == 100
    assert df["Dispatch_Unit"].iloc[0] == 1


def test_create_df_pmc_rejects_wide_result_file(tmp_path) -> None:
    """Only the rocpd long format is supported, so a wide counter file errors
    out with a re-profile message instead of being read as counter data."""
    wide_csv = (
        "GPU_ID,Dispatch_ID,Grid_Size,Workgroup_Size,LDS_Per_Workgroup,"
        "Scratch_Per_Workitem,Arch_VGPR,Accum_VGPR,SGPR,Kernel_Name,"
        "Start_Timestamp,End_Timestamp,Kernel_ID,SQ_WAVES,SQ_BUSY_CYCLES\n"
        "0,0,256,64,0,0,8,0,16,kernel_a,10,20,0,4,100\n"
    )
    common.write_gzip_csv(tmp_path / "results_pmc_perf_0.csv.gz", wide_csv)

    with pytest.raises(SystemExit):
        create_df_pmc(str(tmp_path), verbose=0)


def test_create_df_pmc_missing_file_returns_empty(tmp_path) -> None:
    assert create_df_pmc(str(tmp_path), verbose=0).empty


def test_create_df_pmc_errors_on_header_only_result_file(tmp_path) -> None:
    """A pass that recorded no dispatch is bad profiling output, not empty data."""
    common.write_gzip_csv(tmp_path / "results_pmc_perf_0.csv.gz", ROCPD_COUNTER_HEADER)

    with pytest.raises(SystemExit):
        create_df_pmc(str(tmp_path), verbose=0)


def test_create_df_pmc_errors_on_empty_pass_beside_a_good_one(tmp_path) -> None:
    """One unreadable pass fails the run rather than analyzing a partial set."""
    common.write_gzip_csv(
        tmp_path / "results_pmc_perf_0.csv.gz",
        ROCPD_COUNTER_HEADER + ROCPD_COUNTER_ROW_PREFIX + "SQ_WAVES,4\n",
    )
    (tmp_path / "results_pmc_perf_1.csv.gz").write_bytes(b"")

    with pytest.raises(SystemExit):
        create_df_pmc(str(tmp_path), verbose=0)


def test_create_df_pmc_errors_on_truncated_result_file(tmp_path) -> None:
    """A profile run killed mid-write leaves a partial gzip behind."""
    rows = "".join(f"{ROCPD_COUNTER_ROW_PREFIX}SQ_WAVES,{i}\n" for i in range(2000))
    whole = gzip.compress((ROCPD_COUNTER_HEADER + rows).encode("utf-8"))
    (tmp_path / "results_pmc_perf_0.csv.gz").write_bytes(whole[: len(whole) // 2])

    with pytest.raises(SystemExit):
        create_df_pmc(str(tmp_path), verbose=0)


# =============================================================================
# create_df_pmc: native tool artifacts
# =============================================================================


def test_create_df_pmc_prefers_native_artifacts_over_result_files(tmp_path) -> None:
    """A native run writes both lanes; the rocpd results are not read."""
    common.write_gzip_csv(
        tmp_path / "results_pmc_perf_0.csv.gz",
        ROCPD_COUNTER_HEADER + ROCPD_COUNTER_ROW_PREFIX + "SQ_WAVES,4\n",
    )
    write_native_process(tmp_path, "pmc_perf_0", 100, [(1, "SQ_WAVES", 42)], [1])

    df = create_df_pmc(str(tmp_path), verbose=0)

    assert df["SQ_WAVES"].tolist() == [42]


def test_create_df_pmc_lines_native_counter_sets_up_by_dispatch(tmp_path) -> None:
    """Counters split across passes land on the same dispatch row."""
    write_native_process(tmp_path, "pmc_perf_0", 100, [(1, "SQ_WAVES", 4)], [1])
    write_native_process(tmp_path, "pmc_perf_1", 200, [(1, "SQ_BUSY_CYCLES", 100)], [1])

    df = create_df_pmc(str(tmp_path), verbose=0)

    assert len(df) == 1
    assert df["SQ_WAVES"].iloc[0] == 4
    assert df["SQ_BUSY_CYCLES"].iloc[0] == 100


def test_create_df_pmc_keeps_native_dispatch_order_past_nine(tmp_path) -> None:
    """Ids are numbers, so dispatch 10 does not sort before dispatch 2."""
    dispatch_ids = range(1, 13)
    write_native_process(
        tmp_path,
        "pmc_perf_0",
        100,
        [(d, "SQ_WAVES", d) for d in dispatch_ids],
        dispatch_ids,
    )

    df = create_df_pmc(str(tmp_path), verbose=0)

    assert df["SQ_WAVES"].tolist() == list(dispatch_ids)


def test_create_df_pmc_errors_when_no_native_counter_joins(tmp_path) -> None:
    """Native artifacts whose counters match no dispatch are bad output."""
    write_native_process(tmp_path, "pmc_perf_0", 100, [(2, "SQ_WAVES", 4)], [1])

    with pytest.raises(SystemExit):
        create_df_pmc(str(tmp_path), verbose=0)


def test_create_df_pmc_gives_processes_on_one_gpu_one_gpu_id(tmp_path) -> None:
    """The dispatch CSV carries the node id, which every process shares."""
    write_native_process(
        tmp_path, "pmc_perf_0", 100, [(1, "SQ_WAVES", 4)], [1], gpu_id=2
    )
    write_native_process(
        tmp_path, "pmc_perf_0", 200, [(1, "SQ_WAVES", 8)], [1], gpu_id=2
    )

    df = create_df_pmc(str(tmp_path), verbose=0)

    assert df["GPU_ID"].tolist() == [0, 0]


def test_create_df_pmc_numbers_gpus_in_node_order_across_processes(tmp_path) -> None:
    """GPU ids follow node ids, not the order processes are read in."""
    write_native_process(
        tmp_path, "pmc_perf_0", 100, [(1, "SQ_WAVES", 4)], [1], gpu_id=3
    )
    write_native_process(
        tmp_path, "pmc_perf_0", 200, [(1, "SQ_WAVES", 8)], [1], gpu_id=2
    )

    df = create_df_pmc(str(tmp_path), verbose=0)

    assert dict(zip(df["SQ_WAVES"], df["GPU_ID"])) == {4: 1, 8: 0}


# Native agents CSV
# =============================================================================


def test_build_node_to_gpu_map_numbers_gpus_in_node_order() -> None:
    agents = pd.DataFrame({"node_id": [5, 2, 3]})

    assert build_node_to_gpu_map(agents) == {2: 0, 3: 1, 5: 2}


def test_load_node_to_gpu_map_reads_the_agents_csv(tmp_path) -> None:
    agents_csv = tmp_path / "agents_pmc_perf_0_100.csv.gz"
    common.write_gzip_csv(
        agents_csv,
        "node_id,logical_node_id,name,product_name\n"
        '3,1,"gfx942","AMD Instinct MI300X"\n'
        '2,0,"gfx942","AMD Instinct MI300X"\n',
    )

    assert load_node_to_gpu_map(agents_csv) == {2: 0, 3: 1}


def test_load_node_to_gpu_map_missing_file_maps_nothing(tmp_path) -> None:
    assert load_node_to_gpu_map(tmp_path / "agents_pmc_perf_0_100.csv.gz") == {}


def test_load_kernel_short_names_dedupes_repeated_symbols(tmp_path):
    """A symbol repeats per process and per run, and folds to one entry."""
    pd.DataFrame(
        [("vecCopy(double*)", "vecCopy"), ("vecCopy(double*)", "vecCopy")],
        columns=KERNEL_SYMBOLS_COLUMNS,
    ).to_csv(tmp_path / "rocpd_kernel_symbols_pmc_perf_0.csv.gz", index=False)
    pd.DataFrame(
        [("vecCopy(double*)", "vecCopy"), ("vecAdd()", "vecAdd")],
        columns=KERNEL_SYMBOLS_COLUMNS,
    ).to_csv(tmp_path / "rocpd_kernel_symbols_pmc_perf_1.csv.gz", index=False)

    assert load_kernel_short_names(str(tmp_path), []) == {
        "vecCopy(double*)": "vecCopy",
        "vecAdd()": "vecAdd",
    }


def test_load_kernel_short_names_prefers_the_native_symbols(tmp_path):
    """A native run writes both shapes; the rocpd one is not read."""
    pd.DataFrame(
        [("vecCopy(double*)", "stale")],
        columns=KERNEL_SYMBOLS_COLUMNS,
    ).to_csv(tmp_path / "rocpd_kernel_symbols_pmc_perf_0.csv.gz", index=False)
    for prefix, columns in (
        ("counters", ["dispatch_id"]),
        ("dispatch", ["dispatch_id"]),
    ):
        pd.DataFrame([(1,)], columns=columns).to_csv(
            tmp_path / f"{prefix}_run0_100.csv.gz", index=False
        )
    pd.DataFrame(
        [(7, "vecCopy(double*)", "vecCopy")],
        columns=["kernel_id", "kernel_name", "kernel_short_name"],
    ).to_csv(tmp_path / "kernel_symbols_run0_100.csv.gz", index=False)

    assert load_kernel_short_names(str(tmp_path), []) == {
        "vecCopy(double*)": "vecCopy",
    }


def test_load_kernel_short_names_falls_back_to_the_sampling_results(tmp_path):
    """A PC-sampling-only workload has no rocpd db, so its JSON carries them."""
    tool_data_records = [
        {
            "kernel_symbols": [
                {
                    "formatted_kernel_name": "vecCopy(double*)",
                    "truncated_kernel_name": "vecCopy",
                }
            ]
        },
        {
            "kernel_symbols": [
                {
                    "formatted_kernel_name": "vecAdd()",
                    "truncated_kernel_name": "vecAdd",
                }
            ]
        },
    ]

    assert load_kernel_short_names(str(tmp_path), tool_data_records) == {
        "vecCopy(double*)": "vecCopy",
        "vecAdd()": "vecAdd",
    }


def test_load_kernel_short_names_prefers_the_profiled_csv(tmp_path):
    """A counter run that also sampled takes the CSV, which covers every kernel."""
    pd.DataFrame(
        [("vecCopy(double*)", "vecCopy")], columns=KERNEL_SYMBOLS_COLUMNS
    ).to_csv(tmp_path / "rocpd_kernel_symbols_pmc_perf_0.csv.gz", index=False)
    tool_data_records = [
        {
            "kernel_symbols": [
                {
                    "formatted_kernel_name": "vecAdd()",
                    "truncated_kernel_name": "vecAdd",
                }
            ]
        }
    ]

    assert load_kernel_short_names(str(tmp_path), tool_data_records) == {
        "vecCopy(double*)": "vecCopy"
    }


def test_load_kernel_short_names_falls_back_past_an_empty_csv(tmp_path):
    """A failed extract leaves the file behind, which is a miss, not a mapping."""
    gzip.open(tmp_path / "rocpd_kernel_symbols_pmc_perf_0.csv.gz", "wt").close()
    tool_data_records = [
        {
            "kernel_symbols": [
                {
                    "formatted_kernel_name": "vecAdd()",
                    "truncated_kernel_name": "vecAdd",
                }
            ]
        }
    ]

    assert load_kernel_short_names(str(tmp_path), tool_data_records) == {
        "vecAdd()": "vecAdd"
    }


def test_load_kernel_short_names_tolerates_a_record_without_symbols(tmp_path):
    """The fallback runs when things went wrong, so a bare record is not fatal."""
    assert load_kernel_short_names(str(tmp_path), [{"metadata": {"pid": 1}}]) == {}


@pytest.mark.misc
def test_is_single_panel_config_accepts_shared_gfx115x_dir(tmp_path):
    (tmp_path / "gfx115x").mkdir()

    supported_archs = {
        "gfx1150": "rdna35_point_1",
        "gfx1151": "rdna35_halo",
        "gfx1152": "rdna35_point_2",
        "gfx1153": "rdna35_gorgon_point",
    }

    assert is_single_panel_config(str(tmp_path), supported_archs) is False
