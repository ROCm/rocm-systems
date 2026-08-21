# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pc_sampling.pc_sampling_analysis import (
    aggregate_pc_sample_records,
    detect_pc_sampling_method,
    enrich_with_metadata,
    load_aggregated_pc_sampling,
    load_pc_sample_records,
)
from rocprof_compute_analyze.analysis_db import db_analysis
from utils import schema
from utils.file_io import (
    build_agent_to_gpu_map_from_json,
    load_pc_sampling_results,
    process_pc_sampling_kernel_trace,
    process_pc_sampling_kernel_traces,
)
from utils.parser import (
    PMC_DISPATCH_INFO_TABLE_ID,
    PMC_KERNEL_TOP_TABLE_ID,
    load_non_mertrics_table,
    load_pc_sampling_data,
    load_table_data,
    nullify_unevaluated_metric_values,
)
from utils.utils_common import is_only_pc_sampling

PC_SAMPLING_WORKLOAD = "tests/workloads/vcopy_pc_sampling_only/MI350"
# The capture-host paths the fixture's instruction comments name.
VCOPY_SOURCE = "/app/projects/rocprofiler-compute/sample/vcopy.cpp"
HIP_RUNTIME_SOURCE = (
    "/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel/include/hip/"
    "amd_detail/amd_hip_runtime.h"
)

PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_"
INST_PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_"
HOST_TRAP_DISPLAY_COLUMNS = [
    "pid",
    "source_line",
    "instruction",
    "code_object_id",
    "offset",
    "count",
    "Kernel_Name",
]
STOCHASTIC_DISPLAY_COLUMNS = [
    "pid",
    "source_line",
    "instruction",
    "code_object_id",
    "offset",
    "count",
    "count_issued",
    "count_stalled",
    "stall_reason",
    "Kernel_Name",
]


# ── Helpers for building synthetic JSON / records ────────────


def make_record(
    code_object_id: int,
    offset: int,
    inst_index: int,
    dispatch_id: int,
    wave_issued: bool = True,
    stall_reason: str | None = None,
    inst_type: str | None = None,
    exec_mask: int | None = None,
    wave_cnt: int | None = None,
) -> dict:
    snapshot = {}
    if stall_reason is not None:
        snapshot["stall_reason"] = stall_reason
    record = {
        "pc": {
            "code_object_id": code_object_id,
            "code_object_offset": offset,
        },
        "dispatch_id": dispatch_id,
        "wave_issued": wave_issued,
        "snapshot": snapshot,
    }
    if inst_type is not None:
        record["inst_type"] = inst_type
    if exec_mask is not None:
        record["exec_mask"] = exec_mask
    if wave_cnt is not None:
        record["wave_cnt"] = wave_cnt
    return {"inst_index": inst_index, "record": record}


def make_host_trap_record(
    code_object_id: int,
    offset: int,
    inst_index: int,
    dispatch_id: int,
    exec_mask: int | None = None,
    wave_cnt: int | None = None,
) -> dict:
    """A host_trap sample: no wave_issued / snapshot (no issue/stall info)."""
    record = {
        "pc": {
            "code_object_id": code_object_id,
            "code_object_offset": offset,
        },
        "dispatch_id": dispatch_id,
    }
    if exec_mask is not None:
        record["exec_mask"] = exec_mask
    if wave_cnt is not None:
        record["wave_cnt"] = wave_cnt
    return {
        "inst_index": inst_index,
        "record": record,
    }


def make_dispatch(
    dispatch_id: int,
    kernel_id: int,
    agent_handle: int = 1,
    start: int = 0,
    end: int = 0,
) -> dict:
    """A kernel_dispatch buffer record mapping a dispatch to a kernel."""
    return {
        "start_timestamp": start,
        "end_timestamp": end,
        "dispatch_info": {
            "dispatch_id": dispatch_id,
            "kernel_id": kernel_id,
            "agent_id": {"handle": agent_handle},
        },
    }


def make_kernel_symbol(
    kernel_id: int,
    code_object_id: int,
    formatted_kernel_name: str,
) -> dict:
    """A kernel_symbols entry mapping kernel/code-object ids to a name."""
    return {
        "kernel_id": kernel_id,
        "code_object_id": code_object_id,
        "formatted_kernel_name": formatted_kernel_name,
    }


def make_agent(handle: int, node_id: int, agent_type: int) -> dict:
    """An agents entry (type 1 == CPU, type 2 == GPU)."""
    return {"id": {"handle": handle}, "type": agent_type, "node_id": node_id}


def make_code_object(
    code_object_id: int,
    load_base: int = 0,
) -> dict:
    """A code_objects catalog entry carrying load_base."""
    return {"code_object_id": code_object_id, "load_base": load_base}


def make_tool_data(
    stochastic: list | None = None,
    host_trap: list | None = None,
    instructions: list | None = None,
    comments: list | None = None,
    kernel_symbols: list | None = None,
    kernel_dispatch: list | None = None,
    agents: list | None = None,
    code_objects: list | None = None,
    pid: int | None = 42,
) -> dict:
    """Build a single ``rocprofiler-sdk-tool[0]`` dict for the analyze paths."""
    return {
        "metadata": {"pid": pid},
        "buffer_records": {
            "pc_sample_stochastic": stochastic if stochastic is not None else [],
            "pc_sample_host_trap": host_trap if host_trap is not None else [],
            "kernel_dispatch": kernel_dispatch if kernel_dispatch is not None else [],
        },
        "strings": {
            "pc_sample_instructions": (
                instructions if instructions is not None else []
            ),
            "pc_sample_comments": comments if comments is not None else [],
        },
        "kernel_symbols": kernel_symbols if kernel_symbols is not None else [],
        "agents": agents if agents is not None else [],
        "code_objects": code_objects if code_objects is not None else [],
    }


def write_results_json(path: Path, **kwargs) -> Path:
    """Write a results json wrapping a single tool record built from kwargs."""
    path.write_text(json.dumps({"rocprofiler-sdk-tool": [make_tool_data(**kwargs)]}))
    return path


def sample_tool_data_kwargs() -> dict:
    """Minimal populated tool-data kwargs with one kernel and one sample."""
    return {
        "stochastic": [make_record(5, 0x10, 0, dispatch_id=0, wave_issued=True)],
        "instructions": ["v_mov"],
        "comments": ["/s/a.cpp:1"],
        "kernel_symbols": [make_kernel_symbol(100, 5, "vecCopy")],
        "kernel_dispatch": [make_dispatch(0, 100)],
    }


def make_display_row_tool_data(
    method: str,
    code_object_id: int,
    kernel_name: str,
    instruction: str | None,
    source_line: str,
    pid: int,
    offset: int = 0x10,
    sample_count: int = 1,
) -> dict:
    """Build one process-scoped display row from caller-controlled metadata."""
    inst_index = 1 if instruction is None else 0
    instructions = ["unused"] if instruction is None else [instruction]
    comments = (
        ["/src/unused.cpp:0", source_line] if instruction is None else [source_line]
    )
    if method == "host_trap":
        samples = [
            make_host_trap_record(code_object_id, offset, inst_index, dispatch_id=0)
            for _ in range(sample_count)
        ]
    else:
        samples = [
            make_record(code_object_id, offset, inst_index, dispatch_id=0)
            for _ in range(sample_count)
        ]

    return make_tool_data(
        **{method: samples},
        instructions=instructions,
        comments=comments,
        kernel_symbols=[make_kernel_symbol(100, code_object_id, kernel_name)],
        kernel_dispatch=[make_dispatch(0, 100)],
        code_objects=[make_code_object(code_object_id)],
        pid=pid,
    )


def make_pc_sampling_workload(kernel_name: str = "vecCopy") -> schema.Workload:
    """Build a workload that selects *kernel_name* from the kernel-top table."""
    return schema.Workload(
        filter_kernel_ids=[0],
        dfs={PMC_KERNEL_TOP_TABLE_ID: pd.DataFrame({"Kernel_Name": [kernel_name]})},
    )


def make_multiprocess_dispatch_tool_data() -> list[dict]:
    """Build colliding process-local dispatch IDs for one shared kernel."""
    first_tool_data = make_tool_data(
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[
            make_dispatch(0, 100, start=10, end=20),
            make_dispatch(1, 100, start=30, end=50),
        ],
        pid=101,
    )
    second_tool_data = make_tool_data(
        kernel_symbols=[make_kernel_symbol(200, 7, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 200, start=60, end=90)],
        pid=202,
    )
    return [first_tool_data, second_tool_data]


# ═══════════════════════════════════════════════════════════════
# is_only_pc_sampling
# ═══════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "filter_blocks, expected",
    [
        ([], False),
        (["21"], True),
        (["pc_sampling"], True),
        (["21", "pc_sampling"], True),
        (["21", "2"], False),
        (["2"], False),
    ],
)
def test_is_only_pc_sampling(filter_blocks: list[str], expected: bool) -> None:
    """True only when every requested block is PC sampling (21 / pc_sampling)."""
    assert is_only_pc_sampling(filter_blocks) is expected


@pytest.mark.parametrize(
    "filter_blocks, expected",
    [
        ([], False),
        (["21"], True),
        (["pc_sampling"], True),
        (["21", "pc_sampling"], True),
        (["21", "2"], True),
        (["2"], False),
    ],
)
def test_pc_sampling_collected(
    filter_blocks: list[str], expected: bool, tmp_path: Path
) -> None:
    """True when any collected block is PC sampling, including mixed runs."""
    instance = make_db_analysis(str(tmp_path))
    instance._profiling_config = {"filter_blocks": filter_blocks}
    assert instance.pc_sampling_collected() is expected


# ═══════════════════════════════════════════════════════════════
# detect_pc_sampling_method
# ═══════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "stochastic, host_trap, expected",
    [
        (None, None, None),
        ([make_record(1, 0x10, 0, dispatch_id=0)], None, "stochastic"),
        (None, [make_record(1, 0x10, 0, dispatch_id=0)], "host_trap"),
        (
            [make_record(1, 0x10, 0, dispatch_id=0)],
            [make_record(1, 0x10, 0, dispatch_id=0)],
            "stochastic",
        ),
    ],
)
def test_detect_pc_sampling_method(
    stochastic: list | None,
    host_trap: list | None,
    expected: str | None,
) -> None:
    """Detection prioritizes stochastic and returns None when no samples exist."""
    tool_data = make_tool_data(stochastic=stochastic, host_trap=host_trap)
    assert detect_pc_sampling_method(tool_data) == expected


# ═══════════════════════════════════════════════════════════════
# load_pc_sample_records
# ═══════════════════════════════════════════════════════════════


def test_load_pc_sample_records_empty() -> None:
    """No samples yield an empty df that still carries the normalized columns."""
    df = load_pc_sample_records(make_tool_data())
    assert df.empty
    assert "kernel_id" in df.columns
    assert "stall_reason" in df.columns


@pytest.mark.parametrize("placement", ["stochastic", "host_trap", "mixed"])
def test_load_pc_sample_records_flattens_both_arrays(placement: str) -> None:
    """Both sample arrays are flattened; kernel_id resolved via dispatch."""
    s0 = make_record(5, 0x10, 0, dispatch_id=0)
    s1 = make_record(5, 0x20, 1, dispatch_id=1)
    if placement == "stochastic":
        kwargs = {"stochastic": [s0, s1]}
    elif placement == "host_trap":
        kwargs = {"host_trap": [s0, s1]}
    else:
        kwargs = {"stochastic": [s0], "host_trap": [s1]}
    tool_data = make_tool_data(
        kernel_dispatch=[make_dispatch(0, 100), make_dispatch(1, 101)],
        **kwargs,
    )
    df = load_pc_sample_records(tool_data)
    assert len(df) == 2
    by_offset = dict(zip(df["code_object_offset"], df["kernel_id"]))
    assert by_offset[0x10] == 100
    assert by_offset[0x20] == 101


def test_load_pc_sample_records_missing_snapshot() -> None:
    """A record without a snapshot key yields a None stall_reason, not an error."""
    record = {
        "inst_index": 0,
        "record": {
            "pc": {"code_object_id": 1, "code_object_offset": 0x10},
            "dispatch_id": 0,
            "wave_issued": False,
        },
    }
    df = load_pc_sample_records(make_tool_data(stochastic=[record]))
    assert len(df) == 1
    assert df.iloc[0]["stall_reason"] is None


def test_load_pc_sample_records_skips_incomplete_pc() -> None:
    """Records missing code_object_id / offset / inst_index are skipped."""
    valid = make_record(1, 0x10, 0, dispatch_id=0)
    invalid = {
        "inst_index": 0,
        "record": {
            "pc": {"code_object_id": None, "code_object_offset": 0x10},
            "dispatch_id": 1,
            "wave_issued": True,
            "snapshot": {},
        },
    }
    df = load_pc_sample_records(make_tool_data(stochastic=[valid, invalid]))
    assert len(df) == 1


def test_load_pc_sample_records_unmapped_dispatch_kernel_id_none() -> None:
    """A dispatch_id absent from kernel_dispatch leaves kernel_id None."""
    df = load_pc_sample_records(
        make_tool_data(stochastic=[make_record(1, 0x10, 0, dispatch_id=99)])
    )
    assert df.iloc[0]["kernel_id"] is None


def test_load_pc_sample_records_missing_wave_measurements_are_none() -> None:
    """A record may omit either wave measurement without failing to normalize."""
    sample = make_record(1, 0x10, 0, dispatch_id=0)
    assert "exec_mask" not in sample["record"]
    assert "wave_cnt" not in sample["record"]

    records = load_pc_sample_records(make_tool_data(stochastic=[sample]))

    assert {"exec_mask", "wave_cnt"}.issubset(records.columns)
    assert records.iloc[0]["exec_mask"] is None
    assert records.iloc[0]["wave_cnt"] is None


# ═══════════════════════════════════════════════════════════════
# aggregate_pc_sample_records
# ═══════════════════════════════════════════════════════════════


def test_aggregate_empty_records_returns_columns() -> None:
    """Aggregating an empty record df returns the expected (empty) columns."""
    empty = load_pc_sample_records(make_tool_data())
    result = aggregate_pc_sample_records(
        empty, group_by=["code_object_id", "code_object_offset"], sys_info=None
    )
    assert result.empty
    for column in ("count", "count_issued", "count_stalled", "stall_reason"):
        assert column in result.columns
    # kernel_id / inst_index are carried since they are not group keys.
    assert "kernel_id" in result.columns
    assert "inst_index" in result.columns


def test_aggregate_counts_issued_and_stalled() -> None:
    """A mix of issued and stalled samples produces the documented counts."""
    records = load_pc_sample_records(
        make_tool_data(
            stochastic=[
                make_record(1, 0x10, 0, dispatch_id=0, wave_issued=True),
                make_record(
                    1,
                    0x10,
                    0,
                    dispatch_id=1,
                    wave_issued=False,
                    stall_reason=f"{PREFIX}WAITCNT",
                ),
            ]
        )
    )
    result = aggregate_pc_sample_records(
        records, group_by=["code_object_id", "code_object_offset"], sys_info=None
    )
    row = result.iloc[0]
    assert row["count"] == 2
    assert row["count_issued"] == 1
    assert row["count_stalled"] == 1
    assert row["stall_reason"] == {"WAITCNT": 1}


def test_aggregate_host_trap_counts_are_none() -> None:
    """Without wave_issued info, issued/stalled counts and reasons are None."""
    records = load_pc_sample_records(
        make_tool_data(host_trap=[make_host_trap_record(1, 0x10, 0, dispatch_id=0)])
    )
    result = aggregate_pc_sample_records(
        records, group_by=["code_object_id", "code_object_offset"], sys_info=None
    )
    row = result.iloc[0]
    assert row["count"] == 1
    assert row["count_issued"] is None
    assert row["count_stalled"] is None
    assert row["stall_reason"] is None


def test_aggregate_unknown_stall_key_dropped() -> None:
    """A stall reason outside the canonical key set is dropped."""
    records = load_pc_sample_records(
        make_tool_data(
            stochastic=[
                make_record(
                    1,
                    0x10,
                    0,
                    dispatch_id=0,
                    wave_issued=False,
                    stall_reason=f"{PREFIX}NOT_A_REAL_KEY",
                )
            ]
        )
    )
    result = aggregate_pc_sample_records(
        records, group_by=["code_object_id", "code_object_offset"], sys_info=None
    )
    assert result.iloc[0]["stall_reason"] == {}


def test_aggregate_group_by_kernel_id_separates_shared_code_object() -> None:
    """Grouping by kernel_id keeps two kernels in one code object distinct."""
    records = load_pc_sample_records(
        make_tool_data(
            stochastic=[
                make_record(5, 0x10, 0, dispatch_id=0),
                make_record(5, 0x20, 1, dispatch_id=1),
            ],
            kernel_dispatch=[make_dispatch(0, 100), make_dispatch(1, 101)],
        )
    )
    result = aggregate_pc_sample_records(
        records,
        group_by=["code_object_id", "code_object_offset", "kernel_id"],
        sys_info=None,
    )
    assert len(result) == 2
    assert set(result["kernel_id"]) == {100, 101}


def test_aggregate_active_threads_reports_full_wave_for_all_ones_mask() -> None:
    """A fully-active wave64 reports 100 percent, the physical upper bound."""
    all_ones_wave64 = 2**64 - 1
    records = load_pc_sample_records(
        make_tool_data(
            stochastic=[
                make_record(
                    1,
                    0x10,
                    0,
                    dispatch_id=0,
                    exec_mask=all_ones_wave64,
                    wave_cnt=8,
                )
            ]
        )
    )

    result = aggregate_pc_sample_records(
        records,
        group_by=["code_object_id", "code_object_offset"],
        sys_info={"wave_size": "64", "max_waves_per_cu": "32"},
    )

    assert all_ones_wave64.bit_count() == 64
    assert result.iloc[0]["active_thread_percent"] == 100.0


def test_aggregate_wave_means_include_every_sample_regardless_of_issue_state() -> None:
    """Wave measurements are independent of issue, stall, and instruction state."""
    records = load_pc_sample_records(
        make_tool_data(
            stochastic=[
                make_record(
                    1,
                    0x10,
                    0,
                    dispatch_id=0,
                    wave_issued=True,
                    inst_type=f"{INST_PREFIX}VALU",
                    exec_mask=0b1,
                    wave_cnt=2,
                ),
                make_record(
                    1,
                    0x10,
                    0,
                    dispatch_id=0,
                    wave_issued=False,
                    stall_reason=f"{PREFIX}WAITCNT",
                    inst_type=f"{INST_PREFIX}VALU",
                    exec_mask=0b11,
                    wave_cnt=7,
                ),
                make_record(
                    1,
                    0x10,
                    0,
                    dispatch_id=0,
                    wave_issued=True,
                    inst_type=f"{INST_PREFIX}NO_INST",
                    exec_mask=0b11111111,
                    wave_cnt=20,
                ),
            ]
        )
    )

    row = aggregate_pc_sample_records(
        records,
        group_by=["code_object_id", "code_object_offset"],
        sys_info={"wave_size": "64", "max_waves_per_cu": "32"},
    ).iloc[0]

    assert row["active_thread_percent"] == pytest.approx((11 / 3) / 64 * 100), (
        "active_thread_percent must include all samples; filtering by "
        "wave_issued, stall_reason, or inst_type is wrong"
    )
    assert row["wave_occupancy_percent"] == pytest.approx((29 / 3) / 32 * 100), (
        "wave_occupancy_percent must include all samples; filtering by "
        "wave_issued, stall_reason, or inst_type is wrong"
    )


@pytest.mark.parametrize(
    (
        "method",
        "exec_mask",
        "wave_cnt",
        "expected_active_thread_percent",
        "expected_wave_occupancy_percent",
    ),
    [
        pytest.param("host_trap", 0b1111, None, 6.25, None, id="host-mask-only"),
        pytest.param("stochastic", 0b11, 8, 3.125, 25.0, id="stochastic-both"),
        pytest.param("stochastic", None, None, None, None, id="stochastic-neither"),
    ],
)
def test_aggregate_wave_measurements_follow_field_presence_not_method(
    method: str,
    exec_mask: int | None,
    wave_cnt: int | None,
    expected_active_thread_percent: float | None,
    expected_wave_occupancy_percent: float | None,
) -> None:
    """Each sample contributes the measurements it carries, for either method."""
    if method == "host_trap":
        sample = make_host_trap_record(
            1,
            0x10,
            0,
            dispatch_id=0,
            exec_mask=exec_mask,
            wave_cnt=wave_cnt,
        )
    else:
        sample = make_record(
            1,
            0x10,
            0,
            dispatch_id=0,
            exec_mask=exec_mask,
            wave_cnt=wave_cnt,
        )
    tool_data = make_tool_data(**{method: [sample]})
    records = load_pc_sample_records(tool_data)

    row = aggregate_pc_sample_records(
        records,
        group_by=["code_object_id", "code_object_offset"],
        sys_info={"wave_size": "64", "max_waves_per_cu": "32"},
    ).iloc[0]

    if expected_active_thread_percent is None:
        assert row["active_thread_percent"] is None
    else:
        assert row["active_thread_percent"] == expected_active_thread_percent
    if expected_wave_occupancy_percent is None:
        assert row["wave_occupancy_percent"] is None
    else:
        assert row["wave_occupancy_percent"] == expected_wave_occupancy_percent


def test_aggregate_wave_measurements_exclude_missing_fields_independently() -> None:
    """Missing one field never prevents the other field from contributing."""
    records = load_pc_sample_records(
        make_tool_data(
            stochastic=[
                make_record(1, 0x10, 0, dispatch_id=0, exec_mask=0b1),
                make_record(1, 0x10, 0, dispatch_id=0, wave_cnt=16),
                make_record(
                    1,
                    0x10,
                    0,
                    dispatch_id=0,
                    exec_mask=0b111,
                    wave_cnt=8,
                ),
            ]
        )
    )

    row = aggregate_pc_sample_records(
        records,
        group_by=["code_object_id", "code_object_offset"],
        sys_info={"wave_size": "64", "max_waves_per_cu": "32"},
    ).iloc[0]

    assert row["active_thread_percent"] == 3.125, (
        "exec_mask mean must use exactly the samples carrying exec_mask"
    )
    assert row["wave_occupancy_percent"] == 37.5, (
        "wave_cnt mean must use exactly the samples carrying wave_cnt"
    )


def test_aggregate_wave_measurements_are_none_without_sys_info() -> None:
    """Absent system information leaves both values unknown."""
    records = load_pc_sample_records(
        make_tool_data(
            stochastic=[
                make_record(
                    1,
                    0x10,
                    0,
                    dispatch_id=0,
                    exec_mask=0b1111,
                    wave_cnt=8,
                )
            ]
        )
    )
    group_by = ["code_object_id", "code_object_offset"]

    result = aggregate_pc_sample_records(records, group_by=group_by, sys_info=None)

    assert {
        "active_thread_percent",
        "wave_occupancy_percent",
    }.issubset(result.columns)
    assert result.iloc[0]["active_thread_percent"] is None
    assert result.iloc[0]["wave_occupancy_percent"] is None


@pytest.mark.parametrize(
    ("sys_info", "expected_active_thread_percent", "expected_wave_occupancy_percent"),
    [
        pytest.param(
            {"max_waves_per_cu": "32"},
            None,
            25.0,
            id="missing-wave-size",
        ),
        pytest.param(
            {"wave_size": "64"},
            6.25,
            None,
            id="missing-max-waves",
        ),
        pytest.param(
            {"wave_size": "", "max_waves_per_cu": "32"},
            None,
            25.0,
            id="blank-wave-size",
        ),
        pytest.param(
            {"wave_size": "many", "max_waves_per_cu": "32"},
            None,
            25.0,
            id="nonnumeric-wave-size",
        ),
        pytest.param(
            {"wave_size": "0", "max_waves_per_cu": "32"},
            None,
            25.0,
            id="zero-wave-size",
        ),
        pytest.param(
            {"wave_size": "64", "max_waves_per_cu": ""},
            6.25,
            None,
            id="blank-max-waves",
        ),
        pytest.param(
            {"wave_size": "64", "max_waves_per_cu": "many"},
            6.25,
            None,
            id="nonnumeric-max-waves",
        ),
        pytest.param(
            {"wave_size": "64", "max_waves_per_cu": "0"},
            6.25,
            None,
            id="zero-max-waves",
        ),
    ],
)
def test_aggregate_wave_measurements_require_usable_denominators(
    sys_info: dict[str, str],
    expected_active_thread_percent: float | None,
    expected_wave_occupancy_percent: float | None,
) -> None:
    """Each percentage is None unless its own denominator is usable."""
    records = load_pc_sample_records(
        make_tool_data(
            stochastic=[
                make_record(
                    1,
                    0x10,
                    0,
                    dispatch_id=0,
                    exec_mask=0b1111,
                    wave_cnt=8,
                )
            ]
        )
    )

    row = aggregate_pc_sample_records(
        records,
        group_by=["code_object_id", "code_object_offset"],
        sys_info=sys_info,
    ).iloc[0]

    if expected_active_thread_percent is None:
        assert row["active_thread_percent"] is None
    else:
        assert row["active_thread_percent"] == expected_active_thread_percent
    if expected_wave_occupancy_percent is None:
        assert row["wave_occupancy_percent"] is None
    else:
        assert row["wave_occupancy_percent"] == expected_wave_occupancy_percent


# ═══════════════════════════════════════════════════════════════
# enrich_with_metadata
# ═══════════════════════════════════════════════════════════════


def make_aggregated_row(inst_index: int = 0, kernel_id: int = 100) -> pd.DataFrame:
    """A one-row aggregated df suitable for enrichment."""
    return pd.DataFrame([{"inst_index": inst_index, "kernel_id": kernel_id}])


def test_enrich_attach_subset_only_adds_requested_columns() -> None:
    """Only the requested attach columns are added."""
    tool_data = make_tool_data(
        instructions=["v_mov"],
        comments=["/s/a.cpp:1"],
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
    )
    df = enrich_with_metadata(make_aggregated_row(), tool_data, attach={"instruction"})
    assert "instruction" in df.columns
    assert "source_line" not in df.columns
    assert "kernel_name" not in df.columns


def test_enrich_source_line_out_of_range_is_na() -> None:
    """An inst_index past the comment table yields the 'N/A' sentinel, not ''."""
    tool_data = make_tool_data(instructions=["v_mov"], comments=["/s/a.cpp:1"])
    df = enrich_with_metadata(
        make_aggregated_row(inst_index=5), tool_data, attach={"source_line"}
    )
    assert df.iloc[0]["source_line"] == "N/A"


def test_enrich_empty_source_line_is_na() -> None:
    """An empty comment string yields the 'N/A' sentinel, not ''."""
    tool_data = make_tool_data(instructions=["v_mov"], comments=[""])
    df = enrich_with_metadata(
        make_aggregated_row(inst_index=0), tool_data, attach={"source_line"}
    )
    assert df.iloc[0]["source_line"] == "N/A"


def test_enrich_kernel_name_unmapped_is_none() -> None:
    """A kernel_id absent from kernel_symbols maps to a None kernel_name."""
    tool_data = make_tool_data(
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
    )
    df = enrich_with_metadata(
        make_aggregated_row(kernel_id=999), tool_data, attach={"kernel_name"}
    )
    assert df.iloc[0]["kernel_name"] is None


# ═══════════════════════════════════════════════════════════════
# load_aggregated_pc_sampling
# ═══════════════════════════════════════════════════════════════


def test_load_aggregated_pc_sampling_happy_path() -> None:
    """The helper loads, aggregates, enriches and returns the code-object tree."""
    tool_data = make_tool_data(
        stochastic=[make_record(5, 0x10, 0, dispatch_id=0, wave_issued=True)],
        instructions=["v_mov"],
        comments=["/s/a.cpp:1"],
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
        code_objects=[make_code_object(5)],
    )
    line = load_aggregated_pc_sampling(tool_data, sys_info=None)[0].instruction_lines[0]
    assert line.total_count == 1
    assert line.instruction == "v_mov"
    assert line.source == "/s/a.cpp:1"
    assert line.kernel_name == "vecCopy"


@pytest.mark.parametrize("method", ["host_trap", "stochastic"])
def test_wave_measurements_are_carried_but_not_added_to_terminal_frame(
    method: str,
) -> None:
    """Database records carry wave metrics without changing terminal columns."""
    if method == "host_trap":
        sample = make_host_trap_record(
            5,
            0x10,
            0,
            dispatch_id=0,
            exec_mask=0b1111,
            wave_cnt=8,
        )
    else:
        sample = make_record(
            5,
            0x10,
            0,
            dispatch_id=0,
            exec_mask=0b1111,
            wave_cnt=8,
        )
    tool_data = make_tool_data(
        **{method: [sample]},
        instructions=["v_mov"],
        comments=["/s/a.cpp:1"],
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
        code_objects=[make_code_object(5)],
    )

    display_frame = load_pc_sampling_data(schema.Workload(), "count", [tool_data])
    assert {
        "active_thread_percent",
        "wave_occupancy_percent",
    }.isdisjoint(display_frame.columns)

    line = load_aggregated_pc_sampling(
        tool_data,
        sys_info={"wave_size": "64", "max_waves_per_cu": "32"},
    )[0].instruction_lines[0]
    assert line.active_thread_percent == 6.25
    assert line.wave_occupancy_percent == 25.0


# ═══════════════════════════════════════════════════════════════
# load_pc_sampling_data
# ═══════════════════════════════════════════════════════════════


def setup_pc_sampling_data(
    method: str = "host_trap",
) -> dict:
    """Build tool data for filtered PC sampling tests.

    vecCopy (kernel_id 100) and vecAdd (kernel_id 101) share code object 5,
    so kernels are distinguished only by kernel_id via dispatch attribution.
    """
    samples = [
        make_record(5, 0x10, 0, dispatch_id=0),
        make_record(
            5,
            0x20,
            1,
            dispatch_id=1,
            wave_issued=False,
            stall_reason=f"{PREFIX}WAITCNT",
        ),
        make_record(5, 0x30, 2, dispatch_id=2),
    ]

    key = "host_trap" if method == "host_trap" else "stochastic"
    kwargs = {key: samples}
    return make_tool_data(
        instructions=["v_mov_b32", "s_waitcnt", "v_add_f32"],
        comments=[
            "/src/vcopy.cpp:42",
            "/src/vcopy.cpp:43",
            "/src/vadd.cpp:30",
        ],
        kernel_symbols=[
            make_kernel_symbol(100, 5, "vecCopy"),
            make_kernel_symbol(101, 5, "vecAdd"),
        ],
        kernel_dispatch=[
            make_dispatch(0, 100),
            make_dispatch(1, 100),
            make_dispatch(2, 101),
        ],
        **kwargs,
    )


@pytest.mark.parametrize(
    "method, sorting_type",
    [
        ("host_trap", "offset"),
        ("host_trap", "count"),
        ("stochastic", "offset"),
        ("stochastic", "count"),
    ],
)
def test_load_pc_sampling_data_filtered_schema_and_sort(
    method: str,
    sorting_type: str,
) -> None:
    """Column projection follows method; row order follows sorting_type."""
    expected_columns = (
        HOST_TRAP_DISPLAY_COLUMNS
        if method == "host_trap"
        else STOCHASTIC_DISPLAY_COLUMNS
    )
    tool_data = setup_pc_sampling_data(method=method)
    df = load_pc_sampling_data(
        make_pc_sampling_workload(),
        sorting_type,
        [tool_data],
    )
    assert not df.empty
    assert list(df.columns) == expected_columns
    for _, row in df.iterrows():
        assert row["code_object_id"] == 5
    if sorting_type == "count":
        counts = df["count"].tolist()
        assert counts == sorted(counts, reverse=True)
    else:
        offsets = df["offset"].tolist()
        assert offsets == sorted(offsets)


def test_load_pc_sampling_data_filtered_offset_sort_is_numeric() -> None:
    """Offset sort orders by numeric value, not lexicographic hex string."""
    tool_data = make_tool_data(
        host_trap=[
            make_record(5, 0x100, 0, dispatch_id=0),
            make_record(5, 0x20, 1, dispatch_id=1),
        ],
        instructions=["a", "b"],
        comments=["/src/f.cpp:1", "/src/f.cpp:2"],
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100), make_dispatch(1, 100)],
    )
    df = load_pc_sampling_data(
        make_pc_sampling_workload(),
        "offset",
        [tool_data],
    )
    # 0x20 (32) must precede 0x100 (256); lexicographic order would invert them.
    assert df["offset"].tolist() == ["0x20", "0x100"]


@pytest.mark.parametrize("num_rows, expected_rows", [(1, 1), (0, 2), (None, 2)])
def test_load_pc_sampling_data_filtered_num_rows_limit(
    num_rows: int | None,
    expected_rows: int,
) -> None:
    """num_rows caps the table after sorting; 0 or None keeps every row."""
    df = load_pc_sampling_data(
        make_pc_sampling_workload(),
        "count",
        [setup_pc_sampling_data()],
        num_rows=num_rows,
    )
    assert len(df) == expected_rows


def make_pc_sampling_guard_data(
    instructions: list | None,
    comments: list | None,
    indices: tuple[int, int] = (0, 1),
) -> dict:
    """Build tool data with caller-controlled instruction/comment tables."""
    samples = [
        make_record(5, 0x10, indices[0], dispatch_id=0),
        make_record(5, 0x20, indices[1], dispatch_id=1),
    ]
    return make_tool_data(
        host_trap=samples,
        instructions=instructions,
        comments=comments,
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100), make_dispatch(1, 100)],
    )


@pytest.mark.parametrize(
    "instructions, comments, instruction_none, source_line_na, indices",
    [
        pytest.param(
            ["v_mov"],
            ["/s/a.cpp:1", "/s/a.cpp:2"],
            "some",
            "none",
            (0, 1),
            id="instructions_short",
        ),
        pytest.param(
            ["v_mov", "v_add"],
            ["/s/a.cpp:1"],
            "none",
            "some",
            (0, 1),
            id="comments_short",
        ),
        pytest.param(
            ["v_mov"],
            ["/s/a.cpp:1", "/s/a.cpp:2", "/s/a.cpp:3", "/s/a.cpp:4", "/s/a.cpp:5"],
            "all",
            "none",
            (3, 4),
            id="instructions_all_out_of_range",
        ),
    ],
)
def test_load_pc_sampling_data_filtered_out_of_range_index_guards(
    instructions: list | None,
    comments: list | None,
    instruction_none: str,
    source_line_na: str,
    indices: tuple[int, int],
) -> None:
    """An inst_index past a string table yields None / 'N/A', not an error.

    instruction stays None when out of range; source_line uses the "N/A"
    sentinel so display code does not suppress the table.
    """
    tool_data = make_pc_sampling_guard_data(instructions, comments, indices)
    df = load_pc_sampling_data(
        make_pc_sampling_workload(),
        "offset",
        [tool_data],
    )
    assert not df.empty
    assert_none_kind(df["instruction"], instruction_none)
    assert_na_kind(df["source_line"], source_line_na)


@pytest.mark.parametrize(
    "instructions, comments",
    [
        pytest.param(None, ["/s/a.cpp:1", "/s/a.cpp:2"], id="empty_instructions"),
        pytest.param(["v_mov", "v_add"], None, id="empty_comments"),
    ],
)
def test_load_pc_sampling_data_filtered_empty_string_table_warns_and_skips(
    instructions: list | None,
    comments: list | None,
) -> None:
    """Empty instruction/comment table warns with the kernel name, no exit."""
    tool_data = make_pc_sampling_guard_data(instructions, comments)
    with patch("utils.parser.console_warning") as console_warning_mock:
        df = load_pc_sampling_data(
            make_pc_sampling_workload(),
            "offset",
            [tool_data],
        )
    assert df.empty
    console_warning_mock.assert_called_once()
    assert "vecCopy" in console_warning_mock.call_args.args[0]


def assert_none_kind(column: pd.Series, kind: str) -> None:
    """Assert how many entries in *column* are None: all / some / none."""
    if kind == "all":
        assert column.isna().all()
    elif kind == "some":
        assert column.isna().any() and not column.isna().all()
    else:
        assert not column.isna().any()


def assert_na_kind(column: pd.Series, kind: str) -> None:
    """Assert how many entries in *column* are the 'N/A' sentinel: all/some/none."""
    is_na = column == "N/A"
    if kind == "all":
        assert is_na.all()
    elif kind == "some":
        assert is_na.any() and not is_na.all()
    else:
        assert not is_na.any()


def test_load_pc_sampling_data_filtered_multi_dispatch_groupby() -> None:
    """Two dispatch IDs at one (code_object_id, offset) collapse to a summed row."""
    samples = [
        make_record(
            5,
            0x10,
            0,
            dispatch_id=0,
            wave_issued=False,
            stall_reason=f"{PREFIX}WAITCNT",
        ),
        make_record(
            5,
            0x10,
            1,
            dispatch_id=1,
            wave_issued=False,
            stall_reason=f"{PREFIX}ALU_DEPENDENCY",
        ),
    ]
    tool_data = make_tool_data(
        stochastic=samples,
        instructions=["v_mov", "v_add"],
        comments=["/s/a.cpp:1", "/s/a.cpp:2"],
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100), make_dispatch(1, 100)],
    )
    df = load_pc_sampling_data(
        make_pc_sampling_workload(),
        "count",
        [tool_data],
    )
    assert len(df) == 1
    row = df.iloc[0]
    assert row["count"] == 2
    assert row["count_stalled"] == 2
    assert row["count_issued"] == 0
    assert {reason for reason, _ in row["stall_reason"]} == {
        "WAITCNT",
        "ALU_DEPENDENCY",
    }


def test_load_pc_sampling_data_filtered_kernel_not_found() -> None:
    """
    Return an empty DataFrame when the requested kernel name
    is absent from the kernel symbols.
    """
    tool_data = setup_pc_sampling_data()
    df = load_pc_sampling_data(
        make_pc_sampling_workload("nonexistent"),
        "offset",
        [tool_data],
    )
    assert df.empty


def test_load_pc_sampling_data_no_tool_data() -> None:
    """Return an empty DataFrame when no parsed tool data is provided."""
    df = load_pc_sampling_data(schema.Workload(), "count", [])
    assert df.empty


@pytest.mark.parametrize("method", ["host_trap", "stochastic"])
def test_load_pc_sampling_data_separates_code_objects_within_one_record(
    method: str,
) -> None:
    """Two code objects under one kernel name stay separate display rows."""
    if method == "host_trap":
        samples = [
            make_host_trap_record(5, 0x10, 0, dispatch_id=0),
            make_host_trap_record(5, 0x10, 0, dispatch_id=0),
            make_host_trap_record(7, 0x10, 0, dispatch_id=1),
            make_host_trap_record(7, 0x10, 0, dispatch_id=1),
        ]
    else:
        samples = [
            make_record(5, 0x10, 0, dispatch_id=0, wave_issued=True),
            make_record(
                5,
                0x10,
                0,
                dispatch_id=0,
                wave_issued=False,
                stall_reason=f"{PREFIX}WAITCNT",
            ),
            make_record(
                7,
                0x10,
                0,
                dispatch_id=1,
                wave_issued=False,
                stall_reason=f"{PREFIX}ALU_DEPENDENCY",
            ),
            make_record(
                7,
                0x10,
                0,
                dispatch_id=1,
                wave_issued=False,
                stall_reason=f"{PREFIX}WAITCNT",
            ),
        ]
    tool_data = make_tool_data(
        **{method: samples},
        instructions=["v_mov"],
        comments=["/src/shared.cpp:10"],
        kernel_symbols=[
            make_kernel_symbol(100, 5, "sharedKernel"),
            make_kernel_symbol(101, 7, "sharedKernel"),
        ],
        kernel_dispatch=[make_dispatch(0, 100), make_dispatch(1, 101)],
    )

    df = load_pc_sampling_data(
        schema.Workload(),
        "offset",
        [tool_data],
    )

    # The same offset in two code objects can be different code, so the counts
    # are never summed together.
    assert len(df) == 2
    expected_columns = (
        HOST_TRAP_DISPLAY_COLUMNS
        if method == "host_trap"
        else STOCHASTIC_DISPLAY_COLUMNS
    )
    assert list(df.columns) == expected_columns
    assert list(df["code_object_id"]) == [5, 7]
    assert list(df["count"]) == [2, 2]
    assert set(df["Kernel_Name"]) == {"sharedKernel"}
    if method == "stochastic":
        assert list(df["count_issued"]) == [1, 0]
        assert list(df["count_stalled"]) == [1, 2]
        assert list(df["stall_reason"]) == [
            [("WAITCNT", 1)],
            [("ALU_DEPENDENCY", 1), ("WAITCNT", 1)],
        ]


@pytest.mark.parametrize("method", ["stochastic", "host_trap"])
def test_load_pc_sampling_data_no_filter_schema_parity(method: str) -> None:
    """No-filter has the same columns as the single-kernel view, with more rows."""
    samples = [
        make_record(5, 0x10, 0, dispatch_id=0),
        make_record(5, 0x10, 0, dispatch_id=0),
        make_record(5, 0x20, 1, dispatch_id=1),
    ]
    # vecCopy (kernel 100) and vecAdd (kernel 101) share code object 5 at distinct
    # offsets, so each row's kernel resolves via dispatch correlation.
    tool_data = make_tool_data(
        instructions=["v_mov_b32 v0 v1", "s_waitcnt vmcnt(0)"],
        comments=["/src/vcopy.cpp:42", "/src/vadd.cpp:99"],
        kernel_symbols=[
            make_kernel_symbol(100, 5, "vecCopy"),
            make_kernel_symbol(101, 5, "vecAdd"),
        ],
        kernel_dispatch=[make_dispatch(0, 100), make_dispatch(1, 101)],
        **{method: samples},
    )
    kernel_top_df = pd.DataFrame({"Kernel_Name": ["vecCopy", "vecAdd"]})
    no_filter = load_pc_sampling_data(
        schema.Workload(),
        "offset",
        [tool_data],
    )
    # Index 1 selects the second kernel-top row, not the first.
    single = load_pc_sampling_data(
        schema.Workload(
            filter_kernel_ids=[1], dfs={PMC_KERNEL_TOP_TABLE_ID: kernel_top_df}
        ),
        "offset",
        [tool_data],
    )
    assert list(no_filter.columns) == list(single.columns)
    assert "Kernel_Name" in no_filter.columns
    assert set(no_filter["Kernel_Name"]) == {"vecCopy", "vecAdd"}
    assert set(single["Kernel_Name"]) == {"vecAdd"}
    assert len(no_filter) > len(single)
    by_kernel = dict(zip(no_filter["source_line"], no_filter["Kernel_Name"]))
    assert by_kernel[".../vcopy.cpp:42"] == "vecCopy"
    assert by_kernel[".../vadd.cpp:99"] == "vecAdd"


def test_load_pc_sampling_data_multiple_kernels_error() -> None:
    """Return an empty DataFrame and log an error when >1 kernel ID is filtered."""
    tool_data = make_tool_data(stochastic=[make_record(100, 0x10, 0, dispatch_id=0)])
    workload = schema.Workload(filter_kernel_ids=[0, 1])
    with patch("utils.parser.console_error"):
        df = load_pc_sampling_data(workload, "count", [tool_data])
    assert df.empty


def test_load_pc_sampling_data_single_kernel_valid() -> None:
    """Return per-kernel data when exactly one valid kernel ID is filtered."""
    tool_data = make_tool_data(
        stochastic=[make_record(100, 0x10, 0, dispatch_id=0)],
        instructions=["v_mov_b32"],
        comments=["/src/vcopy.cpp:42"],
        kernel_symbols=[make_kernel_symbol(100, 100, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
    )
    workload = schema.Workload(
        filter_kernel_ids=[0],
        dfs={PMC_KERNEL_TOP_TABLE_ID: pd.DataFrame({"Kernel_Name": ["vecCopy"]})},
    )
    df = load_pc_sampling_data(workload, "count", [tool_data])
    assert not df.empty


def test_load_pc_sampling_data_single_kernel_out_of_bounds() -> None:
    """Return an empty DataFrame when the filtered kernel ID exceeds kernel-top."""
    tool_data = make_tool_data(
        stochastic=[make_record(100, 0x10, 0, dispatch_id=0)],
        kernel_symbols=[make_kernel_symbol(100, 100, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
    )
    workload = schema.Workload(
        filter_kernel_ids=[99],
        dfs={
            PMC_KERNEL_TOP_TABLE_ID: pd.DataFrame({
                "Kernel_Name": ["vecCopy", "vecAdd"]
            })
        },
    )
    df = load_pc_sampling_data(workload, "count", [tool_data])
    assert df.empty


def test_load_pc_sampling_data_method_not_detected() -> None:
    """Return an empty DataFrame when neither pc_sample array is populated."""
    tool_data = make_tool_data(
        kernel_symbols=[make_kernel_symbol(100, 100, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
    )
    workload = schema.Workload(
        filter_kernel_ids=[0],
        dfs={PMC_KERNEL_TOP_TABLE_ID: pd.DataFrame({"Kernel_Name": ["vecCopy"]})},
    )
    df = load_pc_sampling_data(workload, "count", [tool_data])
    assert df.empty


def test_load_pc_sampling_data_detects_method_after_empty_record() -> None:
    """A zero-sample first process does not hide later sampled processes."""
    empty_tool_data = make_tool_data(
        kernel_dispatch=[make_dispatch(0, 100)],
        pid=101,
    )
    sampled_tool_data = make_tool_data(
        stochastic=[make_record(5, 0x10, 0, dispatch_id=0)],
        instructions=["v_mov"],
        comments=["/s/a.cpp:1"],
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
        code_objects=[make_code_object(5)],
        pid=202,
    )

    expected = load_pc_sampling_data(
        schema.Workload(),
        "count",
        [sampled_tool_data],
    )
    actual = load_pc_sampling_data(
        schema.Workload(),
        "count",
        [empty_tool_data, sampled_tool_data],
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_load_pc_sampling_data_rejects_conflicting_methods() -> None:
    """Records from one invocation must not mix PC sampling methods."""
    host_trap_tool_data = setup_pc_sampling_data(method="host_trap")
    stochastic_tool_data = setup_pc_sampling_data(method="stochastic")

    with patch("utils.parser.console_error") as console_error_mock:
        df = load_pc_sampling_data(
            schema.Workload(),
            "count",
            [host_trap_tool_data, stochastic_tool_data],
        )

    assert df.empty
    console_error_mock.assert_called_once()
    assert "conflicting" in console_error_mock.call_args.args[0]
    assert console_error_mock.call_args.kwargs == {"exit": False}


@pytest.mark.parametrize(
    "populated, expected_column_count",
    [
        ("host_trap", 7),  # host_trap-only is detected
        ("both", 10),  # stochastic wins when both arrays are populated
    ],
)
def test_load_pc_sampling_data_method_detection(
    populated: str,
    expected_column_count: int,
) -> None:
    """Single-kernel method detection: host_trap-only and stochastic-priority."""
    kwargs = {"host_trap": [make_record(5, 0x10, 0, dispatch_id=0)]}
    if populated == "both":
        kwargs["stochastic"] = [make_record(5, 0x10, 0, dispatch_id=0)]
    tool_data = make_tool_data(
        instructions=["v_mov"],
        comments=["/s/a.cpp:1"],
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
        **kwargs,
    )
    workload = schema.Workload(
        filter_kernel_ids=[0],
        dfs={PMC_KERNEL_TOP_TABLE_ID: pd.DataFrame({"Kernel_Name": ["vecCopy"]})},
    )
    df = load_pc_sampling_data(workload, "count", [tool_data])
    assert len(df.columns) == expected_column_count


def test_load_pc_sampling_data_no_filter_instruction_out_of_range() -> None:
    """No-filter: an inst_index past the instruction table yields a None entry."""
    tool_data = make_tool_data(
        stochastic=[make_record(5, 0x10, 1, dispatch_id=0)],
        instructions=["v_mov"],  # len 1; inst_index 1 is out of range
        comments=["/s/a.cpp:1", "/s/a.cpp:2"],  # len 2; inst_index 1 in range
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
    )
    df = load_pc_sampling_data(schema.Workload(), "count", [tool_data])
    assert not df.empty
    assert pd.isna(df.iloc[0]["instruction"])
    assert df.iloc[0]["source_line"] == ".../a.cpp:2"


def test_load_pc_sampling_data_preserves_display_identity_boundaries() -> None:
    """Keep rows distinct when kernel, instruction, or source metadata differs."""
    tool_data_records = [
        make_display_row_tool_data(
            "host_trap",
            code_object_id=5,
            kernel_name="sharedKernel",
            instruction="v_add",
            source_line="/src/shared.cpp:10",
            pid=101,
        ),
        make_display_row_tool_data(
            "host_trap",
            code_object_id=7,
            kernel_name="sharedKernel",
            instruction="v_mul",
            source_line="/src/shared.cpp:10",
            pid=202,
        ),
        make_display_row_tool_data(
            "host_trap",
            code_object_id=9,
            kernel_name="sharedKernel",
            instruction="v_add",
            source_line="/src/other.cpp:20",
            pid=303,
        ),
        make_display_row_tool_data(
            "host_trap",
            code_object_id=11,
            kernel_name="otherKernel",
            instruction="v_add",
            source_line="/src/shared.cpp:10",
            pid=404,
        ),
    ]

    df = load_pc_sampling_data(
        schema.Workload(),
        "offset",
        tool_data_records,
    )

    actual_rows = set(
        df[["Kernel_Name", "offset", "instruction", "source_line", "count"]].itertuples(
            index=False, name=None
        )
    )
    assert actual_rows == {
        ("sharedKernel", "0x10", "v_add", ".../shared.cpp:10", 1),
        ("sharedKernel", "0x10", "v_mul", ".../shared.cpp:10", 1),
        ("sharedKernel", "0x10", "v_add", ".../other.cpp:20", 1),
        ("otherKernel", "0x10", "v_add", ".../shared.cpp:10", 1),
    }


def test_load_pc_sampling_data_retains_missing_instruction_metadata() -> None:
    """Keep per-process rows even when their instruction metadata is missing."""
    tool_data_records = [
        make_display_row_tool_data(
            "host_trap",
            code_object_id=5,
            kernel_name="sharedKernel",
            instruction=None,
            source_line="/src/shared.cpp:10",
            pid=101,
        ),
        make_display_row_tool_data(
            "host_trap",
            code_object_id=7,
            kernel_name="sharedKernel",
            instruction=None,
            source_line="/src/shared.cpp:10",
            pid=202,
        ),
    ]

    df = load_pc_sampling_data(
        schema.Workload(),
        "count",
        tool_data_records,
    )

    assert len(df) == 2
    assert list(df["pid"]) == [101, 202]
    assert list(df["count"]) == [1, 1]
    assert df["instruction"].isna().all()


def test_load_pc_sampling_data_keeps_multi_process_rows_separate() -> None:
    """Rows from separate result records stay separate, keyed by process."""
    first_tool_data = make_tool_data(
        stochastic=[
            make_record(
                5,
                0x10,
                0,
                dispatch_id=0,
                wave_issued=False,
                stall_reason=f"{PREFIX}WAITCNT",
            ),
            make_record(
                5,
                0x10,
                0,
                dispatch_id=0,
                wave_issued=False,
                stall_reason=f"{PREFIX}WAITCNT",
            ),
            make_record(6, 0x20, 1, dispatch_id=1, wave_issued=True),
        ],
        instructions=["v_mov", "v_add"],
        comments=["/src/shared.cpp:10", "/src/distinct.cpp:20"],
        kernel_symbols=[
            make_kernel_symbol(100, 5, "sharedKernel"),
            make_kernel_symbol(101, 6, "distinctKernel"),
        ],
        kernel_dispatch=[make_dispatch(0, 100), make_dispatch(1, 101)],
        code_objects=[make_code_object(5), make_code_object(6)],
        pid=101,
    )
    second_tool_data = make_tool_data(
        stochastic=[
            make_record(7, 0x10, 0, dispatch_id=0, wave_issued=True),
            make_record(
                7,
                0x10,
                0,
                dispatch_id=0,
                wave_issued=False,
                stall_reason=f"{PREFIX}ALU_DEPENDENCY",
            ),
            make_record(
                7,
                0x10,
                0,
                dispatch_id=0,
                wave_issued=False,
                stall_reason=f"{PREFIX}WAITCNT",
            ),
        ],
        instructions=["v_mov"],
        comments=["/src/shared.cpp:10"],
        kernel_symbols=[make_kernel_symbol(200, 7, "sharedKernel")],
        kernel_dispatch=[make_dispatch(0, 200)],
        code_objects=[make_code_object(7)],
        pid=202,
    )

    df = load_pc_sampling_data(
        schema.Workload(),
        "count",
        [first_tool_data, second_tool_data],
    )

    assert set(df["Kernel_Name"]) == {"sharedKernel", "distinctKernel"}
    shared_rows = df[df["Kernel_Name"] == "sharedKernel"]
    # Both processes sampled offset 0x10 under the same kernel name, in code
    # objects that only happen to be numbered differently. The rows stay apart.
    assert list(shared_rows["pid"]) == [202, 101]
    assert list(shared_rows["offset"]) == ["0x10", "0x10"]
    assert list(shared_rows["code_object_id"]) == [7, 5]
    assert list(shared_rows["count"]) == [3, 2]
    assert list(shared_rows["count_issued"]) == [1, 0]
    assert list(shared_rows["count_stalled"]) == [2, 2]
    assert list(shared_rows["stall_reason"]) == [
        [("ALU_DEPENDENCY", 1), ("WAITCNT", 1)],
        [("WAITCNT", 2)],
    ]


def test_load_pc_sampling_data_applies_top_n_across_process_rows() -> None:
    """Apply the row limit to the combined per-process rows."""
    first_tool_data = make_tool_data(
        host_trap=[
            *[make_host_trap_record(5, 0x10, 0, dispatch_id=0) for _ in range(3)],
            *[make_host_trap_record(6, 0x20, 1, dispatch_id=1) for _ in range(5)],
        ],
        instructions=["shared", "first_leader"],
        comments=["/src/shared.cpp:10", "/src/first.cpp:20"],
        kernel_symbols=[
            make_kernel_symbol(100, 5, "sharedKernel"),
            make_kernel_symbol(101, 6, "firstLeader"),
        ],
        kernel_dispatch=[make_dispatch(0, 100), make_dispatch(1, 101)],
        pid=101,
    )
    second_tool_data = make_tool_data(
        host_trap=[
            *[make_host_trap_record(7, 0x10, 0, dispatch_id=0) for _ in range(3)],
            *[make_host_trap_record(8, 0x30, 1, dispatch_id=1) for _ in range(4)],
        ],
        instructions=["shared", "second_leader"],
        comments=["/src/shared.cpp:10", "/src/second.cpp:30"],
        kernel_symbols=[
            make_kernel_symbol(200, 7, "sharedKernel"),
            make_kernel_symbol(201, 8, "secondLeader"),
        ],
        kernel_dispatch=[make_dispatch(0, 200), make_dispatch(1, 201)],
        pid=202,
    )

    df = load_pc_sampling_data(
        schema.Workload(),
        "count",
        [first_tool_data, second_tool_data],
        num_rows=1,
    )

    # Per-process rows are ranked as they are; nothing is summed first, so the
    # single largest row wins rather than a cross-process total.
    assert len(df) == 1
    assert df.iloc[0]["Kernel_Name"] == "firstLeader"
    assert df.iloc[0]["pid"] == 101
    assert df.iloc[0]["count"] == 5


# ═══════════════════════════════════════════════════════════════
# nullify_unevaluated_metric_values
# ═══════════════════════════════════════════════════════════════


def test_nullify_unevaluated_metrics_metric_table_nullified() -> None:
    """
    Replace Value/Avg/Min/Max with 'N/A' in metric tables
    while preserving Metric_ID and Metric.
    """
    df = pd.DataFrame({
        "Metric_ID": ["1.1.0", "1.1.1"],
        "Metric": ["Wavefronts", "VALU Insts"],
        "Value": [
            "AVG(SQ_WAVES)",
            "AVG(SQ_INSTS_VALU)",
        ],
        "Avg": ["formula1", "formula2"],
        "Min": ["formula3", "formula4"],
        "Max": ["formula5", "formula6"],
    })
    workload = schema.Workload(
        dfs={10: df},
        dfs_type={10: "metric_table"},
    )
    nullify_unevaluated_metric_values(workload)
    for col in ["Value", "Avg", "Min", "Max"]:
        assert (workload.dfs[10][col] == "N/A").all()
    assert workload.dfs[10]["Metric_ID"].iloc[0] == "1.1.0"
    assert workload.dfs[10]["Metric"].iloc[0] == "Wavefronts"


def test_nullify_unevaluated_metrics_non_metric_table_untouched() -> None:
    """Leave non-metric-table DataFrames unchanged."""
    df = pd.DataFrame({"Value": [42, 99]})
    workload = schema.Workload(
        dfs={20: df},
        dfs_type={20: "raw_csv_table"},
    )
    nullify_unevaluated_metric_values(workload)
    assert workload.dfs[20]["Value"].tolist() == [42, 99]


def test_nullify_unevaluated_metrics_empty_df_skipped() -> None:
    """Skip empty DataFrames without error even when typed as metric_table."""
    df = pd.DataFrame()
    workload = schema.Workload(
        dfs={30: df},
        dfs_type={30: "metric_table"},
    )
    nullify_unevaluated_metric_values(workload)
    assert workload.dfs[30].empty


# ═══════════════════════════════════════════════════════════════
# load_pc_sampling_results
# ═══════════════════════════════════════════════════════════════


def test_load_pc_sampling_results_loads_all_pid_files_in_numeric_order(
    tmp_path: Path,
) -> None:
    """Load only numeric PID result files in PID order."""
    write_results_json(tmp_path / "101_ps_file_results.json", pid=101)
    write_results_json(tmp_path / "3_ps_file_results.json", pid=3)
    write_results_json(tmp_path / "20_ps_file_results.json", pid=20)

    nested_path = tmp_path / "nested"
    nested_path.mkdir()
    nested_result_path = nested_path / "1_ps_file_results.json"
    write_results_json(nested_result_path, pid=1)
    (tmp_path / "1_ps_file_results.json").symlink_to(nested_result_path)
    write_results_json(tmp_path / "worker_ps_file_results.json", pid=2)
    write_results_json(tmp_path / "4_ps_file_results.json.backup", pid=4)
    (tmp_path / "5_ps_file_results.json").mkdir()

    tool_data_records = load_pc_sampling_results(str(tmp_path))

    assert [record["metadata"]["pid"] for record in tool_data_records] == [
        1,
        3,
        20,
        101,
    ]


@pytest.mark.parametrize(
    "directory_exists",
    [
        pytest.param(True, id="empty_directory"),
        pytest.param(False, id="missing_directory"),
    ],
)
def test_load_pc_sampling_results_returns_empty_without_result_files(
    tmp_path: Path,
    directory_exists: bool,
) -> None:
    """Return no records when the directory or its result files are absent."""
    workload_path = tmp_path if directory_exists else tmp_path / "missing"

    assert load_pc_sampling_results(str(workload_path)) == []


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param({}, id="missing_pid"),
        pytest.param({"pid": None}, id="null_pid"),
    ],
)
def test_load_pc_sampling_results_single_record_requires_process_id(
    tmp_path: Path,
    metadata: dict,
) -> None:
    """Require a concrete metadata PID for every result record."""
    tool_data = make_tool_data()
    tool_data["metadata"] = metadata
    result_path = tmp_path / "42_ps_file_results.json"
    result_path.write_text(
        json.dumps({"rocprofiler-sdk-tool": [tool_data]}), encoding="utf-8"
    )

    with pytest.raises(SystemExit):
        load_pc_sampling_results(str(tmp_path))


def test_load_pc_sampling_results_rejects_duplicate_process_ids(
    tmp_path: Path,
) -> None:
    """Reject result records that share the same metadata PID."""
    write_results_json(tmp_path / "101_ps_file_results.json", pid=101)
    write_results_json(tmp_path / "202_ps_file_results.json", pid=101)

    with pytest.raises(SystemExit):
        load_pc_sampling_results(str(tmp_path))


# ═══════════════════════════════════════════════════════════════
# process_pc_sampling_kernel_trace
# ═══════════════════════════════════════════════════════════════


def test_process_pc_sampling_none_returns_empty() -> None:
    """Return empty DataFrame with expected columns when tool_data is None."""
    df = process_pc_sampling_kernel_trace(None)
    assert df.empty
    assert list(df.columns) == [
        "Dispatch_Id",
        "PID",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "GPU_ID",
    ]


def test_process_pc_sampling_with_agent_info() -> None:
    """Verify column selection, GPU mapping, kernel names, and timestamps."""
    tool_data = make_tool_data(
        kernel_symbols=[
            make_kernel_symbol(12, 2, "vecCopy"),
            make_kernel_symbol(13, 2, "vecAdd"),
            make_kernel_symbol(14, 2, "vecMul"),
        ],
        kernel_dispatch=[
            make_dispatch(
                1, 12, agent_handle=20, start=1981199661678356, end=1981199662835032
            ),
            make_dispatch(2, 13, agent_handle=30, start=2000, end=3000),
            make_dispatch(3, 14, agent_handle=99, start=4000, end=5000),
        ],
        agents=[
            make_agent(handle=10, node_id=1, agent_type=1),
            make_agent(handle=20, node_id=2, agent_type=2),
            make_agent(handle=30, node_id=3, agent_type=2),
        ],
        pid=42,
    )

    df = process_pc_sampling_kernel_trace(tool_data)

    # Correct shape and columns
    assert len(df) == 3
    assert list(df.columns) == [
        "Dispatch_Id",
        "PID",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "GPU_ID",
    ]

    # Multi-GPU mapping: handle 20 -> GPU 0, handle 30 -> GPU 1, unknown -> 0
    assert df["GPU_ID"].tolist() == [0, 1, 0]
    assert df["PID"].tolist() == [42, 42, 42]
    assert df["Kernel_Name"].tolist() == ["vecCopy", "vecAdd", "vecMul"]

    # Timestamps passed through unchanged
    assert df["Start_Timestamp"].iloc[0] == 1981199661678356
    assert df["End_Timestamp"].iloc[0] == 1981199662835032


def test_process_pc_sampling_no_gpu_agents() -> None:
    """Default GPU_ID to 0 when no GPU agents are present."""
    tool_data = make_tool_data(
        kernel_symbols=[make_kernel_symbol(12, 2, "vecCopy")],
        kernel_dispatch=[make_dispatch(1, 12, agent_handle=99, start=1000, end=2000)],
        agents=[make_agent(handle=10, node_id=1, agent_type=1)],
    )
    df = process_pc_sampling_kernel_trace(tool_data)
    assert len(df) == 1
    assert df["GPU_ID"].iloc[0] == 0


def test_process_pc_sampling_unmapped_kernel_id() -> None:
    """A dispatch whose kernel_id is absent from kernel_symbols maps to None."""
    tool_data = make_tool_data(
        kernel_symbols=[make_kernel_symbol(12, 2, "vecCopy")],
        kernel_dispatch=[make_dispatch(1, 999, agent_handle=20, start=1, end=2)],
        agents=[make_agent(handle=20, node_id=2, agent_type=2)],
    )
    df = process_pc_sampling_kernel_trace(tool_data)
    assert len(df) == 1
    assert df.iloc[0]["Kernel_Name"] is None


# ═══════════════════════════════════════════════════════════════
# build_agent_to_gpu_map_from_json
# ═══════════════════════════════════════════════════════════════


def test_build_agent_to_gpu_map_single_gpu() -> None:
    """Map one GPU agent to GPU index 0, ignoring CPU agents."""
    agents = [
        make_agent(handle=10, node_id=1, agent_type=1),
        make_agent(handle=20, node_id=2, agent_type=2),
    ]
    assert build_agent_to_gpu_map_from_json(agents) == {20: 0}


def test_build_agent_to_gpu_map_two_gpus() -> None:
    """Assign sequential GPU indices to multiple GPU agents sorted by node_id."""
    agents = [
        make_agent(handle=10, node_id=1, agent_type=1),
        make_agent(handle=30, node_id=3, agent_type=2),
        make_agent(handle=20, node_id=2, agent_type=2),
    ]
    assert build_agent_to_gpu_map_from_json(agents) == {20: 0, 30: 1}


def test_build_agent_to_gpu_map_no_gpu_agents() -> None:
    """Return an empty map when no GPU agents are present."""
    agents = [
        make_agent(handle=10, node_id=1, agent_type=1),
        make_agent(handle=11, node_id=2, agent_type=1),
    ]
    assert build_agent_to_gpu_map_from_json(agents) == {}


def test_build_agent_to_gpu_map_empty() -> None:
    """Return an empty map when the agents list is empty."""
    assert build_agent_to_gpu_map_from_json([]) == {}


# ═══════════════════════════════════════════════════════════════
# load_pc_sampling_tool_data / calc_dispatch_data
# ═══════════════════════════════════════════════════════════════


def make_db_analysis(workload_path: str) -> db_analysis:
    """Construct a db_analysis whose only populated state is _runs."""
    instance = db_analysis.__new__(db_analysis)
    instance._runs = {workload_path: schema.Workload()}
    return instance


def test_load_pc_sampling_tool_data_gate(tmp_path: Path) -> None:
    """Tool data loads whenever PC sampling was collected, else returns empty."""
    write_results_json(
        tmp_path / "42_ps_file_results.json", **sample_tool_data_kwargs()
    )
    instance = make_db_analysis(str(tmp_path))

    instance._profiling_config = {"filter_blocks": ["21", "2"]}  # mixed
    assert instance.pc_sampling_collected() is True
    assert instance.pc_sampling_only() is False
    assert instance.load_pc_sampling_tool_data(str(tmp_path))

    instance._profiling_config = {"filter_blocks": ["21"]}  # pc sampling only
    assert instance.pc_sampling_only() is True
    assert instance.load_pc_sampling_tool_data(str(tmp_path))

    instance._profiling_config = {"filter_blocks": ["2"]}  # counters only
    assert instance.pc_sampling_collected() is False
    assert instance.load_pc_sampling_tool_data(str(tmp_path)) == []


def test_pc_sampling_kernel_traces_renumber_colliding_dispatch_ids() -> None:
    """Give every process's dispatch an id unique across the workload."""
    combined_trace = process_pc_sampling_kernel_traces(
        make_multiprocess_dispatch_tool_data()
    )

    # Both records carry a process-local dispatch 0.
    assert combined_trace["Dispatch_Id"].tolist() == [0, 1, 2]
    assert combined_trace["PID"].tolist() == [101, 101, 202]
    assert combined_trace["Dispatch_Id"].is_unique


def test_pc_sampling_kernel_traces_preserve_single_process_ids() -> None:
    """Leave a lone process's dispatch ordering untouched."""
    single_record = make_multiprocess_dispatch_tool_data()[:1]

    combined_trace = process_pc_sampling_kernel_traces(single_record)

    assert combined_trace["Dispatch_Id"].tolist() == [0, 1]


def test_pc_sampling_multiprocess_dispatch_statistics_include_every_row(
    tmp_path: Path,
) -> None:
    """Include every process-local dispatch in workload statistics."""
    workload = schema.Workload()
    args = argparse.Namespace(time_unit="ns")
    instance = make_db_analysis(str(tmp_path))

    instance.build_pc_sampling_only_workload(
        workload,
        str(tmp_path),
        args,
        make_multiprocess_dispatch_tool_data(),
    )

    assert workload.raw_pmc["Dispatch_ID"].tolist() == [0, 1, 2]
    assert workload.raw_pmc["PID"].tolist() == [101, 101, 202]
    kernel_top = workload.dfs[PMC_KERNEL_TOP_TABLE_ID].iloc[0]
    assert kernel_top["Count"] == 3
    assert kernel_top["Sum(ns)"] == 60
    dispatch_info = workload.dfs[PMC_DISPATCH_INFO_TABLE_ID]
    assert dispatch_info["Dispatch_ID"].tolist() == [0, 1, 2]
    assert dispatch_info["PID"].tolist() == [101, 101, 202]


def test_pc_sampling_dispatch_filter_selects_one_process(
    tmp_path: Path,
) -> None:
    """Resolve a filtered dispatch ID to exactly one process's dispatch."""
    workload = schema.Workload(filter_dispatch_ids=["2"])
    args = argparse.Namespace(time_unit="ns")
    instance = make_db_analysis(str(tmp_path))

    instance.build_pc_sampling_only_workload(
        workload,
        str(tmp_path),
        args,
        make_multiprocess_dispatch_tool_data(),
    )

    # Dispatch 2 is the second process's only dispatch; before renumbering it
    # shared id 0 with the first process and both were selected.
    kernel_top = workload.dfs[PMC_KERNEL_TOP_TABLE_ID].iloc[0]
    assert kernel_top["Count"] == 1
    assert kernel_top["Sum(ns)"] == 30
    dispatch_info = workload.dfs[PMC_DISPATCH_INFO_TABLE_ID]
    assert dispatch_info["Dispatch_ID"].tolist() == [2]
    assert dispatch_info["PID"].tolist() == [202]


def test_load_table_data_forwards_pc_sampling_tool_data() -> None:
    """load_table_data forwards tool data to load_non_mertrics_table."""
    sentinel = [{"sentinel": True}]
    args = argparse.Namespace(debug=False)
    workload = schema.Workload()
    workload.sys_info = pd.DataFrame([{"gpu_arch": "gfx942"}])
    with patch("utils.parser.load_non_mertrics_table") as mock_load_non_metrics:
        with patch("utils.parser.eval_metric"):
            with patch("utils.parser.apply_filters"):
                load_table_data(
                    workload=workload,
                    dir_path="dir",
                    is_gui=False,
                    args=args,
                    dfs_expressions={},
                    pc_sampling_tool_data=sentinel,
                )
    mock_load_non_metrics.assert_called_once()
    assert mock_load_non_metrics.call_args.args[3] is sentinel


def test_load_non_mertrics_table_populates_pc_sampling_from_tool_data(
    tmp_path: Path,
) -> None:
    """A ``from_pc_sampling`` table is populated when tool data is provided."""
    args = argparse.Namespace(pc_sampling_sorting_type="count", pc_sampling_rows=10)
    workload = schema.Workload()
    workload.dfs = {2101: pd.DataFrame({"from_pc_sampling": [True]})}
    tool_data = make_tool_data(**sample_tool_data_kwargs())
    load_non_mertrics_table(
        workload, str(tmp_path), args, pc_sampling_tool_data=[tool_data]
    )
    assert not workload.dfs[2101].empty


def test_load_non_mertrics_table_pc_sampling_empty_without_tool_data(
    tmp_path: Path,
) -> None:
    """Without tool data the ``from_pc_sampling`` table stays empty (no crash)."""
    args = argparse.Namespace(pc_sampling_sorting_type="count", pc_sampling_rows=10)
    workload = schema.Workload()
    workload.dfs = {2101: pd.DataFrame({"from_pc_sampling": [True]})}
    load_non_mertrics_table(workload, str(tmp_path), args)
    assert workload.dfs[2101].empty


# ═══════════════════════════════════════════════════════════════
# aggregate_pc_sample_records / load_aggregated_pc_sampling
# ═══════════════════════════════════════════════════════════════


def test_aggregate_adds_inst_type_dict() -> None:
    """The shared aggregation carries a per-group inst_type count dict."""
    tool_data = make_tool_data(
        stochastic=[
            make_record(5, 0x10, 0, dispatch_id=0, inst_type=f"{INST_PREFIX}VALU"),
            make_record(5, 0x10, 0, dispatch_id=0, inst_type=f"{INST_PREFIX}VALU"),
            make_record(5, 0x10, 0, dispatch_id=0, inst_type=f"{INST_PREFIX}FLAT"),
        ],
    )
    records_df = load_pc_sample_records(tool_data)
    aggregated = aggregate_pc_sample_records(
        records_df, group_by=["code_object_id", "code_object_offset"], sys_info=None
    )
    assert aggregated.iloc[0]["inst_type"] == {"VALU": 2, "FLAT": 1}


def test_normalize_missing_tool_data_returns_empty() -> None:
    """An empty tool record yields no code-object records."""
    assert load_aggregated_pc_sampling(make_tool_data(), sys_info=None) == []


def test_normalize_groups_by_code_object_with_catalog() -> None:
    """Records group per code object carrying load_base."""
    tool_data = make_tool_data(
        stochastic=[
            make_record(5, 0x10, 0, dispatch_id=0, inst_type=f"{INST_PREFIX}VALU"),
            make_record(5, 0x20, 1, dispatch_id=1, inst_type=f"{INST_PREFIX}FLAT"),
            make_record(6, 0x30, 2, dispatch_id=2, inst_type=f"{INST_PREFIX}VALU"),
        ],
        instructions=["v_mov", "v_add", "v_sub"],
        comments=["/s/a.cpp:1", "/s/a.cpp:2", "/s/b.cpp:1"],
        kernel_symbols=[
            make_kernel_symbol(100, 5, "vecCopy"),
            make_kernel_symbol(101, 5, "vecAdd"),
            make_kernel_symbol(102, 6, "vecSub"),
        ],
        kernel_dispatch=[
            make_dispatch(0, 100),
            make_dispatch(1, 101),
            make_dispatch(2, 102),
        ],
        code_objects=[
            make_code_object(5, load_base=0x1000),
            make_code_object(6, load_base=0x2000),
        ],
    )
    records = {
        r.code_object_id: r
        for r in load_aggregated_pc_sampling(tool_data, sys_info=None)
    }
    assert records[5].load_base == 0x1000
    assert len(records[5].instruction_lines) == 2
    assert len(records[6].instruction_lines) == 1


def test_normalize_attributes_line_to_kernel_via_dispatch() -> None:
    """Each offset in a shared code object is attributed to its dispatch kernel."""
    # code object 5 holds vecCopy (kernel 100) and vecAdd (kernel 101) at
    # distinct offsets; names resolve via kernel_id (dispatch correlation).
    tool_data = make_tool_data(
        stochastic=[
            make_record(5, 0x10, 0, dispatch_id=0, inst_type=f"{INST_PREFIX}VALU"),
            make_record(5, 0x20, 1, dispatch_id=1, inst_type=f"{INST_PREFIX}FLAT"),
        ],
        instructions=["v_mov", "v_add"],
        comments=["/s/a.cpp:1", "/s/a.cpp:2"],
        kernel_symbols=[
            make_kernel_symbol(100, 5, "vecCopy"),
            make_kernel_symbol(101, 5, "vecAdd"),
        ],
        kernel_dispatch=[make_dispatch(0, 100), make_dispatch(1, 101)],
        code_objects=[make_code_object(5)],
    )
    lines = load_aggregated_pc_sampling(tool_data, sys_info=None)[0].instruction_lines
    by_offset = {line.code_object_offset: line.kernel_name for line in lines}
    assert by_offset[0x10] == "vecCopy"
    assert by_offset[0x20] == "vecAdd"


def test_normalize_instruction_line_counts_and_dicts() -> None:
    """A sampled line carries totals, issue/stall counts, and typed dicts."""
    tool_data = make_tool_data(
        stochastic=[
            make_record(
                5,
                0x10,
                0,
                dispatch_id=0,
                wave_issued=True,
                inst_type=f"{INST_PREFIX}VALU",
            ),
            make_record(
                5,
                0x10,
                0,
                dispatch_id=0,
                wave_issued=False,
                stall_reason=f"{PREFIX}WAITCNT",
                inst_type=f"{INST_PREFIX}VALU",
            ),
        ],
        instructions=["v_mov"],
        comments=["/s/a.cpp:1"],
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
        code_objects=[make_code_object(5)],
    )
    line = load_aggregated_pc_sampling(tool_data, sys_info=None)[0].instruction_lines[0]
    assert line.code_object_offset == 0x10
    assert line.instruction == "v_mov"
    assert line.total_count == 2
    assert line.issue_count == 1
    assert line.stall_count == 1
    assert line.stall_reasons == {"WAITCNT": 1}
    assert line.inst_types == {"VALU": 2}


def test_normalize_host_trap_line_has_null_issue_stall() -> None:
    """host_trap lines have no issue/stall info and no stall-reason dict."""
    tool_data = make_tool_data(
        host_trap=[make_host_trap_record(5, 0x10, 0, dispatch_id=0)],
        instructions=["v_mov"],
        comments=["/s/a.cpp:1"],
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        code_objects=[make_code_object(5)],
    )
    line = load_aggregated_pc_sampling(tool_data, sys_info=None)[0].instruction_lines[0]
    assert line.total_count == 1
    assert line.issue_count is None
    assert line.stall_count is None
    assert line.stall_reasons == {}


def test_load_pc_sampling_data_no_debug_info_source_line_na() -> None:
    """Empty comment strings yield 'N/A' source lines across a multi-row table."""
    # Without debug info every comment is "", which must not collapse the table.
    tool_data = make_tool_data(
        stochastic=[
            make_record(5, 0x10, 0, dispatch_id=0, wave_issued=True),
            make_record(5, 0x20, 1, dispatch_id=0, wave_issued=True),
        ],
        instructions=["v_mov", "v_add"],
        comments=["", ""],
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
    )
    df = load_pc_sampling_data(schema.Workload(), "count", [tool_data])
    assert len(df) == 2
    assert (df["source_line"] == "N/A").all()


def test_normalize_unmapped_dispatch_yields_none_kernel() -> None:
    """A sample whose dispatch has no kernel mapping yields a None kernel_name."""
    tool_data = make_tool_data(
        stochastic=[make_record(999, 0x10, 0, dispatch_id=0)],
        instructions=["v_mov"],
        comments=["/s/a.cpp:1"],
        kernel_symbols=[make_kernel_symbol(100, 100, "vecCopy")],
        code_objects=[make_code_object(999)],
    )
    record = load_aggregated_pc_sampling(tool_data, sys_info=None)[0]
    assert record.code_object_id == 999
    assert record.instruction_lines[0].kernel_name is None


def test_calc_dispatch_data_uses_provided_tool_data(tmp_path: Path) -> None:
    """calc_dispatch_data builds PC-sampling dispatch rows from the provided map."""
    tool_data = make_tool_data(
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100, agent_handle=20, start=10, end=20)],
        agents=[make_agent(handle=20, node_id=2, agent_type=2)],
    )
    instance = make_db_analysis(str(tmp_path))
    instance._profiling_config = {"filter_blocks": ["21"]}  # pc_sampling_only -> True
    result = instance.calc_dispatch_data({str(tmp_path): [tool_data]})
    df = result[str(tmp_path)]
    assert list(df.columns) == [
        "dispatch_id",
        "kernel_name",
        "gpu_id",
        "start_timestamp",
        "end_timestamp",
    ]
    assert df.iloc[0]["kernel_name"] == "vecCopy"
    assert df.iloc[0]["gpu_id"] == 0


def test_calc_dispatch_data_warns_for_empty_pc_sampling_results(
    tmp_path: Path,
) -> None:
    """An empty collection emits the missing-results warning."""
    instance = make_db_analysis(str(tmp_path))
    instance._profiling_config = {"filter_blocks": ["21"]}

    with patch("utils.file_io.console_warning") as warning:
        result = instance.calc_dispatch_data({str(tmp_path): []})

    assert result[str(tmp_path)].empty
    warning.assert_called_once_with(
        "PC sampling results not found. Cannot build dispatch data."
    )


def test_calc_dispatch_data_stitches_pc_sampling_tool_records(
    tmp_path: Path,
) -> None:
    """Combine dispatch rows from all PC sampling tool records."""
    instance = make_db_analysis(str(tmp_path))
    instance._profiling_config = {"filter_blocks": ["21"]}

    result = instance.calc_dispatch_data({
        str(tmp_path): make_multiprocess_dispatch_tool_data()
    })

    df = result[str(tmp_path)]
    assert df["kernel_name"].tolist() == ["vecCopy", "vecCopy", "vecCopy"]
    assert df["dispatch_id"].tolist() == [0, 1, 2]
