# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pc_sampling import pc_sampling_analysis
from pc_sampling.pc_sampling_analysis import (
    aggregate_instruction_lines,
    aggregate_pc_sample_records,
    detect_pc_sampling_method,
    enrich_with_metadata,
    load_aggregated_pc_sampling,
    load_pc_sample_records,
)
from rocprof_compute_analyze.analysis_db import db_analysis, pc_sampling_dispatch_trace
from utils import schema
from utils.file_io import (
    build_agent_to_gpu_map_from_json,
    load_pc_sampling_results,
    process_pc_sampling_kernel_trace,
    process_pc_sampling_kernel_traces,
)
from utils.parser import (
    PMC_KERNEL_TOP_TABLE_ID,
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


@pytest.fixture(autouse=True)
def reset_wave_size_clamp_warning():
    """The capped-percentage warning fires once per process, so reset it."""
    pc_sampling_analysis.WAVE_SIZE_CLAMP_WARNED = False


HOST_TRAP_DISPLAY_COLUMNS = [
    "pid",
    "source_line",
    "instruction",
    "instruction_type",
    "code_object_id",
    "offset",
    "count",
    "active_thread_percent",
    "Kernel_Name",
]
STOCHASTIC_DISPLAY_COLUMNS = [
    "pid",
    "source_line",
    "instruction",
    "instruction_type",
    "code_object_id",
    "offset",
    "count",
    "active_thread_percent",
    "issue_count",
    "stall_count",
    "wave_occupancy_percent",
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


def make_instruction_line_record(
    total_count: int,
    issue_count: int | None,
    stall_reasons: dict[str, int],
    active_thread_percent: float | None,
    wave_occupancy_percent: float | None,
) -> pc_sampling_analysis.InstructionLineRecord:
    """Build one aggregated instruction line with the given counts."""
    return pc_sampling_analysis.InstructionLineRecord(
        code_object_offset=0x10,
        kernel_name="k",
        instruction="v_mov",
        source="/s/a.cpp:1",
        total_count=total_count,
        issue_count=issue_count,
        stall_count=None if issue_count is None else total_count - issue_count,
        stall_reasons=stall_reasons,
        inst_types={},
        active_thread_percent=active_thread_percent,
        wave_occupancy_percent=wave_occupancy_percent,
    )


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
    assert pd.isna(records.iloc[0]["exec_mask"])
    assert pd.isna(records.iloc[0]["wave_cnt"])


def test_load_pc_sample_records_keeps_full_width_exec_mask() -> None:
    """A full wave64 mask survives normalization bit for bit."""
    all_ones_wave64 = 2**64 - 1
    records = load_pc_sample_records(
        make_tool_data(
            stochastic=[
                make_record(1, 0x10, 0, dispatch_id=0, exec_mask=all_ones_wave64),
                make_record(1, 0x10, 1, dispatch_id=0),
            ]
        )
    )

    assert records.iloc[0]["exec_mask"] == all_ones_wave64


# ═══════════════════════════════════════════════════════════════
# aggregate_pc_sample_records
# ═══════════════════════════════════════════════════════════════


def test_aggregate_empty_records_returns_columns() -> None:
    """Aggregating an empty record df returns the expected (empty) columns."""
    empty = load_pc_sample_records(make_tool_data())
    result = aggregate_pc_sample_records(
        empty, group_by=["code_object_id", "code_object_offset"], sys_info={}
    )
    assert result.empty
    for column in ("count", "issue_count", "stall_count", "stall_reason"):
        assert column in result.columns
    # kernel_id / inst_index are carried since they are not group keys.
    assert "kernel_id" in result.columns
    assert "inst_index" in result.columns


def test_aggregate_counts_issued_and_stalled() -> None:
    """A mix of issued and stalled samples produces the documented counts."""
    records = load_pc_sample_records(
        make_tool_data(
            stochastic=[
                make_record(
                    1,
                    0x10,
                    0,
                    dispatch_id=0,
                    wave_issued=True,
                    stall_reason=f"{PREFIX}OTHER_WAIT",
                ),
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
        records, group_by=["code_object_id", "code_object_offset"], sys_info={}
    )
    row = result.iloc[0]
    assert row["count"] == 2
    assert row["issue_count"] == 1
    assert row["stall_count"] == 1
    assert row["stall_reason"] == {"WAITCNT": 1}


def test_aggregate_host_trap_counts_are_none() -> None:
    """Without wave_issued info, issued/stalled counts and reasons are None."""
    records = load_pc_sample_records(
        make_tool_data(host_trap=[make_host_trap_record(1, 0x10, 0, dispatch_id=0)])
    )
    result = aggregate_pc_sample_records(
        records, group_by=["code_object_id", "code_object_offset"], sys_info={}
    )
    row = result.iloc[0]
    assert row["count"] == 1
    assert row["issue_count"] is None
    assert row["stall_count"] is None
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
        records, group_by=["code_object_id", "code_object_offset"], sys_info={}
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
        sys_info={},
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

    assert bin(all_ones_wave64).count("1") == 64
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

    aggregated = aggregate_pc_sample_records(
        records,
        group_by=["code_object_id", "code_object_offset"],
        sys_info={"wave_size": "64", "max_waves_per_cu": "32"},
    )
    row = aggregated.iloc[0]

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

    aggregated = aggregate_pc_sample_records(
        records,
        group_by=["code_object_id", "code_object_offset"],
        sys_info={"wave_size": "64", "max_waves_per_cu": "32"},
    )
    row = aggregated.iloc[0]

    if expected_active_thread_percent is None:
        assert pd.isna(row["active_thread_percent"])
    else:
        assert row["active_thread_percent"] == expected_active_thread_percent
    if expected_wave_occupancy_percent is None:
        assert pd.isna(row["wave_occupancy_percent"])
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

    aggregated = aggregate_pc_sample_records(
        records,
        group_by=["code_object_id", "code_object_offset"],
        sys_info={"wave_size": "64", "max_waves_per_cu": "32"},
    )
    row = aggregated.iloc[0]

    assert row["active_thread_percent"] == 3.125, (
        "exec_mask mean must use exactly the samples carrying exec_mask"
    )
    assert row["wave_occupancy_percent"] == 37.5, (
        "wave_cnt mean must use exactly the samples carrying wave_cnt"
    )


def test_aggregate_instruction_lines_sums_counts_and_weights_percentages() -> None:
    """Counts are summed, and each percentage is weighted by sample count.

    An instruction line without a percentage adds no weight, and a count that
    no instruction line has stays unknown.
    """
    totals = aggregate_instruction_lines([
        make_instruction_line_record(3, 1, {"WAITCNT": 2}, None, 50.0),
        make_instruction_line_record(5, 4, {"NONE": 1}, 80.0, 10.0),
    ])
    host_trap_totals = aggregate_instruction_lines([
        make_instruction_line_record(2, None, {}, 50.0, None)
    ])

    assert totals == pc_sampling_analysis.SourceLineRecord(
        total_count=8,
        issue_count=5,
        stall_count=3,
        stall_reasons={"WAITCNT": 2, "NONE": 1},
        active_thread_percent=80.0,
        wave_occupancy_percent=(50.0 * 3 + 10.0 * 5) / 8,
    )
    assert (host_trap_totals.issue_count, host_trap_totals.stall_count) == (None, None)
    assert host_trap_totals.wave_occupancy_percent is None


def test_aggregate_wave_measurements_are_none_without_sys_info() -> None:
    """System information without the machine specs leaves both unknown."""
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

    result = aggregate_pc_sample_records(records, group_by=group_by, sys_info={})

    assert {
        "active_thread_percent",
        "wave_occupancy_percent",
    }.issubset(result.columns)
    assert pd.isna(result.iloc[0]["active_thread_percent"])
    assert pd.isna(result.iloc[0]["wave_occupancy_percent"])


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

    aggregated = aggregate_pc_sample_records(
        records,
        group_by=["code_object_id", "code_object_offset"],
        sys_info=sys_info,
    )
    row = aggregated.iloc[0]

    if expected_active_thread_percent is None:
        assert pd.isna(row["active_thread_percent"])
    else:
        assert row["active_thread_percent"] == expected_active_thread_percent
    if expected_wave_occupancy_percent is None:
        assert pd.isna(row["wave_occupancy_percent"])
    else:
        assert row["wave_occupancy_percent"] == expected_wave_occupancy_percent


def test_aggregate_caps_active_threads_at_a_full_wave() -> None:
    """A wave64 mask on a wave32 machine reports 100 percent, and warns."""
    records = load_pc_sample_records(
        make_tool_data(
            stochastic=[
                make_record(1, 0x10, 0, dispatch_id=0, exec_mask=2**64 - 1),
                make_record(1, 0x20, 1, dispatch_id=0, exec_mask=0b11),
            ]
        )
    )

    with patch("pc_sampling.pc_sampling_analysis.console_warning") as warning_mock:
        result = aggregate_pc_sample_records(
            records,
            group_by=["code_object_id", "code_object_offset"],
            sys_info={"wave_size": "32"},
        )

    by_offset = result.set_index("code_object_offset")["active_thread_percent"]
    assert by_offset[0x10] == 100.0
    assert by_offset[0x20] == 6.25
    warning_mock.assert_called_once()
    assert "capped at 100" in warning_mock.call_args.args[0]


def test_aggregate_warns_about_capping_only_once() -> None:
    """A second workload with the same mismatch does not repeat the warning."""
    records = load_pc_sample_records(
        make_tool_data(
            stochastic=[make_record(1, 0x10, 0, dispatch_id=0, exec_mask=2**64 - 1)]
        )
    )
    aggregate_pc_sample_records(
        records,
        group_by=["code_object_id", "code_object_offset"],
        sys_info={"wave_size": "32"},
    )

    with patch("pc_sampling.pc_sampling_analysis.console_warning") as warning_mock:
        aggregate_pc_sample_records(
            records,
            group_by=["code_object_id", "code_object_offset"],
            sys_info={"wave_size": "32"},
        )

    warning_mock.assert_not_called()


def test_aggregate_caps_wave_occupancy_at_max_waves_per_cu() -> None:
    """wave_cnt exceeding max_waves_per_cu is capped at 100 percent."""
    records = load_pc_sample_records(
        make_tool_data(
            stochastic=[
                make_record(1, 0x10, 0, dispatch_id=0, exec_mask=0b1, wave_cnt=40),
                make_record(1, 0x20, 1, dispatch_id=0, exec_mask=0b1, wave_cnt=4),
            ]
        )
    )

    result = aggregate_pc_sample_records(
        records,
        group_by=["code_object_id", "code_object_offset"],
        sys_info={"wave_size": "64", "max_waves_per_cu": "32"},
    )

    by_offset = result.set_index("code_object_offset")["wave_occupancy_percent"]
    assert by_offset[0x10] == 100.0
    assert by_offset[0x20] == pytest.approx(4 / 32 * 100)


def test_aggregate_does_not_warn_when_every_mask_fits() -> None:
    """A mask within the wave size stays silent."""
    records = load_pc_sample_records(
        make_tool_data(
            stochastic=[make_record(1, 0x10, 0, dispatch_id=0, exec_mask=0b11)]
        )
    )

    with patch("pc_sampling.pc_sampling_analysis.console_warning") as warning_mock:
        aggregate_pc_sample_records(
            records,
            group_by=["code_object_id", "code_object_offset"],
            sys_info={"wave_size": "64"},
        )

    warning_mock.assert_not_called()


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
    line = load_aggregated_pc_sampling(tool_data, {})[0].instruction_lines[0]
    assert line.total_count == 1
    assert line.instruction == "v_mov"
    assert line.source == "/s/a.cpp:1"
    assert line.kernel_name == "vecCopy"


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


def make_db_analysis(workload_path: str) -> db_analysis:
    """Construct a db_analysis whose only populated state is _runs."""
    instance = db_analysis.__new__(db_analysis)
    instance._runs = {workload_path: schema.Workload()}
    return instance


def test_pc_sampling_kernel_traces_renumber_colliding_dispatch_ids() -> None:
    """Give every process's dispatch an id unique across the workload."""
    combined_trace = process_pc_sampling_kernel_traces(
        make_multiprocess_dispatch_tool_data()
    )

    # Both records carry a process-local dispatch 0.
    assert combined_trace["Dispatch_Id"].tolist() == [1, 2, 3]
    assert combined_trace["PID"].tolist() == [101, 101, 202]
    assert combined_trace["Dispatch_Id"].is_unique


def test_pc_sampling_kernel_traces_preserve_single_process_ids() -> None:
    """Leave a lone process's dispatch ordering untouched."""
    single_record = make_multiprocess_dispatch_tool_data()[:1]

    combined_trace = process_pc_sampling_kernel_traces(single_record)

    assert combined_trace["Dispatch_Id"].tolist() == [1, 2]


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
        records_df, group_by=["code_object_id", "code_object_offset"], sys_info={}
    )
    assert aggregated.iloc[0]["inst_type"] == {"VALU": 2, "FLAT": 1}


def test_normalize_missing_tool_data_returns_empty() -> None:
    """An empty tool record yields no code-object records."""
    assert load_aggregated_pc_sampling(make_tool_data(), {}) == []


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
    records = {r.code_object_id: r for r in load_aggregated_pc_sampling(tool_data, {})}
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
    lines = load_aggregated_pc_sampling(tool_data, {})[0].instruction_lines
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
    line = load_aggregated_pc_sampling(tool_data, {})[0].instruction_lines[0]
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
    line = load_aggregated_pc_sampling(tool_data, {})[0].instruction_lines[0]
    assert line.total_count == 1
    assert line.issue_count is None
    assert line.stall_count is None
    assert line.stall_reasons == {}


def test_normalize_unmapped_dispatch_yields_none_kernel() -> None:
    """A sample whose dispatch has no kernel mapping yields a None kernel_name."""
    tool_data = make_tool_data(
        stochastic=[make_record(999, 0x10, 0, dispatch_id=0)],
        instructions=["v_mov"],
        comments=["/s/a.cpp:1"],
        kernel_symbols=[make_kernel_symbol(100, 100, "vecCopy")],
        code_objects=[make_code_object(999)],
    )
    record = load_aggregated_pc_sampling(tool_data, {})[0]
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
    result = instance.calc_dispatch_data({
        str(tmp_path): pc_sampling_dispatch_trace([tool_data])
    })
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
        result = instance.calc_dispatch_data({
            str(tmp_path): pc_sampling_dispatch_trace([])
        })

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
        str(tmp_path): pc_sampling_dispatch_trace(
            make_multiprocess_dispatch_tool_data()
        )
    })

    df = result[str(tmp_path)]
    assert df["kernel_name"].tolist() == ["vecCopy", "vecCopy", "vecCopy"]
    assert df["dispatch_id"].tolist() == [1, 2, 3]
