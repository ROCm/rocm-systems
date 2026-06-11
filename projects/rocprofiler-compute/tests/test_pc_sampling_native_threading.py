# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""End-to-end threading tests for the optional ``native_code_object_map``
parameter through the PC sampling dispatcher.

``utils.parser.load_pc_sampling_data`` forwards the native map to
``load_pc_sampling_data_per_kernel`` -> ``enrich_with_metadata`` on BOTH the
all-kernels path (no ``-k`` filter) and the single-kernel ``-k`` path. These
tests prove the per-instruction table's ``instruction`` / ``source_line``
columns are sourced from the native map (keyed by
``(code_object_id, code_object_offset)``) rather than the rocprofiler-sdk
string tables on each path. The sdk string tables are deliberately set to text
distinct from the native text so a fallback would be visible.
"""

import pandas as pd

from utils import schema
from utils.parser import PMC_KERNEL_TOP_TABLE_ID, load_pc_sampling_data

# ── Helpers for building synthetic records / tool_data ───────


def make_record(
    code_object_id: int,
    offset: int,
    inst_index: int,
    dispatch_id: int,
) -> dict:
    """A host_trap PC sample carrying the pc fields the native map keys on."""
    return {
        "inst_index": inst_index,
        "record": {
            "pc": {
                "code_object_id": code_object_id,
                "code_object_offset": offset,
            },
            "dispatch_id": dispatch_id,
        },
    }


def make_dispatch(dispatch_id: int, kernel_id: int) -> dict:
    """A kernel_dispatch buffer record mapping a dispatch to a kernel."""
    return {
        "start_timestamp": 0,
        "end_timestamp": 0,
        "dispatch_info": {
            "dispatch_id": dispatch_id,
            "kernel_id": kernel_id,
            "agent_id": {"handle": 1},
        },
    }


def make_kernel_symbol(
    kernel_id: int, code_object_id: int, formatted_kernel_name: str
) -> dict:
    """A kernel_symbols entry mapping kernel/code-object ids to a name."""
    return {
        "kernel_id": kernel_id,
        "code_object_id": code_object_id,
        "formatted_kernel_name": formatted_kernel_name,
    }


def make_tool_data() -> dict:
    """A single ``rocprofiler-sdk-tool[0]`` dict with two kernels.

    vecCopy (kernel 100) and vecAdd (kernel 101) share code object 5 at
    distinct offsets; each row's kernel resolves via dispatch correlation.
    The sdk instruction/comment tables hold text DISTINCT from the native map
    so any fallback to the sdk tables would be visible in the assertions.
    """
    samples = [
        make_record(5, 0x10, 0, dispatch_id=0),
        make_record(5, 0x20, 1, dispatch_id=1),
    ]
    return {
        "buffer_records": {
            "pc_sample_stochastic": [],
            "pc_sample_host_trap": samples,
            "kernel_dispatch": [
                make_dispatch(0, 100),
                make_dispatch(1, 101),
            ],
        },
        "strings": {
            "pc_sample_instructions": ["sdk_mov", "sdk_add"],
            "pc_sample_comments": ["/sdk/x.cpp:1", "/sdk/x.cpp:2"],
        },
        "kernel_symbols": [
            make_kernel_symbol(100, 5, "vecCopy"),
            make_kernel_symbol(101, 5, "vecAdd"),
        ],
        "agents": [],
    }


def make_native_map() -> dict:
    """Native map keyed by (code_object_id, code_object_offset).

    Names/comments are distinct from the sdk string tables in make_tool_data.
    """
    return {
        (5, 0x10): {"name": "native_mov", "comment": "/native/a.cpp:42"},
        (5, 0x20): {"name": "native_add", "comment": "/native/b.cpp:99"},
    }


# ═══════════════════════════════════════════════════════════════
# native_code_object_map threads through load_pc_sampling_data
# ═══════════════════════════════════════════════════════════════


def test_native_map_threads_all_kernels_path() -> None:
    """No ``-k`` filter: the native map drives instruction/source_line."""
    tool_data = make_tool_data()
    native_map = make_native_map()
    df = load_pc_sampling_data(
        schema.Workload(),
        "ps_file",
        "offset",
        tool_data,
        native_code_object_map=native_map,
    )
    assert not df.empty
    # instruction comes from the native names, not the sdk strings.
    assert set(df["instruction"]) == {"native_mov", "native_add"}
    # source_line comes from the native comments (trimmed to .../<basename>).
    assert set(df["source_line"]) == {".../a.cpp:42", ".../b.cpp:99"}
    # the sdk text never leaks through.
    assert "sdk_mov" not in set(df["instruction"])
    assert "sdk_add" not in set(df["instruction"])


def test_native_map_threads_single_kernel_path() -> None:
    """Single ``-k`` filter: the native map still drives both columns."""
    tool_data = make_tool_data()
    native_map = make_native_map()
    workload = schema.Workload(
        filter_kernel_ids=[0],
        dfs={
            PMC_KERNEL_TOP_TABLE_ID: pd.DataFrame({
                "Kernel_Name": ["vecCopy", "vecAdd"]
            })
        },
    )
    df = load_pc_sampling_data(
        workload,
        "ps_file",
        "offset",
        tool_data,
        native_code_object_map=native_map,
    )
    assert not df.empty
    # kernel_index 0 -> vecCopy, which owns offset 0x10 only.
    assert set(df["Kernel_Name"]) == {"vecCopy"}
    assert df["instruction"].tolist() == ["native_mov"]
    assert df["source_line"].tolist() == [".../a.cpp:42"]
    # the sdk text never leaks through.
    assert "sdk_mov" not in df["instruction"].tolist()
