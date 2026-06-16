# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Native code-object ISA enrichment for PC sampling analysis."""

import json
from pathlib import Path

import pandas as pd

from utils import schema
from utils.file_io import load_native_code_object_map
from utils.parser import PMC_KERNEL_TOP_TABLE_ID, load_pc_sampling_data
from utils.pc_sampling_analysis import SOURCE_LINE_MISSING, enrich_with_metadata

# ── Helpers: native code-object-info JSON ────────────────────


def make_instruction(
    name: str,
    code_obj_offset: int,
    virtual_address: int = 0,
    size: int = 4,
    comment: str | None = "",
) -> dict:
    """Build one instruction entry; omit comment when comment is None."""
    inst = {
        "name": name,
        "code_obj_offset": code_obj_offset,
        "virtual_address": virtual_address,
        "size": size,
    }
    if comment is not None:
        inst["comment"] = comment
    return inst


def make_code_object(
    code_object_id: int,
    symbol_name: str,
    code_object_offset: int,
    instructions: list,
) -> dict:
    """Build one code_objects entry with a single symbol holding instructions."""
    return {
        "id": code_object_id,
        "symbols": [
            {
                "name": symbol_name,
                "code_object_offset": code_object_offset,
                "instructions": instructions,
            }
        ],
    }


def write_code_obj_info(path: Path, code_objects: list) -> Path:
    """Write a *code_obj_info.json file wrapping the given code objects."""
    path.write_text(json.dumps({"code_objects": code_objects}))
    return path


# ── Helpers: enrich_with_metadata inputs ─────────────────────


def make_strings_tool_data(
    instructions: list | None = None,
    comments: list | None = None,
    kernel_symbols: list | None = None,
) -> dict:
    """Minimal tool_data carrying the sdk string tables used as fallback."""
    return {
        "strings": {
            "pc_sample_instructions": (
                instructions if instructions is not None else []
            ),
            "pc_sample_comments": comments if comments is not None else [],
        },
        "kernel_symbols": kernel_symbols if kernel_symbols is not None else [],
    }


def make_native_row(
    inst_index: int,
    code_object_id: int,
    code_object_offset: int,
    kernel_id: int = 100,
) -> dict:
    """One aggregated row carrying the columns the native map is keyed on."""
    return {
        "inst_index": inst_index,
        "code_object_id": code_object_id,
        "code_object_offset": code_object_offset,
        "kernel_id": kernel_id,
    }


# ── Helpers: load_pc_sampling_data threading ─────────────────


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


def make_threading_tool_data() -> dict:
    """Two kernels sharing code object 5; sdk text distinct from the native map."""
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
    """Native map keyed by (code_object_id, offset), distinct from sdk text."""
    return {
        (5, 0x10): {"name": "native_mov", "comment": "/native/a.cpp:42"},
        (5, 0x20): {"name": "native_add", "comment": "/native/b.cpp:99"},
    }


# ═══════════════════════════════════════════════════════════════
# load_native_code_object_map
# ═══════════════════════════════════════════════════════════════


def test_single_file_single_code_object_multiple_instructions(
    tmp_path: Path,
) -> None:
    """One file with several instructions yields one entry per instruction."""
    write_code_obj_info(
        tmp_path / "1234_code_obj_info.json",
        [
            make_code_object(
                code_object_id=2,
                symbol_name="vecCopy",
                code_object_offset=137000,
                instructions=[
                    make_instruction("v_mov_b32 v0, v1", 137008, comment="/s/a.cpp:1"),
                    make_instruction(
                        "s_waitcnt vmcnt(0)", 137012, comment="/s/a.cpp:2"
                    ),
                    make_instruction(
                        "v_add_f32 v2, v0, v1", 137016, comment="/s/a.cpp:3"
                    ),
                ],
            )
        ],
    )
    result = load_native_code_object_map(str(tmp_path))
    assert result[(2, 137008)] == {
        "name": "v_mov_b32 v0, v1",
        "comment": "/s/a.cpp:1",
    }
    assert result[(2, 137012)] == {
        "name": "s_waitcnt vmcnt(0)",
        "comment": "/s/a.cpp:2",
    }
    assert result[(2, 137016)] == {
        "name": "v_add_f32 v2, v0, v1",
        "comment": "/s/a.cpp:3",
    }
    assert len(result) == 3


def test_two_files_disjoint_code_objects_merged(tmp_path: Path) -> None:
    """Two files with disjoint code objects merge into a single map."""
    write_code_obj_info(
        tmp_path / "100_code_obj_info.json",
        [
            make_code_object(
                code_object_id=1,
                symbol_name="kernelA",
                code_object_offset=0,
                instructions=[make_instruction("v_mov", 0x10, comment="a:1")],
            )
        ],
    )
    write_code_obj_info(
        tmp_path / "200_code_obj_info.json",
        [
            make_code_object(
                code_object_id=2,
                symbol_name="kernelB",
                code_object_offset=0,
                instructions=[make_instruction("v_add", 0x20, comment="b:1")],
            )
        ],
    )
    result = load_native_code_object_map(str(tmp_path))
    assert result[(1, 0x10)] == {"name": "v_mov", "comment": "a:1"}
    assert result[(2, 0x20)] == {"name": "v_add", "comment": "b:1"}
    assert len(result) == 2


def test_bare_code_obj_info_file_is_picked_up(tmp_path: Path) -> None:
    """A file named exactly code_obj_info.json (no pid prefix) is matched."""
    write_code_obj_info(
        tmp_path / "code_obj_info.json",
        [
            make_code_object(
                code_object_id=7,
                symbol_name="bare",
                code_object_offset=0,
                instructions=[make_instruction("s_endpgm", 0x40, comment="end:1")],
            )
        ],
    )
    result = load_native_code_object_map(str(tmp_path))
    assert result[(7, 0x40)] == {"name": "s_endpgm", "comment": "end:1"}


def test_numeric_id_and_offset_yield_int_tuple_keys(tmp_path: Path) -> None:
    """JSON-number id/offset produce integer-tuple keys."""
    write_code_obj_info(
        tmp_path / "5678_code_obj_info.json",
        [
            make_code_object(
                code_object_id=2,
                symbol_name="vecCopy",
                code_object_offset=137000,
                instructions=[
                    make_instruction("v_mov_b32", 137008, comment="/s/a.cpp:1")
                ],
            )
        ],
    )
    result = load_native_code_object_map(str(tmp_path))
    assert (2, 137008) in result
    key = next(iter(result))
    assert isinstance(key[0], int)
    assert isinstance(key[1], int)


def test_empty_dir_returns_empty_dict(tmp_path: Path) -> None:
    """A directory with no matching json files returns an empty dict."""
    result = load_native_code_object_map(str(tmp_path))
    assert result == {}


def test_nonexistent_dir_returns_empty_dict(tmp_path: Path) -> None:
    """A path that does not exist returns an empty dict without raising."""
    missing = tmp_path / "does_not_exist"
    result = load_native_code_object_map(str(missing))
    assert result == {}


def test_string_id_and_offset_are_coerced_to_int_tuple(tmp_path: Path) -> None:
    """JSON-string id/offset are coerced to an integer-tuple key."""
    (tmp_path / "4321_code_obj_info.json").write_text(
        json.dumps({
            "code_objects": [
                {
                    "id": "2",
                    "symbols": [
                        {
                            "name": "vecCopy",
                            "code_object_offset": 137000,
                            "instructions": [
                                {
                                    "name": "v_mov_b32",
                                    "code_obj_offset": "137008",
                                    "virtual_address": 0,
                                    "size": 4,
                                    "comment": "/s/a.cpp:1",
                                }
                            ],
                        }
                    ],
                }
            ]
        })
    )
    result = load_native_code_object_map(str(tmp_path))
    assert (2, 137008) in result
    key = next(iter(result))
    assert isinstance(key[0], int)
    assert isinstance(key[1], int)
    assert result[(2, 137008)] == {"name": "v_mov_b32", "comment": "/s/a.cpp:1"}


def test_missing_comment_defaults_to_empty_string(tmp_path: Path) -> None:
    """An instruction without a comment key stores comment as ''."""
    write_code_obj_info(
        tmp_path / "9_code_obj_info.json",
        [
            make_code_object(
                code_object_id=3,
                symbol_name="noComment",
                code_object_offset=0,
                instructions=[make_instruction("v_nop", 0x50, comment=None)],
            )
        ],
    )
    result = load_native_code_object_map(str(tmp_path))
    assert result[(3, 0x50)]["name"] == "v_nop"
    assert result[(3, 0x50)]["comment"] == ""


# ═══════════════════════════════════════════════════════════════
# enrich_with_metadata (native map)
# ═══════════════════════════════════════════════════════════════


def test_enrich_native_present_sources_both_columns() -> None:
    """A non-empty native map drives instruction/source_line per row key."""
    aggregated = pd.DataFrame([
        make_native_row(inst_index=0, code_object_id=5, code_object_offset=0x10),
        make_native_row(inst_index=1, code_object_id=5, code_object_offset=0x20),
    ])
    # sdk tables deliberately hold DIFFERENT text so a fallback would be visible.
    tool_data = make_strings_tool_data(
        instructions=["sdk_inst_0", "sdk_inst_1"],
        comments=["sdk_src_0", "sdk_src_1"],
    )
    native_code_object_map = {
        (5, 0x10): {"name": "v_native_mov", "comment": "/native/a.cpp:1"},
        (5, 0x20): {"name": "v_native_add", "comment": "/native/a.cpp:2"},
    }
    df = enrich_with_metadata(
        aggregated,
        tool_data,
        attach={"instruction", "source_line"},
        native_code_object_map=native_code_object_map,
    )
    assert df["instruction"].tolist() == ["v_native_mov", "v_native_add"]
    assert df["source_line"].tolist() == ["/native/a.cpp:1", "/native/a.cpp:2"]


def test_enrich_native_present_comment_verbatim() -> None:
    """A populated native comment is used verbatim as the source_line."""
    aggregated = pd.DataFrame([
        make_native_row(inst_index=0, code_object_id=7, code_object_offset=0x40),
    ])
    tool_data = make_strings_tool_data(instructions=["sdk"], comments=["sdk_src"])
    native_code_object_map = {
        (7, 0x40): {"name": "v_x", "comment": "/native/b.cpp:99"},
    }
    df = enrich_with_metadata(
        aggregated,
        tool_data,
        attach={"instruction", "source_line"},
        native_code_object_map=native_code_object_map,
    )
    assert df.iloc[0]["source_line"] == "/native/b.cpp:99"


def test_enrich_fallback_none_uses_sdk_strings() -> None:
    """native_code_object_map=None falls back to the sdk tables by inst_index."""
    aggregated = pd.DataFrame([
        make_native_row(inst_index=1, code_object_id=5, code_object_offset=0x20),
    ])
    tool_data = make_strings_tool_data(
        instructions=["sdk_inst_0", "sdk_inst_1"],
        comments=["sdk_src_0", "sdk_src_1"],
    )
    df = enrich_with_metadata(
        aggregated,
        tool_data,
        attach={"instruction", "source_line"},
        native_code_object_map=None,
    )
    assert df.iloc[0]["instruction"] == "sdk_inst_1"
    assert df.iloc[0]["source_line"] == "sdk_src_1"


def test_enrich_fallback_empty_dict_uses_sdk_strings() -> None:
    """An empty native map is falsy-equivalent to None: sdk tables are used."""
    aggregated = pd.DataFrame([
        make_native_row(inst_index=0, code_object_id=5, code_object_offset=0x10),
    ])
    tool_data = make_strings_tool_data(
        instructions=["sdk_inst_0", "sdk_inst_1"],
        comments=["sdk_src_0", "sdk_src_1"],
    )
    df = enrich_with_metadata(
        aggregated,
        tool_data,
        attach={"instruction", "source_line"},
        native_code_object_map={},
    )
    assert df.iloc[0]["instruction"] == "sdk_inst_0"
    assert df.iloc[0]["source_line"] == "sdk_src_0"


def test_enrich_native_miss_yields_none_and_sentinel() -> None:
    """A row key absent from a non-empty native map yields None / 'N/A'."""
    aggregated = pd.DataFrame([
        make_native_row(inst_index=0, code_object_id=5, code_object_offset=0x99),
    ])
    tool_data = make_strings_tool_data(
        instructions=["sdk_inst_0"],
        comments=["sdk_src_0"],
    )
    # Map is non-empty but does NOT contain (5, 0x99).
    native_code_object_map = {
        (5, 0x10): {"name": "v_native_mov", "comment": "/native/a.cpp:1"},
    }
    df = enrich_with_metadata(
        aggregated,
        tool_data,
        attach={"instruction", "source_line"},
        native_code_object_map=native_code_object_map,
    )
    assert df.iloc[0]["instruction"] is None
    assert df.iloc[0]["source_line"] == SOURCE_LINE_MISSING


def test_enrich_native_empty_comment_yields_sentinel() -> None:
    """A native record with an empty comment yields the 'N/A' sentinel."""
    aggregated = pd.DataFrame([
        make_native_row(inst_index=0, code_object_id=5, code_object_offset=0x10),
    ])
    tool_data = make_strings_tool_data(
        instructions=["sdk_inst_0"], comments=["sdk_src_0"]
    )
    native_code_object_map = {
        (5, 0x10): {"name": "v_native_mov", "comment": ""},
    }
    df = enrich_with_metadata(
        aggregated,
        tool_data,
        attach={"instruction", "source_line"},
        native_code_object_map=native_code_object_map,
    )
    # name still flows through; only the empty comment is sentinel-ized.
    assert df.iloc[0]["instruction"] == "v_native_mov"
    assert df.iloc[0]["source_line"] == SOURCE_LINE_MISSING


# ═══════════════════════════════════════════════════════════════
# native map threading through load_pc_sampling_data
# ═══════════════════════════════════════════════════════════════


def test_native_map_threads_all_kernels_path() -> None:
    """No -k filter: the native map drives instruction/source_line."""
    tool_data = make_threading_tool_data()
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
    """Single -k filter: the native map still drives both columns."""
    tool_data = make_threading_tool_data()
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
