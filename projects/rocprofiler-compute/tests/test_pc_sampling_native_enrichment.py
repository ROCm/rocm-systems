# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests pinning the native-code-object enrichment contract for
``enrich_with_metadata``.

These tests target the NEW optional trailing parameter
``native_code_object_map`` added to
``utils.pc_sampling_analysis.enrich_with_metadata``. When a non-empty native
map is supplied the ``instruction`` / ``source_line`` columns are sourced from
the native record keyed by ``(code_object_id, code_object_offset)``; when it is
None / empty the loader falls back to the rocprofiler-sdk string tables indexed
by ``inst_index`` (current behavior).
"""

import pandas as pd

from utils.pc_sampling_analysis import SOURCE_LINE_MISSING, enrich_with_metadata

# ── Helpers ──────────────────────────────────────────────────


def make_tool_data(
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


# ═══════════════════════════════════════════════════════════════
# Native present (criterion 3): columns sourced from the native map
# ═══════════════════════════════════════════════════════════════


def test_enrich_native_present_sources_both_columns() -> None:
    """A non-empty native map drives instruction/source_line per row key."""
    aggregated = pd.DataFrame([
        make_native_row(inst_index=0, code_object_id=5, code_object_offset=0x10),
        make_native_row(inst_index=1, code_object_id=5, code_object_offset=0x20),
    ])
    # sdk tables deliberately hold DIFFERENT text so a fallback would be visible.
    tool_data = make_tool_data(
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
    tool_data = make_tool_data(instructions=["sdk"], comments=["sdk_src"])
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


# ═══════════════════════════════════════════════════════════════
# Fallback (criterion 4): None / {} -> sdk string tables by inst_index
# ═══════════════════════════════════════════════════════════════


def test_enrich_fallback_none_uses_sdk_strings() -> None:
    """native_code_object_map=None falls back to the sdk tables by inst_index."""
    aggregated = pd.DataFrame([
        make_native_row(inst_index=1, code_object_id=5, code_object_offset=0x20),
    ])
    tool_data = make_tool_data(
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
    tool_data = make_tool_data(
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


# ═══════════════════════════════════════════════════════════════
# Miss in native mode (criterion 5): key absent -> None / sentinel
# ═══════════════════════════════════════════════════════════════


def test_enrich_native_miss_yields_none_and_sentinel() -> None:
    """A row key absent from a non-empty native map yields None / 'N/A'."""
    aggregated = pd.DataFrame([
        make_native_row(inst_index=0, code_object_id=5, code_object_offset=0x99),
    ])
    tool_data = make_tool_data(
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


# ═══════════════════════════════════════════════════════════════
# Empty comment (criterion 6): native comment "" -> sentinel
# ═══════════════════════════════════════════════════════════════


def test_enrich_native_empty_comment_yields_sentinel() -> None:
    """A native record with an empty comment yields the 'N/A' sentinel."""
    aggregated = pd.DataFrame([
        make_native_row(inst_index=0, code_object_id=5, code_object_offset=0x10),
    ])
    tool_data = make_tool_data(instructions=["sdk_inst_0"], comments=["sdk_src_0"])
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
