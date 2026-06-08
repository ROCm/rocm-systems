# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for ``db_analysis.calc_pc_sampling_data`` native dependency wiring.

These drive ``calc_pc_sampling_data`` end-to-end against a synthetic workload
directory (a ``ps_file_results.json`` plus an optional ``*_code_obj_info.json``)
and assert the returned per-workload DataFrame contract, with particular focus
on the ``dependencies`` column threaded in from
``utils.parser.parse_waitcnt_dependencies`` on the native path and forced empty
on the rocprofiler-sdk-string fallback path.
"""

import json
from pathlib import Path
from typing import Any, Optional

import common  # noqa: F401  (adds src/ to sys.path)

from rocprof_compute_analyze.analysis_db import db_analysis

_EXPECTED_COLS = {
    "offset",
    "count",
    "count_issued",
    "count_stalled",
    "stall_reason",
    "kernel_name",
    "instruction",
    "source_line",
    "source_file",
    "line",
    "dependencies",
}


def _make_record(
    code_object_id: int,
    offset: int,
    inst_index: int,
    dispatch_id: int,
    wave_issued: bool = True,
    stall_reason: Optional[str] = None,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    if stall_reason is not None:
        snapshot["stall_reason"] = stall_reason
    return {
        "inst_index": inst_index,
        "record": {
            "pc": {
                "code_object_id": code_object_id,
                "code_object_offset": offset,
            },
            "dispatch_id": dispatch_id,
            "wave_issued": wave_issued,
            "snapshot": snapshot,
        },
    }


def _write_results_json(
    path: Path,
    stochastic: Optional[list] = None,
    host_trap: Optional[list] = None,
    instructions: Optional[list] = None,
    comments: Optional[list] = None,
    kernel_symbols: Optional[list] = None,
) -> Path:
    data = {
        "rocprofiler-sdk-tool": [
            {
                "buffer_records": {
                    "pc_sample_stochastic": stochastic if stochastic else [],
                    "pc_sample_host_trap": host_trap if host_trap else [],
                },
                "strings": {
                    "pc_sample_instructions": instructions if instructions else [],
                    "pc_sample_comments": comments if comments else [],
                },
                "kernel_symbols": kernel_symbols if kernel_symbols else [],
            }
        ]
    }
    path.write_text(json.dumps(data))
    return path


def _write_code_obj_info(
    path: Path,
    code_object_id: int,
    instructions: list[dict],
) -> Path:
    data = {
        "code_objects": [
            {
                "id": code_object_id,
                "symbols": [
                    {
                        "name": "vecCopy",
                        "code_object_offset": 0,
                        "size": sum(i.get("size", 0) for i in instructions),
                        "instructions": instructions,
                    }
                ],
            }
        ]
    }
    path.write_text(json.dumps(data))
    return path


class _StubRuns:
    """Minimal db_analysis stand-in exposing only ``_runs``.

    ``calc_pc_sampling_data`` iterates ``self._runs.keys()`` (workload-path
    strings) and reads files from disk, so the method can be invoked unbound on
    this stub without the heavy real constructor.
    """

    def __init__(self, workload_paths: list[str]) -> None:
        self._runs = {path: None for path in workload_paths}


def _native_instructions() -> list[dict]:
    """Disassembly with a vmcnt producer at 0 and an s_waitcnt at offset 4.

    Instruction ranges: [0,4) producer, [4,8) s_waitcnt vmcnt(0).
    parse_waitcnt_dependencies should map offset 4 -> [0].
    """
    return [
        {
            "name": "global_load_dwordx4 v[0:3], v[4:5]",
            "comment": "/home/u/kernel.hip:42",
            "code_obj_offset": 0,
            "size": 4,
        },
        {
            "name": "s_waitcnt vmcnt(0)",
            "comment": "/home/u/kernel.hip:43",
            "code_obj_offset": 4,
            "size": 4,
        },
    ]


def test_native_branch_attributes_and_attaches_dependencies(tmp_path: Path) -> None:
    """Native path: in-range samples get ISA/source attribution + deps column."""
    samples = [
        _make_record(2, 0, inst_index=0, dispatch_id=0),  # producer offset
        _make_record(2, 4, inst_index=1, dispatch_id=1),  # waitcnt offset
    ]
    _write_results_json(
        tmp_path / "ps_file_results.json",
        host_trap=samples,
        instructions=["UNUSED_0", "UNUSED_1"],
        comments=["unused:0", "unused:1"],
        kernel_symbols=[{"code_object_id": 2, "formatted_kernel_name": "vecCopy"}],
    )
    _write_code_obj_info(
        tmp_path / "9999_code_obj_info.json",
        code_object_id=2,
        instructions=_native_instructions(),
    )

    stub = _StubRuns([str(tmp_path)])
    result = db_analysis.calc_pc_sampling_data(stub)

    df = result[str(tmp_path)]
    assert _EXPECTED_COLS.issubset(set(df.columns))

    # Producer offset 0 -> native instruction + source, no dependencies.
    row0 = df[df["offset"] == 0].iloc[0]
    assert row0["instruction"] == "global_load_dwordx4 v[0:3], v[4:5]"
    assert row0["source_file"] == "/home/u/kernel.hip"
    assert row0["line"] == "42"
    assert row0["kernel_name"] == "vecCopy"
    assert row0["dependencies"] == []

    # s_waitcnt offset 4 -> depends on the producer at offset 0.
    row4 = df[df["offset"] == 4].iloc[0]
    assert row4["instruction"] == "s_waitcnt vmcnt(0)"
    assert row4["source_file"] == "/home/u/kernel.hip"
    assert row4["line"] == "43"
    assert row4["dependencies"] == [0]


def test_native_branch_out_of_range_offset_unattributed(tmp_path: Path) -> None:
    """An offset outside every native range yields empty ISA/source columns."""
    samples = [
        _make_record(2, 4, inst_index=0, dispatch_id=0),  # in range
        _make_record(2, 64, inst_index=1, dispatch_id=1),  # past all ranges
    ]
    _write_results_json(
        tmp_path / "ps_file_results.json",
        stochastic=samples,
        instructions=["UNUSED_0", "UNUSED_1"],
        comments=["unused:0", "unused:1"],
        kernel_symbols=[{"code_object_id": 2, "formatted_kernel_name": "vecCopy"}],
    )
    _write_code_obj_info(
        tmp_path / "9999_code_obj_info.json",
        code_object_id=2,
        instructions=_native_instructions(),
    )

    stub = _StubRuns([str(tmp_path)])
    df = db_analysis.calc_pc_sampling_data(stub)[str(tmp_path)]

    # In-range offset 4 resolves.
    row4 = df[df["offset"] == 4].iloc[0]
    assert row4["instruction"] == "s_waitcnt vmcnt(0)"
    assert row4["dependencies"] == [0]

    # Out-of-range offset 64 -> all native attribution columns None / empty deps.
    row64 = df[df["offset"] == 64].iloc[0]
    assert row64["instruction"] is None
    assert row64["source_file"] is None
    assert row64["line"] is None
    assert row64["source_line"] is None
    assert row64["dependencies"] == []


def test_fallback_branch_has_empty_dependencies(tmp_path: Path) -> None:
    """Fallback path (no code_obj_info.json): deps column present and all empty."""
    samples = [
        _make_record(100, 0x10, inst_index=0, dispatch_id=0),
        _make_record(100, 0x20, inst_index=1, dispatch_id=1),
    ]
    _write_results_json(
        tmp_path / "ps_file_results.json",
        stochastic=samples,
        instructions=["v_mov_b32", "s_waitcnt"],
        comments=["/src/vcopy.cpp:42", "/src/vcopy.cpp:43"],
        kernel_symbols=[{"code_object_id": 100, "formatted_kernel_name": "vecCopy"}],
    )

    stub = _StubRuns([str(tmp_path)])
    df = db_analysis.calc_pc_sampling_data(stub)[str(tmp_path)]

    assert _EXPECTED_COLS.issubset(set(df.columns))

    # SDK strings by inst_index; source_file/line unset; deps all empty.
    row0 = df[df["offset"] == 0x10].iloc[0]
    assert row0["instruction"] == "v_mov_b32"
    assert row0["source_line"] == "/src/vcopy.cpp:42"
    assert row0["source_file"] is None
    assert row0["line"] is None
    assert row0["kernel_name"] == "vecCopy"

    assert all(dep == [] for dep in df["dependencies"])
