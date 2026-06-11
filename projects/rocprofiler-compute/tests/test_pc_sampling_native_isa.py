# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import json
from pathlib import Path

from utils.file_io import load_native_code_object_map

# ── Helpers for building synthetic code-object-info JSON ──────


def make_instruction(
    name: str,
    code_obj_offset: int,
    virtual_address: int = 0,
    size: int = 4,
    comment: str | None = "",
) -> dict:
    """Build a single instruction entry; omit comment when comment is None."""
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
    """Build a single code_objects entry with one symbol holding instructions."""
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
    """JSON-number id/offset produce integer-tuple keys; types are int."""
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
    """JSON-string id/offset are coerced to an integer-tuple key; types are int."""
    # id and code_obj_offset are written as JSON STRINGS, exercising the
    # int(...) coercion in load_native_code_object_map.
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
