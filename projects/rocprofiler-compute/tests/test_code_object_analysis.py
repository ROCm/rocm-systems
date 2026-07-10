# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import json

from pc_sampling.code_object_analysis import (
    CodeObjectDisassembly,
    CodeObjectInstruction,
    load_code_object_disassemblies,
    parse_code_object_info,
)


def make_instruction(virtual_address, name="s_nop", comment=""):
    """Build one code_obj_info instruction dict."""
    return {
        "code_obj_offset": virtual_address - 0x1000,
        "comment": comment,
        "name": name,
        "size": 4,
        "virtual_address": virtual_address,
    }


def make_symbol(name, instructions):
    """Build one code_obj_info symbol dict owning the given instructions."""
    return {
        "code_object_offset": 0,
        "instructions": instructions,
        "name": name,
        "size": sum(inst["size"] for inst in instructions),
        "virtual_address": instructions[0]["virtual_address"] if instructions else 0,
    }


def make_code_object(code_object_id, symbols):
    """Build one code_obj_info code object dict."""
    return {"id": code_object_id, "symbols": symbols}


def make_code_obj_info(code_objects):
    """Build a full code_obj_info.json payload."""
    return {"code_objects": code_objects}


def test_parse_returns_one_entry_per_code_object():
    data = make_code_obj_info([
        make_code_object(1, [make_symbol("a", [make_instruction(0x1000)])]),
        make_code_object(2, [make_symbol("b", [make_instruction(0x2000)])]),
    ])

    disassemblies = parse_code_object_info(data)

    assert [d.code_object_id for d in disassemblies] == [1, 2]


def test_parse_flattens_all_symbol_instructions():
    data = make_code_obj_info([
        make_code_object(
            1,
            [
                make_symbol("a", [make_instruction(0x1000), make_instruction(0x1004)]),
                make_symbol("b", [make_instruction(0x1008)]),
            ],
        )
    ])

    disassemblies = parse_code_object_info(data)

    # Both symbols' instructions are flattened into one per-object list.
    assert len(disassemblies[0].instructions) == 3


def test_parse_captures_virtual_address_instruction_and_comment():
    data = make_code_obj_info([
        make_code_object(
            7,
            [
                make_symbol(
                    "kern",
                    [make_instruction(0x2040, name="v_mov_b32", comment="src.cpp:5")],
                )
            ],
        )
    ])

    disassemblies = parse_code_object_info(data)

    assert disassemblies[0].instructions[0] == CodeObjectInstruction(
        virtual_address=0x2040,
        instruction="v_mov_b32",
        comment="src.cpp:5",
    )


def test_parse_empty_dict_returns_empty_list():
    assert parse_code_object_info({}) == []


def test_parse_code_object_without_symbols_yields_no_instructions():
    data = make_code_obj_info([{"id": 7}])
    assert parse_code_object_info(data) == [
        CodeObjectDisassembly(code_object_id=7, instructions=[])
    ]


def _write(path, code_objects):
    path.write_text(json.dumps(make_code_obj_info(code_objects)), encoding="utf-8")


def test_load_discovers_files_and_parses_pid(tmp_path):
    _write(tmp_path / "123_code_obj_info.json", [make_code_object(1, [])])
    (tmp_path / "sub").mkdir()
    _write(tmp_path / "sub" / "456_code_obj_info.json", [make_code_object(2, [])])

    result = load_code_object_disassemblies(str(tmp_path))

    assert set(result) == {123, 456}
    assert result[123][0].code_object_id == 1
    assert result[456][0].code_object_id == 2


def test_load_skips_file_without_pid_prefix(tmp_path):
    _write(tmp_path / "code_obj_info.json", [make_code_object(1, [])])

    assert load_code_object_disassemblies(str(tmp_path)) == {}


def test_load_skips_malformed_json(tmp_path):
    (tmp_path / "123_code_obj_info.json").write_text("{not json", encoding="utf-8")

    assert load_code_object_disassemblies(str(tmp_path)) == {}
