# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import json
from pathlib import Path

from common import ROOT

from pc_sampling.code_object_analysis import (
    CodeObjectDisassembly,
    CodeObjectInstruction,
    load_code_object_disassemblies,
    parse_code_object_info,
)

CODE_OBJ_INFO_JSON = (
    Path(ROOT) / "workloads/rocflop_pcsampling/MI300X_A1/455902_code_obj_info.json"
)


def test_parse_returns_one_entry_per_code_object():
    data = json.loads(CODE_OBJ_INFO_JSON.read_text(encoding="utf-8"))
    disassemblies = parse_code_object_info(data)

    assert [d.code_object_id for d in disassemblies] == [
        code_object["id"] for code_object in data["code_objects"]
    ]


def test_parse_flattens_all_symbol_instructions():
    data = json.loads(CODE_OBJ_INFO_JSON.read_text(encoding="utf-8"))
    disassemblies = parse_code_object_info(data)

    for disassembly, code_object in zip(disassemblies, data["code_objects"]):
        expected_count = sum(
            len(symbol["instructions"]) for symbol in code_object["symbols"]
        )
        assert len(disassembly.instructions) == expected_count


def test_parse_captures_virtual_address_instruction_and_comment():
    data = json.loads(CODE_OBJ_INFO_JSON.read_text(encoding="utf-8"))
    disassemblies = parse_code_object_info(data)

    first_symbol_inst = data["code_objects"][0]["symbols"][0]["instructions"][0]
    first_parsed = disassemblies[0].instructions[0]

    assert first_parsed == CodeObjectInstruction(
        virtual_address=first_symbol_inst["virtual_address"],
        instruction=first_symbol_inst["name"],
        comment=first_symbol_inst["comment"],
    )


def test_parse_empty_dict_returns_empty_list():
    assert parse_code_object_info({}) == []


def test_parse_code_object_without_symbols_yields_no_instructions():
    data = {"code_objects": [{"id": 7}]}
    assert parse_code_object_info(data) == [
        CodeObjectDisassembly(code_object_id=7, instructions=[])
    ]


def _write(path, code_objects):
    path.write_text(json.dumps({"code_objects": code_objects}), encoding="utf-8")


def test_load_discovers_files_and_parses_pid(tmp_path):
    _write(tmp_path / "123_code_obj_info.json", [{"id": 1, "symbols": []}])
    (tmp_path / "sub").mkdir()
    _write(tmp_path / "sub" / "456_code_obj_info.json", [{"id": 2, "symbols": []}])

    result = load_code_object_disassemblies(str(tmp_path))

    assert set(result) == {123, 456}
    assert result[123][0].code_object_id == 1
    assert result[456][0].code_object_id == 2


def test_load_skips_file_without_pid_prefix(tmp_path):
    _write(tmp_path / "code_obj_info.json", [{"id": 1, "symbols": []}])

    assert load_code_object_disassemblies(str(tmp_path)) == {}


def test_load_skips_malformed_json(tmp_path):
    (tmp_path / "123_code_obj_info.json").write_text("{not json", encoding="utf-8")

    assert load_code_object_disassemblies(str(tmp_path)) == {}
