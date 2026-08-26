# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import json
from pathlib import Path

import common
import pytest
import yaml

from pc_sampling import code_object_analysis
from pc_sampling.code_object_analysis import (
    CodeObjectDisassembly,
    CodeObjectInstruction,
    CodeObjectSymbol,
    InstructionPipelines,
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


def test_parse_keeps_symbols_separate_with_their_own_instructions():
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

    assert [symbol.name for symbol in disassemblies[0].symbols] == ["a", "b"]
    assert [len(symbol.instructions) for symbol in disassemblies[0].symbols] == [2, 1]


def test_parse_captures_symbol_virtual_address():
    data = make_code_obj_info([
        make_code_object(1, [make_symbol("kern", [make_instruction(0x2040)])])
    ])

    assert parse_code_object_info(data)[0].symbols[0].virtual_address == 0x2040


def test_parse_captures_virtual_address_instruction_and_source():
    """Parse an instruction's virtual address, opcode, and source text."""
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

    assert disassemblies[0].symbols[0].instructions[0] == CodeObjectInstruction(
        virtual_address=0x2040,
        instruction="v_mov_b32",
        source="src.cpp:5",
    )


def test_parse_empty_dict_returns_empty_list():
    assert parse_code_object_info({}) == []


def test_parse_code_object_without_symbols_yields_no_symbols():
    data = make_code_obj_info([{"id": 7}])
    assert parse_code_object_info(data) == [
        CodeObjectDisassembly(code_object_id=7, symbols=[])
    ]


def test_parse_symbol_without_instructions_yields_empty_instruction_list():
    data = make_code_obj_info([
        make_code_object(7, [{"name": "kern", "virtual_address": 0x1000}])
    ])
    assert parse_code_object_info(data)[0].symbols == [
        CodeObjectSymbol(name="kern", virtual_address=0x1000, instructions=[])
    ]


def test_load_discovers_files_and_parses_pid():
    workload_dir = Path(common.get_output_dir())
    workload_dir.mkdir(parents=True, exist_ok=True)
    try:
        (workload_dir / "123_code_obj_info.json").write_text(
            json.dumps(make_code_obj_info([make_code_object(1, [])])), encoding="utf-8"
        )
        (workload_dir / "sub").mkdir()
        (workload_dir / "sub" / "456_code_obj_info.json").write_text(
            json.dumps(make_code_obj_info([make_code_object(2, [])])), encoding="utf-8"
        )

        result = load_code_object_disassemblies(str(workload_dir))

        assert set(result) == {123, 456}
        assert result[123][0].code_object_id == 1
        assert result[456][0].code_object_id == 2
    finally:
        common.clean_output_dir(True, str(workload_dir))


def test_load_skips_file_without_pid_prefix():
    workload_dir = Path(common.get_output_dir())
    workload_dir.mkdir(parents=True, exist_ok=True)
    try:
        (workload_dir / "code_obj_info.json").write_text(
            json.dumps(make_code_obj_info([make_code_object(1, [])])), encoding="utf-8"
        )

        assert load_code_object_disassemblies(str(workload_dir)) == {}
    finally:
        common.clean_output_dir(True, str(workload_dir))


def test_load_skips_malformed_json():
    workload_dir = Path(common.get_output_dir())
    workload_dir.mkdir(parents=True, exist_ok=True)
    try:
        (workload_dir / "123_code_obj_info.json").write_text(
            "{not json", encoding="utf-8"
        )

        assert load_code_object_disassemblies(str(workload_dir)) == {}
    finally:
        common.clean_output_dir(True, str(workload_dir))


# =============================================================================
# Execution pipeline lookup
# =============================================================================


@pytest.fixture
def pipeline_table(tmp_path, monkeypatch):
    """Point the loader at a small generated table."""
    analysis_configs = tmp_path / "rocprof_compute_soc" / "analysis_configs"
    analysis_configs.mkdir(parents=True)
    (analysis_configs / "instruction_pipelines.yaml").write_text(
        yaml.safe_dump({
            "commit": "0" * 40,
            "pipelines": {
                "VALU": ["v_mov_b32_e32"],
                "INTERNAL": ["s_waitcnt"],
                "MATRIX": ["v_mfma_f32_16x16x16f16"],
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(InstructionPipelines, "table", None)
    monkeypatch.setattr(code_object_analysis.config, "rocprof_compute_home", tmp_path)


@pytest.mark.parametrize(
    "instruction, expected",
    [
        ("v_mov_b32_e32 v1, 0", "VALU"),
        ("s_waitcnt", "INTERNAL"),
        ("s_waitcnt lgkmcnt(0)", "INTERNAL"),
        ("v_mfma_f32_16x16x16f16 a[0:3], v0, v1, a[0:3]", "MATRIX"),
        ("v_not_a_real_instruction v0", None),
        (None, None),
        ("", None),
    ],
)
def test_lookup_reads_the_leading_mnemonic(pipeline_table, instruction, expected):
    assert InstructionPipelines.lookup(instruction) == expected


def test_lookup_without_a_table_leaves_every_type_unset(tmp_path, monkeypatch):
    """A missing table degrades to empty types instead of failing analyze."""
    monkeypatch.setattr(InstructionPipelines, "table", None)
    monkeypatch.setattr(code_object_analysis.config, "rocprof_compute_home", tmp_path)

    assert InstructionPipelines.lookup("v_mov_b32_e32 v1, 0") is None
    assert InstructionPipelines.lookup("s_waitcnt") is None
