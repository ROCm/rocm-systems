#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the instruction execution pipeline table generator.

The classification rules are exercised against records shaped like real
llvm-tblgen output but written here, so these tests need neither LLVM nor a
checkout of it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from instruction_pipeline_generator import (  # noqa: E402
    InstructionRecord,
    build_pipeline_table,
    classify,
    find_rule_conflicts,
    find_uncovered_mnemonics,
    instruction_records,
    merge_flags_by_spelling,
    parse_corpus_mnemonics,
    spellings_for_record,
)


def make_dump(records):
    """Build a tblgen JSON dump holding the given instruction records."""
    dump = {"!instanceof": {"Instruction": list(records)}}
    dump.update(records)
    return dump


def test_matrix_wins_over_valu():
    """An MFMA record sets IsMAI and VALU; the pipeline is the matrix one."""
    assert classify("v_mfma_f32_16x16x16f16", {"IsMAI", "VALU"}) == "MATRIX"
    assert classify("v_wmma_f32_16x16x16_f16", {"IsWMMA", "VALU"}) == "MATRIX"
    assert classify("v_swmmac_f32_16x16x32_f16", {"IsSWMMAC", "VALU"}) == "MATRIX"


def test_branch_wins_over_scalar():
    """A conditional branch is scalar-encoded but runs on the branch pipeline."""
    assert classify("s_cbranch_scc1", {"isBranch", "SALU"}) == "BRANCH"


def test_every_flag_reaches_its_pipeline():
    assert classify("v_add_f32_e32", {"VALU"}) == "VALU"
    assert classify("s_add_i32", {"SALU"}) == "SCALAR"
    assert classify("s_load_dwordx4", {"SMRD"}) == "SCALAR"
    assert classify("ds_read_b32", {"DS"}) == "LDS"
    assert classify("lds_direct_load", {"LDSDIR"}) == "LDS"
    assert classify("buffer_load_dword", {"MUBUF", "VALU"}) == "VMEM"
    assert classify("tbuffer_load_format_x", {"MTBUF", "VALU"}) == "VMEM"
    assert classify("image_load", {"MIMG", "VALU"}) == "VMEM"
    assert classify("exp", {"EXP"}) == "EXP"


def test_name_rules_split_the_flat_encoding():
    """global, scratch and flat share one encoding but not one pipeline."""
    assert classify("global_load_dword", {"FLAT", "VALU"}) == "VMEM"
    assert classify("scratch_load_dword", {"FLAT", "VALU"}) == "VMEM"
    assert classify("flat_load_dword", {"FLAT", "VALU"}) == "FLAT"


def test_name_rules_carve_up_the_scalar_encoding():
    """Barrier, message and wait instructions have no flag of their own."""
    assert classify("s_barrier", {"SALU"}) == "BARRIER"
    assert classify("s_wakeup_barrier", {"SALU"}) == "BARRIER"
    assert classify("s_sendmsg", {"SALU"}) == "EXP"
    assert classify("s_waitcnt", {"SALU"}) == "INTERNAL"
    assert classify("s_endpgm", {"SALU"}) == "INTERNAL"


def test_unknown_mnemonic_is_unclassified():
    assert classify("not_an_instruction", set()) is None


def test_spellings_cover_every_printed_form():
    """Short form, encoding-suffixed form and per-family rename all get a key."""
    record = {
        "Mnemonic": "v_add_co_ci_u32",
        "PseudoInstr": "v_addc_u32_e32",
        # The operands follow the name with no separating space.
        "AsmString": "v_add_co_ci_u32$vdst, vcc, $src0, $src1",
    }

    spellings = spellings_for_record("V_ADD_CO_CI_U32_e32_gfx11", record)

    assert spellings == {
        "v_add_co_ci_u32",
        "v_add_co_ci_u32_e32",
        "v_addc_u32_e32",
    }


def test_flags_merge_across_records_sharing_a_name():
    """The variant carrying IsMAI decides the pipeline for all of them."""
    dump = make_dump({
        "V_MFMA_F32_e64_vgprcd": {
            "Mnemonic": "v_mfma_f32_16x16x16f16",
            "VALU": 1,
        },
        "V_MFMA_F32_e64": {
            "Mnemonic": "v_mfma_f32_16x16x16f16",
            "VALU": 1,
            "IsMAI": 1,
        },
    })

    merged_flags = merge_flags_by_spelling(instruction_records(dump))

    assert merged_flags["v_mfma_f32_16x16x16f16"] == {"VALU", "IsMAI"}
    assert build_pipeline_table(merged_flags) == {
        "v_mfma_f32_16x16x16f16": "MATRIX",
        "v_mfma_f32_16x16x16f16_e64": "MATRIX",
    }


def test_records_without_a_printed_name_are_dropped():
    dump = make_dump({"PSEUDO_ONLY": {"Mnemonic": None, "VALU": 1}})

    assert instruction_records(dump) == []


def test_rule_conflict_is_reported():
    """A name rule that swallows an encoding it was never meant to cover."""
    merged_flags = {"s_waitcnt": {"SALU"}, "s_setup_something": {"VALU"}}

    conflicts = find_rule_conflicts(merged_flags)

    assert list(conflicts) == ["s_setup_something"]


def test_flag_rule_outranking_a_name_rule_is_not_a_conflict():
    """A branch keeps its pipeline even though a name rule also matches it."""
    assert find_rule_conflicts({"s_setpc_b64": {"SALU", "isBranch"}}) == {}


def test_uncovered_corpus_mnemonic_is_reported():
    table = {"v_mov_b32_e32": "VALU"}

    assert find_uncovered_mnemonics(table, {"v_mov_b32_e32", "v_add_f32_e32"}) == [
        "v_add_f32_e32"
    ]


def test_corpus_mnemonics_come_from_the_disassembly_lines():
    disassembly = "\n".join([
        "gfx950_copy.hsaco:\tfile format elf64-amdgpu",
        "0000000000001900 <copy>:",
        "\ts_load_dword s3, s[0:1], 0x1c        // 000000001900: C00200C0",
        "\ts_endpgm                             // 00000000191C: BF810000",
    ])

    assert parse_corpus_mnemonics(disassembly) == {"s_load_dword", "s_endpgm"}


def test_record_flags_ignore_fields_that_are_not_pipeline_flags():
    dump = make_dump({
        "V_ADD_F32_e32": {"Mnemonic": "v_add_f32", "VALU": 1, "isCodeGenOnly": 1}
    })

    assert instruction_records(dump) == [
        InstructionRecord(
            record_name="V_ADD_F32_e32",
            spellings=frozenset({"v_add_f32", "v_add_f32_e32"}),
            flags=frozenset({"VALU"}),
        )
    ]
