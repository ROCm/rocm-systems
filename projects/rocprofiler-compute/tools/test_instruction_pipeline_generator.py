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
    Corpus,
    InstructionRecord,
    Pipeline,
    Rules,
    Table,
    TableGen,
)


def make_dump(records):
    """Build a tblgen JSON dump holding the given instruction records."""
    dump = {"!instanceof": {"Instruction": list(records)}}
    dump.update(records)
    return dump


def test_matrix_wins_over_valu():
    """An MFMA record sets IsMAI and VALU; the pipeline is the matrix one."""
    assert Rules.classify("v_mfma_f32_16x16x16f16", {"IsMAI", "VALU"}) == "MATRIX"
    assert Rules.classify("v_wmma_f32_16x16x16_f16", {"IsWMMA", "VALU"}) == "MATRIX"
    assert Rules.classify("v_swmmac_f32_16x16x32_f16", {"IsSWMMAC", "VALU"}) == "MATRIX"


def test_branch_wins_over_scalar():
    """A conditional branch is scalar-encoded but runs on the branch pipeline."""
    assert Rules.classify("s_cbranch_scc1", {"isBranch", "SALU"}) == "BRANCH"


def test_every_flag_reaches_its_pipeline():
    assert Rules.classify("v_add_f32_e32", {"VALU"}) == "VALU"
    assert Rules.classify("s_add_i32", {"SALU"}) == "SCALAR"
    assert Rules.classify("s_load_dwordx4", {"SMRD"}) == "SCALAR"
    assert Rules.classify("ds_read_b32", {"DS"}) == "LDS"
    assert Rules.classify("lds_direct_load", {"LDSDIR"}) == "LDS"
    assert Rules.classify("buffer_load_dword", {"MUBUF", "VALU"}) == "VMEM"
    assert Rules.classify("tbuffer_load_format_x", {"MTBUF", "VALU"}) == "VMEM"
    assert Rules.classify("image_load", {"MIMG", "VALU"}) == "VMEM"
    assert Rules.classify("exp", {"EXP"}) == "EXP"


def test_prefixes_split_the_flat_encoding():
    """global, scratch and flat share one encoding but not one pipeline."""
    assert Rules.classify("global_load_dword", {"FLAT", "VALU"}) == "VMEM"
    assert Rules.classify("scratch_load_dword", {"FLAT", "VALU"}) == "VMEM"
    assert Rules.classify("flat_load_dword", {"FLAT", "VALU"}) == "FLAT"


def test_prefixes_split_the_scalar_encoding():
    """Barrier, message and wait instructions have no flag of their own."""
    assert Rules.classify("s_barrier", {"SALU"}) == "BARRIER"
    assert Rules.classify("s_wakeup_barrier", {"SALU"}) == "BARRIER"
    assert Rules.classify("s_sendmsg", {"SALU"}) == "EXP"
    assert Rules.classify("s_waitcnt", {"SALU"}) == "INTERNAL"
    assert Rules.classify("s_endpgm", {"SALU"}) == "INTERNAL"


def test_unknown_mnemonic_is_unclassified():
    assert Rules.classify("not_an_instruction", set()) is None


def test_mnemonics_cover_every_printed_form():
    """Short form, encoding-suffixed form and per-family rename all get a key."""
    record = {
        "Mnemonic": "v_add_co_ci_u32",
        "PseudoInstr": "v_addc_u32_e32",
        # The operands follow the name with no separating space.
        "AsmString": "v_add_co_ci_u32$vdst, vcc, $src0, $src1",
    }

    mnemonics = TableGen.mnemonics_for_record("V_ADD_CO_CI_U32_e32_gfx11", record)

    assert mnemonics == {
        "v_add_co_ci_u32",
        "v_add_co_ci_u32_e32",
        "v_addc_u32_e32",
    }


def test_flags_merge_across_records_sharing_a_mnemonic():
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

    assert Rules.build_table(TableGen.records(dump)) == {
        "v_mfma_f32_16x16x16f16": "MATRIX",
        "v_mfma_f32_16x16x16f16_e64": "MATRIX",
    }


def test_records_without_a_printed_name_are_dropped():
    dump = make_dump({"PSEUDO_ONLY": {"Mnemonic": None, "VALU": 1}})

    assert TableGen.records(dump) == []


def test_corpus_mnemonics_come_from_the_disassembly_lines():
    disassembly = "\n".join([
        "gfx950_copy.hsaco:\tfile format elf64-amdgpu",
        "0000000000001900 <copy>:",
        "\ts_load_dword s3, s[0:1], 0x1c        // 000000001900: C00200C0",
        "\ts_endpgm                             // 00000000191C: BF810000",
    ])

    assert Corpus.parse_mnemonics(disassembly) == {"s_load_dword", "s_endpgm"}


def test_record_flags_ignore_fields_that_are_not_pipeline_flags():
    dump = make_dump({
        "V_ADD_F32_e32": {"Mnemonic": "v_add_f32", "VALU": 1, "isCodeGenOnly": 1}
    })

    assert TableGen.records(dump) == [
        InstructionRecord(
            record_name="V_ADD_F32_e32",
            mnemonics=frozenset({"v_add_f32", "v_add_f32_e32"}),
            flags=frozenset({"VALU"}),
        )
    ]


def test_document_groups_mnemonics_and_records_the_rules():
    """Mnemonics group under their pipeline, and every rule behind them is kept."""
    document = Table.build_document(
        "0" * 40, {"v_add_f32_e32": Pipeline.VALU, "v_add_f32": Pipeline.VALU}
    )

    assert document["pipelines"] == {"VALU": ["v_add_f32", "v_add_f32_e32"]}
    produced_by_flags = set(document["rules"]["flags"].values())
    produced_by_prefixes = set(document["rules"]["prefixes"])
    assert produced_by_flags | produced_by_prefixes == {p.value for p in Pipeline}
    assert "s_wait" in document["rules"]["prefixes"]["INTERNAL"]
