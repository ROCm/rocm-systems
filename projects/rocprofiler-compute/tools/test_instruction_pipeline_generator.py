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
    Prefixes,
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


def test_every_encoding_reaches_its_pipeline():
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
    """Barrier, message and wait instructions have no encoding of their own."""
    assert Rules.classify("s_barrier", {"SALU"}) == "BARRIER"
    assert Rules.classify("s_wakeup_barrier", {"SALU"}) == "BARRIER"
    assert Rules.classify("s_sendmsg", {"SALU"}) == "EXP"
    assert Rules.classify("s_waitcnt", {"SALU"}) == "INTERNAL"
    assert Rules.classify("s_endpgm", {"SALU"}) == "INTERNAL"


def test_unknown_mnemonic_is_unclassified():
    assert Rules.classify("not_an_instruction", set()) is None


def test_mnemonics_cover_every_printed_form():
    """The name and the per-family rename are kept; PseudoInstr is not a name."""
    record = {
        "Mnemonic": "v_add_co_ci_u32",
        # Identifies the instruction rather than naming it, so it is not a key.
        "PseudoInstr": "v_addc_u32_e32",
        # The operands follow the name with no separating space.
        "AsmString": "v_addc_co_u32$vdst, vcc, $src0, $src1",
    }

    mnemonics = TableGen.mnemonics_for_record(record)

    assert mnemonics == {"v_add_co_ci_u32", "v_addc_co_u32"}


def test_variant_takes_the_encodings_of_its_pseudo():
    """A concrete record can print the name while its pseudo sets IsMAI."""
    dump = make_dump({
        "V_MFMA_F32_16X16X16_F16_e64": {
            "Mnemonic": "v_mfma_f32_16x16x16f16",
            "VALU": 1,
            "IsMAI": 1,
        },
        "V_MFMA_F32_16X16X16_F16_gfx940_acd": {
            "PseudoInstr": "V_MFMA_F32_16X16X16_F16_e64",
            "AsmString": "v_mfma_f32_16x16x16_f16$vdst, $src0",
            "VALU": 1,
        },
    })

    table = Rules.build_table(TableGen.instructions(TableGen.records(dump)))

    # Both spellings of the one instruction answer the same way.
    assert table == {
        "v_mfma_f32_16x16x16f16": "MATRIX",
        "v_mfma_f32_16x16x16_f16": "MATRIX",
    }


def test_record_without_a_pseudo_stands_for_itself():
    """A pseudo has no PseudoInstr of its own and is its own instruction."""
    dump = make_dump({
        "V_ADD_F32_e32": {"Mnemonic": "v_add_f32", "VALU": 1},
        "S_ADD_I32": {"Mnemonic": "s_add_i32", "SALU": 1},
    })

    instructions = TableGen.instructions(TableGen.records(dump))

    assert sorted(sorted(i.mnemonics) for i in instructions) == [
        ["s_add_i32"],
        ["v_add_f32"],
    ]


def test_records_without_a_printed_name_are_dropped():
    dump = make_dump({"NO_NAME": {"Mnemonic": None, "VALU": 1}})

    assert TableGen.records(dump) == []


def test_corpus_mnemonics_come_from_the_disassembly_lines():
    disassembly = "\n".join([
        "gfx950_copy.hsaco:\tfile format elf64-amdgpu",
        "0000000000001900 <copy>:",
        "\ts_load_dword s3, s[0:1], 0x1c        // 000000001900: C00200C0",
        "\ts_endpgm                             // 00000000191C: BF810000",
    ])

    assert Corpus.parse_mnemonics(disassembly) == {"s_load_dword", "s_endpgm"}


def test_record_encodings_ignore_fields_that_are_not_pipeline_encodings():
    dump = make_dump({
        "V_ADD_F32_e32": {"Mnemonic": "v_add_f32", "VALU": 1, "isCodeGenOnly": 1}
    })

    assert TableGen.records(dump) == [
        InstructionRecord(
            record_name="V_ADD_F32_e32",
            pseudo="V_ADD_F32_e32",
            mnemonics=frozenset({"v_add_f32"}),
            encodings=frozenset({"VALU"}),
        )
    ]


def test_compression_keeps_the_shortest_prefix_that_holds():
    """One pipeline below a node means the node stands for all of it."""
    table = {
        "v_add_f32": Pipeline.VALU,
        "v_sub_f32": Pipeline.VALU,
        "s_add_i32": Pipeline.SCALAR,
    }

    prefixes = Prefixes.compress(table)

    # v and s would hold too, but a prefix stops at an underscore.
    assert prefixes == {"v_": Pipeline.VALU, "s_": Pipeline.SCALAR}
    assert Prefixes.mismatches(prefixes, table) == []


def test_a_disagreeing_descendant_gets_its_own_prefix():
    """The longer prefix wins, so a name under another name still resolves."""
    table = {"s_wakeup": Pipeline.INTERNAL, "s_wakeup_barrier": Pipeline.BARRIER}

    prefixes = Prefixes.compress(table)

    assert Prefixes.longest_match(prefixes, "s_wakeup") == Pipeline.INTERNAL
    assert Prefixes.longest_match(prefixes, "s_wakeup_barrier") == Pipeline.BARRIER
    assert Prefixes.mismatches(prefixes, table) == []


def test_mismatches_name_what_a_prefix_answers_differently():
    """The check the generator refuses to write on."""
    table = {"v_add_f32": Pipeline.VALU, "v_add_f64": Pipeline.MATRIX}

    assert Prefixes.mismatches({"v_": Pipeline.VALU}, table) == ["v_add_f64"]


def test_document_groups_prefixes_and_carries_the_overrides():
    """Prefixes group under their pipeline, per architecture and by default."""
    document = Table.build_document("0" * 40, {"v_add_f32": Pipeline.VALU})

    assert document["pipelines"] == {"VALU": ["v_add_f32"]}
    assert document["arch_overrides"]["gfx908"] == {"FLAT": ["global_", "scratch_"]}
    assert "v_mfma_f64_" in document["arch_overrides"]["gfx950"]["VALU"]
    # An architecture with no hardware difference carries no override.
    assert "gfx1250" not in document["arch_overrides"]


def test_overrides_apply_by_prefix_for_one_architecture():
    """An override claims the encoding tails of the name it lists."""
    table = {
        "global_load_dword": Pipeline.VMEM,
        "v_mfma_f64_16x16x4f64_vgprcd_e64": Pipeline.MATRIX,
    }

    assert Rules.apply_overrides(table, "gfx950") == {
        "global_load_dword": Pipeline.FLAT,
        "v_mfma_f64_16x16x4f64_vgprcd_e64": Pipeline.VALU,
    }
    assert Rules.apply_overrides(table, "gfx1250") == table
