# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Alias expansion must cover an encoding's whole don't-care id range.

An encoding format whose encoding field is narrower than the 9-bit
``encoding_id`` (``raw[0] >> 23``) shares that id across a range of don't-care
low bits, so ``emit_all`` emits one table row per alias and lets binary search
find any of them.

The expansion used to build each alias as ``enc_id | i``, which is only the
intended base-plus-offset when the base's don't-care bits are already zero.
They are not: ``src_encoding_order`` comes from
``LegalizationGenerator._dt_index()``, the decode-table slot of the first
opcode the encoding actually *defines*. For ``ENC_VOP2`` that slot is
``op << 2``, so it is zero only when the encoding defines opcode 0. CDNA1-CDNA4
define VOP2 opcode 0 (``V_CNDMASK_B32``) and came out at 0; CDNA5 and RDNA1-4
start at opcode 1 and come out at 4. ``4 | i`` over ``i`` in 0..255 reaches only
the 128 ids that have bit 2 set, and reaches each of them twice -- and bit 2 of
``op << 2`` is bit 0 of the opcode, so every even-opcode VOP2 instruction lost
its row entirely and ``lookup()`` returned nullptr for it.

These tests pin the property the fix restores: the emitted alias ids for one
source record are exactly the ``2**dont_care`` ids starting at the encoding's
own base. They do not assert table-wide row uniqueness, which is a separate
concern with a separate cause.
"""

import collections
import functools
import os
import re
from pathlib import Path

import pytest

from amdisa.__main__ import _PROFILES
from amdisa.legalization import (
    LegalizationAction,
    LegalizationEntry,
    LegalizationGenerator,
)
from amdisa.legalization_codegen import emit_all
from amdisa.parser import Parser

# Width of the encoding_id the generated tables are keyed on.
_ENCODING_ID_BITS = 9

# Taken from the generator's own profile table, minus the bare 'cdna' alias
# that has no ISA XML of its own, so a newly supported ISA is covered here the
# moment it is registered.
_ISAS = {isa: cls for isa, cls in _PROFILES.items() if isa != 'cdna'}

_GENERATED = (
    Path(__file__).resolve().parents[3] / 'rocjitsu/src/rocjitsu/code/dbt/generated'
)

_ROW_RE = re.compile(r'\{\s*(\d+),\s*(\d+),\s*Action')


def _mrisa_dir() -> Path:
    default = (
        Path(__file__).resolve().parents[6] / 'shared' / 'machine-readable-isa' / 'isa'
    )
    return Path(os.environ.get('MRISA_PATH', default))


@functools.lru_cache(maxsize=None)
def _spec(isa: str):
    xml = f'amdgpu_isa_{isa.replace(".", "_")}.xml'
    return Parser(str(_mrisa_dir() / xml), _ISAS[isa]()).parse()


def _narrow_encodings(spec):
    """(name, bits, dt_index) for every encoding with don't-care id bits."""
    narrow = []
    for name, enc in sorted(spec.encoding_map.items()):
        bits = getattr(enc, 'enc_field_bit_cnt', 0)
        if not bits or bits >= _ENCODING_ID_BITS:
            continue
        narrow.append((name, bits, LegalizationGenerator._dt_index(enc, spec)))
    return narrow


def _emit_rows(tmp_path: Path, entries):
    """Run entries through emit_all and read back (opcode, encoding_id) rows."""
    emit_all(tmp_path, [('src', 'dst', entries)])
    text = (tmp_path / 'legalization_src_to_dst.h').read_text()
    return [(int(op), int(enc)) for op, enc in _ROW_RE.findall(text)]


def _entry(mnemonic: str, encoding_order: int, encoding_bits: int, opcode: int):
    return LegalizationEntry(
        src_mnemonic=mnemonic,
        src_encoding=mnemonic,
        src_encoding_order=encoding_order,
        src_encoding_bits=encoding_bits,
        src_opcode=opcode,
        action=LegalizationAction.identity(),
    )


def _table_rows(pair: str):
    path = _GENERATED / f'legalization_{pair}.h'
    assert path.is_file(), f'missing generated table {path}'
    return collections.Counter(
        (int(op), int(enc)) for op, enc in _ROW_RE.findall(path.read_text())
    )


class TestLegalizationAliasExpansion:
    @pytest.mark.parametrize('isa', sorted(_ISAS))
    def test_alias_ids_cover_the_full_dont_care_range(self, isa, tmp_path):
        """Every narrow encoding expands to its complete 2**dont_care range.

        Parameterised over every supported ISA so that a future ISA which
        leaves an encoding's opcode 0 undefined is caught without anyone
        having to remember this failure mode.
        """
        spec = _spec(isa)
        narrow = _narrow_encodings(spec)
        assert narrow, f'{isa} declares no narrow encodings'

        # One entry per encoding, each with a distinct opcode so the emitted
        # rows can be attributed back to the encoding that produced them.
        entries = [
            _entry(name, dt_index, bits, opcode)
            for opcode, (name, bits, dt_index) in enumerate(narrow)
        ]
        by_opcode = collections.defaultdict(list)
        for opcode, encoding_id in _emit_rows(tmp_path, entries):
            by_opcode[opcode].append(encoding_id)

        for opcode, (name, bits, dt_index) in enumerate(narrow):
            dont_care = _ENCODING_ID_BITS - bits
            base = dt_index & ~((1 << dont_care) - 1)
            expected = list(range(base, base + (1 << dont_care)))
            assert sorted(by_opcode[opcode]) == expected, (
                f'{isa} {name}: expanded alias ids do not cover '
                f'[{base}, {base + (1 << dont_care)})'
            )

    @pytest.mark.parametrize('isa', ['rdna4', 'cdna5'])
    def test_vop2_expands_from_the_encoding_base_not_a_defined_opcode(
        self, isa, tmp_path
    ):
        """A dt_index with don't-care bits set still expands from the base.

        These ISAs leave VOP2 opcode 0 undefined, so ``_dt_index`` reports the
        slot of opcode 1 and its don't-care bits are non-zero. This is the
        directly targeted case: without masking, the expansion starts at that
        slot and covers half the range twice over.
        """
        spec = _spec(isa)
        encoding = spec.encoding_map['ENC_VOP2']
        dt_index = LegalizationGenerator._dt_index(encoding, spec)
        assert dt_index != 0, f'{isa} ENC_VOP2 no longer exercises this case'

        dont_care = _ENCODING_ID_BITS - encoding.enc_field_bit_cnt
        rows = _emit_rows(
            tmp_path, [_entry('ENC_VOP2', dt_index, encoding.enc_field_bit_cnt, 0)]
        )
        base = dt_index & ~((1 << dont_care) - 1)
        assert [enc for _, enc in rows] == list(range(base, base + (1 << dont_care)))

    def test_generated_rdna4_to_cdna4_reaches_even_vop2_opcodes(self):
        """V_OR_B32 is even-opcode VOP2, and used to have no row at all.

        Checked against the committed table because a generator fix that is
        never regenerated changes nothing. V_XOR_B32 is the adjacent odd
        opcode, which was always reachable; it is here to show the assertion
        is not vacuous.
        """
        encoding = _spec('rdna4').encoding_map['ENC_VOP2']
        opcodes = {inst.name: inst.opcode for inst in encoding.insts}
        rows = _table_rows('rdna4_to_cdna4')

        for mnemonic in ('V_OR_B32', 'V_XOR_B32'):
            opcode = opcodes[mnemonic]
            slot = encoding.primary_dt_ptrs[opcode]
            for encoding_id in range(slot, slot + 4):
                assert rows[(opcode, encoding_id)] >= 1, (
                    f'rdna4_to_cdna4 has no row for {mnemonic} '
                    f'(opcode {opcode}) at encoding_id {encoding_id}'
                )
