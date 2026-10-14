# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from amdisa.isa_profile import Rdna1Profile, Rdna2Profile
from amdisa.legalization import ExpansionKind, LegalizationGenerator
from amdisa.parser import Parser


@pytest.mark.parametrize(('source', 'target'), [('rdna1', 'rdna2'), ('rdna2', 'rdna1')])
def test_distinct_gfx10_format_tables_require_semantic_expansion(source, target):
    # RDNA1 packed formats such as ID 30 are absent in RDNA2. Instructions
    # alone cannot make a descriptor or immediate valid for the other table.
    isa_dir = Path(__file__).resolve().parents[6] / 'shared/machine-readable-isa/isa'
    specs = [
        (name, Parser(str(isa_dir / f'amdgpu_isa_{name}.xml'), profile).parse(), None)
        for name, profile in [('rdna1', Rdna1Profile()), ('rdna2', Rdna2Profile())]
    ]
    entries = LegalizationGenerator(specs).classify(source, target)
    formatted = [
        entry
        for entry in entries
        if entry.src_mnemonic.startswith(
            (
                'BUFFER_LOAD_FORMAT',
                'BUFFER_STORE_FORMAT',
                'TBUFFER_LOAD_FORMAT',
                'TBUFFER_STORE_FORMAT',
            )
        )
    ]
    assert len(formatted) >= 32
    for entry in formatted:
        assert entry.action.kind == 'expand', entry.src_mnemonic
        assert entry.action.expansion_kind == ExpansionKind.BUFFER_FORMAT_POLICY
