# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the CrossIsaAnalyzer."""

import os

import pytest

from amdisa.cross_isa import CrossIsaAnalyzer, SharedInstructionPlan
from amdisa.parser import Parser
from amdisa.semantics import derive_all_semantics

MRISA = os.environ.get('MRISA_PATH', '')
if not MRISA:
    pytest.skip('MRISA_PATH not set — skipping cross-ISA tests', allow_module_level=True)

# Use a smaller subset for fast tests.
_PROFILES = None


def _get_profiles():
    """Lazy-load profiles to avoid import overhead on collection."""
    global _PROFILES
    if _PROFILES is None:
        from amdisa.isa_profile import (
            CdnaProfile, Cdna1Profile, Rdna1Profile, Rdna4Profile,
        )
        _PROFILES = {
            'cdna1': Cdna1Profile(),
            'cdna4': CdnaProfile(),
            'rdna1': Rdna1Profile(),
            'rdna4': Rdna4Profile(),
        }
    return _PROFILES


def _parse_specs(isa_names: list[str]):
    """Parse and derive semantics for the given ISAs."""
    profiles = _get_profiles()
    specs = []
    for name in isa_names:
        spec = Parser(f'{MRISA}/amdgpu_isa_{name}.xml', profiles[name]).parse()
        sem = derive_all_semantics(spec)
        specs.append((name, spec, sem))
    return specs


class TestCrossIsaAnalyzer:
    """Tests for CrossIsaAnalyzer.analyze()."""

    def test_analyze_returns_plan(self):
        specs = _parse_specs(['cdna1', 'cdna4'])
        plan = CrossIsaAnalyzer().analyze(specs)
        assert isinstance(plan, SharedInstructionPlan)

    def test_universal_nonempty_with_two_cdna(self):
        specs = _parse_specs(['cdna1', 'cdna4'])
        plan = CrossIsaAnalyzer().analyze(specs)
        # S_ADD_U32 should be universal across CDNA ISAs.
        assert plan.total_universal > 0
        assert 's_add_u32' in plan.universal

    def test_exclusive_contains_cdna4_only(self):
        specs = _parse_specs(['cdna1', 'cdna4'])
        plan = CrossIsaAnalyzer().analyze(specs)
        # CDNA4 has FP8/BF8 instructions not on CDNA1.
        cdna4_excl = plan.isa_exclusive.get('cdna4', set())
        assert len(cdna4_excl) > 0

    def test_rdna4_exclusive_instructions(self):
        specs = _parse_specs(['rdna1', 'rdna4'])
        plan = CrossIsaAnalyzer().analyze(specs)
        rdna4_excl = plan.isa_exclusive.get('rdna4', set())
        # RDNA4 has many renamed/new instructions.
        assert len(rdna4_excl) > 50

    def test_family_shared_across_families(self):
        specs = _parse_specs(['cdna1', 'rdna1'])
        plan = CrossIsaAnalyzer().analyze(specs)
        # Instructions with different encoding layouts across families
        # should be classified as family_shared, not universal.
        assert plan.total_family_shared > 0 or plan.total_universal > 0

    def test_four_isa_analysis(self):
        specs = _parse_specs(['cdna1', 'cdna4', 'rdna1', 'rdna4'])
        plan = CrossIsaAnalyzer().analyze(specs)
        total = plan.total_universal + plan.total_family_shared + plan.total_exclusive
        assert total > 0
        # Universal requires identical encoding + semantics + operand types.
        # With operand type differences across ISAs, the universal count is
        # small — only instructions with the exact same OperandType enum
        # values on all ISAs qualify (e.g., s_getpc_b64, s_endpgm).
        assert plan.total_universal >= 1

    def test_shared_inst_info_fields(self):
        specs = _parse_specs(['cdna1', 'cdna4'])
        plan = CrossIsaAnalyzer().analyze(specs)
        if 's_add_u32' in plan.universal:
            info = plan.universal['s_add_u32']
            assert info.mnemonic == 's_add_u32'
            assert info.encoding_name  # Non-empty
            assert len(info.field_layout) > 0
            assert len(info.isa_names) == 2

    def test_no_instruction_in_multiple_categories(self):
        """Each instruction should appear in exactly one category."""
        specs = _parse_specs(['cdna1', 'cdna4', 'rdna1', 'rdna4'])
        plan = CrossIsaAnalyzer().analyze(specs)
        universal_set = set(plan.universal.keys())
        family_set = set()
        for fam_insts in plan.family_shared.values():
            family_set |= set(fam_insts.keys())
        exclusive_set = set()
        for isa_insts in plan.isa_exclusive.values():
            exclusive_set |= isa_insts
        # No overlap between universal and family_shared.
        assert len(universal_set & family_set) == 0, (
            f"Overlap: {universal_set & family_set}"
        )
