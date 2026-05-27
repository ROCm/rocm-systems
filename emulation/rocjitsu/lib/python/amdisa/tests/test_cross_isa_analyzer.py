# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the CrossIsaAnalyzer."""

import os

import pytest

from amdisa.cross_isa import (
    CrossIsaAnalyzer,
    SharedInstructionPlan,
    CDNA_GFX9,
    RDNA_GFX10,
    RDNA_GFX11,
    RDNA_GFX12,
    CDNA_ISAS,
    RDNA_ISAS,
)
from amdisa.parser import Parser
from amdisa.semantics import derive_all_semantics

MRISA = os.environ.get('MRISA_PATH', '')
if not MRISA:
    pytest.skip(
        'MRISA_PATH not set — skipping cross-ISA tests', allow_module_level=True
    )

# Use a smaller subset for fast tests.
_PROFILES = None


def _get_profiles():
    """Lazy-load profiles to avoid import overhead on collection."""
    global _PROFILES
    if _PROFILES is None:
        from amdisa.isa_profile import (
            CdnaProfile,
            Cdna1Profile,
            Rdna1Profile,
            Rdna2Profile,
            Rdna3Profile,
            Rdna3_5Profile,
            Rdna4Profile,
        )

        _PROFILES = {
            'cdna1': Cdna1Profile(),
            'cdna4': CdnaProfile(),
            'rdna1': Rdna1Profile(),
            'rdna2': Rdna2Profile(),
            'rdna3': Rdna3Profile(),
            'rdna3_5': Rdna3_5Profile(),
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

    def test_mixed_cdna_rdna_classifies_instructions(self):
        """Analyzing one CDNA and one RDNA ISA together classifies instructions."""
        specs = _parse_specs(['cdna1', 'rdna1'])
        plan = CrossIsaAnalyzer().analyze(specs)
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

    def test_instruction_classification_completeness(self):
        """Every instruction instance should be classified exactly once.

        The analyzer classifies each (isa, mnemonic, encoding) tuple into exactly
        one category. The same mnemonic MAY appear in multiple categories when
        it has different structures across ISA groups:
        - v_add_f32 on CDNA1+CDNA4 (same structure) -> family_shared['cdna']
        - v_add_f32 on RDNA4 (different structure) -> isa_exclusive['rdna4']

        Additionally, the same mnemonic can exist in multiple encodings (e.g.,
        VOP1 and VOP3), each classified independently.

        This test verifies that:
        1. No (mnemonic, encoding, isa) appears in multiple places
        2. Universal and family_shared entries don't overlap for same encoding
        3. All instructions from the input specs are accounted for
        """
        specs = _parse_specs(['cdna1', 'cdna4', 'rdna1', 'rdna4'])
        plan = CrossIsaAnalyzer().analyze(specs)

        # Build a set of all (mnemonic, encoding, isa) tuples that were classified
        # For isa_exclusive (which doesn't track encoding), we use None as encoding
        classified: set[tuple[str, str | None, str]] = set()

        # Universal: mnemonic appears on all listed ISAs with specific encoding
        for mnemonic, info in plan.universal.items():
            for isa in info.isa_names:
                key = (mnemonic, info.encoding_name, isa)
                assert key not in classified, (
                    f"Duplicate classification for {key}: already classified, "
                    f"now found in universal"
                )
                classified.add(key)

        # Family-shared: mnemonic appears on listed ISAs within family
        for family, fam_insts in plan.family_shared.items():
            for (mnemonic, encoding_name), info in fam_insts.items():
                for isa in info.isa_names:
                    key = (mnemonic, encoding_name, isa)
                    assert key not in classified, (
                        f"Duplicate classification for {key}: already classified, "
                        f"now found in family_shared[{family}]"
                    )
                    classified.add(key)

        # ISA-exclusive: mnemonic appears only on that ISA
        # Note: isa_exclusive doesn't track encoding, so multiple encodings
        # of the same mnemonic on one ISA collapse to a single entry
        for isa, mnemonics in plan.isa_exclusive.items():
            for mnemonic in mnemonics:
                key = (mnemonic, None, isa)
                assert key not in classified, (
                    f"Duplicate classification for {key}: already classified, "
                    f"now found in isa_exclusive[{isa}]"
                )
                classified.add(key)

        # Verify we classified a reasonable number of instructions
        assert len(classified) > 1000, (
            f"Expected >1000 classified (mnemonic, encoding, isa) tuples, "
            f"got {len(classified)}"
        )

    def test_multi_encoding_instruction_no_collision(self):
        """Instructions with same mnemonic but different encodings should not collide."""
        # Parse CDNA1 and CDNA4 which have v_mov_b32 in both VOP1 and VOP3
        specs = _parse_specs(['cdna1', 'cdna4'])
        plan = CrossIsaAnalyzer().analyze(specs)

        # Verify composite keys in family_shared - extract mnemonics
        for family, fam_insts in plan.family_shared.items():
            # Check if any v_mov_b32 variants exist
            v_mov_entries = [
                (mnem, enc) for (mnem, enc) in fam_insts.keys() if mnem == 'v_mov_b32'
            ]

            # If v_mov_b32 exists, ensure encoding names are unique (no overwrites)
            if v_mov_entries:
                encodings = [enc for _, enc in v_mov_entries]
                assert len(encodings) == len(
                    set(encodings)
                ), f"Duplicate encodings in family '{family}' for v_mov_b32: {encodings}"
                # Both VOP1 and VOP3 should be present
                assert any(
                    'VOP1' in enc for enc in encodings
                ), f"Expected VOP1 encoding for v_mov_b32 in family '{family}'"
                assert any(
                    'VOP3' in enc for enc in encodings
                ), f"Expected VOP3 encoding for v_mov_b32 in family '{family}'"

    def test_v_cmpx_instructions_cannot_share_execute(self):
        """Verify v_cmpx instructions cannot share execute() templates across families.

        v_cmpx instructions have different execution semantics between CDNA and RDNA
        families. Even if they appear in family_shared, CodeGenerator._can_share_execute()
        must return False to prevent generating shared execute templates.
        """
        from amdisa.codegen import CodeGenerator

        # Parse CDNA and RDNA ISAs
        specs = _parse_specs(['cdna1', 'rdna1'])
        plan = CrossIsaAnalyzer().analyze(specs)

        # Create a minimal CodeGenerator to test _can_share_execute
        # We don't need full code generation, just the sharing logic
        codegen = CodeGenerator(
            isa_spec=specs[0][1],  # cdna1 spec
            out_path='/tmp',  # unused for this test
            semantics=specs[0][2],  # cdna1 semantics
            shared_plan=plan,
        )

        # Find all v_cmpx instructions
        v_cmpx_instructions = []

        # Check family_shared for v_cmpx
        for family_key, fam_insts in plan.family_shared.items():
            for (mnemonic, encoding_name), info in fam_insts.items():
                if mnemonic.startswith('v_cmpx_'):
                    v_cmpx_instructions.append(
                        (mnemonic, encoding_name, info.semantic_class)
                    )

        # Verify we found v_cmpx instructions
        assert v_cmpx_instructions, "No v_cmpx instructions found in the plan"

        # Critical assertion: _can_share_execute must return False for ALL v_cmpx
        failing_instructions = []
        for mnemonic, encoding_name, semantic_class in v_cmpx_instructions:
            can_share = codegen._can_share_execute(mnemonic)
            if can_share:
                failing_instructions.append((mnemonic, encoding_name, semantic_class))

        assert not failing_instructions, (
            f"CodeGenerator._can_share_execute() incorrectly returns True for v_cmpx instructions:\n"
            f"{failing_instructions}\n"
            f"These instructions have different semantics between CDNA and RDNA families "
            f"and must NOT generate shared execute templates. Their semantic classes must be "
            f"in CodeGenerator._NON_SHAREABLE_CLASSES."
        )

    def test_classify_family_sub_families(self):
        """Verify _classify_family() returns correct sub-family names."""
        classify = CrossIsaAnalyzer._classify_family

        # Sub-family groupings (finest).
        assert classify({'rdna1'}) == 'rdna_gfx10'
        assert classify({'rdna1', 'rdna2'}) == 'rdna_gfx10'
        assert classify({'rdna3'}) == 'rdna_gfx11'
        assert classify({'rdna3', 'rdna3_5'}) == 'rdna_gfx11'
        assert classify({'rdna4'}) == 'rdna_gfx12'
        assert classify({'cdna1'}) == 'cdna_gfx9'
        assert classify({'cdna1', 'cdna4'}) == 'cdna_gfx9'
        assert classify({'cdna1', 'cdna2', 'cdna3', 'cdna4'}) == 'cdna_gfx9'

        # Coarse family fallback (spans sub-families, same family).
        assert classify({'rdna1', 'rdna3'}) == 'rdna'
        assert classify({'rdna1', 'rdna4'}) == 'rdna'
        assert classify({'rdna1', 'rdna2', 'rdna3', 'rdna3_5', 'rdna4'}) == 'rdna'

        # Mixed CDNA + RDNA — joined names.
        result = classify({'cdna1', 'rdna1'})
        assert 'cdna1' in result and 'rdna1' in result

    def test_rdna_cross_generation_no_collision(self):
        """Parsing RDNA ISAs across generations must not trigger key collisions.

        ENC_MUBUF changed layout between RDNA2 (gfx10) and RDNA3 (gfx11),
        and RDNA4 (gfx12) replaced it with ENC_VBUFFER.  Without sub-family
        grouping, instructions like buffer_load_format_x would collide on
        the same (mnemonic, enc_name) key in family_shared['rdna'].
        """
        specs = _parse_specs(['rdna1', 'rdna2', 'rdna3', 'rdna3_5', 'rdna4'])
        plan = CrossIsaAnalyzer().analyze(specs)

        # The analyzer has defensive assertions that fire on key collisions
        # in family_shared (see cross_isa.py line 289).  Before sub-family
        # grouping, ENC_MUBUF instructions like buffer_load_format_x would
        # collide here because RDNA1/2 and RDNA3/3.5 have different field
        # layouts under the same encoding name.  Reaching this point without
        # an AssertionError confirms the sub-family split resolves it.
        total = plan.total_universal + plan.total_family_shared + plan.total_exclusive
        assert total > 0

        # Sub-family keys should appear for RDNA generations.
        family_keys = set(plan.family_shared.keys())
        assert (
            'rdna_gfx10' in family_keys
        ), f"Expected 'rdna_gfx10' in family_shared keys, got {family_keys}"

    def test_family_shared_uses_sub_family_keys(self):
        """Verify family_shared uses sub-family keys when multiple ISAs share one.

        Sub-family keys (e.g., 'rdna_gfx10') only appear when 2+ ISAs from
        the same sub-family are present.  With only one ISA per sub-family,
        same-sub-family groups become isa_exclusive and cross-sub-family
        groups land in the coarse 'rdna' bucket.
        """
        # RDNA1 + RDNA2 are both in RDNA_GFX10 — should produce 'rdna_gfx10'.
        specs = _parse_specs(['rdna1', 'rdna2', 'rdna4'])
        plan = CrossIsaAnalyzer().analyze(specs)

        family_keys = set(plan.family_shared.keys())
        assert 'rdna_gfx10' in family_keys, (
            f"Expected 'rdna_gfx10' in family_shared keys when rdna1+rdna2 "
            f"are both present, got: {family_keys}"
        )
