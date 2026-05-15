# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the helper function registry and resolver."""

import os

import pytest

from amdisa.sema_ast import (
    ExecModel,
    SemaBlock,
    SemaNode,
    SemaNodeKind,
    SemaType,
)
from amdisa.sema_helpers import (
    HELPER_REGISTRY,
    HelperTreatment,
    list_unresolved,
    resolve_helper,
)

_MRISA = os.environ.get('MRISA_PATH', os.path.expanduser('~/rocm-dev/mrisa'))
SEMA_XML_PATH = os.path.join(_MRISA, 'amdgpu_isa_cdna4.semantics.xml')
_HAS_SEMA_XML = os.path.isfile(SEMA_XML_PATH)


class TestHelperRegistry:
    def test_registry_has_135_entries(self):
        assert len(HELPER_REGISTRY) == 135

    def test_address_calc_entries_are_inline_cpp(self):
        for name in ['CalcBufferAddr', 'CalcFlatAddr', 'CalcGlobalAddr',
                      'CalcDsAddr', 'CalcScalarGlobalAddr',
                      'CalcScalarBufferAddr', 'CalcScratchAddr']:
            treatment, cpp = HELPER_REGISTRY[name]
            assert treatment == HelperTreatment.INLINE_CPP
            assert cpp is not None

    def test_system_entries_are_opaque_nop(self):
        for name in ['CheckBarrierComplete', 'PrefetchScalarData',
                      'PrefetchScalarInst', 'ReallocVgprs',
                      'WaitIdleExceptStoreCnt', 'nop', 's_nop',
                      'InCluster', 'InWorkgroup']:
            treatment, cpp = HELPER_REGISTRY[name]
            assert treatment == HelperTreatment.OPAQUE_NOP
            assert cpp is None

    def test_recursive_entries_have_no_cpp_name(self):
        recursive = [
            (name, t, cpp) for name, (t, cpp) in HELPER_REGISTRY.items()
            if t == HelperTreatment.RECURSIVE
        ]
        assert len(recursive) >= 37
        for name, _, cpp in recursive:
            assert cpp is None, f"{name} should have cpp=None"

    def test_all_entries_have_valid_treatment(self):
        for name, (treatment, _) in HELPER_REGISTRY.items():
            assert isinstance(treatment, HelperTreatment), (
                f"{name} has invalid treatment {treatment}"
            )

    def test_type_conversion_entries(self):
        for name in ['f16_to_f32', 'f32_to_bf16', 'fp8_to_f32', 'bf8_to_f16']:
            treatment, cpp = HELPER_REGISTRY[name]
            assert treatment == HelperTreatment.INLINE_CPP
            assert cpp is not None

    def test_scaled_conversion_entries(self):
        for name in ['f32_to_fp8_scale', 'bf16_to_fp4_sr_scale',
                      'f16_to_bf6_scale']:
            treatment, cpp = HELPER_REGISTRY[name]
            assert treatment == HelperTreatment.INLINE_CPP
            assert 'scale' in cpp

    def test_fp_classification_entries(self):
        for name in ['isNAN', 'isQuietNAN', 'isSignalNAN', 'cvtToQuietNAN']:
            assert name in HELPER_REGISTRY
            assert HELPER_REGISTRY[name][0] == HelperTreatment.INLINE_CPP


class TestResolveHelper:
    def test_known_inline(self):
        treatment, cpp = resolve_helper('CalcBufferAddr')
        assert treatment == HelperTreatment.INLINE_CPP
        assert cpp == 'calc_buffer_addr'

    def test_known_opaque(self):
        treatment, cpp = resolve_helper('nop')
        assert treatment == HelperTreatment.OPAQUE_NOP
        assert cpp is None

    def test_known_recursive(self):
        treatment, cpp = resolve_helper('v_add_nc_u32')
        assert treatment == HelperTreatment.RECURSIVE
        assert cpp is None

    def test_unknown_returns_inline_with_name(self):
        treatment, cpp = resolve_helper('some_future_helper')
        assert treatment == HelperTreatment.INLINE_CPP
        assert cpp == 'some_future_helper'

    def test_depth_limit_raises(self):
        with pytest.raises(ValueError, match="depth exceeded"):
            resolve_helper('v_add_nc_u32', depth=4)

    def test_cycle_detection(self):
        body_a = SemaNode(SemaNodeKind.SEQ, children=(
            SemaNode(SemaNodeKind.CALL, call_name='v_max_num_f32', children=(
                SemaNode(SemaNodeKind.ID, id_name='v_max_num_f32'),
            )),
        ))
        body_b = SemaNode(SemaNodeKind.SEQ, children=(
            SemaNode(SemaNodeKind.CALL, call_name='v_add_nc_u32', children=(
                SemaNode(SemaNodeKind.ID, id_name='v_add_nc_u32'),
            )),
        ))
        fake_blocks = {
            'V_ADD_NC_U32': SemaBlock('V_ADD_NC_U32', ExecModel.VECTOR, body_a),
            'V_MAX_NUM_F32': SemaBlock('V_MAX_NUM_F32', ExecModel.VECTOR, body_b),
        }
        with pytest.raises(ValueError, match="Cycle detected"):
            resolve_helper(
                'v_add_nc_u32', all_blocks=fake_blocks,
                visited=set(), depth=0,
            )


class TestListUnresolved:
    def test_empty_blocks(self):
        assert list_unresolved({}) == []

    def test_all_resolved(self):
        body = SemaNode(SemaNodeKind.SEQ, children=(
            SemaNode(SemaNodeKind.CALL, call_name='CalcBufferAddr', children=(
                SemaNode(SemaNodeKind.ID, id_name='CalcBufferAddr'),
            )),
        ))
        blocks = {'TEST': SemaBlock('TEST', ExecModel.SCALAR, body)}
        assert list_unresolved(blocks) == []

    def test_unresolved_detected(self):
        body = SemaNode(SemaNodeKind.SEQ, children=(
            SemaNode(SemaNodeKind.CALL, call_name='SomeNewHelper', children=(
                SemaNode(SemaNodeKind.ID, id_name='SomeNewHelper'),
            )),
        ))
        blocks = {'TEST': SemaBlock('TEST', ExecModel.SCALAR, body)}
        assert list_unresolved(blocks) == ['SomeNewHelper']


@pytest.mark.skipif(not _HAS_SEMA_XML, reason="Semantics XML not available")
class TestSemaXmlCoverage:
    """Verify all .call targets are in the registry."""

    @pytest.fixture(scope='class')
    def blocks(self):
        from amdisa.sema_parser import parse_semantics_xml
        return parse_semantics_xml(SEMA_XML_PATH)

    def test_no_unresolved_helpers(self, blocks):
        unresolved = list_unresolved(blocks)
        assert unresolved == [], (
            f"Unresolved .call targets: {unresolved}"
        )

    def test_call_target_count(self, blocks):
        targets = set()
        for block in blocks.values():
            for node in block.body.walk():
                if node.kind == SemaNodeKind.CALL and node.call_name:
                    targets.add(node.call_name)
        assert len(targets) == 135

    def test_no_real_cycles(self, blocks):
        for name, (treatment, _) in HELPER_REGISTRY.items():
            if treatment == HelperTreatment.RECURSIVE:
                resolve_helper(name, all_blocks=blocks, visited=set(), depth=0)
