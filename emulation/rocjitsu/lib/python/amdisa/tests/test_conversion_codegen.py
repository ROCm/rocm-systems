# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for conversions lowered through shared/conversion.h."""

import pytest

from amdisa.codegen.execute import conversion
from amdisa.codegen.execute.sema_lower import lower_sema_block
from amdisa.codegen.execute.simd_codegen import local_coverage_probe, simd_probe_line
from amdisa.sema_derive import derive_sema_block
from amdisa.sema_enrich import enrich_block
from amdisa.semantics import derive_semantics

_INTEGER_TO_FLOAT = [
    ('V_CVT_F32_I32', 'amdgpu::conversion::I32, amdgpu::fp_format::F32'),
    ('V_CVT_F32_U32', 'amdgpu::conversion::U32, amdgpu::fp_format::F32'),
    ('V_CVT_F16_I16', 'amdgpu::conversion::I16, amdgpu::fp_format::F16'),
    ('V_CVT_F16_U16', 'amdgpu::conversion::U16, amdgpu::fp_format::F16'),
]


def _lower(name, enc, fields=frozenset()):
    block = derive_sema_block(derive_semantics(name, enc))
    if fields:
        block = enrich_block(block, enc_field_names=frozenset(fields))
    return lower_sema_block(block)


@pytest.mark.parametrize(('name', 'formats'), _INTEGER_TO_FLOAT)
def test_integer_to_float_vop3_passes_output_modifiers_to_one_stage_object(
    name, formats
):
    cpp = _lower(name, 'ENC_VOP3', {'clamp', 'omod'})
    stages = (
        f'const auto conversion = amdgpu::conversion_to_float<{formats}>'
        '(wf, 0u, 0u, inst_.omod, inst_.clamp);'
    )
    assert stages in cpp
    # Bits go straight to the destination: no host float, rounding or rescaling.
    assert 'conversion(' in cpp.split(stages, 1)[1]
    for host in (
        'static_cast<float>',
        'f32_to_f16',
        'apply_omod',
        'clamp_floating_result',
    ):
        assert host not in cpp


@pytest.mark.parametrize(('name', 'formats'), _INTEGER_TO_FLOAT)
def test_integer_to_float_vop1_scales_f16_results_by_sdwa_omod(name, formats):
    cpp = _lower(name, 'ENC_VOP1')
    if 'F16' in name:
        source = formats.split(',')[0]
        expected = f'amdgpu::sdwa_conversion_to_f16<{source}>(*this, wf);'
    else:
        expected = f'amdgpu::conversion_to_float<{formats}>(wf);'
    assert f'const auto conversion = {expected}' in cpp


@pytest.mark.parametrize(
    ('dtype', 'formats'),
    [
        ('f32_i32', 'amdgpu::conversion::I32, amdgpu::fp_format::F32'),
        ('f32_u32', 'amdgpu::conversion::U32, amdgpu::fp_format::F32'),
    ],
)
def test_f32_integer_conversion_simd_keeps_output_modifiers(dtype, formats):
    vop1 = simd_probe_line(f'v_cvt_{dtype}_vop1')
    vop3 = simd_probe_line(f'v_cvt_{dtype}_vop3')
    assert vop1 == (
        f'  ROCJITSU_TRY_SIMD_VOP1_UNARY(uint32_t, uint32_t, '
        f'amdgpu::conversion_to_float<{formats}>(wf));'
    )
    assert vop3 == (
        f'  ROCJITSU_TRY_SIMD_VOP1_UNARY(uint32_t, uint32_t, '
        f'amdgpu::conversion_to_float<{formats}>(wf, 0u, 0u, inst.inst_.omod, inst.inst_.clamp));'
    )


@pytest.mark.parametrize('dtype', ['f16_i16', 'f16_u16'])
def test_f16_integer_conversion_simd_writes_true16_halves(dtype):
    stages = conversion.stages_expr(
        dtype, ('0u', '0u', 'inst.inst_.omod', 'inst.inst_.clamp')
    )
    vop3 = simd_probe_line(f'v_cvt_{dtype}_vop3', true16_vop3=True)
    assert vop3 == (
        f'  if (amdgpu::try_execute_words_simd<1, false, true, 1>(inst, wf, {stages})) return;'
    )
    e32 = local_coverage_probe(
        f'v_cvt_{dtype}_vop1', e32=True, e32_half_dst=True, e32_half_inputs=1
    )
    vop1_stages = conversion.stages_expr(dtype)
    assert e32 == (
        f'  if (amdgpu::try_execute_words_simd<1, true, true, 1>(inst, wf, {vop1_stages})) return;'
    )
    # Without true16 halves the result is written zero-extended.
    assert simd_probe_line(f'v_cvt_{dtype}_vop1').startswith(
        '  ROCJITSU_TRY_SIMD_VOP1_UNARY(uint32_t, uint32_t,'
    )


_FLOAT_TO_FLOAT = [
    ('V_CVT_F16_F32', 'amdgpu::fp_format::F32, amdgpu::fp_format::F16'),
    ('V_CVT_F32_F64', 'amdgpu::fp_format::F64, amdgpu::fp_format::F32'),
    ('V_CVT_F64_F32', 'amdgpu::fp_format::F32, amdgpu::fp_format::F64'),
]


@pytest.mark.parametrize(('name', 'formats'), _FLOAT_TO_FLOAT)
def test_float_to_float_vop3_passes_every_modifier_to_one_stage_object(name, formats):
    cpp = _lower(name, 'ENC_VOP3', {'abs', 'neg', 'clamp', 'omod'})
    stages = (
        f'const auto conversion = amdgpu::conversion_to_float<{formats}>'
        '(wf, inst_.abs, inst_.neg, inst_.omod, inst_.clamp);'
    )
    assert stages in cpp
    for host in (
        'static_cast<float>',
        'static_cast<double>',
        'f32_to_f16',
        'apply_to_float',
    ):
        assert host not in cpp


def test_f64_conversion_simd_uses_raw_lane_glue():
    assert simd_probe_line('v_cvt_f32_f64_vop1') == (
        '  ROCJITSU_TRY_SIMD_CONVERSION_FROM_F64(amdgpu::conversion_to_float<'
        'amdgpu::fp_format::F64, amdgpu::fp_format::F32>(wf));'
    )
    assert simd_probe_line('v_cvt_f64_f32_vop3') == (
        '  ROCJITSU_TRY_SIMD_CONVERSION_TO_F64(amdgpu::conversion_to_float<'
        'amdgpu::fp_format::F32, amdgpu::fp_format::F64>'
        '(wf, inst.inst_.abs, inst.inst_.neg, inst.inst_.omod, inst.inst_.clamp));'
    )
    f16 = simd_probe_line('v_cvt_f16_f32_vop3', true16_vop3=True)
    assert f16.startswith('  if (amdgpu::try_execute_words_simd<1, false, true, 0>(')


@pytest.mark.parametrize('name', ['fp8', 'bf8'])
def test_fp8_decode_simd_probe_uses_the_shared_ocp_decoder(name):
    layout = 'Fp8' if name == 'fp8' else 'Bf8'
    probe = local_coverage_probe(f'v_cvt_f32_{name}_vop3')
    assert (
        f'amdgpu::conversion::decode_fp8<amdgpu::conversion::{layout}>(byte)' in probe
    )
    assert ('util::fp8_e5m3_to_f32_simd' in probe) == (name == 'fp8')
    assert 'util::fp8_e4m3_to_f32_simd' not in probe
    assert 'util::bf8_e5m2_to_f32_simd' not in probe


def test_fp8_encodes_keep_fnuz_helpers_on_cdna3():
    assert not conversion.uses_ocp_fp8('cdna3')
    for arch in ('cdna4', 'cdna5', 'rdna4'):
        assert conversion.uses_ocp_fp8(arch)
    assert conversion.to_fp8_expr('bf8', True, False, 'inst_') == (
        'amdgpu::conversion_to_fp8<amdgpu::conversion::Bf8>(wf, 0u, inst_.neg)'
    )
