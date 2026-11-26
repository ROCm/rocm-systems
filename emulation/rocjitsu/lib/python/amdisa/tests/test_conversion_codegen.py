# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for conversions lowered through shared/conversion.h."""

import pytest

from amdisa.codegen.execute import conversion, fp8_formats
from amdisa.codegen.execute.sema_lower import lower_sema_block
from amdisa.codegen.execute.simd_codegen import local_coverage_probe, simd_probe_line
from amdisa.codegen.execute.vector_special import (
    gen_vector_cvt_pk,
)
from amdisa.sema_derive import derive_sema_block
from amdisa.sema_enrich import enrich_block
from amdisa.semantics import (
    INTEGER_SOURCE_CONVERSIONS,
    TO_FLOAT_CONVERSIONS,
    derive_semantics,
)


def _formats(source, result):
    def name(fmt):
        prefix = 'fp_format' if fmt.startswith('F') else 'conversion'
        return f'amdgpu::{prefix}::{fmt}'

    return f'{name(source)}, {name(result)}'


# Every conversion the semantic table routes through shared/conversion.h.
_INTEGER_TO_FLOAT = [
    (f'V_CVT_{dtype.upper()}', _formats(*formats))
    for dtype, formats in TO_FLOAT_CONVERSIONS.items()
    if dtype in INTEGER_SOURCE_CONVERSIONS and not dtype.startswith('f16_')
]
_FLOAT_TO_FLOAT = [
    (f'V_CVT_{dtype.upper()}', _formats(*formats))
    for dtype, formats in TO_FLOAT_CONVERSIONS.items()
    if dtype not in INTEGER_SOURCE_CONVERSIONS and not dtype.startswith('f16_')
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
    # Retained emulator order for SDWA F16 results; see sdwa_conversion_to_f16.
    cpp = _lower(name, 'ENC_VOP1')
    if 'F16' in name:
        source = formats.split(',')[0]
        expected = f'amdgpu::sdwa_conversion_to_f16<{source}>(*this, wf);'
    else:
        expected = f'amdgpu::conversion_to_float<{formats}>(wf);'
    assert f'const auto conversion = {expected}' in cpp


@pytest.mark.parametrize('dtype', ['f32_i32', 'f32_u32'])
def test_f32_integer_conversion_simd_passes_the_scalar_stage_object(dtype):
    # The probe passes the scalar body's object, OMOD and CLAMP included, so a
    # modified VOP3 encoding stays on the SIMD path.
    probe = '  ROCJITSU_TRY_SIMD_VOP1_UNARY(uint32_t, uint32_t, conversion);'
    assert simd_probe_line(f'v_cvt_{dtype}_vop1') == probe
    assert simd_probe_line(f'v_cvt_{dtype}_vop3') == probe


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


@pytest.mark.parametrize(
    'dtype', ['f16_i16', 'f16_u16', 'i16_f16', 'u16_f16', 'f16_f32', 'f32_f16']
)
def test_half_conversions_have_no_legacy_simd_probe(dtype):
    for encoding in ('vop1', 'vop3'):
        assert simd_probe_line(f'v_cvt_{dtype}_{encoding}') is None


@pytest.mark.parametrize(
    'name', ['V_CVT_F16_I16', 'V_CVT_F16_U16', 'V_CVT_F16_F32', 'V_CVT_F32_F16']
)
def test_half_conversion_requires_arithmetic_lowering(name):
    with pytest.raises(ValueError, match='require rj-arith lowering'):
        _lower(name, 'ENC_VOP1')


def test_share_stages_hoists_the_scalar_declaration_above_the_probe():
    declaration = (
        '  const auto conversion = amdgpu::conversion_pk_u8(wf, inst_.abs, inst_.neg);'
    )
    body = f'  uint64_t exec = wf.exec();\n{declaration}\n  for (;;) {{}}\n'
    probe = '  ROCJITSU_TRY_SIMD_VOP3_TERNARY_INT(uint32_t, conversion);'
    shared_probe, rest = conversion.share_stages(probe, body)
    assert shared_probe == f'{declaration}\n{probe}'
    assert rest == '  uint64_t exec = wf.exec();\n  for (;;) {}\n'
    # Probes that resolve nothing shared are left alone.
    other = (
        '  ROCJITSU_TRY_SIMD_VOP1_UNARY(uint32_t, uint32_t, amdgpu::conversion::Fp8{});'
    )
    assert conversion.share_stages(other, body) == (other, body)
    assert conversion.share_stages(None, body) == (None, body)
    with pytest.raises(ValueError, match='declares none'):
        conversion.share_stages(probe, '  for (;;) {}\n')


@pytest.mark.parametrize('name', ['fp8', 'bf8'])
def test_fp8_decode_simd_probe_follows_the_target_policy(name):
    layout = 'Fp8' if name == 'fp8' else 'Bf8'
    probe = local_coverage_probe(f'v_cvt_f32_{name}_vop3', arch_name='rdna4')
    assert (
        f'amdgpu::conversion::decode_fp8<amdgpu::conversion::{layout}>(byte)' in probe
    )
    assert ('util::fp8_e5m3_to_f32_simd' in probe) == (name == 'fp8')
    assert 'util::fp8_e4m3_to_f32_simd' not in probe
    assert 'util::bf8_e5m2_to_f32_simd' not in probe
    # The SIMD decode matches the scalar body's util helpers elsewhere.
    helper = (
        'util::fp8_e4m3_to_f32_simd' if name == 'fp8' else 'util::bf8_e5m2_to_f32_simd'
    )
    for arch in ('cdna4', 'cdna5', None):
        util = local_coverage_probe(f'v_cvt_f32_{name}_vop3', arch_name=arch)
        assert 'amdgpu::conversion::decode_fp8' not in util
        assert f'{helper}(byte)' in util


@pytest.mark.parametrize(
    'arch, policy, shared_narrowing, shared_widening',
    [
        ('cdna3', fp8_formats.Fp8Policy.FNUZ, False, False),
        ('cdna4', fp8_formats.Fp8Policy.OCP_UTIL, False, False),
        ('cdna5', fp8_formats.Fp8Policy.OCP_SHARED_NARROWING, True, False),
        ('rdna4', fp8_formats.Fp8Policy.OCP_SHARED, True, True),
        ('rdna3', fp8_formats.DEFAULT_FP8_POLICY, False, False),
        ('cdna1', fp8_formats.DEFAULT_FP8_POLICY, False, False),
        ('', fp8_formats.DEFAULT_FP8_POLICY, False, False),
        (None, fp8_formats.DEFAULT_FP8_POLICY, False, False),
    ],
)
def test_fp8_policy_selects_narrowing_and_widening(
    arch, policy, shared_narrowing, shared_widening
):
    assert fp8_formats.fp8_policy(arch) is policy
    assert fp8_formats.uses_shared_fp8_narrowing(arch) == shared_narrowing
    assert fp8_formats.uses_shared_fp8_widening(arch) == shared_widening
    assert fp8_formats.uses_fnuz_fp8(arch) == (arch == 'cdna3')


@pytest.mark.parametrize('fmt, helper', [('fp8', 'fp8_e4m3'), ('bf8', 'bf8_e5m2')])
def test_cdna5_uses_shared_narrowing_for_non_scaled_ocp_conversions(fmt, helper):
    packed = gen_vector_cvt_pk(
        ['dst'],
        ['a', 'b'],
        'vector_cvt_pk',
        f'{fmt}_f32',
        arch_name='cdna5',
        is_vop3=True,
        fp8_format_select='inst_.clamp',
    )
    assert (
        f'amdgpu::conversion_to_fp8<amdgpu::conversion::{fmt.title()}>(wf, 0u, inst_.neg)'
        in packed
    )
    assert ('util::f32_to_fp8_e5m3_rne_mode' in packed) == (fmt == 'fp8')
    widened = gen_vector_cvt_pk(
        ['dst'], ['a'], 'vector_cvt_pk', f'f32_{fmt}', arch_name='cdna5'
    )
    assert f'util::{helper}_to_f32' in widened
