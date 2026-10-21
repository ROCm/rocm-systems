# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for conversions lowered through shared/conversion.h."""

import pytest

from amdisa.codegen.execute import conversion
from amdisa.codegen.execute.sema_lower import lower_sema_block
from amdisa.codegen.execute.simd_codegen import local_coverage_probe, simd_probe_line
from amdisa.sema_derive import derive_sema_block
from amdisa.sema_enrich import enrich_block
from amdisa.semantics import TO_FLOAT_CONVERSIONS, derive_semantics


def _formats(source, result):
    def name(fmt):
        prefix = 'fp_format' if fmt.startswith('F') else 'conversion'
        return f'amdgpu::{prefix}::{fmt}'

    return f'{name(source)}, {name(result)}'


# Every conversion the semantic table routes through shared/conversion.h.
_INTEGER_TO_FLOAT = [
    (f'V_CVT_{dtype.upper()}', _formats(*formats))
    for dtype, formats in TO_FLOAT_CONVERSIONS.items()
    if conversion.is_integer_source(dtype)
]
_FLOAT_TO_FLOAT = [
    (f'V_CVT_{dtype.upper()}', _formats(*formats))
    for dtype, formats in TO_FLOAT_CONVERSIONS.items()
    if not conversion.is_integer_source(dtype)
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


@pytest.mark.parametrize('dtype', ['f16_i16', 'f16_u16'])
def test_f16_integer_conversion_simd_writes_true16_halves(dtype):
    vop3 = simd_probe_line(f'v_cvt_{dtype}_vop3', true16_vop3=True)
    assert vop3 == (
        '  if (amdgpu::try_execute_words_simd<1, false, true, 1>(inst, wf, conversion))'
        ' return;'
    )
    e32 = local_coverage_probe(
        f'v_cvt_{dtype}_vop1', e32=True, e32_half_dst=True, e32_half_inputs=1
    )
    assert e32 == (
        '  if (amdgpu::try_execute_words_simd<1, true, true, 1>(inst, wf, conversion)) return;'
    )
    # Without true16 halves the result is written zero-extended.
    assert simd_probe_line(f'v_cvt_{dtype}_vop1').startswith(
        '  ROCJITSU_TRY_SIMD_VOP1_UNARY(uint32_t, uint32_t,'
    )


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
        '  ROCJITSU_TRY_SIMD_CONVERSION_FROM_F64(conversion);'
    )
    # A floating source's ABS/NEG/OMOD/CLAMP travel in the shared object.
    assert simd_probe_line('v_cvt_f64_f32_vop3') == (
        '  ROCJITSU_TRY_SIMD_CONVERSION_TO_F64(conversion);'
    )
    f16 = simd_probe_line('v_cvt_f16_f32_vop3', true16_vop3=True)
    assert f16 == (
        '  if (amdgpu::try_execute_words_simd<1, false, true, 0>(inst, wf, conversion))'
        ' return;'
    )


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
def test_fp8_decode_simd_probe_uses_the_shared_ocp_decoder(name):
    layout = 'Fp8' if name == 'fp8' else 'Bf8'
    probe = local_coverage_probe(f'v_cvt_f32_{name}_vop3')
    assert (
        f'amdgpu::conversion::decode_fp8<amdgpu::conversion::{layout}>(byte)' in probe
    )
    assert ('util::fp8_e5m3_to_f32_simd' in probe) == (name == 'fp8')
    assert 'util::fp8_e4m3_to_f32_simd' not in probe
    assert 'util::bf8_e5m2_to_f32_simd' not in probe
    # CDNA4's SIMD decode matches its scalar body's util helpers.
    cdna4 = local_coverage_probe(f'v_cvt_f32_{name}_vop3', arch_name='cdna4')
    assert 'amdgpu::conversion::decode_fp8' not in cdna4
    helper = (
        'util::fp8_e4m3_to_f32_simd' if name == 'fp8' else 'util::bf8_e5m2_to_f32_simd'
    )
    assert f'{helper}(byte)' in cdna4


def test_fp8_encodes_keep_util_helpers_on_cdna3_and_cdna4():
    # CDNA3 uses FNUZ encodings; CDNA4 keeps its manual's FP16_OVFL rule.
    for arch in ('cdna3', 'cdna4'):
        assert not conversion.uses_ocp_fp8(arch)
    for arch in ('cdna5', 'rdna4'):
        assert conversion.uses_ocp_fp8(arch)
    assert conversion.to_fp8_expr('bf8', True, False, 'inst_') == (
        'amdgpu::conversion_to_fp8<amdgpu::conversion::Bf8>(wf, 0u, inst_.neg)'
    )
