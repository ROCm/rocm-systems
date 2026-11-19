# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace
import hashlib, pytest, re
from amdisa.codegen.execute import float_fract, float_ldexp
from amdisa.codegen.execute.sema_lower import _INLINE_UNARY_OPS, lower_sema_block
from amdisa.codegen.execute.simd_codegen import (
    SIMD_VOP1_UNARY,
    local_coverage_probe,
    simd_probe_line,
)
from amdisa.codegen.execute.vector_special import (
    _TRIG_PREOP_CHUNK_BITS,
    _TRIG_PREOP_TWO_OVER_PI_CHUNKS,
    _TRIG_PREOP_VALID_BITS,
    gen_vector_bitop3,
    gen_vector_cvt_pk,
    gen_vector_div_fixup,
    gen_vector_mad_32_16,
    gen_vector_movrel,
    gen_vector_mullit,
    gen_vector_perm_pk16,
    gen_vector_permlane,
    gen_vector_qsad,
    gen_vector_sat_pack,
    gen_vector_swaprel,
    gen_vector_trig_preop,
)
from amdisa.codegen.config import CodegenConfig
from amdisa.codegen._generator import CodeGenerator
from amdisa.gpuisa import Instruction, Operand
from amdisa.isa_profile import Cdna5Profile, Rdna3Profile, Rdna4Profile
from amdisa.sema_derive import derive_sema_block
from amdisa.sema_enrich import enrich_block
from amdisa.semantics import derive_semantics

"""Tests for vector-special execute code generation."""


def test_relative_vgpr_ops_use_unsigned_packed_m0_fields():
    move = gen_vector_movrel(
        ['vdst'],
        ['src0'],
        'srcdst2',
        True,
        supports_dpp=True,
        supports_dpp8=True,
        supports_sdwa=True,
    )
    swap = gen_vector_swaprel(['vdst'], ['src0'], True)
    full_width = gen_vector_movrel(['vdst'], ['src0'], 'src', True)

    for cpp in (move, swap):
        assert 'wf.m0() & 0x3ffu' in cpp
        assert '(wf.m0() >> 16) & 0x3ffu' in cpp
        assert 'resolved_vgpr_offset(wf' in cpp
    assert 'amdgpu::arith::execute_move32' in move
    assert 'DppPlan rel_dpp_plan' in move
    assert 'apply_dpp(\n        rel_src, rel_dpp_plan' in move
    assert 'apply_dpp8(rel_src' in move
    assert 'stage_source(rel_src' in move
    assert 'rel_staged_src_binding(rel_src' in move
    assert 'amdgpu::arith::execute_swap<false>' in swap
    assert 'std::optional<uint32_t>' in move
    assert 'Operand rel_src' in move
    assert 'Operand rel_dst' in move
    assert 'rel_src_offset > 255u' not in swap
    assert 'rel_dst_offset > 255u' not in swap
    assert 'rel_src_valid ? rel_src_index : 0u' in move
    assert 'if (!rel_dst_valid) return;' in move
    assert 'amdgpu::relative_vgpr_index(rel_src_base, wf.m0(),' in full_width
    assert 'static_cast<int32_t>(wf.m0())' not in full_width


@pytest.mark.parametrize(
    ('generator', 'args'),
    [
        (gen_vector_movrel, (['vdst'], ['src0'], 'unknown')),
        (gen_vector_perm_pk16, (['vdst'], ['src0', 'src1', 'src2'], 'unknown', True)),
    ],
)
def test_unsupported_vector_special_operations_fail_during_generation(generator, args):
    with pytest.raises(ValueError):
        generator(*args)


def test_trig_preop_two_over_pi_table_integrity():
    table_bytes = b''.join(
        chunk.to_bytes(3, byteorder='big') for chunk in _TRIG_PREOP_TWO_OVER_PI_CHUNKS
    )

    assert _TRIG_PREOP_CHUNK_BITS == 24
    assert _TRIG_PREOP_VALID_BITS == 1201
    assert len(_TRIG_PREOP_TWO_OVER_PI_CHUNKS) == 51
    assert (
        hashlib.sha256(table_bytes).hexdigest()
        == '6945fb5ae75f6a3e8dec95553821e5886324e9e96181eb7944d828f508eee5b9'
    )


def test_permlane_is_not_shared_across_profiles():
    assert 'vector_permlane16' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'vector_permlanex16' in CodeGenerator._NON_SHAREABLE_CLASSES


def test_arch_local_execute_bodies_are_not_shared():
    assert 'scalar_getreg' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'scalar_setreg' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'scalar_setreg_imm' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'vector_movrel' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'vector_swaprel' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'vector_perm_pk16' in CodeGenerator._NON_SHAREABLE_CLASSES
    assert 'vector_qsad' in CodeGenerator._NON_SHAREABLE_CLASSES


@pytest.mark.parametrize('module', [float_fract, float_ldexp])
def test_raw_operation_policies_reject_unknown_formats(module):
    with pytest.raises(ValueError, match='bf16'):
        module.policy_expr('bf16', *(['false'] if module is float_ldexp else []))


@pytest.mark.parametrize(
    'template',
    [
        'v_add_f32_vop2',
        'v_sub_f32_vop2',
        'v_subrev_f32_vop2',
        'v_mul_f32_vop2',
        'v_fmac_f32_vop2',
        'v_fmaak_f32_vop2',
        'v_fmamk_f32_vop2',
        'v_mac_f32_vop2',
        'v_add_f32_vop3',
        'v_sub_f32_vop3',
        'v_subrev_f32_vop3',
        'v_mul_f32_vop3',
        'v_fma_f32_vop3',
        'v_fmac_f32_vop3',
    ],
)
def test_f32_mode_aware_probes_share_one_guest_environment_policy(template):
    probe = simd_probe_line(template)

    # One ungated probe taking a single guest FP environment for the instruction,
    # with the arithmetic on the unscoped cores instead of scoping each SIMD chunk.
    assert probe.count('ROCJITSU_TRY_SIMD_') == 1
    assert probe.lstrip().startswith('ROCJITSU_TRY_SIMD_')
    assert '_SCOPED(' in probe
    assert 'native_arithmetic_matches' not in probe
    assert (
        'amdgpu::fma_f32_simd_unscoped(' in probe
        or 'amdgpu::binary_f32_simd_unscoped<' in probe
    )
    # The scoped cores must not also scope per chunk.
    assert 'amdgpu::fma_f32_simd(' not in probe
    assert 'amdgpu::binary_f32_simd<' not in probe


_GENERATED = (
    Path(__file__).resolve().parents[3]
    / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
)


_SCOPED_CORES = (
    'amdgpu::fma_f32_simd(',
    'amdgpu::binary_f32_simd<',
    'amdgpu::binary_f16_simd<',
)


_UNSCOPED_CORE = re.compile(r'amdgpu::\w+_simd_unscoped\b')


_PROBE = re.compile(r'ROCJITSU_TRY_SIMD_\w+\(|try_execute_words_simd<[^>]*>\(')


# Probes that install one guest FP environment for the whole instruction.
_INSTRUCTION_SCOPED_PROBE = re.compile(
    r'ROCJITSU_TRY_SIMD_\w+_SCOPED\('
    r'|ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_F32(?:_SELECTORS)?\('
    r'|try_execute_words_simd<(?:[^,>]+,){4}\s*true>\('
)


def _probe_calls(text: str):
    """Yield each SIMD probe's opener and its full parenthesized call."""
    for match in _PROBE.finditer(text):
        depth, end = 1, match.end()
        while depth:
            depth += {'(': 1, ')': -1}.get(text[end], 0)
            end += 1
        yield match.group(0), text[match.start() : end]


def test_true16_vop3_simd_probe_leaves_unsupported_b16_scalar():
    assert simd_probe_line('v_not_b16_vop3', true16_vop3=True) is None
    assert simd_probe_line('v_cndmask_b16_vop3', true16_vop3=True) is None
    assert simd_probe_line('v_cndmask_b16_vop3', true16_vop3=False) is None


def test_gfx1250_true16_execute_bodies_are_arch_local():
    codegen = object.__new__(CodeGenerator)
    codegen.config = CodegenConfig()
    codegen.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        profile=Cdna5Profile(),
    )
    mov_b16 = Instruction(
        'V_MOV_B16',
        'ENC_VOP1',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
        ],
    )
    or_b16 = Instruction(
        'V_OR_B16',
        'ENC_VOP3',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
            Operand('src1', 16, 'OPR_SRC', True, False, False, False, 2),
        ],
    )
    add_f16 = Instruction(
        'V_ADD_F16',
        'ENC_VOP2',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
            Operand('vsrc1', 16, 'OPR_VGPR', True, False, False, False, 2),
        ],
    )
    not_b16 = Instruction(
        'V_NOT_B16',
        'ENC_VOP1',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
        ],
    )
    cndmask_b16 = Instruction(
        'V_CNDMASK_B16',
        'ENC_VOP3',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
            Operand('src1', 16, 'OPR_SRC', True, False, False, False, 2),
            Operand('src2', 64, 'OPR_SREG', True, False, False, False, 3),
        ],
    )

    assert codegen._requires_arch_local_execute(mov_b16, 'ENC_VOP1')
    assert codegen._requires_arch_local_execute(not_b16, 'ENC_VOP1')
    assert codegen._requires_arch_local_execute(add_f16, 'ENC_VOP2')
    assert codegen._requires_arch_local_execute(or_b16, 'ENC_VOP3')
    assert codegen._requires_arch_local_execute(cndmask_b16, 'ENC_VOP3')
    assert not codegen._can_force_shared_simd_probe(mov_b16, 'ENC_VOP1')


def test_gfx1250_non_true16_simd_probe_can_still_be_shared():
    codegen = object.__new__(CodeGenerator)
    codegen.config = CodegenConfig()
    codegen.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        profile=Cdna5Profile(),
    )
    mov_b32 = Instruction(
        'V_MOV_B32',
        'ENC_VOP1',
        0,
        [
            Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 32, 'OPR_SRC', True, False, False, False, 1),
        ],
    )

    assert not codegen._requires_arch_local_execute(mov_b32, 'ENC_VOP1')
    assert not codegen._can_force_shared_simd_probe(mov_b32, 'ENC_VOP1')


def test_gfx1250_true16_e32_dst_reg_uses_physical_vgpr():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(arch_name='cdna5', profile=Cdna5Profile())
    fmac_f16 = Instruction(
        'V_FMAC_F16',
        'ENC_VOP2',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
            Operand('vsrc1', 16, 'OPR_VGPR', True, False, False, False, 2),
        ],
    )
    add_f32 = Instruction(
        'V_ADD_F32',
        'ENC_VOP2',
        0,
        [
            Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 32, 'OPR_SRC', True, False, False, False, 1),
            Operand('vsrc1', 32, 'OPR_VGPR', True, False, False, False, 2),
        ],
    )

    assert codegen._e32_true16_dst_reg_expr(fmac_f16, 'ENC_VOP2') == (
        '(inst_.vdst & 0x7fu)'
    )
    assert codegen._e32_true16_dst_reg_expr(add_f32, 'ENC_VOP2') == 'inst_.vdst'


def test_rdna4_true16_execute_bodies_are_arch_local():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(arch_name='rdna4', profile=Rdna4Profile())
    add_f16 = Instruction(
        'V_ADD_F16',
        'ENC_VOP2',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
            Operand('vsrc1', 16, 'OPR_VGPR', True, False, False, False, 2),
        ],
    )
    cndmask_b16 = Instruction(
        'V_CNDMASK_B16',
        'ENC_VOP3',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
            Operand('src1', 16, 'OPR_SRC', True, False, False, False, 2),
            Operand('src2', 64, 'OPR_SREG', True, False, False, False, 3),
        ],
    )

    assert codegen._requires_arch_local_execute(add_f16, 'ENC_VOP2')
    assert codegen._requires_arch_local_execute(cndmask_b16, 'ENC_VOP3')
    assert codegen._e32_true16_dst_reg_expr(add_f16, 'ENC_VOP2') == (
        '(inst_.vdst & 0x7fu)'
    )


def test_rdna3_vop3_true16_execute_bodies_are_arch_local():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(arch_name='rdna3', profile=Rdna3Profile())
    cndmask_b16 = Instruction(
        'V_CNDMASK_B16',
        'ENC_VOP3',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
            Operand('src1', 16, 'OPR_SRC', True, False, False, False, 2),
            Operand('src2', 64, 'OPR_SREG', True, False, False, False, 3),
        ],
    )
    or_b16 = Instruction(
        'V_OR_B16',
        'ENC_VOP3',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
            Operand('src1', 16, 'OPR_SRC', True, False, False, False, 2),
        ],
    )
    add_f16 = Instruction(
        'V_ADD_F16',
        'ENC_VOP2',
        0,
        [
            Operand('vdst', 16, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 16, 'OPR_SRC', True, False, False, False, 1),
            Operand('vsrc1', 16, 'OPR_VGPR', True, False, False, False, 2),
        ],
    )

    assert codegen._requires_arch_local_execute(cndmask_b16, 'ENC_VOP3')
    assert codegen._requires_arch_local_execute(or_b16, 'ENC_VOP3')
    assert codegen._requires_arch_local_execute(add_f16, 'ENC_VOP2')


def test_gfx1250_bitop3_b16_uses_true16_helpers():
    body = gen_vector_bitop3(
        ['vdst'], ['src0', 'src1', 'src2'], 'b16', true16_opsel='inst_.opsel'
    )

    assert 'read_vop3_true16_src(src0, wf, lane, inst_.opsel, 0)' in body
    assert 'read_vop3_true16_src(src1, wf, lane, inst_.opsel, 1)' in body
    assert 'read_vop3_true16_src(src2, wf, lane, inst_.opsel, 2)' in body
    assert 'write_vop3_true16_dst(vdst, wf, lane, inst_.opsel, result, true)' in body


@pytest.mark.parametrize('op', ['i16', 'u16'])
def test_pknorm_f16_spelling_matches_pk_norm(op):
    # GFX9 and RDNA1/2 spell V_CVT_PK_NORM_*_F16 as V_CVT_PKNORM_*_F16.
    pknorm = derive_semantics(f'V_CVT_PKNORM_{op.upper()}_F16', 'ENC_VOP3')
    pk_norm = derive_semantics(f'V_CVT_PK_NORM_{op.upper()}_F16', 'ENC_VOP3')
    assert (pknorm.semantic_class, pknorm.operation, pknorm.data_type) == (
        'vector_cvt_pknorm',
        op,
        'f16',
    )
    assert (pknorm.semantic_class, pknorm.operation, pknorm.data_type) == (
        pk_norm.semantic_class,
        pk_norm.operation,
        pk_norm.data_type,
    )
    assert simd_probe_line(f'v_cvt_pknorm_{op}_f16_vop3') == simd_probe_line(
        f'v_cvt_pk_norm_{op}_f16_vop3'
    )


def test_perm_codegen_covers_multiword_results():
    perm = gen_vector_perm_pk16(['vdst'], ['src0', 'src1', 'src2'], 'b6', True)
    assert 'uint32_t packed[3]' in perm
    assert 'source_bit = index * 6u' in perm
    assert 'Operand dst_word = vdst;' in perm
    assert 'dst_word.encoding_value_' in perm
    assert 'read_lane64(src1, lane)' in perm


@pytest.mark.parametrize(
    'mnemonic',
    [
        'min',
        'max',
        'min3',
        'max3',
        'med3',
        'minmax',
        'maxmin',
        'min_num',
        'max_num',
        'minimum',
        'maximum',
    ],
)
@pytest.mark.parametrize('true16', [False, True])
def test_half_selection_has_no_legacy_simd_probe(mnemonic, true16):
    for encoding in ('vop2', 'vop3'):
        assert (
            simd_probe_line(f'v_{mnemonic}_f16_{encoding}', true16_vop3=true16) is None
        )


def test_packed_rtz_rejects_retired_fallbacks():
    for name in ('v_cvt_pkrtz_f16_f32', 'v_cvt_pk_rtz_f16_f32'):
        for encoding in ('vop2', 'vop3'):
            assert simd_probe_line(f'{name}_{encoding}') is None
            assert local_coverage_probe(f'{name}_{encoding}') is None
    with pytest.raises(
        ValueError, match='Packed RTZ conversion requires rj-arith lowering'
    ):
        gen_vector_cvt_pk(['vdst'], ['src0', 'src1'], 'vector_cvt_pkrtz_f16_f32', None)


@pytest.mark.parametrize('op', ['u16_f32', 'i16_f32'])
def test_packed_integer_rejects_retired_fallback(op):
    with pytest.raises(
        ValueError, match='Packed integer conversion requires rj-arith lowering'
    ):
        gen_vector_cvt_pk(['vdst'], ['src0', 'src1'], 'vector_cvt_pk', op, is_vop3=True)


def test_saturating_pack_rejects_retired_fallback():
    with pytest.raises(
        ValueError, match='Saturating packing requires rj-arith lowering'
    ):
        gen_vector_sat_pack(['vdst'], ['src0'], 'u8_i16')


@pytest.mark.parametrize('dtype', ['f32', 'f16'])
@pytest.mark.parametrize('op', ['u16', 'i16'])
def test_normalized_conversion_rejects_retired_fallback(dtype, op):
    with pytest.raises(
        ValueError, match='Normalized packing requires rj-arith lowering'
    ):
        gen_vector_cvt_pk(
            ['vdst'], ['src0', 'src1'], 'vector_cvt_pknorm', op, dtype=dtype
        )
    for prefix in ('pk_norm', 'pknorm'):
        name = f'v_cvt_{prefix}_{op}_{dtype}_vop3'
        assert simd_probe_line(name) is None
        assert local_coverage_probe(name) is None


@pytest.mark.parametrize(
    'cls, sources',
    [
        ('vector_cvt_pk_u8_f32', ['a', 'b', 'c']),
        ('vector_cvt_pk_u8_f32', ['a', 'b']),
        ('vector_pack_b32_f16', ['a', 'b']),
    ],
)
def test_byte_and_halfword_packing_reject_retired_fallbacks(cls, sources):
    with pytest.raises(
        ValueError, match='Byte and halfword packing requires rj-arith lowering'
    ):
        gen_vector_cvt_pk(['dst'], sources, cls, None, is_vop3=True)
    for name in ('v_cvt_pk_u8_f32_vop3', 'v_pack_b32_f16_vop3'):
        assert simd_probe_line(name) is None
        assert local_coverage_probe(name) is None


def test_trig_preop_preserves_special_fp_rules():
    trig = gen_vector_trig_preop(['vdst'], ['src0', 'src1'], is_vop3=True, has_abs=True)
    assert '0xA2F983u' in trig
    assert 'kTwoOverPiChunks' in trig
    assert 'kChunkBits = 24u' in trig
    assert 'kValidBits = 1201u' in trig
    assert 'selector * 53u' in trig
    assert 'if (bit >= kValidBits) return 0' in trig
    assert 'exponent > 1077u' in trig
    assert 'exponent >= 1968u' in trig
    assert 'scale_u53_f64_rtz(segment, scale)' in trig
    assert 'amdgpu::fp_mode::effective_omod' in trig
    assert 'amdgpu::fp_mode::finish_f64' in trig
    assert re.search(r'finish_f64\([^;]*effective_omod', trig, flags=re.DOTALL)
    assert not re.search(r'finish_f64\([^;]*inst_\.omod', trig, flags=re.DOTALL)
