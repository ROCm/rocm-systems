# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Matrix instruction execute body generators.

Free functions that emit C++ execute_impl bodies for matrix
instructions: WMMA, SWMMAC, MFMA, and AccVGPR read/write. Opted-in
targets dispatch to rj-arith; other targets use the legacy mma_exec.h helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from amdisa.codegen.execute.fp8_formats import fp8_helper_name
from amdisa.codegen.execute.required_vgpr import append_required_vgpr_offset
from amdisa.codegen.execute.rj_arith_matrix import gen_dense_wmma
from amdisa.codegen.execute.rj_arith_sparse_matrix import gen_swmmac
from amdisa.isa_profile import MatrixLayout, SwmmacLayout

if TYPE_CHECKING:
    from amdisa.codegen.execute import ExecuteContext

# Input families that have specialized (compile-time M/N/K) MFMA kernels in the
# hand-maintained mma_exec.h. CDNA MFMA and RDNA WMMA both flow through the
# GFX9 MFMA-layout helpers (exec_f32/exec_i32_i8), so the same spec templates
# apply to both; fixed Wave32 WMMA uses its own spec helpers (handled below).
# The spec templates fall back to the generic runtime path for unsupported
# shapes / when stdx::simd is unavailable, so emitting them is always safe.
_MFMA_F32_SPEC = {'F32': 'f32', 'XF32': 'f32', 'F16': 'f16', 'BF16': 'bf16'}
_F8_FIXED = frozenset({'FP8_FP8', 'FP8_BF8', 'BF8_FP8', 'BF8_BF8'})


def _matrix_base(supports_gpr_idx: bool, base: str, role: str) -> str:
    """Apply MODE.GPR_IDX_EN to an architectural-VGPR MMA base."""
    if not supports_gpr_idx:
        return base
    return (
        f'amdgpu::apply_gpr_idx_to_mma_base(wf, vb, {base}, '
        f'amdgpu::VgprMsbRole::{role})'
    )


def _mma_targ(M: int, N: int, K: int, B: int, *, batch_optional: bool) -> str:
    """Spec template argument list, dropping a defaulted BATCH==1."""
    if batch_optional and B == 1:
        return f'{M}, {N}, {K}'
    return f'{M}, {N}, {K}, {B}'


def gen_accvgpr_read(dst: list[str], src: list[str]) -> str:
    """Generate V_ACCVGPR_READ: copy ACCVGPR → VGPR."""
    # In our model, ACCVGPRs are just VGPRs in the accumulator range.
    # The operand resolution already handles the mapping.
    return (
        f'  uint64_t exec = wf.exec();\n'
        f'  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {{\n'
        f'    if (!(exec & (1ULL << lane))) continue;\n'
        f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));\n'
        f'  }}'
    )


def gen_accvgpr_write(dst: list[str], src: list[str]) -> str:
    """Generate V_ACCVGPR_WRITE: copy VGPR → ACCVGPR."""
    return (
        f'  uint64_t exec = wf.exec();\n'
        f'  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {{\n'
        f'    if (!(exec & (1ULL << lane))) continue;\n'
        f'    amdgpu::RegisterAccess(wf).write_lane({dst[0]}, lane, amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));\n'
        f'  }}'
    )


def gen_mfma(ctx: ExecuteContext) -> str:
    """Generate MFMA / SMFMAC matrix multiply-accumulate.

    Opted-in targets use rj-arith or reject a missing binding. Other targets
    use the mma_exec.h helpers which implement the exact GFX9 register
    mapping formulas. The helpers handle cross-lane data movement, WAR
    hazard avoidance (buffered writes), and inline constant accumulator
    initialization without clobbering overlapping source operands.
    """
    if (rj_arith_body := gen_dense_wmma(ctx) or gen_swmmac(ctx)) is not None:
        return rj_arith_body
    if ctx.profile.uses_rj_arith_matrix:
        raise ValueError(
            f'Missing rj-arith matrix binding: {ctx.arch_name} {ctx.inst.name}'
        )
    inst, dst, src = ctx.inst, ctx.dst_ops, ctx.src_ops
    arch_name = ctx.arch_name
    supports_gpr_idx = ctx.profile.supports_gpr_idx
    op_sel_hi_2_expr = ctx.op_sel_hi_2_expr
    name = inst.name
    d, s0, s1, s2 = dst[0], src[0], src[1], src[2]

    import re

    m = re.match(
        r'V_(?:S?MFMA[C]?|S?WMMA[C]?)_(F32|I32|F64|F16|BF16|BF16F32|BF8|FP8)_(\d+)X(\d+)X(\d+)'
        r'(?:_\d+B)?_?(F32|XF32|F16|BF16|I8|IU8|IU4|F64|FP8|BF8'
        r'|BF8_BF8|BF8_FP8|FP8_BF8|FP8_FP8'
        r'|F16_FP8|F16_BF8|BF16_FP8|BF16_BF8'
        r'|F8_F6_F4|F8F6F4|F4)?'
        r'(?:_1K)?$',
        name,
    )

    unimplemented = (
        '  wf.report_instruction_execution_error(\n'
        '      amdgpu::InstructionExecutionError::UnimplementedInstruction);\n'
        '  return;'
    )
    if not m:
        return unimplemented

    result_type = m.group(1)  # F32, I32, F64, BF16F32
    M, N, K = int(m.group(2)), int(m.group(3)), int(m.group(4))
    input_type = m.group(5)  # F32, XF32, F16, BF16, I8, F64, etc.
    is_swmmac = name.startswith('V_SWMMAC_')

    if inst.operands and inst.operands[0].fieldless:
        raise ValueError(
            f'{name}: fieldless operand at index 0 breaks the '
            f'accumulator-is-operand[0] assumption in gen_mfma. A fieldless '
            f'operand on a matrix instruction is unhandled. Decide how it '
            f'should participate in the dst-width / operand mapping before '
            f'regenerating.'
        )
    dst_bits = inst.operands[0].size if inst.operands else 0
    dst_regs = max(1, dst_bits // 32)

    # SMFMAC: sparse matrix FMA with 4:2 structured sparsity.
    # Sparse inputs and metadata are distributed across the whole wave.
    if 'SMFMAC' in name:
        L = []
        L.append(f'  auto &cu = wf.cu();')
        L.append(f'  uint32_t vb = wf.vgpr_alloc().base;')
        if ctx.profile.uses_vgpr_msb_indexing:
            append_required_vgpr_offset(L, 'dst_off', d, uses_vgpr_msb_indexing=True)
            L.append('  uint32_t dst = vb + *dst_off;')
            append_required_vgpr_offset(L, 's0b_off', s0, uses_vgpr_msb_indexing=True)
            L.append('  uint32_t s0b = vb + *s0b_off;')
            append_required_vgpr_offset(L, 's1b_off', s1, uses_vgpr_msb_indexing=True)
            L.append('  uint32_t s1b = vb + *s1b_off;')
            append_required_vgpr_offset(L, 'index_off', s2, uses_vgpr_msb_indexing=True)
            L.append(f'  uint32_t idx = vb + *index_off;')
        else:
            if 'acc_cd' in ctx.enc_field_names:
                dst_base = f'amdgpu::dst_base(vb, {d}.encoding_value_, inst_.acc_cd)'
            else:
                dst_base = f'amdgpu::dst_base(vb, {d}.encoding_value_, 1)'
            L.append(
                f'  uint32_t dst = {_matrix_base(supports_gpr_idx, dst_base, "Dst")};'
            )
            L.append(
                f'  uint32_t s0b = {_matrix_base(supports_gpr_idx, f"amdgpu::src_base(vb, {s0}.encoding_value_)", "Src0")};'
            )
            L.append(
                f'  uint32_t s1b = {_matrix_base(supports_gpr_idx, f"amdgpu::src_base(vb, {s1}.encoding_value_)", "Src1")};'
            )
            L.append(
                f'  uint32_t idx = {_matrix_base(supports_gpr_idx, f"amdgpu::src_base(vb, {s2}.encoding_value_)", "Src2")};'
            )

        flags = 'cu.arithmetic_flags()'
        if arch_name == 'cdna4':
            flags += ' | (wf.fp16_ovfl() ? RJ_ARITH_FP16_OVFL : 0)'

        if (
            arch_name in ('cdna3', 'cdna4')
            and result_type == 'F32'
            and input_type in ('F16', 'BF16', *_F8_FIXED)
        ):
            bits = 16 if input_type in ('F16', 'BF16') else 8
            suffix = (
                '_gfx950'
                if bits == 8 and arch_name == 'cdna4' and M * K == 1024
                else ''
            )
            entry = f'rj_arith_{name.lower()}_w64{suffix}'
            L.append('  uint64_t controls = inst_.cbsz | (uint64_t(inst_.abid) << 3);')
            L.append(
                f'  amdgpu::arith::exec_sparse_matrix<{M*K*bits//4096}, {M*K*bits//2048}, {M*M//64}, 1, 64>'
                f'(wf, {entry}, {flags}, controls, dst, s0b, s1b, idx);'
            )
            return '\n'.join(L)

        if (
            arch_name in ('cdna3', 'cdna4')
            and result_type == 'I32'
            and input_type == 'I8'
        ):
            api = 'rj_arith_' + name.lower() + '_w64'
            L.append('  uint64_t controls = inst_.cbsz | (uint64_t(inst_.abid) << 3);')
            L.append(
                f'  amdgpu::arith::exec_sparse_matrix<{M*K//512}, {M*K//256}, {M*N//64}, 1, 64>('
            )
            L.append(f'      wf, {api}, {flags}, controls, dst, s0b, s1b, idx);')
            return '\n'.join(L)

        raise ValueError(
            f'{name}: missing rj-arith sparse matrix binding for {arch_name}'
        )

    # Compute number of blocks from output register count and matrix dims.
    if result_type == 'F64':
        B = 64 * (dst_regs // 2) // (M * N)
    else:
        B = 64 * dst_regs // (M * N)

    # Determine input element size in bits and extract functions.
    _INPUT_BITS = {
        'F32': 32,
        'XF32': 32,
        'F16': 16,
        'BF16': 16,
        'I8': 8,
        'IU8': 8,
        'IU4': 4,
        'F64': 64,
        'FP8': 8,
        'BF8': 8,
        'FP8_FP8': 8,
        'FP8_BF8': 8,
        'BF8_FP8': 8,
        'BF8_BF8': 8,
        'F16_FP8': 8,
        'F16_BF8': 8,
        'BF16_FP8': 8,
        'BF16_BF8': 8,
        'F8_F6_F4': 8,
        'F8F6F4': 8,
        'F4': 4,
    }
    in_bits = _INPUT_BITS.get(input_type, 32)

    # Map input types to amdgpu::extract_* function names.
    _EXTRACT_A = {
        'F32': 'amdgpu::extract_f32',
        'XF32': 'amdgpu::extract_f32',
        'F16': 'amdgpu::extract_f16',
        'BF16': 'amdgpu::extract_bf16',
        'FP8_FP8': 'amdgpu::extract_fp8',
        'FP8_BF8': 'amdgpu::extract_fp8',
        'BF8_FP8': 'amdgpu::extract_bf8',
        'BF8_BF8': 'amdgpu::extract_bf8',
        'F8_F6_F4': 'amdgpu::extract_fp8',
        'F8F6F4': 'amdgpu::extract_fp8',
        'F4': 'amdgpu::extract_fp4',
    }
    _EXTRACT_B = {
        'F32': 'amdgpu::extract_f32',
        'XF32': 'amdgpu::extract_f32',
        'F16': 'amdgpu::extract_f16',
        'BF16': 'amdgpu::extract_bf16',
        'FP8_FP8': 'amdgpu::extract_fp8',
        'FP8_BF8': 'amdgpu::extract_bf8',
        'BF8_FP8': 'amdgpu::extract_fp8',
        'BF8_BF8': 'amdgpu::extract_bf8',
        'F8_F6_F4': 'amdgpu::extract_fp8',
        'F8F6F4': 'amdgpu::extract_fp8',
        'F4': 'amdgpu::extract_fp4',
    }

    L = []
    L.append(f'  auto &cu = wf.cu();')
    L.append(f'  uint32_t vb = wf.vgpr_alloc().base;')
    arch = arch_name
    is_dense_wmma = name.startswith('V_WMMA_')
    matrix_layout = ctx.profile.matrix_layout
    swmmac_layout = ctx.profile.swmmac_layout if is_swmmac else SwmmacLayout.NONE
    uses_supported_swmmac_layout = swmmac_layout is not SwmmacLayout.NONE
    uses_fixed_wave_swmmac_layout = swmmac_layout is SwmmacLayout.FIXED_WAVE
    uses_arith_sparse_integer = (
        arch == 'cdna5'
        and uses_fixed_wave_swmmac_layout
        and (M, N, K, result_type, input_type) == (16, 16, 128, 'I32', 'IU8')
    )

    uses_runtime_wave_swmmac_layout = swmmac_layout is SwmmacLayout.RUNTIME_WAVE
    uses_rdna_swmmac_arithmetic = (
        uses_runtime_wave_swmmac_layout
        and (M, N) == (16, 16)
        and (
            (
                result_type == 'I32'
                and (input_type, K) in {('IU8', 32), ('IU4', 32), ('IU4', 64)}
            )
            or (
                K == 32
                and (
                    (result_type, input_type)
                    in {
                        ('F32', 'F16'),
                        ('F32', 'BF16'),
                        ('F16', 'F16'),
                        ('BF16', 'BF16'),
                    }
                    or (result_type == 'F32' and input_type in _F8_FIXED)
                )
            )
        )
    )
    if uses_runtime_wave_swmmac_layout and not uses_rdna_swmmac_arithmetic:
        # Reject unsupported sparse shapes before resolving operands or index keys.
        return unimplemented
    # Fixed-wave bit zero is rejected below; the other bits are reuse hints.
    swmmac_index_key_expr = (
        '0u' if uses_fixed_wave_swmmac_layout else 'inst_.opsel & 0x3u'
    )
    uses_fixed_wave32_split_k_dense_layout = (
        matrix_layout is MatrixLayout.WMMA_SPLIT_K
        and ctx.profile.wave_size == 32
        and ctx.profile.wave_size_max == 32
        and is_dense_wmma
    )
    uses_plain_vgpr_dst = matrix_layout is not MatrixLayout.MFMA_ACCUMULATOR and (
        is_dense_wmma or uses_supported_swmmac_layout
    )
    uses_gfx11_wmma_layout = (
        matrix_layout is MatrixLayout.WMMA_REPLICATED_HALFWAVE and is_dense_wmma
    )
    uses_gfx12_wmma_layout = (
        matrix_layout is MatrixLayout.WMMA_SPLIT_K
        and ctx.profile.wave_size < ctx.profile.wave_size_max
        and is_dense_wmma
    )
    uses_rdna_wmma_arithmetic = (
        (uses_gfx11_wmma_layout or uses_gfx12_wmma_layout)
        and input_type in ('IU4', 'IU8')
        and (
            (M, N, K) == (16, 16, 16)
            or (
                uses_gfx12_wmma_layout
                and input_type == 'IU4'
                and (M, N, K) == (16, 16, 32)
            )
        )
    )
    is_gfx1251_f64_wmma = (
        arch == 'cdna5'
        and is_dense_wmma
        and result_type == 'F64'
        and input_type == 'F64'
        and (M, N, K) == (16, 16, 4)
    )
    if is_gfx1251_f64_wmma:
        # Reject an architecturally illegal execution state before resolving or
        # reading any instruction operand. Execution callbacks report failures
        # through the wavefront instead of using exceptions for control flow.
        L.append(
            '  if (!amdgpu::is_gfx1251_wmma_execution_state_valid('
            'wf.wf_size(), wf.exec())) [[unlikely]] {'
        )
        L.append('    wf.report_instruction_execution_error(')
        L.append('        amdgpu::InstructionExecutionError::UnsupportedOperandValue);')
        L.append('    return;')
        L.append('  }')
    swmmac_index_entries = 32 if is_swmmac and K >= 128 and in_bits <= 8 else 16
    if ctx.profile.uses_vgpr_msb_indexing:
        append_required_vgpr_offset(L, 'dst_off', d, uses_vgpr_msb_indexing=True)
        L.append('  uint32_t dst = vb + *dst_off;')
        append_required_vgpr_offset(L, 'src0_base_off', s0, uses_vgpr_msb_indexing=True)
        L.append('  uint32_t src0_base = vb + *src0_base_off;')
        append_required_vgpr_offset(L, 'src1_base_off', s1, uses_vgpr_msb_indexing=True)
        L.append('  uint32_t src1_base = vb + *src1_base_off;')
        src0_base_expr = 'src0_base'
        src1_base_expr = 'src1_base'
        if uses_supported_swmmac_layout:
            if uses_fixed_wave_swmmac_layout and not uses_arith_sparse_integer:
                L.append(f'  uint32_t const_acc = amdgpu::ACC_FROM_VGPR;')
                L.append(f'  uint32_t s2 = dst;')
            append_required_vgpr_offset(L, 'index_off', s2, uses_vgpr_msb_indexing=True)
            L.append(f'  uint32_t index_base = vb + *index_off;')
            if not uses_arith_sparse_integer:
                L.append(f'  uint32_t index_key = {swmmac_index_key_expr};')
            index_base_expr = 'index_base'
            index_key_expr = 'index_key'
        else:
            const_acc_type = 'uint64_t' if is_gfx1251_f64_wmma else 'uint32_t'
            L.append(f'  {const_acc_type} const_acc;')
            L.append(
                f'  auto src2_off = Isa::resolved_vgpr_offset(wf, {s2}.opr_type_, '
                f'{s2}.encoding_value_, {s2}.vgpr_msb_role());'
            )
            L.append(f'  uint32_t s2 = dst;')
            L.append(f'  if (src2_off) {{')
            L.append(f'    const_acc = amdgpu::ACC_FROM_VGPR;')
            L.append(f'    s2 = vb + *src2_off;')
            L.append(f'  }} else {{')
            read_acc = 'read_scalar64' if is_gfx1251_f64_wmma else 'read_scalar'
            L.append(f'    const_acc = amdgpu::RegisterAccess(wf).{read_acc}({s2});')
            L.append(f'  }}')
    else:
        # ACC_CD selects VGPRs or AccVGPRs for the C and D matrices. Encodings
        # without the field always use AccVGPRs.
        if 'acc_cd' in ctx.enc_field_names:
            L.append(
                f'  uint32_t dst = {_matrix_base(supports_gpr_idx, f"amdgpu::dst_base(vb, {d}.encoding_value_, inst_.acc_cd)", "Dst")};'
            )
        elif uses_plain_vgpr_dst:
            L.append(f'  uint32_t dst = vb + {d}.encoding_value_;')
        else:
            L.append(
                f'  uint32_t dst = {_matrix_base(supports_gpr_idx, f"amdgpu::dst_base(vb, {d}.encoding_value_, 1)", "Dst")};'
            )
        L.append(
            f'  uint32_t src0_base = {_matrix_base(supports_gpr_idx, f"amdgpu::src_base(vb, {s0}.encoding_value_)", "Src0")};'
        )
        L.append(
            f'  uint32_t src1_base = {_matrix_base(supports_gpr_idx, f"amdgpu::src_base(vb, {s1}.encoding_value_)", "Src1")};'
        )
        src0_base_expr = 'src0_base'
        src1_base_expr = 'src1_base'
        if uses_supported_swmmac_layout:
            if uses_fixed_wave_swmmac_layout:
                L.append(f'  uint32_t const_acc = amdgpu::ACC_FROM_VGPR;')
                L.append(f'  uint32_t s2 = dst;')
            L.append(
                f'  uint32_t index_base = amdgpu::src_base(vb, {s2}.encoding_value_);'
            )
            L.append(f'  uint32_t index_key = {swmmac_index_key_expr};')
            index_base_expr = 'index_base'
            index_key_expr = 'index_key'
        else:
            L.append(f'  uint32_t const_acc;')
            L.append(f'  uint32_t s2 = {arch_name}::resolve_acc(vb, dst,')
            L.append(
                f'      {s2}.encoding_value_, const_acc,'
                f' [&] {{ return amdgpu::RegisterAccess(wf).read_scalar({s2}); }});'
            )
            if supports_gpr_idx:
                L.append(f'  if ({s2}.decoded_vgpr())')
                L.append(
                    '    s2 = amdgpu::apply_gpr_idx_to_mma_base('
                    'wf, vb, s2, amdgpu::VgprMsbRole::Src2);'
                )

    if arch_name == 'rdna3_5' and is_dense_wmma:
        L.extend(
            [
                '  const amdgpu::RegisterAccess registers(wf);',
                f'  src0_base = registers.gfx115_wmma_source_base({s0}, src0_base);',
                f'  src1_base = registers.gfx115_wmma_source_base({s1}, src1_base);',
                f'  s2 = registers.gfx115_wmma_source_base({s2}, s2);',
            ]
        )

    # FP64 BLGP is a negate mask, unlike the lane permutation used for FP32.
    if (
        arch in ('cdna3', 'cdna4')
        and result_type == input_type == 'F64'
        and (M, N, K, B) in {(16, 16, 4, 1), (4, 4, 4, 4)}
    ):
        api = 'rj_arith_' + name.lower() + '_w64'
        outputs = M * N * B // 32
        L.append(
            '  const bool compatibility = (cu.arithmetic_flags() & RJ_ARITH_SEMANTICS_MASK)'
        )
        L.append('      == RJ_ARITH_SEMANTICS_COMPATIBILITY;')
        L.append('  // Preserve legacy FP32 widening only in Compatibility.')
        L.append('  std::optional<uint64_t> inline_c;')
        L.append(f'  if (!{s2}.decoded_vgpr()) {{')
        L.append('    inline_c = compatibility')
        L.append(
            '        ? std::bit_cast<uint64_t>(double(std::bit_cast<float>(const_acc)))'
        )
        L.append(f'        : amdgpu::RegisterAccess(wf).read_scalar64({s2});')
        L.append('  }')
        L.append(
            f'  amdgpu::arith::exec_dense_fp_matrix<2, {outputs}, 64, {outputs}, false, true>('
        )
        L.append(f'      wf, {api}, cu.arithmetic_flags(), inst_.blgp,')
        L.append(f'      dst, {src0_base_expr}, {src1_base_expr}, s2,')
        L.append('      inline_c);')
        return '\n'.join(L)

    # Dense integer matrix formats share the observed whole-wave register bridge.
    is_dense_i8_mfma = (
        arch in ('cdna3', 'cdna4')
        and result_type == 'I32'
        and input_type == 'I8'
        and M == N
        and (M, K, B)
        in {
            (32, 4, 2),
            (16, 4, 4),
            (4, 4, 16),
            (32, 16, 1),
            (16, 32, 1),
            (16, 64, 1),
            (32, 32, 1),
        }
    )
    is_dense_iu8_wmma = uses_fixed_wave32_split_k_dense_layout and (
        M,
        N,
        K,
        result_type,
        input_type,
    ) == (16, 16, 64, 'I32', 'IU8')
    if is_dense_i8_mfma or is_dense_iu8_wmma:
        lanes = 32 if is_dense_iu8_wmma else 64
        api = 'rj_arith_' + name.lower() + f'_w{lanes}'
        if is_dense_iu8_wmma:
            L.append(
                '  uint64_t controls = (inst_.neg & 3) | ((inst_.opsel & 1u) ? RJ_ARITH_SWMMAC_INDEX_KEY_1 : 0);'
            )
            L.append('  if (inst_.clamp) controls |= RJ_ARITH_WMMA_CLAMP;')
            inputs, outputs = 8, 8
        else:
            L.append(
                '  uint64_t controls = inst_.cbsz | (uint64_t(inst_.abid) << 3) | (uint64_t(inst_.blgp) << 7);'
            )
            inputs, outputs = M * K * B // 256, M * M * B // 64
        if is_dense_i8_mfma:
            L.append(
                '  const uint64_t a_lanes = inst_.cbsz == 0 ? ~uint64_t{0} : amdgpu::mfma_permuted_lane_mask([&](uint32_t lane) {'
            )
            L.append(
                '    return amdgpu::permute_a_lane(lane, inst_.cbsz, inst_.abid); });'
            )
            L.append(
                '  const uint64_t b_lanes = inst_.blgp == 0 ? ~uint64_t{0} : amdgpu::mfma_permuted_lane_mask([&](uint32_t lane) {'
            )
            L.append('    return amdgpu::permute_b_lane(lane, inst_.blgp); });')
        L.append(
            f'  amdgpu::arith::exec_dense_fp_matrix<{inputs}, {outputs}, {lanes}>('
        )
        flags = 'cu.arithmetic_flags()'
        # MI300 §7.1 Table 28 and CDNA4 §7.1.2 specify I32 saturation. The model
        # follows those control tables over the statements in MI300 §7.3 and
        # CDNA4 §7.4 that integer operations ignore MODE.
        if is_dense_i8_mfma:
            flags += ' | (wf.fp16_ovfl() ? RJ_ARITH_FP16_OVFL : 0)'
        L.append(f'      wf, {api}, {flags}, controls,')
        L.append(f'      dst, {src0_base_expr}, {src1_base_expr}, s2,')
        # Inline -1 is valid C even though its bits equal the legacy sentinel.
        inline_acc = '!src2_off' if is_dense_iu8_wmma else f'!{s2}.decoded_vgpr()'
        L.append(
            f'      {inline_acc} ? std::optional<uint32_t>(const_acc) : std::nullopt'
            + (', a_lanes, b_lanes);' if is_dense_i8_mfma else ');')
        )
        return '\n'.join(L)

    # Carry the existing MFMA register/accumulator resolution into the library bridge.
    # CDNA1/2 and other instruction families retain their original executors.
    if (
        arch in ('cdna3', 'cdna4')
        and name.startswith('V_MFMA_F32_')
        and M == N
        and (
            input_type in ('F16', 'BF16')
            and (M, K, B)
            in {
                (32, 4, 2),
                (16, 4, 4),
                (4, 4, 16),
                (32, 8, 1),
                (16, 16, 1),
                (16, 32, 1),
                (32, 16, 1),
            }
            or input_type == 'F32'
            and (M, K, B)
            in {(32, 1, 2), (16, 1, 4), (4, 1, 16), (32, 2, 1), (16, 4, 1)}
            or input_type == 'XF32'
            and (M, K, B) in {(16, 8, 1), (32, 4, 1)}
            or input_type in _F8_FIXED
            and (M, K, B) in {(16, 32, 1), (32, 16, 1)}
        )
    ):
        api = 'rj_arith_' + name.lower() + '_w64'
        if arch == 'cdna4' and (
            input_type in _F8_FIXED
            or input_type == 'XF32'
            or input_type in ('F16', 'BF16')
            and (M, K, B) not in {(16, 32, 1), (32, 16, 1)}
        ):
            api += '_gfx950'
        L.append(
            '  uint64_t controls = inst_.cbsz | (uint64_t(inst_.abid) << RJ_ARITH_MFMA_ABID_SHIFT)'
        )
        L.append('      | (uint64_t(inst_.blgp) << RJ_ARITH_MFMA_BLGP_SHIFT);')
        L.append(
            '  const uint64_t a_lanes = inst_.cbsz == 0 ? ~uint64_t{0} : amdgpu::mfma_permuted_lane_mask([&](uint32_t lane) {'
        )
        L.append('    return amdgpu::permute_a_lane(lane, inst_.cbsz, inst_.abid); });')
        L.append(
            '  const uint64_t b_lanes = inst_.blgp == 0 ? ~uint64_t{0} : amdgpu::mfma_permuted_lane_mask([&](uint32_t lane) {'
        )
        L.append('    return amdgpu::permute_b_lane(lane, inst_.blgp); });')
        L.append(
            f'  amdgpu::arith::exec_dense_fp_matrix<{M * K * B * in_bits // 2048}, {M * N * B // 64}, 64>('
        )
        flags = 'cu.arithmetic_flags()'
        # CDNA4 §7.1.2 specifies finite F32 overflow saturation. MI300's table
        # also lists it, but gfx942 retains its existing floating-point behavior
        # pending a physical mode-toggle measurement on that target.
        # gfx942 also forwards the bit to enforce exact-model qualification;
        # its approximate paths continue to ignore it.
        if (arch == 'cdna3' and input_type != 'F32') or (
            arch == 'cdna4' and input_type in ('F32', 'BF16')
        ):
            flags += ' | (wf.fp16_ovfl() ? RJ_ARITH_FP16_OVFL : 0)'
        L.append(f'      wf, {api}, {flags}, controls,')
        L.append(f'      dst, {src0_base_expr}, {src1_base_expr}, s2,')
        L.append(
            f'      !{s2}.decoded_vgpr() ? std::optional<uint32_t>(const_acc) : std::nullopt, a_lanes, b_lanes);'
        )
        return '\n'.join(L)

    if (
        arch == 'cdna4'
        and input_type in ('F8_F6_F4', 'F8F6F4')
        and (M, N, K) in ((16, 16, 128), (32, 32, 64))
    ):
        L.append(
            '  auto registers_for = [](uint32_t format) -> uint32_t { return format < 2 ? 8 : format < 4 ? 6 : 4; };'
        )
        L.append(
            '  if (inst_.cbsz > 4 || inst_.blgp > 4) { wf.report_instruction_execution_error(amdgpu::InstructionExecutionError::UnsupportedOperandValue); return; }'
        )
        L.append(
            '  uint64_t controls = inst_.cbsz | (uint64_t(inst_.blgp) << RJ_ARITH_MATRIX_FORMAT_B_SHIFT);'
        )
        L.append(f'  amdgpu::arith::exec_dense_fp_matrix<8, {M*M//64}, 64>(')
        L.append(
            f'      wf, rj_arith_{name.lower()}_w64, cu.arithmetic_flags(), controls, dst, {src0_base_expr}, {src1_base_expr}, s2,'
        )
        L.append(
            f'      !{s2}.decoded_vgpr() ? std::optional<uint32_t>(const_acc) : std::nullopt,'
        )
        L.append(
            '      ~uint64_t{0}, ~uint64_t{0}, registers_for(inst_.cbsz), registers_for(inst_.blgp));'
        )
        return '\n'.join(L)

    if (
        arch == 'cdna5'
        and uses_fixed_wave_swmmac_layout
        and (M, N) == (16, 16)
        and (
            K == 64
            and (result_type, input_type)
            in {
                ('F32', 'F16'),
                ('F32', 'BF16'),
                ('F16', 'F16'),
                ('BF16', 'BF16'),
                ('BF16F32', 'BF16'),
            }
            or K == 128
            and input_type in _F8_FIXED
            and result_type in ('F32', 'F16')
        )
    ):
        # These forms have tied register accumulators and a fixed metadata set.
        L = [
            line
            for line in L
            if line
            not in (
                '  uint32_t const_acc = amdgpu::ACC_FROM_VGPR;',
                '  uint32_t s2 = dst;',
                '  uint32_t index_key = 0u;',
            )
        ]
        if K == 64:
            # CDNA5 specifies A/B NEG as MBZ; do not silently accept disputed encodings.
            L.extend(
                [
                    '  if ((inst_.neg | inst_.neg_hi) & 3u) {',
                    '    wf.report_instruction_execution_error(amdgpu::InstructionExecutionError::UnsupportedOperandValue);',
                    '    return;',
                    '  }',
                ]
            )
        outputs = 8 if result_type == 'F32' else 4
        accumulators = 8 if result_type in ('F32', 'BF16F32') else 4
        indices = 2 if K == 128 else 1
        api = 'rj_arith_' + name.lower() + '_w32'
        flags = 'cu.arithmetic_flags() | (wf.fp16_ovfl() ? RJ_ARITH_FP16_OVFL : 0)'
        L.append(
            f'  amdgpu::arith::exec_sparse_matrix<8, 16, {outputs}, {indices}, 32, {accumulators}>('
        )
        L.append(
            f'      wf, {api}, {flags}, (inst_.opsel & 1u) ? RJ_ARITH_SWMMAC_INDEX_KEY_1 : 0, dst, {src0_base_expr}, {src1_base_expr}, {index_base_expr});'
        )
        return '\n'.join(L)

    is_split_half = (M, N, K) == (16, 16, 32) and (result_type, input_type) in {
        ('F32', 'F16'),
        ('F32', 'BF16'),
        ('F16', 'F16'),
        ('BF16', 'BF16'),
        ('BF16F32', 'BF16'),
    }
    is_split_f8 = (
        (M, N) == (16, 16)
        and K in (64, 128)
        and input_type in _F8_FIXED
        and result_type in ('F32', 'F16')
    )
    is_split_f32 = (M, N, K, result_type, input_type) == (16, 16, 4, 'F32', 'F32')
    if uses_fixed_wave32_split_k_dense_layout and (
        is_split_half or is_split_f8 or is_split_f32
    ):
        api = 'rj_arith_' + name.lower() + '_w32'
        L.append('  uint64_t controls = (inst_.neg & 4) ? RJ_ARITH_WMMA_NEG_C : 0;')
        L.append('  if (inst_.neg_hi & 4) controls |= RJ_ARITH_WMMA_ABS_C;')
        factors = 2 if is_split_f32 else K // 8 if is_split_f8 else 8
        if result_type in ('F16', 'BF16'):
            bf16 = 'true' if result_type == 'BF16' else 'false'
            L.append('  if (!src2_off &&')
            L.append(
                '      (cu.arithmetic_flags() & RJ_ARITH_SEMANTICS_MASK) != RJ_ARITH_SEMANTICS_COMPATIBILITY) {'
            )
            L.append(
                f'    const_acc = amdgpu::dot_packed16::inline_word<{bf16}>(const_acc, inst_.src2);'
            )
            L.append('  }')
            bridge_shape = f'{factors}, 4, 32, 4, true'
        elif result_type == 'BF16F32':
            bridge_shape = f'{factors}, 4, 32, 8'
        else:
            bridge_shape = f'{factors}, 8, 32'
        L.append('  uint64_t flags = cu.arithmetic_flags();')
        if result_type in ('F16', 'BF16', 'BF16F32'):
            L.append('  if (wf.fp16_ovfl()) flags |= RJ_ARITH_FP16_OVFL;')
        L.append(f'  amdgpu::arith::exec_dense_fp_matrix<{bridge_shape}>(')
        L.append(f'      wf, {api}, flags, controls,')
        L.append(f'      dst, {src0_base_expr}, {src1_base_expr}, s2,')
        L.append(
            '      !src2_off ? std::optional<uint32_t>(const_acc) : std::nullopt);'
        )
        return '\n'.join(L)

    if is_gfx1251_f64_wmma:
        L.append(f'  amdgpu::exec_wmma_f64_16x16x4_f64(cu, dst,')
        L.append(
            f'      src0_base, src1_base, s2, const_acc, inst_.neg, inst_.neg_hi, wf.exec());'
        )
    elif result_type == 'F64':
        L.append(f'  amdgpu::exec_f64(cu, {M}, {N}, {K}, {B}, dst,')
        L.append(f'                 {src0_base_expr},')
        L.append(f'                 {src1_base_expr},')
        neg = 'inst_.blgp' if arch in ('cdna3', 'cdna4') else '0u'
        L.append(f'                 s2, const_acc, {neg});')
    elif result_type == 'I32':
        if uses_rdna_wmma_arithmetic:
            gfx12 = str(uses_gfx12_wmma_layout).lower()
            shape_arg = f', {K}' if K != 16 else ''
            L.append(
                f'  amdgpu::exec_rdna_wmma_i32<{gfx12}, {in_bits}{shape_arg}>(cu, wf.wf_size(), dst,'
                f' {src0_base_expr}, {src1_base_expr}, s2,'
                f' {s2}.encoding_value_ < 256 ? std::optional<uint32_t>(const_acc) : std::nullopt,'
                f' /*a_signed=*/(inst_.neg & 0x1u) != 0,'
                f' /*b_signed=*/(inst_.neg & 0x2u) != 0, inst_.clamp);'
            )
            return '\n'.join(L)

        if uses_arith_sparse_integer:
            api = 'rj_arith_' + name.lower() + '_w32'
            L.append(
                '  uint64_t controls = (inst_.neg & 3) | ((inst_.opsel & 1u) ? RJ_ARITH_SWMMAC_INDEX_KEY_1 : 0);'
            )
            L.append('  if (inst_.clamp) controls |= RJ_ARITH_WMMA_CLAMP;')
            L.append('  amdgpu::arith::exec_sparse_matrix<8, 16, 8, 2, 32>(')
            L.append(
                f'      wf, {api}, cu.arithmetic_flags(), controls, dst, {src0_base_expr}, {src1_base_expr}, {index_base_expr});'
            )
            return '\n'.join(L)

        if input_type != 'I8':
            raise ValueError(
                f'{name}: missing rj-arith integer matrix binding for {arch_name}'
            )
        L.append(f'  amdgpu::exec_i32_i8(cu, {M}, {N}, {K}, {B}, dst,')
        L.append(f'                     {src0_base_expr},')
        L.append(f'                     {src1_base_expr},')
        L.append(f'                     s2, const_acc,')
        L.append(f'                     inst_.cbsz, inst_.abid, inst_.blgp);')
    else:
        if uses_fixed_wave32_split_k_dense_layout and input_type in (
            'F8_F6_F4',
            'F8F6F4',
            'F4',
        ):
            if input_type == 'F4':
                L.append('  uint32_t matrix_a_fmt = 4, matrix_b_fmt = 4;')
                L.append('  uint64_t controls = 0;')
            else:
                L.append('  uint32_t matrix_a_fmt = inst_.opsel;')
                L.append(
                    f'  uint32_t matrix_b_fmt = ({op_sel_hi_2_expr} << 2) | inst_.opsel_hi;'
                )
                L.append(
                    '  if (matrix_a_fmt > 4 || matrix_b_fmt > 4) { wf.report_instruction_execution_error(amdgpu::InstructionExecutionError::UnsupportedOperandValue); return; }'
                )
                L.append(
                    '  uint64_t controls = matrix_a_fmt | (uint64_t(matrix_b_fmt) << 3);'
                )
            L.append('  if (inst_.neg & 4) controls |= RJ_ARITH_WMMA_MIXED_NEG_C;')
            L.append('  if (inst_.neg_hi & 4) controls |= RJ_ARITH_WMMA_MIXED_ABS_C;')
            L.append(
                '  auto registers_for = [](uint32_t fmt) { return fmt < 2 ? 16u : fmt < 4 ? 12u : 8u; };'
            )
            api = (
                f'rj_arith_v_wmma_f32_{M}x{N}x{K}_'
                + ('f4' if input_type == 'F4' else 'f8f6f4')
                + '_w32'
            )
            L.append(f'  amdgpu::arith::exec_dense_fp_matrix<16, {M*N//32}, 32>(')
            L.append(
                f'      wf, {api}, cu.arithmetic_flags(), controls, dst, {src0_base_expr}, {src1_base_expr}, s2,'
            )
            L.append(
                '      !src2_off ? std::optional<uint32_t>(const_acc) : std::nullopt,'
            )
            L.append(
                f'      0xffffffffULL, 0xffffffffULL, registers_for(matrix_a_fmt) * {M//16}, registers_for(matrix_b_fmt));'
            )
            return '\n'.join(L)

        if (
            uses_gfx11_wmma_layout
            and result_type in ('F32', 'F16', 'BF16')
            and input_type in ('F16', 'BF16')
            and (M, N, K) == (16, 16, 16)
        ):
            bf16 = str(input_type == 'BF16').lower()
            packed_arg = ', true' if result_type != 'F32' else ''
            if packed_arg:
                L.append(f'  if ({s2}.encoding_value_ < 256)')
                L.append(
                    f'    const_acc = amdgpu::dot_packed16::inline_word<{bf16}>'
                    f'(const_acc, {s2}.encoding_value_);'
                )
            L.append(
                f'  amdgpu::exec_gfx11_wmma_dot2<{bf16}{packed_arg}>(cu, wf.wf_size(), dst,'
                f' {src0_base_expr}, {src1_base_expr}, s2,'
                f' {s2}.encoding_value_ < 256 ? std::optional<uint32_t>(const_acc) : std::nullopt,'
                f" inst_.neg, inst_.neg_hi{', (inst_.op_sel >> 2) & 1u, wf.fp16_ovfl()' if packed_arg else ''});"
            )
            return '\n'.join(L)
        ea = fp8_helper_name(arch, _EXTRACT_A.get(input_type, 'amdgpu::extract_f32'))
        eb = fp8_helper_name(arch, _EXTRACT_B.get(input_type, 'amdgpu::extract_f32'))
        # CDNA1-4 VOP3P_MFMA encoding has cbsz/abid/blgp fields for A-matrix
        # broadcast and B-matrix lane permutation; RDNA WMMA does not, so
        # pass 0 (the spec templates require the args explicitly).
        has_blgp = arch in ('cdna1', 'cdna2', 'cdna3', 'cdna4')
        cbsz = 'inst_.cbsz' if has_blgp else '0u'
        abid = 'inst_.abid' if has_blgp else '0u'
        blgp = 'inst_.blgp' if has_blgp else '0u'
        s0b = src0_base_expr
        s1b = src1_base_expr
        denorm_arg = ''
        if arch == 'cdna2' and input_type in _MFMA_F32_SPEC:
            mode = 'wf.fp_denorm_mode_f32()' if input_type == 'F32' else '0u'
            denorm_arg = f', amdgpu::MfmaF32Denorm{{{mode}}}'
        if N % 16 == 0 and input_type in _MFMA_F32_SPEC and has_blgp:
            fam = _MFMA_F32_SPEC[input_type]
            targ = _mma_targ(M, N, K, B, batch_optional=(fam != 'f32'))
            L.append(
                f'  amdgpu::exec_f32_mfma_{fam}_spec<{targ}>(cu, dst, {s0b}, {s1b}, s2,'
                f' const_acc, {cbsz}, {abid}, {blgp}{denorm_arg});'
            )
        elif has_blgp:
            L.append(
                f'  amdgpu::exec_f32<{M}, {N}, {K}, {B}>(cu, {in_bits}, dst, {s0b}, {s1b},'
                f' s2, {ea}, {eb}, const_acc, inst_.cbsz, inst_.abid, inst_.blgp{denorm_arg});'
            )
        else:
            L.append(
                f'  amdgpu::exec_f32<{M}, {N}, {K}, {B}>(cu, {in_bits}, dst, {s0b}, {s1b},'
                f' s2, {ea}, {eb}, const_acc);'
            )

    return '\n'.join(L)
