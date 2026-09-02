# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Matrix instruction execute body generators.

Free functions that emit C++ execute_impl bodies for matrix
instructions: MFMA (matrix fused multiply-add), AccVGPR read/write.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from amdisa.codegen.execute.fp8_formats import FNUZ_FP8_ARCHES, fp8_helper_name
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


def _f8_bools(input_type: str) -> tuple[str, str]:
    """(A_FP8, B_FP8) C++ bool literals for a fixed f8 input family."""
    parts = input_type.split('_')
    return (
        'true' if parts[0] == 'FP8' else 'false',
        'true' if parts[1] == 'FP8' else 'false',
    )


def _fixed_wave32_wmma_spec(
    result_type: str, input_type: str, M: int, N: int, K: int
) -> str | None:
    """Specialized fixed Wave32 dense-WMMA kernel name for a given shape,
    or None to use the generic runtime path. Returns the full callee including
    template args; the call site supplies (cu, dst, s0, s1, s2, const_acc)."""
    if input_type in _F8_FIXED:
        a_fp8, b_fp8 = _f8_bools(input_type)
        if result_type == 'F32':
            return f'exec_wmma_f32_f8_spec<{M}, {N}, {K}, {a_fp8}, {b_fp8}>'
        if result_type == 'F16':
            return f'exec_wmma_f16_f8_spec<{M}, {N}, {K}, {a_fp8}, {b_fp8}>'
        return None
    if result_type == 'F16':
        if input_type == 'F16' and (M, N, K) == (16, 16, 32):
            return f'exec_wmma_f16_spec<{M}, {N}, {K}>'
        return None
    if result_type == 'BF16':
        if input_type == 'BF16' and (M, N, K) == (16, 16, 32):
            return f'exec_wmma_bf16_spec<{M}, {N}, {K}>'
        return None
    if result_type == 'BF16F32':
        if input_type == 'BF16' and (M, N, K) == (16, 16, 32):
            return 'exec_wmma_bf16f32_16x16x32_bf16'
        return None
    # result F32
    if input_type in ('F32', 'XF32') and N % 16 == 0:
        return f'exec_wmma_f32_f32_spec<{M}, {N}, {K}>'
    if input_type == 'F16' and (M, N, K) == (16, 16, 32):
        return 'exec_wmma_f32_16x16x32_f16'
    if input_type == 'BF16' and (M, N, K) == (16, 16, 32):
        return 'exec_wmma_f32_16x16x32_bf16'
    return None


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

    Uses the mma_exec.h helpers which implement the exact GFX9 register
    mapping formulas. The helpers handle cross-lane data movement, WAR
    hazard avoidance (buffered writes), and inline constant accumulator
    initialization without clobbering overlapping source operands.
    """
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

    if not m:
        return (
            f'  // MFMA stub: {name}\n'
            f'  (void)wf;\n'
            f'  throw util::UnimplementedInst(mnemonic());'
        )

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
    # F32-result variants use cross-lane exec_smfmac_* helpers from
    # mma_exec.h; I32-result variants fall through to the per-lane stub.
    if 'SMFMAC' in name and result_type == 'F32':
        _SMFMAC_READ = {
            'F16': 'amdgpu::smfmac_read_f16',
            'BF16': 'amdgpu::smfmac_read_bf16',
        }
        _SMFMAC_FP8_READ = {
            'FP8': fp8_helper_name(arch_name, 'amdgpu::smfmac_read_fp8'),
            'BF8': fp8_helper_name(arch_name, 'amdgpu::smfmac_read_bf8'),
        }
        L = []
        L.append(f'  auto &cu = wf.cu();')
        L.append(f'  uint32_t vb = wf.vgpr_alloc().base;')
        if ctx.profile.uses_vgpr_msb_indexing:
            L.append(
                f'  uint32_t dst = vb + *Isa::resolved_vgpr_offset(wf, {d}.opr_type_, '
                f'{d}.encoding_value_, {d}.vgpr_msb_role());'
            )
            L.append(
                f'  uint32_t s0b = vb + *Isa::resolved_vgpr_offset(wf, {s0}.opr_type_, '
                f'{s0}.encoding_value_, {s0}.vgpr_msb_role());'
            )
            L.append(
                f'  uint32_t s1b = vb + *Isa::resolved_vgpr_offset(wf, {s1}.opr_type_, '
                f'{s1}.encoding_value_, {s1}.vgpr_msb_role());'
            )
            L.append(
                f'  auto index_off = Isa::resolved_vgpr_offset(wf, {s2}.opr_type_, '
                f'{s2}.encoding_value_, {s2}.vgpr_msb_role());'
            )
            L.append(f'  if (!index_off)')
            L.append(f'    throw util::UnimplementedInst(mnemonic());')
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

        if input_type in ('F16', 'BF16'):
            read_fn = _SMFMAC_READ[input_type]
            L.append(
                f'  amdgpu::exec_smfmac_f32_{M}x{N}x{K}_f16(cu, dst, s0b, s1b, idx, {read_fn});'
            )
        else:
            parts = input_type.split('_')
            read_a = _SMFMAC_FP8_READ.get(parts[0], 'amdgpu::smfmac_read_fp8')
            read_b = _SMFMAC_FP8_READ.get(parts[1], 'amdgpu::smfmac_read_fp8')
            L.append(
                f'  amdgpu::exec_smfmac_f32_{M}x{N}x{K}_fp8(cu, dst, s0b, s1b, idx, {read_a},\n'
                f'                                       {read_b});'
            )
        return '\n'.join(L)

    if 'SMFMAC' in name:
        L = []
        L.append(f'  // SMFMAC stub: {name} (I32 result, no cross-lane helper)')
        L.append(f'  (void)wf;')
        L.append(f'  throw util::UnimplementedInst(mnemonic());')
        return '\n'.join(L)

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
    uses_runtime_wave_swmmac_layout = swmmac_layout is SwmmacLayout.RUNTIME_WAVE
    # Fixed-wave OPSEL bits are matrix-reuse hints, not sparse index selectors.
    swmmac_index_key_expr = (
        '0u' if uses_fixed_wave_swmmac_layout else 'inst_.opsel & 0x1u'
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
    swmmac_index_entries = 32 if is_swmmac and K >= 128 and in_bits <= 8 else 16
    if ctx.profile.uses_vgpr_msb_indexing:
        L.append(
            f'  uint32_t dst = vb + *Isa::resolved_vgpr_offset(wf, {d}.opr_type_, '
            f'{d}.encoding_value_, {d}.vgpr_msb_role());'
        )
        L.append(
            f'  uint32_t src0_base = vb + *Isa::resolved_vgpr_offset(wf, {s0}.opr_type_, '
            f'{s0}.encoding_value_, {s0}.vgpr_msb_role());'
        )
        L.append(
            f'  uint32_t src1_base = vb + *Isa::resolved_vgpr_offset(wf, {s1}.opr_type_, '
            f'{s1}.encoding_value_, {s1}.vgpr_msb_role());'
        )
        src0_base_expr = 'src0_base'
        src1_base_expr = 'src1_base'
        if uses_supported_swmmac_layout:
            L.append(f'  uint32_t const_acc = amdgpu::ACC_FROM_VGPR;')
            L.append(f'  uint32_t s2 = dst;')
            L.append(
                f'  auto index_off = Isa::resolved_vgpr_offset(wf, {s2}.opr_type_, '
                f'{s2}.encoding_value_, {s2}.vgpr_msb_role());'
            )
            L.append(f'  if (!index_off)')
            L.append(f'    throw util::UnimplementedInst(mnemonic());')
            L.append(f'  uint32_t index_base = vb + *index_off;')
            L.append(f'  uint32_t index_key = {swmmac_index_key_expr};')
            index_base_expr = 'index_base'
            index_key_expr = 'index_key'
        else:
            L.append(f'  uint32_t const_acc;')
            L.append(
                f'  auto src2_off = Isa::resolved_vgpr_offset(wf, {s2}.opr_type_, '
                f'{s2}.encoding_value_, {s2}.vgpr_msb_role());'
            )
            L.append(f'  uint32_t s2 = dst;')
            L.append(f'  if (src2_off) {{')
            L.append(f'    const_acc = amdgpu::ACC_FROM_VGPR;')
            L.append(f'    s2 = vb + *src2_off;')
            L.append(f'  }} else {{')
            L.append(f'    const_acc = amdgpu::RegisterAccess(wf).read_scalar({s2});')
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
            L.append(f'  uint32_t s2 = amdgpu::resolve_acc(vb, dst,')
            L.append(
                f'      {s2}.encoding_value_, const_acc,'
                f' [&] {{ return amdgpu::RegisterAccess(wf).read_scalar({s2}); }});'
            )
            if supports_gpr_idx:
                L.append('  if (const_acc == amdgpu::ACC_FROM_VGPR)')
                L.append(
                    '    s2 = amdgpu::apply_gpr_idx_to_mma_base('
                    'wf, vb, s2, amdgpu::VgprMsbRole::Src2);'
                )

    if (
        arch == 'cdna5'
        and is_dense_wmma
        and result_type == 'F64'
        and input_type == 'F64'
        and (M, N, K) == (16, 16, 4)
    ):
        L.append(f'  amdgpu::exec_wmma_f64_16x16x4_f64(cu, dst,')
        L.append(f'      src0_base, src1_base, s2, const_acc, inst_.neg, inst_.neg_hi,')
        L.append(
            f'      wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64(), wf.exec());'
        )
    elif result_type == 'F64':
        L.append(f'  amdgpu::exec_f64(cu, {M}, {N}, {K}, {B}, dst,')
        L.append(f'                 {src0_base_expr},')
        L.append(f'                 {src1_base_expr},')
        neg = 'inst_.blgp' if arch in ('cdna3', 'cdna4') else '0u'
        L.append(f'                 s2, const_acc, {neg});')
    elif result_type == 'I32':

        def append_signed_extractors(suffix: str) -> None:
            L.append(
                f'  auto extract_a = [&](auto &cu, uint32_t base, const amdgpu::InputLoc &loc) {{'
            )
            L.append(
                f'    return (inst_.neg & 0x1u) ? amdgpu::extract_i{suffix}(cu, base, loc)'
                f' : amdgpu::extract_u{suffix}(cu, base, loc);'
            )
            L.append(f'  }};')
            L.append(
                f'  auto extract_b = [&](auto &cu, uint32_t base, const amdgpu::InputLoc &loc) {{'
            )
            L.append(
                f'    return (inst_.neg & 0x2u) ? amdgpu::extract_i{suffix}(cu, base, loc)'
                f' : amdgpu::extract_u{suffix}(cu, base, loc);'
            )
            L.append(f'  }};')

        if uses_fixed_wave_swmmac_layout:
            if input_type in ('IU4', 'IU8'):
                suffix = '4' if input_type == 'IU4' else '8'
                append_signed_extractors(suffix)
            else:
                L.append(f'  auto extract_a = amdgpu::extract_i8;')
                L.append(f'  auto extract_b = amdgpu::extract_i8;')
            L.append(
                f'  amdgpu::exec_swmmac_i32(cu, {M}, {N}, {K}, {in_bits}, dst,'
                f' {src0_base_expr}, {src1_base_expr}, s2, {index_base_expr},'
                f' {swmmac_index_entries}, {index_key_expr},'
                f' extract_a, extract_b, inst_.clamp, const_acc);'
            )
            return '\n'.join(L)
        if uses_fixed_wave32_split_k_dense_layout:
            # Fixed Wave32 split-K IU WMMA overloads neg_lo: bit set means
            # signed extension, bit clear means unsigned.
            # The iu8 16x16x64 WMMA has a specialized kernel taking the
            # per-operand signedness directly.
            if input_type == 'IU8' and (M, N, K) == (16, 16, 64):
                L.append(
                    f'  amdgpu::exec_wmma_i32_16x16x64_iu8(cu, dst,'
                    f' {src0_base_expr}, {src1_base_expr}, s2,'
                    f' /*a_signed=*/(inst_.neg & 0x1u) != 0,'
                    f' /*b_signed=*/(inst_.neg & 0x2u) != 0, inst_.clamp, const_acc);'
                )
                return '\n'.join(L)
            if input_type in ('IU4', 'IU8'):
                suffix = '4' if input_type == 'IU4' else '8'
                append_signed_extractors(suffix)
            else:
                L.append(f'  auto extract_a = amdgpu::extract_i8;')
                L.append(f'  auto extract_b = amdgpu::extract_i8;')
            L.append(
                f'  amdgpu::exec_wmma_i32(cu, {M}, {N}, {K}, {in_bits}, dst,'
                f' {src0_base_expr}, {src1_base_expr}, s2, extract_a, extract_b,'
                f' inst_.clamp, const_acc);'
            )
        elif uses_runtime_wave_swmmac_layout:
            if input_type in ('IU4', 'IU8'):
                suffix = '4' if input_type == 'IU4' else '8'
                append_signed_extractors(suffix)
            else:
                L.append(f'  auto extract_a = amdgpu::extract_i8;')
                L.append(f'  auto extract_b = amdgpu::extract_i8;')
            L.append(
                f'  amdgpu::exec_swmmac_i32(cu, {M}, {N}, {K}, {in_bits}, dst,'
                f' {src0_base_expr}, {src1_base_expr}, s2, {index_base_expr},'
                f' {swmmac_index_entries}, {index_key_expr}, extract_a, extract_b,'
                f' inst_.clamp, const_acc, wf.wf_size());'
            )
        elif uses_gfx11_wmma_layout:
            if input_type in ('IU4', 'IU8'):
                suffix = '4' if input_type == 'IU4' else '8'
                append_signed_extractors(suffix)
            else:
                L.append(f'  auto extract_a = amdgpu::extract_i8;')
                L.append(f'  auto extract_b = amdgpu::extract_i8;')
            L.append(
                f'  amdgpu::exec_gfx11_wmma_i32(cu, wf.wf_size(), {M}, {N}, {K}, {in_bits}, dst,'
                f' {src0_base_expr}, {src1_base_expr}, s2, extract_a, extract_b,'
                f' inst_.clamp, const_acc);'
            )
        elif uses_gfx12_wmma_layout:
            if input_type in ('IU4', 'IU8'):
                suffix = '4' if input_type == 'IU4' else '8'
                append_signed_extractors(suffix)
            else:
                L.append(f'  auto extract_a = amdgpu::extract_i8;')
                L.append(f'  auto extract_b = amdgpu::extract_i8;')
            L.append(
                f'  amdgpu::exec_wmma_i32(cu, {M}, {N}, {K}, {in_bits}, dst,'
                f' {src0_base_expr}, {src1_base_expr}, s2, extract_a, extract_b,'
                f' inst_.clamp, const_acc, wf.wf_size());'
            )
        else:
            has_blgp = arch in ('cdna1', 'cdna2', 'cdna3', 'cdna4')
            if not has_blgp:
                suffix = '4' if input_type == 'IU4' else '8'
                append_signed_extractors(suffix)
                L.append(
                    f'  amdgpu::exec_i32_mixed(cu, {M}, {N}, {K}, {B}, {in_bits}, dst,'
                )
                L.append(f'                        {src0_base_expr},')
                L.append(f'                        {src1_base_expr},')
                L.append(
                    f'                        s2, extract_a, extract_b, const_acc, inst_.clamp);'
                )
            elif input_type == 'IU4':
                L.append(
                    f'  amdgpu::exec_i32_mixed(cu, {M}, {N}, {K}, {B}, {in_bits}, dst,'
                )
                L.append(f'                        {src0_base_expr},')
                L.append(f'                        {src1_base_expr},')
                L.append(
                    f'                        s2, amdgpu::extract_u4, amdgpu::extract_u4, const_acc);'
                )
            else:
                L.append(f'  amdgpu::exec_i32_i8(cu, {M}, {N}, {K}, {B}, dst,')
                L.append(f'                     {src0_base_expr},')
                L.append(f'                     {src1_base_expr},')
                L.append(f'                     s2, const_acc,')
                L.append(f'                     inst_.cbsz, inst_.abid, inst_.blgp);')
    else:
        # F32, F16, and BF16 matrix results accumulate in f32. Fixed Wave32 WMMA
        # uses a Wave32 layout; CDNA MFMA uses the GFX9 MFMA layout helpers.
        if uses_fixed_wave32_split_k_dense_layout and input_type in (
            'F8_F6_F4',
            'F8F6F4',
        ):
            L.append(f'  uint32_t matrix_a_fmt = inst_.opsel;')
            L.append(
                f'  uint32_t matrix_b_fmt = ({op_sel_hi_2_expr} << 2) | '
                f'inst_.opsel_hi;'
            )
            L.append(
                f'  bool dispatched = amdgpu::dispatch_matrix_fmt_pair('
                f'matrix_a_fmt, matrix_b_fmt,'
            )
            L.append(
                f'      [&](uint32_t a_bits, uint32_t b_bits, auto extract_a, auto extract_b) {{'
            )
            L.append(
                f'        amdgpu::exec_wmma_f32_mixed(cu, {M}, {N}, {K}, a_bits, b_bits, dst,'
            )
            L.append(f'            {src0_base_expr},')
            L.append(f'            {src1_base_expr},')
            L.append(
                f'            s2, extract_a, extract_b, const_acc,'
                f' amdgpu::wmma_c_modifier(inst_.neg, inst_.neg_hi));'
            )
            L.append(f'      }});')
            L.append(f'  if (!dispatched)')
            L.append(f'    throw util::UnimplementedInst(mnemonic());')
            return '\n'.join(L)

        ea = fp8_helper_name(arch, _EXTRACT_A.get(input_type, 'amdgpu::extract_f32'))
        eb = fp8_helper_name(arch, _EXTRACT_B.get(input_type, 'amdgpu::extract_f32'))
        # CDNA1-4 VOP3P_MFMA encoding has cbsz/abid/blgp fields for
        # A-matrix broadcast and B-matrix lane permutation. RDNA does
        # not have MFMA (only WMMA), so these fields don't exist.
        if uses_fixed_wave_swmmac_layout:
            if result_type == 'F16':
                exec_fn = 'exec_swmmac_f16'
            elif result_type == 'BF16':
                exec_fn = 'exec_swmmac_bf16'
            else:
                exec_fn = 'exec_swmmac_f32'
            L.append(
                f'  amdgpu::{exec_fn}(cu, {M}, {N}, {K}, {in_bits}, dst,'
                f' {src0_base_expr}, {src1_base_expr}, s2, {index_base_expr},'
                f' {swmmac_index_entries}, {index_key_expr},'
                f' {ea}, {eb}, const_acc);'
            )
        elif uses_fixed_wave32_split_k_dense_layout:
            # Dense WMMA: a specialized Wave32 kernel where one exists, else
            # the generic exec_wmma_* runtime path.
            spec = _fixed_wave32_wmma_spec(result_type, input_type, M, N, K)
            if spec is not None:
                if result_type in ('F32', 'BF16F32'):
                    L.append(
                        f'  amdgpu::{spec}(cu, dst, {src0_base_expr}, {src1_base_expr},'
                        f' s2, const_acc,'
                        f' amdgpu::wmma_c_modifier(inst_.neg, inst_.neg_hi));'
                    )
                elif spec.startswith('exec_wmma_f16_f8_spec'):
                    L.append(
                        f'  amdgpu::{spec}(cu, dst, {src0_base_expr}, {src1_base_expr},'
                        f' s2, const_acc,'
                        f' wf.fp16_ovfl());'
                    )
                else:
                    L.append(
                        f'  amdgpu::{spec}(cu, dst, {src0_base_expr}, {src1_base_expr},'
                        f' s2, const_acc);'
                    )
            else:
                if result_type == 'F16':
                    exec_fn = 'exec_wmma_f16'
                elif result_type == 'BF16':
                    exec_fn = 'exec_wmma_bf16'
                else:
                    exec_fn = 'exec_wmma_f32'
                if exec_fn == 'exec_wmma_f32':
                    L.append(
                        f'  amdgpu::{exec_fn}(cu, {M}, {N}, {K}, {in_bits}, dst,'
                        f' {src0_base_expr}, {src1_base_expr}, s2, {ea}, {eb}, const_acc,'
                        f' amdgpu::wmma_c_modifier(inst_.neg, inst_.neg_hi));'
                    )
                else:
                    L.append(
                        f'  amdgpu::{exec_fn}(cu, {M}, {N}, {K}, {in_bits}, dst,'
                        f' {src0_base_expr}, {src1_base_expr}, s2, {ea}, {eb}, const_acc);'
                    )
        elif uses_runtime_wave_swmmac_layout:
            if result_type == 'F16':
                exec_fn = 'exec_swmmac_f16'
            elif result_type == 'BF16':
                exec_fn = 'exec_swmmac_bf16'
            else:
                exec_fn = 'exec_swmmac_f32'
            L.append(
                f'  amdgpu::{exec_fn}(cu, {M}, {N}, {K}, {in_bits}, dst, {src0_base_expr},'
                f' {src1_base_expr}, s2, {index_base_expr}, {swmmac_index_entries},'
                f' {index_key_expr},'
                f' {ea}, {eb}, const_acc, wf.wf_size());'
            )
        elif (
            uses_gfx11_wmma_layout
            and result_type == 'F32'
            and input_type not in ('F8_F6_F4', 'F8F6F4')
        ):
            L.append(
                f'  amdgpu::exec_gfx11_wmma_f32(cu, wf.wf_size(), {M}, {N}, {K}, {in_bits}, dst,'
                f' {src0_base_expr}, {src1_base_expr}, s2, {ea}, {eb}, const_acc,'
                f' amdgpu::wmma_c_modifier(inst_.neg, inst_.neg_hi));'
            )
        elif uses_gfx12_wmma_layout and input_type not in ('F8_F6_F4', 'F8F6F4'):
            if result_type == 'F16':
                L.append(
                    f'  amdgpu::exec_wmma_f16(cu, {M}, {N}, {K}, {in_bits}, dst,'
                    f' {src0_base_expr}, {src1_base_expr}, s2, {ea}, {eb}, const_acc,'
                    f' wf.wf_size());'
                )
            elif result_type == 'BF16':
                L.append(
                    f'  amdgpu::exec_wmma_bf16(cu, {M}, {N}, {K}, {in_bits}, dst,'
                    f' {src0_base_expr}, {src1_base_expr}, s2, {ea}, {eb}, const_acc,'
                    f' wf.wf_size());'
                )
            else:
                L.append(
                    f'  amdgpu::exec_wmma_f32(cu, {M}, {N}, {K}, {in_bits}, dst,'
                    f' {src0_base_expr}, {src1_base_expr}, s2, {ea}, {eb}, const_acc,'
                    f' amdgpu::wmma_c_modifier(inst_.neg, inst_.neg_hi), wf.wf_size());'
                )
        elif input_type in ('F8_F6_F4', 'F8F6F4'):
            # f8f6f4 MFMA: cbsz/blgp encode data format, NOT lane
            # permutation. Use dispatch_matrix_fmt_pair to select the
            # correct extract functions and bit widths. Scaled MFMA is a
            # distinct VOP3PX2 compound instruction generated by
            # _generator.py; a dense suffix never consumes prefix fields.
            L.append(f'  uint32_t s0b = {src0_base_expr};')
            L.append(f'  uint32_t s1b = {src1_base_expr};')
            L.append(
                '  bool dispatched = amdgpu::dispatch_matrix_fmt_pair(inst_.cbsz, inst_.blgp,'
            )
            L.append('      [&](uint32_t a_bits, uint32_t b_bits, auto ea, auto eb) {')
            L.append(
                f'        amdgpu::exec_f32_mixed(cu, {M}, {N}, {K}, {B}, a_bits, b_bits,'
            )
            L.append(
                f'                               dst, s0b, s1b, s2, ea, eb, const_acc);'
            )
            L.append('      });')
            L.append('  if (!dispatched)')
            L.append('    throw util::UnimplementedInst(mnemonic());')
        elif uses_gfx11_wmma_layout and result_type in ('F16', 'BF16'):
            exec_fn = (
                'exec_gfx11_wmma_f16'
                if result_type == 'F16'
                else 'exec_gfx11_wmma_bf16'
            )
            L.append(
                f'  amdgpu::{exec_fn}(cu, wf.wf_size(), {M}, {N}, {K}, {in_bits}, dst,'
            )
            L.append(f'      {src0_base_expr},')
            L.append(f'      {src1_base_expr}, s2,' ' (inst_.op_sel >> 2) & 0x1u,')
            L.append(f'      {ea}, {eb}, const_acc);')
        elif uses_gfx12_wmma_layout and result_type in ('F16', 'BF16'):
            exec_fn = 'exec_wmma_f16' if result_type == 'F16' else 'exec_wmma_bf16'
            L.append(f'  amdgpu::{exec_fn}(cu, {M}, {N}, {K}, {in_bits}, dst,')
            L.append(f'      {src0_base_expr},')
            L.append(
                f'      {src1_base_expr}, s2, {ea}, {eb}, const_acc,' f' wf.wf_size());'
            )
        else:
            # CDNA1-4 VOP3P_MFMA encoding has cbsz/abid/blgp fields for A-matrix
            # broadcast and B-matrix lane permutation; RDNA WMMA does not, so
            # pass 0 (the spec templates require the args explicitly).
            has_blgp = arch in ('cdna1', 'cdna2', 'cdna3', 'cdna4')
            cbsz = 'inst_.cbsz' if has_blgp else '0u'
            abid = 'inst_.abid' if has_blgp else '0u'
            blgp = 'inst_.blgp' if has_blgp else '0u'
            s0b = src0_base_expr
            s1b = src1_base_expr
            if N % 16 == 0 and input_type in _F8_FIXED and has_blgp:
                a_fp8, b_fp8 = _f8_bools(input_type)
                fnuz = ', true' if arch in FNUZ_FP8_ARCHES else ''
                L.append(
                    f'  amdgpu::exec_f32_mfma_f8_spec<{M}, {N}, {K}, {a_fp8}, {b_fp8}{fnuz}>('
                    f'cu, dst, {s0b}, {s1b}, s2, const_acc, {cbsz}, {abid}, {blgp});'
                )
            elif N % 16 == 0 and input_type in _MFMA_F32_SPEC and has_blgp:
                fam = _MFMA_F32_SPEC[input_type]
                targ = _mma_targ(M, N, K, B, batch_optional=(fam != 'f32'))
                L.append(
                    f'  amdgpu::exec_f32_mfma_{fam}_spec<{targ}>(cu, dst, {s0b}, {s1b}, s2,'
                    f' const_acc, {cbsz}, {abid}, {blgp});'
                )
            elif has_blgp:
                L.append(
                    f'  amdgpu::exec_f32(cu, {M}, {N}, {K}, {B}, {in_bits}, dst, {s0b}, {s1b},'
                    f' s2, {ea}, {eb}, const_acc, inst_.cbsz, inst_.abid, inst_.blgp);'
                )
            else:
                L.append(
                    f'  amdgpu::exec_f32(cu, {M}, {N}, {K}, {B}, {in_bits}, dst, {s0b}, {s1b},'
                    f' s2, {ea}, {eb}, const_acc);'
                )

    return '\n'.join(L)
