# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Matrix instruction execute body generators.

Free functions that emit C++ execute_impl bodies for matrix
instructions: MFMA (matrix fused multiply-add), AccVGPR read/write.
"""

from __future__ import annotations

from amdisa.gpuisa import Instruction

# Input families that have specialized (compile-time M/N/K) MFMA kernels in the
# hand-maintained mma_exec.h. CDNA MFMA and RDNA WMMA both flow through the
# GFX9 MFMA-layout helpers (exec_f32/exec_i32_i8), so the same spec templates
# apply to both; gfx1250 WMMA uses its own wave32 spec helpers (handled below).
# The spec templates fall back to the generic runtime path for unsupported
# shapes / when stdx::simd is unavailable, so emitting them is always safe.
_MFMA_F32_SPEC = {'F32': 'f32', 'XF32': 'f32', 'F16': 'f16', 'BF16': 'bf16'}
_F8_FIXED = frozenset({'FP8_FP8', 'FP8_BF8', 'BF8_FP8', 'BF8_BF8'})


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


def _gfx1250_wmma_spec(
    result_type: str, input_type: str, M: int, N: int, K: int
) -> str | None:
    """Specialized gfx1250 (wave32) dense-WMMA kernel name for a given shape,
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
        f'    {dst[0]}.write_lane(wf, lane, {src[0]}.read_lane(wf, lane));\n'
        f'  }}'
    )


def gen_accvgpr_write(dst: list[str], src: list[str]) -> str:
    """Generate V_ACCVGPR_WRITE: copy VGPR → ACCVGPR."""
    return (
        f'  uint64_t exec = wf.exec();\n'
        f'  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {{\n'
        f'    if (!(exec & (1ULL << lane))) continue;\n'
        f'    {dst[0]}.write_lane(wf, lane, {src[0]}.read_lane(wf, lane));\n'
        f'  }}'
    )


def gen_mfma(
    inst: Instruction, dst: list[str], src: list[str], arch_name: str = ""
) -> str:
    """Generate MFMA / SMFMAC matrix multiply-accumulate.

    Uses the mma_exec.h helpers which implement the exact GFX9 register
    mapping formulas. The helpers handle cross-lane data movement, WAR
    hazard avoidance (buffered writes), and inline constant accumulator
    initialization without clobbering overlapping source operands.
    """
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

    result_type = m.group(1)  # F32, I32, F64
    if result_type == 'BF16F32':
        result_type = 'F32'
    M, N, K = int(m.group(2)), int(m.group(3)), int(m.group(4))
    input_type = m.group(5)  # F32, XF32, F16, BF16, I8, F64, etc.
    is_swmmac = name.startswith('V_SWMMAC_')

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
            'FP8': 'amdgpu::smfmac_read_fp8',
            'BF8': 'amdgpu::smfmac_read_bf8',
        }
        L = []
        L.append(f'  auto &cu = wf.cu();')
        L.append(f'  uint32_t vb = wf.vgpr_alloc().base;')
        has_acc_cd = arch_name in ('cdna2', 'cdna3', 'cdna4')
        if has_acc_cd:
            L.append(
                f'  uint32_t dst = amdgpu::dst_base(vb, {d}.encoding_value_, inst_.acc_cd);'
            )
        else:
            L.append(f'  uint32_t dst = amdgpu::dst_base(vb, {d}.encoding_value_, 1);')
        L.append(f'  uint32_t s0b = amdgpu::src_base(vb, {s0}.encoding_value_);')
        L.append(f'  uint32_t s1b = amdgpu::src_base(vb, {s1}.encoding_value_);')
        L.append(f'  uint32_t idx = amdgpu::src_base(vb, {s2}.encoding_value_);')

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
    if arch == 'gfx1250':
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
        if is_swmmac:
            swmmac_index_entries = 32 if K >= 128 and in_bits <= 8 else 16
            L.append(f'  uint32_t const_acc = amdgpu::ACC_FROM_VGPR;')
            L.append(f'  uint32_t s2 = dst;')
            L.append(
                f'  auto index_off = Isa::resolved_vgpr_offset(wf, {s2}.opr_type_, '
                f'{s2}.encoding_value_, {s2}.vgpr_msb_role());'
            )
            L.append(f'  if (!index_off)')
            L.append(f'    throw util::UnimplementedInst(mnemonic());')
            L.append(f'  uint32_t index_base = vb + *index_off;')
            L.append(f'  uint32_t index_key = inst_.opsel & 0x1u;')
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
            L.append(f'    const_acc = {s2}.read_scalar(wf);')
            L.append(f'  }}')
    else:
        # acc_cd field exists in CDNA2/3/4 VOP3P_MFMA encoding (controls
        # AccVGPR bank selection). CDNA1 and RDNA lack this field — default
        # to 1 (always use AccVGPR bank, the CDNA1 behavior).
        has_acc_cd = arch in ('cdna2', 'cdna3', 'cdna4')
        if has_acc_cd:
            L.append(
                f'  uint32_t dst = amdgpu::dst_base(vb, {d}.encoding_value_, inst_.acc_cd);'
            )
        else:
            L.append(f'  uint32_t dst = amdgpu::dst_base(vb, {d}.encoding_value_, 1);')
        L.append(f'  uint32_t const_acc;')
        L.append(f'  uint32_t s2 = amdgpu::resolve_acc(vb, dst,')
        L.append(
            f'      {s2}.encoding_value_, const_acc,'
            f' [&] {{ return {s2}.read_scalar(wf); }});'
        )

    if result_type == 'F64':
        L.append(f'  amdgpu::exec_f64(cu, {M}, {N}, {K}, {B}, dst,')
        if arch == 'gfx1250':
            L.append(f'                 src0_base,')
            L.append(f'                 src1_base,')
        else:
            L.append(f'                 amdgpu::src_base(vb, {s0}.encoding_value_),')
            L.append(f'                 amdgpu::src_base(vb, {s1}.encoding_value_),')
        L.append(f'                 s2, const_acc);')
    elif result_type == 'I32':
        if arch == 'gfx1250':
            # LLVM's gfx1250 IU WMMA convention overloads neg_lo: bit set means
            # signed extension, bit clear means unsigned.
            if is_swmmac:
                if input_type in ('IU4', 'IU8'):
                    suffix = '4' if input_type == 'IU4' else '8'
                    L.append(
                        f'  auto extract_a = (inst_.neg & 0x1u) ? amdgpu::extract_i{suffix}'
                        f' : amdgpu::extract_u{suffix};'
                    )
                    L.append(
                        f'  auto extract_b = (inst_.neg & 0x2u) ? amdgpu::extract_i{suffix}'
                        f' : amdgpu::extract_u{suffix};'
                    )
                else:
                    L.append(f'  auto extract_a = amdgpu::extract_i8;')
                    L.append(f'  auto extract_b = amdgpu::extract_i8;')
                L.append(
                    f'  amdgpu::exec_swmmac_i32(cu, {M}, {N}, {K}, {in_bits}, dst, src0_base,'
                    f' src1_base, s2, index_base, {swmmac_index_entries}, index_key,'
                    f' extract_a, extract_b, inst_.clamp, const_acc);'
                )
                return '\n'.join(L)
            # The iu8 16x16x64 WMMA has a specialized kernel taking the
            # per-operand signedness directly.
            if input_type == 'IU8' and (M, N, K) == (16, 16, 64):
                L.append(
                    f'  amdgpu::exec_wmma_i32_16x16x64_iu8(cu, dst, src0_base, src1_base, s2,'
                    f' /*a_signed=*/(inst_.neg & 0x1u) != 0,'
                    f' /*b_signed=*/(inst_.neg & 0x2u) != 0, inst_.clamp, const_acc);'
                )
                return '\n'.join(L)
            if input_type in ('IU4', 'IU8'):
                suffix = '4' if input_type == 'IU4' else '8'
                L.append(
                    f'  auto extract_a = (inst_.neg & 0x1u) ? amdgpu::extract_i{suffix}'
                    f' : amdgpu::extract_u{suffix};'
                )
                L.append(
                    f'  auto extract_b = (inst_.neg & 0x2u) ? amdgpu::extract_i{suffix}'
                    f' : amdgpu::extract_u{suffix};'
                )
            else:
                L.append(f'  auto extract_a = amdgpu::extract_i8;')
                L.append(f'  auto extract_b = amdgpu::extract_i8;')
            L.append(
                f'  amdgpu::exec_wmma_i32(cu, {M}, {N}, {K}, {in_bits}, dst, src0_base,'
                f' src1_base, s2, extract_a, extract_b, inst_.clamp, const_acc);'
            )
        else:
            has_blgp = arch in ('cdna1', 'cdna2', 'cdna3', 'cdna4')
            if not has_blgp:
                suffix = '4' if input_type == 'IU4' else '8'
                L.append(
                    f'  auto extract_a = (inst_.neg & 0x1u) ? amdgpu::extract_i{suffix}'
                    f' : amdgpu::extract_u{suffix};'
                )
                L.append(
                    f'  auto extract_b = (inst_.neg & 0x2u) ? amdgpu::extract_i{suffix}'
                    f' : amdgpu::extract_u{suffix};'
                )
                L.append(
                    f'  amdgpu::exec_i32_mixed(cu, {M}, {N}, {K}, {B}, {in_bits}, dst,'
                )
                L.append(
                    f'                        amdgpu::src_base(vb, {s0}.encoding_value_),'
                )
                L.append(
                    f'                        amdgpu::src_base(vb, {s1}.encoding_value_),'
                )
                L.append(
                    f'                        s2, extract_a, extract_b, const_acc, inst_.clamp);'
                )
            elif input_type == 'IU4':
                L.append(
                    f'  amdgpu::exec_i32_mixed(cu, {M}, {N}, {K}, {B}, {in_bits}, dst,'
                )
                L.append(
                    f'                        amdgpu::src_base(vb, {s0}.encoding_value_),'
                )
                L.append(
                    f'                        amdgpu::src_base(vb, {s1}.encoding_value_),'
                )
                L.append(
                    f'                        s2, amdgpu::extract_u4, amdgpu::extract_u4, const_acc);'
                )
            else:
                L.append(f'  amdgpu::exec_i32_i8(cu, {M}, {N}, {K}, {B}, dst,')
                L.append(
                    f'                     amdgpu::src_base(vb, {s0}.encoding_value_),'
                )
                L.append(
                    f'                     amdgpu::src_base(vb, {s1}.encoding_value_),'
                )
                L.append(f'                     s2, const_acc,')
                L.append(f'                     inst_.cbsz, inst_.abid, inst_.blgp);')
    else:
        # F32, F16, and BF16 matrix results accumulate in f32. gfx1250 WMMA
        # uses a wave32 layout; CDNA MFMA uses the GFX9 MFMA layout helpers.
        if arch == 'gfx1250' and input_type in ('F8_F6_F4', 'F8F6F4'):
            L.append(f'  uint32_t matrix_a_fmt = inst_.opsel;')
            L.append(f'  uint32_t matrix_b_fmt = (inst_.pad_14 << 2) | inst_.opsel_hi;')
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
            L.append(f'            src0_base,')
            L.append(f'            src1_base,')
            L.append(f'            s2, extract_a, extract_b, const_acc);')
            L.append(f'      }});')
            L.append(f'  if (!dispatched)')
            L.append(f'    throw util::UnimplementedInst(mnemonic());')
            return '\n'.join(L)

        ea = _EXTRACT_A.get(input_type, 'amdgpu::extract_f32')
        eb = _EXTRACT_B.get(input_type, 'amdgpu::extract_f32')
        # CDNA1-4 VOP3P_MFMA encoding has cbsz/abid/blgp fields for
        # A-matrix broadcast and B-matrix lane permutation. RDNA does
        # not have MFMA (only WMMA), so these fields don't exist.
        if arch == 'gfx1250':
            if is_swmmac:
                if result_type == 'F16':
                    exec_fn = 'exec_swmmac_f16'
                elif result_type == 'BF16':
                    exec_fn = 'exec_swmmac_bf16'
                else:
                    exec_fn = 'exec_swmmac_f32'
                L.append(
                    f'  amdgpu::{exec_fn}(cu, {M}, {N}, {K}, {in_bits}, dst, src0_base,'
                    f' src1_base, s2, index_base, {swmmac_index_entries}, index_key,'
                    f' {ea}, {eb}, const_acc);'
                )
            else:
                # Dense WMMA: a specialized wave32 kernel where one exists, else
                # the generic exec_wmma_* runtime path.
                spec = _gfx1250_wmma_spec(result_type, input_type, M, N, K)
                if spec is not None:
                    L.append(
                        f'  amdgpu::{spec}(cu, dst, src0_base, src1_base, s2, const_acc);'
                    )
                else:
                    if result_type == 'F16':
                        exec_fn = 'exec_wmma_f16'
                    elif result_type == 'BF16':
                        exec_fn = 'exec_wmma_bf16'
                    else:
                        exec_fn = 'exec_wmma_f32'
                    L.append(
                        f'  amdgpu::{exec_fn}(cu, {M}, {N}, {K}, {in_bits}, dst, src0_base,'
                        f' src1_base, s2, {ea}, {eb}, const_acc);'
                    )
        elif input_type in ('F8_F6_F4', 'F8F6F4'):
            # f8f6f4 MFMA: cbsz/blgp encode data format, NOT lane
            # permutation. Use dispatch_matrix_fmt_pair to select the
            # correct extract functions and bit widths.
            L.append(f'  uint32_t s0b = amdgpu::src_base(vb, {s0}.encoding_value_);')
            L.append(f'  uint32_t s1b = amdgpu::src_base(vb, {s1}.encoding_value_);')
            L.append('  bool dispatched;')
            L.append('  if (!(inst_.abid & 1u)) {')
            L.append(
                '    dispatched = amdgpu::dispatch_matrix_fmt_pair(inst_.cbsz, inst_.blgp,'
            )
            L.append(
                '        [&](uint32_t a_bits, uint32_t b_bits, auto ea, auto eb) {'
            )
            L.append(
                f'          amdgpu::exec_f32_mixed(cu, {M}, {N}, {K}, {B}, a_bits, b_bits,'
            )
            L.append(
                f'                                 dst, s0b, s1b, s2, ea, eb, const_acc);'
            )
            L.append('        });')
            L.append('  } else {')
            L.append(
                '    uint32_t sa_base = amdgpu::src_base(vb, raw_words_[1] & 0x1FFu);'
            )
            L.append(
                '    uint32_t sb_base = amdgpu::src_base(vb, (raw_words_[1] >> 9) & 0x1FFu);'
            )
            L.append('    dispatched = amdgpu::dispatch_matrix_fmt_pair(')
            L.append(
                '        inst_.cbsz, inst_.blgp, [&](uint32_t a_bits, uint32_t b_bits, auto ea, auto eb) {'
            )
            L.append(
                f'          amdgpu::exec_f32_scaled_mixed(cu, {M}, {N}, {K}, {B}, a_bits, b_bits, dst, s0b, s1b, s2, ea,'
            )
            L.append(
                f'                                        eb, const_acc, sa_base, sb_base);'
            )
            L.append('        });')
            L.append('  }')
            L.append('  if (!dispatched)')
            L.append('    throw util::UnimplementedInst(mnemonic());')
        elif result_type in ('F16', 'BF16') and arch in (
            'rdna3',
            'rdna3_5',
            'rdna4',
        ):
            exec_fn = 'exec_f16_gfx9' if result_type == 'F16' else 'exec_bf16_gfx9'
            L.append(f'  amdgpu::{exec_fn}(cu, {M}, {N}, {K}, {B}, {in_bits}, dst,')
            L.append(f'                 amdgpu::src_base(vb, {s0}.encoding_value_),')
            L.append(f'                 amdgpu::src_base(vb, {s1}.encoding_value_),')
            L.append(f'                 s2, {ea}, {eb}, const_acc);')
        else:
            # CDNA1-4 VOP3P_MFMA encoding has cbsz/abid/blgp fields for A-matrix
            # broadcast and B-matrix lane permutation; RDNA WMMA does not, so
            # pass 0 (the spec templates require the args explicitly).
            has_blgp = arch in ('cdna1', 'cdna2', 'cdna3', 'cdna4')
            cbsz = 'inst_.cbsz' if has_blgp else '0u'
            abid = 'inst_.abid' if has_blgp else '0u'
            blgp = 'inst_.blgp' if has_blgp else '0u'
            s0b = f'amdgpu::src_base(vb, {s0}.encoding_value_)'
            s1b = f'amdgpu::src_base(vb, {s1}.encoding_value_)'
            if N % 16 == 0 and input_type in _F8_FIXED and has_blgp:
                a_fp8, b_fp8 = _f8_bools(input_type)
                L.append(
                    f'  amdgpu::exec_f32_mfma_f8_spec<{M}, {N}, {K}, {a_fp8}, {b_fp8}>('
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
