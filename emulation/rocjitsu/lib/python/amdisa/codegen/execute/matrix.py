# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Matrix instruction execute body generators.

Free functions that emit C++ execute_impl bodies for matrix
instructions: MFMA (matrix fused multiply-add), AccVGPR read/write.
"""

from __future__ import annotations

from amdisa.gpuisa import Instruction


def gen_accvgpr_read(dst: list[str], src: list[str]) -> str:
    """Generate V_ACCVGPR_READ: copy ACCVGPR → VGPR."""
    # In our model, ACCVGPRs are just VGPRs in the accumulator range.
    # The operand resolution already handles the mapping.
    return f'  uint64_t exec = wf.exec();\n' \
           f'  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {{\n' \
           f'    if (!(exec & (1ULL << lane))) continue;\n' \
           f'    {dst[0]}.write_lane(wf, lane, {src[0]}.read_lane(wf, lane));\n' \
           f'  }}'

def gen_accvgpr_write(dst: list[str], src: list[str]) -> str:
    """Generate V_ACCVGPR_WRITE: copy VGPR → ACCVGPR."""
    return f'  uint64_t exec = wf.exec();\n' \
           f'  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {{\n' \
           f'    if (!(exec & (1ULL << lane))) continue;\n' \
           f'    {dst[0]}.write_lane(wf, lane, {src[0]}.read_lane(wf, lane));\n' \
           f'  }}'

def gen_mfma(inst: Instruction, dst: list[str], src: list[str], arch_name: str = "") -> str:
    """Generate MFMA / SMFMAC matrix multiply-accumulate.

    Uses the mfma_exec.h helpers which implement the exact GFX9 register
    mapping formulas. The helpers handle cross-lane data movement, WAR
    hazard avoidance (buffered writes), and inline constant accumulator
    initialization without clobbering overlapping source operands.
    """
    name = inst.name
    d, s0, s1, s2 = dst[0], src[0], src[1], src[2]

    import re
    m = re.match(
        r'V_(?:S?MFMA[C]?|S?WMMA[C]?)_(F32|I32|F64|F16|BF16|BF8|FP8)_(\d+)X(\d+)X(\d+)'
        r'(?:_\d+B)?_?(F32|XF32|F16|BF16|I8|IU8|IU4|F64|FP8|BF8'
        r'|BF8_BF8|BF8_FP8|FP8_BF8|FP8_FP8'
        r'|F16_FP8|F16_BF8|BF16_FP8|BF16_BF8'
        r'|F8_F6_F4|F8F6F4)?'
        r'(?:_1K)?$',
        name)

    if not m:
        return (f'  // MFMA stub: {name}\n'
                f'  (void)wf;\n'
                f'  throw util::UnimplementedInst(mnemonic());')

    result_type = m.group(1)  # F32, I32, F64
    M, N, K = int(m.group(2)), int(m.group(3)), int(m.group(4))
    input_type = m.group(5)   # F32, XF32, F16, BF16, I8, F64, etc.

    dst_bits = inst.operands[0].size if inst.operands else 0
    dst_regs = max(1, dst_bits // 32)

    # SMFMAC: per-lane dot-product functional model (no cross-lane mapping).
    if 'SMFMAC' in name:
        L = []
        L.append(f'  // MFMA: {name} \u2014 {M}x{N}x{K} {input_type}\u2192{result_type}')
        L.append(f'  // D({dst_regs} regs/lane) += A * B, functional model')
        L.append(f'  uint64_t exec = wf.exec();')
        L.append(f'  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {{')
        L.append(f'    if (!(exec & (1ULL << lane)))')
        L.append(f'      continue;')
        L.append(f'    uint32_t a_raw = {s0}.read_lane(wf, lane);')
        L.append(f'    uint32_t b_raw = {s1}.read_lane(wf, lane);')

        if input_type in ('F16', 'BF16'):
            conv = 'util::f16_to_f32' if input_type == 'F16' else 'util::bf16_to_f32'
            L.append(f'    float a0 = {conv}(static_cast<uint16_t>(a_raw));')
            L.append(f'    float a1 = {conv}(static_cast<uint16_t>(a_raw >> 16));')
            L.append(f'    float b0 = {conv}(static_cast<uint16_t>(b_raw));')
            L.append(f'    float b1 = {conv}(static_cast<uint16_t>(b_raw >> 16));')
            L.append(f'    float dot = a0 * b0 + a1 * b1;')
            L.append(f'    float acc = std::bit_cast<float>({s2}.read_lane(wf, lane));')
            L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));')
        elif input_type == 'I8':
            L.append(f'    int32_t dot = 0;')
            L.append(f'    for (int k = 0; k < 8; ++k) {{')
            L.append(f'      int8_t ae = static_cast<int8_t>((a_raw >> (k * 8)) & 0xFF);')
            L.append(f'      int8_t be = static_cast<int8_t>((b_raw >> (k * 8)) & 0xFF);')
            L.append(f'      dot += static_cast<int32_t>(ae) * be;')
            L.append(f'    }}')
            L.append(f'    int32_t acc0 = static_cast<int32_t>({s2}.read_lane(wf, lane));')
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(acc0 + dot));')
            L.append(f'    // Additional result registers would require cross-lane data')
        else:
            # FP8/BF8 variants: input_type is e.g. "BF8_BF8", "BF8_FP8", etc.
            _FP8_CONV = {'BF8': 'util::bf8_e5m2_to_f32', 'FP8': 'util::fp8_e4m3_to_f32'}
            parts = input_type.split('_')
            conv_a = _FP8_CONV.get(parts[0], 'util::fp8_e4m3_to_f32')
            conv_b = _FP8_CONV.get(parts[1], 'util::fp8_e4m3_to_f32')
            L.append(f'    float dot = 0.0f;')
            L.append(f'    for (int k = 0; k < 4; ++k) {{')
            L.append(f'      float ae = {conv_a}(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));')
            L.append(f'      float be = {conv_b}(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));')
            L.append(f'      dot += ae * be;')
            L.append(f'    }}')
            L.append(f'    float acc = std::bit_cast<float>({s2}.read_lane(wf, lane));')
            L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));')

        L.append(f'  }}')
        return '\n'.join(L)

    # Compute number of blocks from output register count and matrix dims.
    if result_type == 'F64':
        B = 64 * (dst_regs // 2) // (M * N)
    else:
        B = 64 * dst_regs // (M * N)

    # Determine input element size in bits and extract functions.
    _INPUT_BITS = {
        'F32': 32, 'XF32': 32, 'F16': 16, 'BF16': 16,
        'I8': 8, 'IU8': 8, 'IU4': 4, 'F64': 64,
        'FP8': 8, 'BF8': 8,
        'FP8_FP8': 8, 'FP8_BF8': 8, 'BF8_FP8': 8, 'BF8_BF8': 8,
        'F16_FP8': 8, 'F16_BF8': 8, 'BF16_FP8': 8, 'BF16_BF8': 8,
        'F8_F6_F4': 8, 'F8F6F4': 8,
    }
    in_bits = _INPUT_BITS.get(input_type, 32)

    # Map input types to amdgpu::extract_* function names.
    _EXTRACT_A = {
        'F32': 'amdgpu::extract_f32', 'XF32': 'amdgpu::extract_f32',
        'F16': 'amdgpu::extract_f16', 'BF16': 'amdgpu::extract_bf16',
        'FP8_FP8': 'amdgpu::extract_fp8', 'FP8_BF8': 'amdgpu::extract_fp8',
        'BF8_FP8': 'amdgpu::extract_bf8', 'BF8_BF8': 'amdgpu::extract_bf8',
        'F8_F6_F4': 'amdgpu::extract_fp8', 'F8F6F4': 'amdgpu::extract_fp8',
    }
    _EXTRACT_B = {
        'F32': 'amdgpu::extract_f32', 'XF32': 'amdgpu::extract_f32',
        'F16': 'amdgpu::extract_f16', 'BF16': 'amdgpu::extract_bf16',
        'FP8_FP8': 'amdgpu::extract_fp8', 'FP8_BF8': 'amdgpu::extract_bf8',
        'BF8_FP8': 'amdgpu::extract_fp8', 'BF8_BF8': 'amdgpu::extract_bf8',
        'F8_F6_F4': 'amdgpu::extract_fp8', 'F8F6F4': 'amdgpu::extract_fp8',
    }

    L = []
    L.append(f'  auto &cu = wf.cu();')
    L.append(f'  uint32_t vb = wf.vgpr_alloc().base;')
    # acc_cd field exists in CDNA2/3/4 VOP3P_MFMA encoding (controls
    # AccVGPR bank selection). CDNA1 and RDNA lack this field — default
    # to 1 (always use AccVGPR bank, the CDNA1 behavior).
    arch = arch_name
    has_acc_cd = arch in ('cdna2', 'cdna3', 'cdna4')
    if has_acc_cd:
        L.append(f'  uint32_t dst = amdgpu::dst_base(vb, {d}.encoding_value_, inst_.acc_cd);')
    else:
        L.append(f'  uint32_t dst = amdgpu::dst_base(vb, {d}.encoding_value_, 1);')
    L.append(f'  uint32_t const_acc;')
    L.append(f'  uint32_t s2 = amdgpu::resolve_acc(vb, dst,')
    L.append(f'      {s2}.encoding_value_, const_acc,'
             f' [&] {{ return {s2}.read_scalar(wf); }});')

    if result_type == 'F64':
        L.append(f'  amdgpu::exec_f64(cu, {M}, {N}, {K}, {B}, dst,')
        L.append(f'                 amdgpu::src_base(vb, {s0}.encoding_value_),')
        L.append(f'                 amdgpu::src_base(vb, {s1}.encoding_value_),')
        L.append(f'                 s2, const_acc);')
    elif result_type == 'I32':
        L.append(f'  amdgpu::exec_i32_i8(cu, {M}, {N}, {K}, {B}, dst,')
        L.append(f'                     amdgpu::src_base(vb, {s0}.encoding_value_),')
        L.append(f'                     amdgpu::src_base(vb, {s1}.encoding_value_),')
        L.append(f'                     s2, const_acc);')
    else:
        # F32, F16, BF16 result types all use exec_f32 (accumulate in f32,
        # WMMA F16/BF16 results are truncated at writeback — handled by the
        # register layout, not by separate exec functions).
        ea = _EXTRACT_A.get(input_type, 'amdgpu::extract_f32')
        eb = _EXTRACT_B.get(input_type, 'amdgpu::extract_f32')
        # CDNA1-4 VOP3P_MFMA encoding has cbsz/abid/blgp fields for
        # A-matrix broadcast and B-matrix lane permutation. RDNA does
        # not have MFMA (only WMMA), so these fields don't exist.
        has_blgp = arch in ('cdna1', 'cdna2', 'cdna3', 'cdna4')
        L.append(f'  amdgpu::exec_f32(cu, {M}, {N}, {K}, {B}, {in_bits}, dst,')
        L.append(f'                 amdgpu::src_base(vb, {s0}.encoding_value_),')
        L.append(f'                 amdgpu::src_base(vb, {s1}.encoding_value_),')
        if has_blgp:
            L.append(f'                 s2, {ea}, {eb}, const_acc,')
            L.append(f'                 inst_.cbsz, inst_.abid, inst_.blgp);')
        else:
            L.append(f'                 s2, {ea}, {eb}, const_acc);')

        if input_type in ('F8_F6_F4', 'F8F6F4'):
            # Rewrite the MFMA call: if ABID[0]=1 (scaling enabled),
            # use exec_f32_scaled which applies per-32-K-block E8M0
            # exponent biases from scale VGPRs in the X2 prefix.
            # Dwords 0-1 of the VOP3PX2 encoding are at inst[-2]/[-1]
            # relative to the MFMA encoding pointer.
            # Dword 1 bits [8:0] = scale_src0, bits [17:9] = scale_src1.
            L_scaled = []
            L_scaled.append('  if (inst_.abid & 1u) {')
            L_scaled.append('    auto *raw = reinterpret_cast<const uint32_t *>(&inst_);')
            L_scaled.append('    uint32_t x2_dw1 = raw[-1];')
            L_scaled.append('    uint32_t scale_src0_enc = x2_dw1 & 0x1FFu;')
            L_scaled.append('    uint32_t scale_src1_enc = (x2_dw1 >> 9) & 0x1FFu;')
            L_scaled.append('    uint32_t sa_base = amdgpu::src_base(vb, scale_src0_enc);')
            L_scaled.append('    uint32_t sb_base = amdgpu::src_base(vb, scale_src1_enc);')
            L_scaled.append(f'    amdgpu::exec_f32_scaled(cu, {M}, {N}, {K}, {B}, {in_bits}, dst,')
            L_scaled.append(f'        amdgpu::src_base(vb, {s0}.encoding_value_),')
            L_scaled.append(f'        amdgpu::src_base(vb, {s1}.encoding_value_),')
            L_scaled.append(f'        s2, {ea}, {eb}, const_acc,')
            L_scaled.append(f'        inst_.cbsz, inst_.abid, inst_.blgp, sa_base, sb_base);')
            L_scaled.append('  }')
            # Replace the unscaled MFMA call with a conditional:
            # if ABID[0]=1 use scaled path, else the existing unscaled path.
            # Find and wrap the existing exec_f32 call in an else block.
            for i, line in enumerate(L):
                if 'amdgpu::exec_f32(' in line:
                    L.insert(i, '  if (!(inst_.abid & 1u)) {')
                    # Find the closing semicolon
                    for j in range(i + 1, len(L)):
                        if L[j].rstrip().endswith(';'):
                            L.insert(j + 1, '  } else {')
                            break
                    break
            L.extend(L_scaled)
            L.append('  }')

    return '\n'.join(L)

