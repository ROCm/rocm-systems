# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vector special operation execute body generators.

Free functions that emit C++ execute_impl bodies for specialized vector
instructions: mbcnt, mad_64_32, mad_32_16, div_fixup, div_scale, div_fmas,
dot products, bitop3, permlane variants, and packed type conversion.
"""

from __future__ import annotations

from amdisa.codegen.execute.vop3_modifiers import (
    vop3_src_mod,
    vop3_dst_mod,
    vop3_dst_mod_f64,
)


def gen_vector_mbcnt(dst: list[str], src: list[str], op: str | None) -> str:
    """Generate V_MBCNT_LO/HI_U32_B32 body."""
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(f'    uint32_t mask = {src[0]}.read_lane(wf, lane);')
    L.append(f'    uint32_t base = {src[1]}.read_lane(wf, lane);')
    if op == 'lo':
        L.append(
            '    uint32_t thread_mask = lane < 32 ? (1u << lane) - 1 : 0xFFFFFFFFu;'
        )
        L.append('    uint32_t count = std::popcount(mask & thread_mask);')
    else:  # hi
        L.append('    uint32_t shift = lane >= 32 ? lane - 32 : 0;')
        L.append('    uint32_t thread_mask = lane >= 32 ? (1u << shift) - 1 : 0;')
        L.append('    uint32_t count = std::popcount(mask & thread_mask);')
    L.append(f'    {dst[0]}.write_lane(wf, lane, base + count);')
    L.append('  }')
    return '\n'.join(L)


def gen_vector_mad_64_32(dst: list[str], src: list[str], dtype: str | None) -> str:
    """Generate V_MAD_U64_U32 / V_MAD_I64_I32 body.

    D.i64 = S0.i32 * S1.i32 + S2.i64 (signed)
    D.u64 = S0.u32 * S1.u32 + S2.u64 (unsigned)

    Sources S0 and S1 are 32-bit; the accumulator S2 and result D are
    64-bit VGPR pairs.
    """
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if dtype == 'i64':
        L.append(
            f'    int64_t s0 = static_cast<int32_t>({src[0]}.read_lane(wf, lane));'
        )
        L.append(
            f'    int64_t s1 = static_cast<int32_t>({src[1]}.read_lane(wf, lane));'
        )
        L.append(
            f'    int64_t s2 = static_cast<int64_t>({src[2]}.read_lane64(wf, lane));'
        )
        L.append('    uint64_t result = static_cast<uint64_t>(s0 * s1 + s2);')
    else:
        L.append(f'    uint64_t s0 = {src[0]}.read_lane(wf, lane);')
        L.append(f'    uint64_t s1 = {src[1]}.read_lane(wf, lane);')
        L.append(f'    uint64_t s2 = {src[2]}.read_lane64(wf, lane);')
        L.append('    uint64_t result = s0 * s1 + s2;')
    L.append(f'    {dst[0]}.write_lane64(wf, lane, result);')
    L.append('  }')
    return '\n'.join(L)


def gen_vector_mad_32_16(dst: list[str], src: list[str], dtype: str | None) -> str:
    """Generate V_MAD_U32_U16 / V_MAD_I32_I16 body.

    D.u32 = S0.u16 * S1.u16 + S2.u32 (unsigned)
    D.i32 = S0.i16 * S1.i16 + S2.i32 (signed)
    """
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if dtype == 'i32':
        L.append(
            f'    int32_t s0 = static_cast<int16_t>({src[0]}.read_lane(wf, lane) & 0xFFFF);'
        )
        L.append(
            f'    int32_t s1 = static_cast<int16_t>({src[1]}.read_lane(wf, lane) & 0xFFFF);'
        )
        L.append(
            f'    int32_t s2 = static_cast<int32_t>({src[2]}.read_lane(wf, lane));'
        )
        L.append(
            f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(s0 * s1 + s2));'
        )
    else:
        L.append(f'    uint32_t s0 = {src[0]}.read_lane(wf, lane) & 0xFFFFu;')
        L.append(f'    uint32_t s1 = {src[1]}.read_lane(wf, lane) & 0xFFFFu;')
        L.append(f'    uint32_t s2 = {src[2]}.read_lane(wf, lane);')
        L.append(f'    {dst[0]}.write_lane(wf, lane, s0 * s1 + s2);')
    L.append('  }')
    return '\n'.join(L)


def gen_vector_div_fixup(
    dst: list[str],
    src: list[str],
    dtype: str | None,
    is_vop3: bool = False,
    has_abs: bool = False,
) -> str:
    """Generate V_DIV_FIXUP body (corrects division result)."""
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if dtype == 'f64':
        L.append(
            f'    double p = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));'
        )
        L.append(
            f'    double b = std::bit_cast<double>({src[1]}.read_lane64(wf, lane));'
        )
        L.append(
            f'    double c = std::bit_cast<double>({src[2]}.read_lane64(wf, lane));'
        )
        if is_vop3:
            L.extend(vop3_src_mod('p', 0, has_abs))
            L.extend(vop3_src_mod('b', 1, has_abs))
            L.extend(vop3_src_mod('c', 2, has_abs))
        L.append('    double result;')
        L.append('    if (std::isnan(b)) result = b;')
        L.append('    else if (std::isnan(c)) result = c;')
        L.append(
            '    else if (c == 0.0 && b == 0.0) result = std::numeric_limits<double>::quiet_NaN();'
        )
        L.append(
            '    else if (std::isinf(c) && std::isinf(b)) result = std::numeric_limits<double>::quiet_NaN();'
        )
        L.append('    else if (b == 0.0) {')
        L.append(
            '      result = std::copysign(std::numeric_limits<double>::infinity(),'
        )
        L.append(
            '                             std::bit_cast<double>(std::bit_cast<uint64_t>(b) ^ std::bit_cast<uint64_t>(c)));'
        )
        L.append('    }')
        L.append(
            '    else if (c == 0.0) result = std::copysign(0.0, std::bit_cast<double>(std::bit_cast<uint64_t>(b) ^ std::bit_cast<uint64_t>(c)));'
        )
        L.append('    else if (std::isinf(c)) {')
        L.append(
            '      result = std::copysign(std::numeric_limits<double>::infinity(),'
        )
        L.append(
            '                             std::bit_cast<double>(std::bit_cast<uint64_t>(b) ^ std::bit_cast<uint64_t>(c)));'
        )
        L.append('    }')
        L.append(
            '    else if (std::isinf(b)) result = std::copysign(0.0, std::bit_cast<double>(std::bit_cast<uint64_t>(b) ^ std::bit_cast<uint64_t>(c)));'
        )
        L.append('    else result = p;')
        if is_vop3:
            L.extend(vop3_dst_mod_f64('result'))
        L.append(
            f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));'
        )
    else:
        L.append(f'    float p = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
        L.append(f'    float b = std::bit_cast<float>({src[1]}.read_lane(wf, lane));')
        L.append(f'    float c = std::bit_cast<float>({src[2]}.read_lane(wf, lane));')
        if is_vop3:
            L.extend(vop3_src_mod('p', 0, has_abs))
            L.extend(vop3_src_mod('b', 1, has_abs))
            L.extend(vop3_src_mod('c', 2, has_abs))
        L.append('    float result;')
        L.append('    if (std::isnan(b)) result = b;')
        L.append('    else if (std::isnan(c)) result = c;')
        L.append(
            '    else if (c == 0.0f && b == 0.0f) result = std::numeric_limits<float>::quiet_NaN();'
        )
        L.append(
            '    else if (std::isinf(c) && std::isinf(b)) result = std::numeric_limits<float>::quiet_NaN();'
        )
        L.append('    else if (b == 0.0f) {')
        L.append('      result = std::copysign(std::numeric_limits<float>::infinity(),')
        L.append(
            '                             std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));'
        )
        L.append('    }')
        L.append(
            '    else if (c == 0.0f) result = std::copysign(0.0f, std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));'
        )
        L.append('    else if (std::isinf(c)) {')
        L.append('      result = std::copysign(std::numeric_limits<float>::infinity(),')
        L.append(
            '                             std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));'
        )
        L.append('    }')
        L.append(
            '    else if (std::isinf(b)) result = std::copysign(0.0f, std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));'
        )
        L.append('    else result = p;')
        if is_vop3:
            L.extend(vop3_dst_mod('result'))
        L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
    L.append('  }')
    return '\n'.join(L)


def gen_vector_div_scale(
    dst: list[str],
    src: list[str],
    dtype: str | None,
    is_vop3: bool = False,
    has_abs: bool = False,
) -> str:
    """Generate V_DIV_SCALE body per ISA pseudocode (CDNA4 p.363-365).

    S1 = denominator, S2 = numerator. S0 selects which to scale
    (S0==S1 → scale denominator, S0==S2 → scale numerator).
    VCC is set when V_DIV_FMAS must apply post-scaling.
    """
    is_f64 = dtype == 'f64'
    scale_exp = 128 if is_f64 else 64
    exp_threshold = 768 if is_f64 else 96
    tiny_exp = 53 if is_f64 else 23
    fp_type = 'double' if is_f64 else 'float'
    zero = '0.0' if is_f64 else '0.0f'
    read_fn = 'read_lane64' if is_f64 else 'read_lane'
    write_fn = 'write_lane64' if is_f64 else 'write_lane'
    cast_to = 'uint64_t' if is_f64 else 'uint32_t'

    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  uint64_t vcc = wf.vcc();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(
        f'    {fp_type} s0 = std::bit_cast<{fp_type}>({src[0]}.{read_fn}(wf, lane));'
    )
    L.append(
        f'    {fp_type} s1 = std::bit_cast<{fp_type}>({src[1]}.{read_fn}(wf, lane));'
    )
    L.append(
        f'    {fp_type} s2 = std::bit_cast<{fp_type}>({src[2]}.{read_fn}(wf, lane));'
    )
    if is_vop3:
        L.extend(vop3_src_mod('s0', 0, has_abs))
        L.extend(vop3_src_mod('s1', 1, has_abs))
        L.extend(vop3_src_mod('s2', 2, has_abs))
    L.append(f'    {fp_type} result = s0;')
    L.append('    bool set_vcc = false;')
    L.append(f'    if (s2 == {zero} || s1 == {zero}) {{')
    L.append(f'      // Zero numerator or denominator: pass through s0 unscaled.')
    L.append(f'      // Special-case handling (0/0, 0/x, x/0) is done by v_div_fixup.')
    L.append('    } else {')
    L.append('      int exp1 = 0, exp2 = 0;')
    L.append('      std::frexp(s1, &exp1);')
    L.append('      std::frexp(s2, &exp2);')
    L.append(f'      if (exp2 - exp1 >= {exp_threshold}) {{')
    L.append('        set_vcc = true;')
    L.append(f'        if (s0 == s1) result = std::ldexp(s0, {scale_exp});')
    L.append(f'      }} else if (std::fpclassify(s1) == FP_SUBNORMAL) {{')
    L.append(f'        result = std::ldexp(s0, {scale_exp});')
    if is_f64:
        L.append(f'      }} else if (std::fpclassify(1.0 / s1) == FP_SUBNORMAL &&')
        L.append(f'                 std::fpclassify(s2 / s1) == FP_SUBNORMAL) {{')
    else:
        L.append(
            f'      }} else if (std::fpclassify(1.0 / static_cast<double>(s1)) == FP_SUBNORMAL &&'
        )
        L.append(f'                 std::fpclassify(s2 / s1) == FP_SUBNORMAL) {{')
    L.append('        set_vcc = true;')
    L.append(f'        if (s0 == s1) result = std::ldexp(s0, {scale_exp});')
    if is_f64:
        L.append(f'      }} else if (std::fpclassify(1.0 / s1) == FP_SUBNORMAL) {{')
    else:
        L.append(
            f'      }} else if (std::fpclassify(1.0 / static_cast<double>(s1)) == FP_SUBNORMAL) {{'
        )
    L.append(f'        result = std::ldexp(s0, -{scale_exp});')
    L.append(f'      }} else if (std::fpclassify(s2 / s1) == FP_SUBNORMAL) {{')
    L.append('        set_vcc = true;')
    L.append(f'        if (s0 == s2) result = std::ldexp(s0, {scale_exp});')
    L.append(f'      }} else if (exp2 <= {tiny_exp}) {{')
    L.append(f'        result = std::ldexp(s0, {scale_exp});')
    L.append('      }')
    L.append('    }')
    L.append('    if (set_vcc) vcc |= (1ULL << lane);')
    L.append('    else vcc &= ~(1ULL << lane);')
    L.append(f'    {dst[0]}.{write_fn}(wf, lane, std::bit_cast<{cast_to}>(result));')
    L.append('  }')
    L.append('  wf.set_vcc(vcc);')
    return '\n'.join(L)


def gen_vector_div_fmas(
    dst: list[str],
    src: list[str],
    dtype: str | None,
    is_vop3: bool = False,
    has_abs: bool = False,
) -> str:
    """Generate V_DIV_FMAS body (FMA with scale based on VCC)."""
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  uint64_t vcc = wf.vcc();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if dtype == 'f64':
        L.append(
            f'    double s0 = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));'
        )
        L.append(
            f'    double s1 = std::bit_cast<double>({src[1]}.read_lane64(wf, lane));'
        )
        L.append(
            f'    double s2 = std::bit_cast<double>({src[2]}.read_lane64(wf, lane));'
        )
        if is_vop3:
            L.extend(vop3_src_mod('s0', 0, has_abs))
            L.extend(vop3_src_mod('s1', 1, has_abs))
            L.extend(vop3_src_mod('s2', 2, has_abs))
        L.append('    double result = std::fma(s0, s1, s2);')
        L.append('    if (vcc & (1ULL << lane)) {')
        L.append('      result = std::ldexp(result, 64);')
        L.append('    }')
        L.append(
            f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));'
        )
    else:
        L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
        L.append(f'    float s1 = std::bit_cast<float>({src[1]}.read_lane(wf, lane));')
        L.append(f'    float s2 = std::bit_cast<float>({src[2]}.read_lane(wf, lane));')
        if is_vop3:
            L.extend(vop3_src_mod('s0', 0, has_abs))
            L.extend(vop3_src_mod('s1', 1, has_abs))
            L.extend(vop3_src_mod('s2', 2, has_abs))
        L.append('    float result = std::fma(s0, s1, s2);')
        L.append('    if (vcc & (1ULL << lane)) {')
        L.append('      result = std::ldexp(result, 32);')
        L.append('    }')
        L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
    L.append('  }')
    return '\n'.join(L)


def gen_vector_dot(
    dst: list[str], src: list[str], op: str | None, dtype: str | None
) -> str:
    """Generate V_DOT*C body (dot product accumulate)."""
    L = []
    d = dst[0] if dst else src[0]
    s0, s1 = (src[0], src[1]) if dst else (src[1], src[2])
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(f'    uint32_t a = {s0}.read_lane(wf, lane);')
    L.append(f'    uint32_t b = {s1}.read_lane(wf, lane);')
    L.append(f'    int32_t acc = static_cast<int32_t>({d}.read_lane(wf, lane));')
    if op == 'dot4c':
        L.append('    for (int i = 0; i < 4; ++i) {')
        L.append('      int8_t ea = static_cast<int8_t>((a >> (i * 8)) & 0xFF);')
        L.append('      int8_t eb = static_cast<int8_t>((b >> (i * 8)) & 0xFF);')
        L.append('      acc += static_cast<int32_t>(ea) * static_cast<int32_t>(eb);')
        L.append('    }')
    elif op == 'dot8c':
        L.append('    for (int i = 0; i < 8; ++i) {')
        L.append('      int32_t ea = static_cast<int32_t>((a >> (i * 4)) & 0xF);')
        L.append('      if (ea & 8) ea |= ~0xF;')
        L.append('      int32_t eb = static_cast<int32_t>((b >> (i * 4)) & 0xF);')
        L.append('      if (eb & 8) eb |= ~0xF;')
        L.append('      acc += ea * eb;')
        L.append('    }')
    elif op == 'dot2c' and dtype == 'f32':
        # V_DOT2C_F32_F16: D.f32 += f16_lo(A)*f16_lo(B) + f16_hi(A)*f16_hi(B)
        L.append('    float a0 = util::f16_to_f32(static_cast<uint16_t>(a & 0xFFFF));')
        L.append(
            '    float a1 = util::f16_to_f32(static_cast<uint16_t>((a >> 16) & 0xFFFF));'
        )
        L.append('    float b0 = util::f16_to_f32(static_cast<uint16_t>(b & 0xFFFF));')
        L.append(
            '    float b1 = util::f16_to_f32(static_cast<uint16_t>((b >> 16) & 0xFFFF));'
        )
        L.append('    float facc = std::bit_cast<float>(static_cast<uint32_t>(acc));')
        L.append('    facc += a0 * b0 + a1 * b1;')
        L.append('    acc = static_cast<int32_t>(std::bit_cast<uint32_t>(facc));')
    elif op == 'dot2c' and dtype == 'i32':
        # V_DOT2C_I32_I16: D.i32 += i16_lo(A)*i16_lo(B) + i16_hi(A)*i16_hi(B)
        L.append('    int16_t a0 = static_cast<int16_t>(a & 0xFFFF);')
        L.append('    int16_t a1 = static_cast<int16_t>((a >> 16) & 0xFFFF);')
        L.append('    int16_t b0 = static_cast<int16_t>(b & 0xFFFF);')
        L.append('    int16_t b1 = static_cast<int16_t>((b >> 16) & 0xFFFF);')
        L.append(
            '    acc += static_cast<int32_t>(a0) * b0 + static_cast<int32_t>(a1) * b1;'
        )
    else:
        L.append(f'    (void)a; (void)b; // unhandled dot variant: {op}/{dtype}')
    L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(acc));')
    L.append('  }')
    return '\n'.join(L)


def gen_vector_dot2c_bf16(dst: list[str], src: list[str]) -> str:
    """Generate V_DOT2C_F32_BF16 body: D.f32 += A.bf16[0]*B.bf16[0] + A.bf16[1]*B.bf16[1]."""
    L = []
    d = dst[0] if dst else src[0]
    s0, s1 = (src[0], src[1]) if dst else (src[1], src[2])
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(f'    uint32_t a = {s0}.read_lane(wf, lane);')
    L.append(f'    uint32_t b = {s1}.read_lane(wf, lane);')
    L.append(f'    float acc = std::bit_cast<float>({d}.read_lane(wf, lane));')
    L.append('    float a0 = util::bf16_to_f32(static_cast<uint16_t>(a & 0xFFFF));')
    L.append(
        '    float a1 = util::bf16_to_f32(static_cast<uint16_t>((a >> 16) & 0xFFFF));'
    )
    L.append('    float b0 = util::bf16_to_f32(static_cast<uint16_t>(b & 0xFFFF));')
    L.append(
        '    float b1 = util::bf16_to_f32(static_cast<uint16_t>((b >> 16) & 0xFFFF));'
    )
    L.append('    acc += a0 * b0 + a1 * b1;')
    L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(acc));')
    L.append('  }')
    return '\n'.join(L)


def gen_vector_bitop3(dst: list[str], src: list[str], dtype: str | None) -> str:
    """Generate V_BITOP3_B32/B16 body: 3-input LUT-based bitwise operation.

    The 8-bit truth table is packed into the VOP3 modifier fields:
      truth_table = (omod << 6) | (abs << 3) | neg
    NOT from any source operand value. Source modifiers are not applied.

    Index bit ordering:
      bit 2 = src0, bit 1 = src1, bit 0 = src2
    """
    nbits = '16' if dtype == 'b16' else '32'
    L = []
    L.append('  uint8_t truth_table = static_cast<uint8_t>')
    L.append('      ((inst_.omod << 6) | (inst_.abs << 3) | inst_.neg);')
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(f'    uint32_t a = {src[0]}.read_lane(wf, lane);')
    L.append(f'    uint32_t b = {src[1]}.read_lane(wf, lane);')
    L.append(f'    uint32_t c = {src[2]}.read_lane(wf, lane);')
    L.append(f'    uint32_t result = 0;')
    L.append(f'    for (int i = 0; i < {nbits}; ++i) {{')
    L.append(
        '      uint32_t idx = (((a >> i) & 1) << 2) | (((b >> i) & 1) << 1) | ((c >> i) & 1);'
    )
    L.append('      result |= ((truth_table >> idx) & 1) << i;')
    L.append('    }')
    L.append(f'    {dst[0]}.write_lane(wf, lane, result);')
    L.append('  }')
    return '\n'.join(L)


def gen_vector_permlane_swap(dst: list[str], src: list[str], stride: int) -> str:
    """Generate V_PERMLANE{16,32}_SWAP_B32.

    For each lane N in [0..stride-1]:
      tmp = src0[N]
      src0[N]        ← vdst[N + stride]
      vdst[N+stride] ← tmp
    vdst[0..stride-1] and src0[stride..] are UNCHANGED.
    EXEC mask is IGNORED.
    Both vdst and src0 are outputs (LLVM: returns {vdst_new, src0_new}).
    """
    L = []
    L.append('  uint32_t tmp_dst[64] = {}, tmp_src[64] = {};')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append(f'    tmp_dst[lane] = {dst[0]}.read_lane(wf, lane);')
    L.append(f'    tmp_src[lane] = {dst[1]}.read_lane(wf, lane);')
    L.append('  }')
    L.append(f'  for (uint32_t lane = 0; lane < {stride}; ++lane) {{')
    L.append(f'    if (lane + {stride} >= wf.wf_size()) break;')
    L.append(f'    {dst[1]}.write_lane(wf, lane, tmp_dst[lane + {stride}]);')
    L.append(f'    {dst[0]}.write_lane(wf, lane + {stride}, tmp_src[lane]);')
    L.append('  }')
    return '\n'.join(L)


def gen_vector_permlane(
    dst: list[str], src: list[str], op: str | None, cross: bool
) -> str:
    """Generate V_PERMLANE16_B32 / V_PERMLANEX16_B32 (imm and var forms).

    For each lane i, read from lane (i & ~0xF) | selector[i & 0xF].
    Immediate form: selector from src1 (low 16 lanes) / src2 (high 16 lanes),
      each is a 2-bit field per sub-lane packed into a 32-bit scalar.
    Var form: selector from low 4 bits of src2 VGPR per lane.
    For permlanex16 (cross=True), XOR bit 4 into the source lane to
    enable cross-16-group fetches.
    """
    is_var = op == 'var'
    L = []
    L.append('  constexpr bool fi = false, bound_ctrl = false;')
    L.append('  uint64_t exec = wf.exec();')
    L.append('  uint32_t snap[64];')
    L.append('  for (uint32_t i = 0; i < wf.wf_size(); ++i)')
    L.append(f'    snap[i] = {src[0]}.read_lane(wf, i);')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if is_var:
        L.append(f'    uint32_t sel = {src[1]}.read_lane(wf, lane) & 0xF;')
    else:
        L.append(f'    uint32_t sel_word = (lane < 32)')
        L.append(f'        ? {src[1]}.read_scalar(wf)')
        L.append(f'        : {src[2]}.read_scalar(wf);')
        L.append('    uint32_t sub = lane & 0xF;')
        L.append('    uint32_t sel = (sel_word >> (sub * 2)) & 0xF;')
    xor_bit = ' ^ 0x10' if cross else ''
    L.append(f'    uint32_t src_lane = (lane & ~0xFu) | ((sel{xor_bit}) & 0xFu);')
    L.append('    if (src_lane >= wf.wf_size()) continue;')
    L.append('    bool src_active = (exec & (1ULL << src_lane)) != 0;')
    L.append('    if (!src_active && !fi) {')
    L.append('      if (bound_ctrl)')
    L.append(f'        {dst[0]}.write_lane(wf, lane, 0);')
    L.append('      continue;')
    L.append('    }')
    L.append(f'    {dst[0]}.write_lane(wf, lane, snap[src_lane]);')
    L.append('  }')
    return '\n'.join(L)


def gen_vector_permlane64(dst: list[str], src: list[str]) -> str:
    """Generate V_PERMLANE64_B32: swap lane i with lane i ^ 32."""
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  uint32_t snap[64];')
    L.append('  for (uint32_t i = 0; i < wf.wf_size(); ++i)')
    L.append(f'    snap[i] = {src[0]}.read_lane(wf, i);')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append('    uint32_t partner = lane ^ 32;')
    L.append('    if (partner < wf.wf_size())')
    L.append(f'      {dst[0]}.write_lane(wf, lane, snap[partner]);')
    L.append('  }')
    return '\n'.join(L)


def gen_vector_cvt_pk(dst: list[str], src: list[str], cls: str, op: str | None) -> str:
    """Generate pack/convert instructions."""
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if cls == 'vector_cvt_pk_u8_f32':
        L.append(
            f'    float fval = std::bit_cast<float>({src[0]}.read_lane(wf, lane));'
        )
        L.append(f'    uint32_t byte_sel = {src[1]}.read_lane(wf, lane) & 3;')
        # V_CVT_PK_U8_F32 has 3 srcs; V_CVT_PKACCUM reads old from dst
        old_src = src[2] if len(src) > 2 else dst[0]
        L.append(f'    uint32_t old = {old_src}.read_lane(wf, lane);')
        L.append(
            '    uint32_t byte = static_cast<uint32_t>(std::clamp(fval, 0.0f, 255.0f));'
        )
        L.append('    uint32_t mask = ~(0xFFu << (byte_sel * 8));')
        L.append(
            f'    {dst[0]}.write_lane(wf, lane, (old & mask) | (byte << (byte_sel * 8)));'
        )
    elif cls == 'vector_cvt_pknorm':
        L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
        L.append(f'    float s1 = std::bit_cast<float>({src[1]}.read_lane(wf, lane));')
        if op == 'i16':
            L.append('    auto cvt_i16 = [](float f) -> int16_t {')
            L.append('      if (std::isnan(f)) return 0;')
            L.append(
                '      return static_cast<int16_t>(std::clamp(f * 32767.0f, -32768.0f, 32767.0f));'
            )
            L.append('    };')
            L.append('    int16_t lo = cvt_i16(s0);')
            L.append('    int16_t hi = cvt_i16(s1);')
        else:  # u16
            L.append('    auto cvt_u16 = [](float f) -> uint16_t {')
            L.append('      if (std::isnan(f)) return 0;')
            L.append(
                '      return static_cast<uint16_t>(std::clamp(f * 65535.0f, 0.0f, 65535.0f));'
            )
            L.append('    };')
            L.append('    uint16_t lo = cvt_u16(s0);')
            L.append('    uint16_t hi = cvt_u16(s1);')
        L.append(
            f'    {dst[0]}.write_lane(wf, lane, (static_cast<uint32_t>(hi) << 16) | (static_cast<uint32_t>(lo) & 0xFFFF));'
        )
    elif cls == 'vector_cvt_pkrtz_f16_f32':
        L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
        L.append(f'    float s1 = std::bit_cast<float>({src[1]}.read_lane(wf, lane));')
        L.append(f'    uint32_t lo = util::f32_to_f16(s0);')
        L.append(f'    uint32_t hi = util::f32_to_f16(s1);')
        L.append(f'    {dst[0]}.write_lane(wf, lane, lo | (hi << 16));')
    elif cls == 'vector_cvt_pk':
        L.append(f'    uint32_t s0 = {src[0]}.read_lane(wf, lane);')
        L.append(f'    uint32_t s1 = {src[1]}.read_lane(wf, lane);')
        if op == 'u16_u32':
            L.append('    uint16_t lo = static_cast<uint16_t>(std::min(s0, 0xFFFFu));')
            L.append('    uint16_t hi = static_cast<uint16_t>(std::min(s1, 0xFFFFu));')
        else:  # i16_i32
            L.append(
                '    int16_t lo = static_cast<int16_t>(std::clamp(static_cast<int32_t>(s0), -32768, 32767));'
            )
            L.append(
                '    int16_t hi = static_cast<int16_t>(std::clamp(static_cast<int32_t>(s1), -32768, 32767));'
            )
        L.append(
            f'    {dst[0]}.write_lane(wf, lane, (static_cast<uint32_t>(static_cast<uint16_t>(hi)) << 16) | static_cast<uint32_t>(static_cast<uint16_t>(lo)));'
        )
    elif cls == 'vector_cvt_pk_f16_f32':
        L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
        L.append(f'    float s1 = std::bit_cast<float>({src[1]}.read_lane(wf, lane));')
        L.append(f'    uint32_t lo = util::f32_to_f16(s0);')
        L.append(f'    uint32_t hi = util::f32_to_f16(s1);')
        L.append(f'    {dst[0]}.write_lane(wf, lane, lo | (hi << 16));')
    elif cls == 'vector_cvt_pk_bf16_f32':
        L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
        L.append(f'    float s1 = std::bit_cast<float>({src[1]}.read_lane(wf, lane));')
        L.append(f'    uint32_t lo = util::f32_to_bf16(s0);')
        L.append(f'    uint32_t hi = util::f32_to_bf16(s1);')
        L.append(f'    {dst[0]}.write_lane(wf, lane, lo | (hi << 16));')
    elif cls == 'vector_pack_b32_f16':
        L.append(f'    uint32_t s0 = {src[0]}.read_lane(wf, lane) & 0xFFFF;')
        L.append(f'    uint32_t s1 = {src[1]}.read_lane(wf, lane) & 0xFFFF;')
        L.append(f'    {dst[0]}.write_lane(wf, lane, s0 | (s1 << 16));')
    elif cls == 'vector_cvt_sr_f16_f32':
        # Stochastic rounding: use src1 as random bits for rounding
        L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
        L.append(
            f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(util::f32_to_f16(s0)));'
        )
    elif cls == 'vector_cvt_sr_bf16_f32':
        L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
        L.append(
            f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(util::f32_to_bf16(s0)));'
        )
    L.append('  }')
    return '\n'.join(L)
