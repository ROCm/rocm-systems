# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packed math execute body generators.

Free functions that emit C++ execute_impl bodies for packed 16-bit
instructions (V_PK_ADD_F16, etc.), packed F32 instructions, mad_mix,
dot product instructions, and V_PK_MOV_B32.

The opsel_exprs parameter carries the C++ expressions for op_sel and
op_sel_hi fields, derived from the ISA profile at call time.
"""

from __future__ import annotations


def gen_pk_binop(dst: list[str], src: list[str], op: str | None, dtype: str | None, opsel_exprs: tuple[str, str] = ('', '')) -> str:
    """Generate packed 16-bit binary op (V_PK_ADD_I16, V_PK_MUL_F16, etc.)."""
    d, s0, s1 = dst[0], src[0], src[1]
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(f'    uint32_t raw0 = {s0}.read_lane(wf, lane);')
    L.append(f'    uint32_t raw1 = {s1}.read_lane(wf, lane);')

    # op_sel: which half of each src for LO result
    # op_sel_hi: which half for HI result (default = hi)
    opsel, opsel_hi = opsel_exprs
    L.append(f'    bool sel0_lo = ({opsel} >> 0) & 1;')
    L.append(f'    bool sel1_lo = ({opsel} >> 1) & 1;')
    L.append(f'    bool sel0_hi = ({opsel_hi} >> 0) & 1;')
    L.append(f'    bool sel1_hi = ({opsel_hi} >> 1) & 1;')

    if dtype == 'f16':
        # FP16: extract as float, operate, pack back
        L.append('    float a_lo = util::f16_to_f32(static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0));')
        L.append('    float b_lo = util::f16_to_f32(static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1));')
        L.append('    float a_hi = util::f16_to_f32(static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0));')
        L.append('    float b_hi = util::f16_to_f32(static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1));')
        # neg modifiers
        L.append('    if (inst_.neg & 1) { a_lo = -a_lo; }')
        L.append('    if (inst_.neg & 2) { b_lo = -b_lo; }')
        L.append('    if (inst_.neg_hi & 1) { a_hi = -a_hi; }')
        L.append('    if (inst_.neg_hi & 2) { b_hi = -b_hi; }')
        f_op_map = {
            'add': ('a_lo + b_lo', 'a_hi + b_hi'),
            'mul': ('a_lo * b_lo', 'a_hi * b_hi'),
            'min': ('std::fmin(a_lo, b_lo)', 'std::fmin(a_hi, b_hi)'),
            'max': ('std::fmax(a_lo, b_lo)', 'std::fmax(a_hi, b_hi)'),
        }
        lo_expr, hi_expr = f_op_map[op]
        L.append(f'    float rlo = {lo_expr};')
        L.append(f'    float rhi = {hi_expr};')
        L.append(f'    {d}.write_lane(wf, lane, util::f32_to_f16(rlo) | (static_cast<uint32_t>(util::f32_to_f16(rhi)) << 16));')
    elif dtype == 'i16':
        L.append('    int16_t a_lo = static_cast<int16_t>(sel0_lo ? (raw0 >> 16) : raw0);')
        L.append('    int16_t b_lo = static_cast<int16_t>(sel1_lo ? (raw1 >> 16) : raw1);')
        L.append('    int16_t a_hi = static_cast<int16_t>(sel0_hi ? (raw0 >> 16) : raw0);')
        L.append('    int16_t b_hi = static_cast<int16_t>(sel1_hi ? (raw1 >> 16) : raw1);')
        i_op_map = {
            'add': ('a_lo + b_lo', 'a_hi + b_hi'),
            'sub': ('a_lo - b_lo', 'a_hi - b_hi'),
            'max': ('a_lo > b_lo ? a_lo : b_lo', 'a_hi > b_hi ? a_hi : b_hi'),
            'min': ('a_lo < b_lo ? a_lo : b_lo', 'a_hi < b_hi ? a_hi : b_hi'),
            'ashr': ('static_cast<int16_t>(b_lo >> (a_lo & 15))',
                     'static_cast<int16_t>(b_hi >> (a_hi & 15))'),
        }
        lo_expr, hi_expr = i_op_map[op]
        L.append(f'    uint16_t rlo = static_cast<uint16_t>({lo_expr});')
        L.append(f'    uint16_t rhi = static_cast<uint16_t>({hi_expr});')
        L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));')
    else:  # u16
        L.append('    uint16_t a_lo = static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0);')
        L.append('    uint16_t b_lo = static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1);')
        L.append('    uint16_t a_hi = static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0);')
        L.append('    uint16_t b_hi = static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1);')
        u_op_map = {
            'add': ('a_lo + b_lo', 'a_hi + b_hi'),
            'sub': ('a_lo - b_lo', 'a_hi - b_hi'),
            'mul': ('a_lo * b_lo', 'a_hi * b_hi'),
            'max': ('a_lo > b_lo ? a_lo : b_lo', 'a_hi > b_hi ? a_hi : b_hi'),
            'min': ('a_lo < b_lo ? a_lo : b_lo', 'a_hi < b_hi ? a_hi : b_hi'),
            'shl': ('static_cast<uint16_t>(b_lo << (a_lo & 15u))',
                    'static_cast<uint16_t>(b_hi << (a_hi & 15u))'),
            'shr': ('static_cast<uint16_t>(b_lo >> (a_lo & 15u))',
                    'static_cast<uint16_t>(b_hi >> (a_hi & 15u))'),
        }
        lo_expr, hi_expr = u_op_map[op]
        L.append(f'    uint16_t rlo = static_cast<uint16_t>({lo_expr});')
        L.append(f'    uint16_t rhi = static_cast<uint16_t>({hi_expr});')
        L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));')

    L.append('  }')
    return '\n'.join(L)

def gen_pk_ternary(dst: list[str], src: list[str], op: str | None, dtype: str | None, op_sel_hi_2_expr: str = '', opsel_exprs: tuple[str, str] = ('', '')) -> str:
    """Generate packed 16-bit ternary op (V_PK_FMA_F16, V_PK_MAD_I16, etc.)."""
    d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(f'    uint32_t raw0 = {s0}.read_lane(wf, lane);')
    L.append(f'    uint32_t raw1 = {s1}.read_lane(wf, lane);')
    L.append(f'    uint32_t raw2 = {s2}.read_lane(wf, lane);')
    opsel, opsel_hi = opsel_exprs
    L.append(f'    bool sel0_lo = ({opsel} >> 0) & 1;')
    L.append(f'    bool sel1_lo = ({opsel} >> 1) & 1;')
    L.append(f'    bool sel2_lo = ({opsel} >> 2) & 1;')
    L.append(f'    bool sel0_hi = ({opsel_hi} >> 0) & 1;')
    L.append(f'    bool sel1_hi = ({opsel_hi} >> 1) & 1;')
    L.append(f'    bool sel2_hi = {op_sel_hi_2_expr};')

    if dtype == 'f16':
        L.append('    float a_lo = util::f16_to_f32(static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0));')
        L.append('    float b_lo = util::f16_to_f32(static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1));')
        L.append('    float c_lo = util::f16_to_f32(static_cast<uint16_t>(sel2_lo ? (raw2 >> 16) : raw2));')
        L.append('    float a_hi = util::f16_to_f32(static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0));')
        L.append('    float b_hi = util::f16_to_f32(static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1));')
        L.append('    float c_hi = util::f16_to_f32(static_cast<uint16_t>(sel2_hi ? (raw2 >> 16) : raw2));')
        L.append('    if (inst_.neg & 1) { a_lo = -a_lo; }')
        L.append('    if (inst_.neg & 2) { b_lo = -b_lo; }')
        L.append('    if (inst_.neg & 4) { c_lo = -c_lo; }')
        L.append('    if (inst_.neg_hi & 1) { a_hi = -a_hi; }')
        L.append('    if (inst_.neg_hi & 2) { b_hi = -b_hi; }')
        L.append('    if (inst_.neg_hi & 4) { c_hi = -c_hi; }')
        if op == 'fma':
            L.append('    float rlo = std::fma(a_lo, b_lo, c_lo);')
            L.append('    float rhi = std::fma(a_hi, b_hi, c_hi);')
        elif op in ('minimum3', 'min3'):
            L.append('    float rlo = std::fmin(std::fmin(a_lo, b_lo), c_lo);')
            L.append('    float rhi = std::fmin(std::fmin(a_hi, b_hi), c_hi);')
        elif op in ('maximum3', 'max3'):
            L.append('    float rlo = std::fmax(std::fmax(a_lo, b_lo), c_lo);')
            L.append('    float rhi = std::fmax(std::fmax(a_hi, b_hi), c_hi);')
        else:  # mad
            L.append('    float rlo = a_lo * b_lo + c_lo;')
            L.append('    float rhi = a_hi * b_hi + c_hi;')
        L.append(f'    {d}.write_lane(wf, lane, util::f32_to_f16(rlo) | (static_cast<uint32_t>(util::f32_to_f16(rhi)) << 16));')
    elif dtype == 'i16':
        L.append('    int16_t a_lo = static_cast<int16_t>(sel0_lo ? (raw0 >> 16) : raw0);')
        L.append('    int16_t b_lo = static_cast<int16_t>(sel1_lo ? (raw1 >> 16) : raw1);')
        L.append('    int16_t c_lo = static_cast<int16_t>(sel2_lo ? (raw2 >> 16) : raw2);')
        L.append('    int16_t a_hi = static_cast<int16_t>(sel0_hi ? (raw0 >> 16) : raw0);')
        L.append('    int16_t b_hi = static_cast<int16_t>(sel1_hi ? (raw1 >> 16) : raw1);')
        L.append('    int16_t c_hi = static_cast<int16_t>(sel2_hi ? (raw2 >> 16) : raw2);')
        L.append('    uint16_t rlo = static_cast<uint16_t>(a_lo * b_lo + c_lo);')
        L.append('    uint16_t rhi = static_cast<uint16_t>(a_hi * b_hi + c_hi);')
        L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));')
    else:  # u16
        L.append('    uint16_t a_lo = static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0);')
        L.append('    uint16_t b_lo = static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1);')
        L.append('    uint16_t c_lo = static_cast<uint16_t>(sel2_lo ? (raw2 >> 16) : raw2);')
        L.append('    uint16_t a_hi = static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0);')
        L.append('    uint16_t b_hi = static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1);')
        L.append('    uint16_t c_hi = static_cast<uint16_t>(sel2_hi ? (raw2 >> 16) : raw2);')
        L.append('    uint16_t rlo = static_cast<uint16_t>(a_lo * b_lo + c_lo);')
        L.append('    uint16_t rhi = static_cast<uint16_t>(a_hi * b_hi + c_hi);')
        L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));')

    L.append('  }')
    return '\n'.join(L)

def gen_pk_binop_f32(dst: list[str], src: list[str], op: str | None, opsel_exprs: tuple[str, str] = ('', '')) -> str:
    """Generate packed F32 binary op (V_PK_ADD_F32, V_PK_MUL_F32).

    Operands are 64-bit VGPR pairs holding two 32-bit floats.
    Uses op_sel/op_sel_hi to select which 32-bit half feeds each lane,
    and neg/neg_hi for per-lane negation.
    """
    d, s0, s1 = dst[0], src[0], src[1]
    opsel, opsel_hi = opsel_exprs
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    # Read each source as a pair of F32 values. For VGPR pairs
    # (encoding 256-511), the low register is read_lane and the high
    # register is the next VGPR. For scalar/constant sources, the same
    # 32-bit value applies to both halves.
    for var, src in [('s0', s0), ('s1', s1)]:
        L.append(f'    uint32_t {var}_lo_w = {src}.read_lane(wf, lane);')
        L.append(f'    uint32_t {var}_hi_w = ({src}.encoding_value_ >= 256 && {src}.encoding_value_ <= 511)')
        L.append(f'        ? wf.cu().read_vgpr(wf.vgpr_alloc().base + static_cast<uint32_t>({src}.encoding_value_ - 256) + 1, lane)')
        L.append(f'        : {var}_lo_w;')
    L.append(f'    bool sel0_lo = ({opsel} >> 0) & 1;')
    L.append(f'    bool sel1_lo = ({opsel} >> 1) & 1;')
    L.append(f'    bool sel0_hi = ({opsel_hi} >> 0) & 1;')
    L.append(f'    bool sel1_hi = ({opsel_hi} >> 1) & 1;')
    L.append('    float a_lo = std::bit_cast<float>(sel0_lo ? s0_hi_w : s0_lo_w);')
    L.append('    float a_hi = std::bit_cast<float>(sel0_hi ? s0_hi_w : s0_lo_w);')
    L.append('    float b_lo = std::bit_cast<float>(sel1_lo ? s1_hi_w : s1_lo_w);')
    L.append('    float b_hi = std::bit_cast<float>(sel1_hi ? s1_hi_w : s1_lo_w);')
    L.append('    if (inst_.neg & 1) a_lo = -a_lo;')
    L.append('    if (inst_.neg & 2) b_lo = -b_lo;')
    L.append('    if (inst_.neg_hi & 1) a_hi = -a_hi;')
    L.append('    if (inst_.neg_hi & 2) b_hi = -b_hi;')
    f_map = {
        'add': ('a_lo + b_lo', 'a_hi + b_hi'),
        'mul': ('a_lo * b_lo', 'a_hi * b_hi'),
    }
    lo_expr, hi_expr = f_map[op]
    L.append(f'    uint32_t rlo = std::bit_cast<uint32_t>({lo_expr});')
    L.append(f'    uint32_t rhi = std::bit_cast<uint32_t>({hi_expr});')
    L.append(f'    {d}.write_lane64(wf, lane, static_cast<uint64_t>(rlo) | (static_cast<uint64_t>(rhi) << 32));')
    L.append('  }')
    return '\n'.join(L)

def gen_pk_ternary_f32(dst: list[str], src: list[str], op: str | None, op_sel_hi_2_expr: str = '', opsel_exprs: tuple[str, str] = ('', '')) -> str:
    """Generate packed F32 ternary op (V_PK_FMA_F32).

    Uses op_sel/op_sel_hi/op_sel_hi_2 to select which 32-bit half
    of each source feeds the low and high FMA lanes.
    """
    d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
    opsel, opsel_hi = opsel_exprs
    opsel_hi_2 = op_sel_hi_2_expr
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    for var, src in [('s0', s0), ('s1', s1), ('s2', s2)]:
        L.append(f'    uint32_t {var}_lo_w = {src}.read_lane(wf, lane);')
        L.append(f'    uint32_t {var}_hi_w = ({src}.encoding_value_ >= 256 && {src}.encoding_value_ <= 511)')
        L.append(f'        ? wf.cu().read_vgpr(wf.vgpr_alloc().base + static_cast<uint32_t>({src}.encoding_value_ - 256) + 1, lane)')
        L.append(f'        : {var}_lo_w;')
    L.append(f'    bool sel0_lo = ({opsel} >> 0) & 1;')
    L.append(f'    bool sel1_lo = ({opsel} >> 1) & 1;')
    L.append(f'    bool sel2_lo = ({opsel} >> 2) & 1;')
    L.append(f'    bool sel0_hi = ({opsel_hi} >> 0) & 1;')
    L.append(f'    bool sel1_hi = ({opsel_hi} >> 1) & 1;')
    L.append(f'    bool sel2_hi = {opsel_hi_2};')
    L.append('    float a_lo = std::bit_cast<float>(sel0_lo ? s0_hi_w : s0_lo_w);')
    L.append('    float a_hi = std::bit_cast<float>(sel0_hi ? s0_hi_w : s0_lo_w);')
    L.append('    float b_lo = std::bit_cast<float>(sel1_lo ? s1_hi_w : s1_lo_w);')
    L.append('    float b_hi = std::bit_cast<float>(sel1_hi ? s1_hi_w : s1_lo_w);')
    L.append('    float c_lo = std::bit_cast<float>(sel2_lo ? s2_hi_w : s2_lo_w);')
    L.append('    float c_hi = std::bit_cast<float>(sel2_hi ? s2_hi_w : s2_lo_w);')
    L.append('    if (inst_.neg & 1) a_lo = -a_lo;')
    L.append('    if (inst_.neg & 2) b_lo = -b_lo;')
    L.append('    if (inst_.neg & 4) c_lo = -c_lo;')
    L.append('    if (inst_.neg_hi & 1) a_hi = -a_hi;')
    L.append('    if (inst_.neg_hi & 2) b_hi = -b_hi;')
    L.append('    if (inst_.neg_hi & 4) c_hi = -c_hi;')
    L.append('    uint32_t rlo = std::bit_cast<uint32_t>(std::fma(a_lo, b_lo, c_lo));')
    L.append('    uint32_t rhi = std::bit_cast<uint32_t>(std::fma(a_hi, b_hi, c_hi));')
    L.append(f'    {d}.write_lane64(wf, lane, static_cast<uint64_t>(rlo) | (static_cast<uint64_t>(rhi) << 32));')
    L.append('  }')
    return '\n'.join(L)

def gen_pk_mov_b32(dst: list[str], src: list[str], opsel_exprs: tuple[str, str] = ('', '')) -> str:
    """Generate V_PK_MOV_B32: move two 32-bit values based on op_sel."""
    d, s0, s1 = dst[0], src[0], src[1]
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    for var, src in [('s0', s0), ('s1', s1)]:
        L.append(f'    uint32_t {var}_lo_w = {src}.read_lane(wf, lane);')
        L.append(f'    uint32_t {var}_hi_w = ({src}.encoding_value_ >= 256 && {src}.encoding_value_ <= 511)')
        L.append(f'        ? wf.cu().read_vgpr(wf.vgpr_alloc().base + static_cast<uint32_t>({src}.encoding_value_ - 256) + 1, lane)')
        L.append(f'        : {var}_lo_w;')
    opsel, opsel_hi = opsel_exprs
    L.append(f'    uint32_t lo = ({opsel} & 1) ? s0_hi_w : s0_lo_w;')
    L.append(f'    uint32_t hi = ({opsel_hi} & 2) ? s1_hi_w : s1_lo_w;')
    L.append(f'    {d}.write_lane64(wf, lane, static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32));')
    L.append('  }')
    return '\n'.join(L)

def gen_mad_mix_f32(dst: list[str], src: list[str], op_sel_hi_2_expr: str = '', opsel_exprs: tuple[str, str] = ('', '')) -> str:
    """Generate V_MAD_MIX_F32: mixed-precision FMA with op_sel selecting f16/f32 per src."""
    d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(f'    uint32_t raw0 = {s0}.read_lane(wf, lane);')
    L.append(f'    uint32_t raw1 = {s1}.read_lane(wf, lane);')
    L.append(f'    uint32_t raw2 = {s2}.read_lane(wf, lane);')
    # op_sel_hi selects f16 vs f32 per source (1=f16, 0=f32)
    # When f16: op_sel[i] selects which half (lo=0, hi=1)
    opsel, opsel_hi = opsel_exprs
    L.append('    float a, b, c;')
    L.append(f'    if ({opsel_hi} & 1) a = util::f16_to_f32(static_cast<uint16_t>(({opsel} & 1) ? (raw0 >> 16) : raw0));')
    L.append('    else a = std::bit_cast<float>(raw0);')
    L.append(f'    if ({opsel_hi} & 2) b = util::f16_to_f32(static_cast<uint16_t>(({opsel} & 2) ? (raw1 >> 16) : raw1));')
    L.append('    else b = std::bit_cast<float>(raw1);')
    L.append(f'    if ({op_sel_hi_2_expr}) c = util::f16_to_f32(static_cast<uint16_t>(({opsel} & 4) ? (raw2 >> 16) : raw2));')
    L.append('    else c = std::bit_cast<float>(raw2);')
    L.append('    if (inst_.neg & 1) a = -a;')
    L.append('    if (inst_.neg & 2) b = -b;')
    L.append('    if (inst_.neg & 4) c = -c;')
    L.append('    float result = a * b + c;')
    L.append('    if (inst_.clamp) result = std::clamp(result, 0.0f, 1.0f);')
    L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
    L.append('  }')
    return '\n'.join(L)

def gen_mad_mix_lo_hi(dst: list[str], src: list[str], is_lo: bool, op_sel_hi_2_expr: str = '', opsel_exprs: tuple[str, str] = ('', '')) -> str:
    """Generate V_MAD_MIXLO_F16 / V_MAD_MIXHI_F16."""
    d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(f'    uint32_t raw0 = {s0}.read_lane(wf, lane);')
    L.append(f'    uint32_t raw1 = {s1}.read_lane(wf, lane);')
    L.append(f'    uint32_t raw2 = {s2}.read_lane(wf, lane);')
    opsel, opsel_hi = opsel_exprs
    L.append('    float a, b, c;')
    L.append(f'    if ({opsel_hi} & 1) a = util::f16_to_f32(static_cast<uint16_t>(({opsel} & 1) ? (raw0 >> 16) : raw0));')
    L.append('    else a = std::bit_cast<float>(raw0);')
    L.append(f'    if ({opsel_hi} & 2) b = util::f16_to_f32(static_cast<uint16_t>(({opsel} & 2) ? (raw1 >> 16) : raw1));')
    L.append('    else b = std::bit_cast<float>(raw1);')
    L.append(f'    if ({op_sel_hi_2_expr}) c = util::f16_to_f32(static_cast<uint16_t>(({opsel} & 4) ? (raw2 >> 16) : raw2));')
    L.append('    else c = std::bit_cast<float>(raw2);')
    L.append('    if (inst_.neg & 1) a = -a;')
    L.append('    if (inst_.neg & 2) b = -b;')
    L.append('    if (inst_.neg & 4) c = -c;')
    L.append('    float result = a * b + c;')
    L.append('    if (inst_.clamp) result = std::clamp(result, 0.0f, 1.0f);')
    L.append(f'    uint16_t h = util::f32_to_f16(result);')
    L.append(f'    uint32_t prev = {d}.read_lane(wf, lane);')
    if is_lo:
        L.append(f'    {d}.write_lane(wf, lane, (prev & 0xFFFF0000u) | h);')
    else:
        L.append(f'    {d}.write_lane(wf, lane, (prev & 0x0000FFFFu) | (static_cast<uint32_t>(h) << 16));')
    L.append('  }')
    return '\n'.join(L)

def gen_dot2(dst: list[str], src: list[str], cls: str, opsel_exprs: tuple[str, str] = ('', '')) -> str:
    """Generate V_DOT2_F32_F16, V_DOT2_I32_I16, V_DOT2_U32_U16.

    Uses op_sel to select which 16-bit half of each source feeds
    element 0 (low) and element 1 (high) of the dot product.
    neg/neg_hi are split per element.
    """
    d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
    opsel, opsel_hi = opsel_exprs
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(f'    uint32_t raw0 = {s0}.read_lane(wf, lane);')
    L.append(f'    uint32_t raw1 = {s1}.read_lane(wf, lane);')
    L.append(f'    bool sel0_lo = ({opsel} >> 0) & 1;')
    L.append(f'    bool sel1_lo = ({opsel} >> 1) & 1;')
    L.append(f'    bool sel0_hi = ({opsel_hi} >> 0) & 1;')
    L.append(f'    bool sel1_hi = ({opsel_hi} >> 1) & 1;')

    if cls == 'dot2_f32_f16':
        L.append('    float a0 = util::f16_to_f32(static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0));')
        L.append('    float a1 = util::f16_to_f32(static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0));')
        L.append('    float b0 = util::f16_to_f32(static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1));')
        L.append('    float b1 = util::f16_to_f32(static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1));')
        L.append('    if (inst_.neg & 1) a0 = -a0;')
        L.append('    if (inst_.neg & 2) b0 = -b0;')
        L.append('    if (inst_.neg_hi & 1) a1 = -a1;')
        L.append('    if (inst_.neg_hi & 2) b1 = -b1;')
        L.append(f'    float acc = std::bit_cast<float>({s2}.read_lane(wf, lane));')
        L.append('    if (inst_.neg & 4) acc = -acc;')
        L.append('    float result = a0 * b0 + a1 * b1 + acc;')
        L.append('    if (inst_.clamp) result = std::clamp(result, 0.0f, 1.0f);')
        L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
    elif cls == 'dot2_i32_i16':
        L.append('    int16_t a0 = static_cast<int16_t>(sel0_lo ? (raw0 >> 16) : raw0);')
        L.append('    int16_t a1 = static_cast<int16_t>(sel0_hi ? (raw0 >> 16) : raw0);')
        L.append('    int16_t b0 = static_cast<int16_t>(sel1_lo ? (raw1 >> 16) : raw1);')
        L.append('    int16_t b1 = static_cast<int16_t>(sel1_hi ? (raw1 >> 16) : raw1);')
        L.append(f'    int32_t acc = static_cast<int32_t>({s2}.read_lane(wf, lane));')
        L.append('    int32_t result = static_cast<int32_t>(a0) * b0 + static_cast<int32_t>(a1) * b1 + acc;')
        L.append('    if (inst_.clamp) result = std::clamp(result, static_cast<int32_t>(0), std::numeric_limits<int32_t>::max());')
        L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(result));')
    else:  # dot2_u32_u16
        L.append('    uint16_t a0 = static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0);')
        L.append('    uint16_t a1 = static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0);')
        L.append('    uint16_t b0 = static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1);')
        L.append('    uint16_t b1 = static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1);')
        L.append(f'    uint32_t acc = {s2}.read_lane(wf, lane);')
        L.append('    uint32_t result = static_cast<uint32_t>(a0) * b0 + static_cast<uint32_t>(a1) * b1 + acc;')
        L.append(f'    {d}.write_lane(wf, lane, result);')

    L.append('  }')
    return '\n'.join(L)

def gen_dot4(dst: list[str], src: list[str], cls: str) -> str:
    """Generate V_DOT4_I32_I8 / V_DOT4_U32_U8."""
    d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(f'    uint32_t raw0 = {s0}.read_lane(wf, lane);')
    L.append(f'    uint32_t raw1 = {s1}.read_lane(wf, lane);')

    if cls == 'dot4_i32_i8':
        L.append(f'    int32_t acc = static_cast<int32_t>({s2}.read_lane(wf, lane));')
        L.append('    int32_t sum = acc;')
        L.append('    for (int i = 0; i < 4; ++i) {')
        L.append('      int8_t a = static_cast<int8_t>((raw0 >> (i * 8)) & 0xFF);')
        L.append('      int8_t b = static_cast<int8_t>((raw1 >> (i * 8)) & 0xFF);')
        L.append('      sum += static_cast<int32_t>(a) * b;')
        L.append('    }')
        L.append('    if (inst_.clamp) sum = std::clamp(sum, static_cast<int32_t>(0), std::numeric_limits<int32_t>::max());')
        L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(sum));')
    elif cls == 'dot4_f32_fp8':
        # FP8 dot product: D.f32 += sum(A.fp8[i] * B.fp8[i]) for i in 0..3
        L.append(f'    float acc = std::bit_cast<float>({s2}.read_lane(wf, lane));')
        L.append('    for (int i = 0; i < 4; ++i) {')
        L.append('      float a = util::fp8_e4m3_to_f32(static_cast<uint8_t>((raw0 >> (i * 8)) & 0xFF));')
        L.append('      float b = util::fp8_e4m3_to_f32(static_cast<uint8_t>((raw1 >> (i * 8)) & 0xFF));')
        L.append('      acc += a * b;')
        L.append('    }')
        L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(acc));')
    else:  # dot4_u32_u8
        L.append(f'    uint32_t acc = {s2}.read_lane(wf, lane);')
        L.append('    uint32_t sum = acc;')
        L.append('    for (int i = 0; i < 4; ++i) {')
        L.append('      uint8_t a = static_cast<uint8_t>((raw0 >> (i * 8)) & 0xFF);')
        L.append('      uint8_t b = static_cast<uint8_t>((raw1 >> (i * 8)) & 0xFF);')
        L.append('      sum += static_cast<uint32_t>(a) * b;')
        L.append('    }')
        L.append(f'    {d}.write_lane(wf, lane, sum);')

    L.append('  }')
    return '\n'.join(L)

def gen_dot8(dst: list[str], src: list[str], cls: str) -> str:
    """Generate V_DOT8_I32_I4 / V_DOT8_U32_U4."""
    d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(f'    uint32_t raw0 = {s0}.read_lane(wf, lane);')
    L.append(f'    uint32_t raw1 = {s1}.read_lane(wf, lane);')

    if cls == 'dot8_i32_i4':
        L.append(f'    int32_t acc = static_cast<int32_t>({s2}.read_lane(wf, lane));')
        L.append('    int32_t sum = acc;')
        L.append('    for (int i = 0; i < 8; ++i) {')
        L.append('      int32_t a = static_cast<int32_t>((raw0 >> (i * 4)) & 0xF);')
        L.append('      if (a & 0x8) a |= ~0xF;')
        L.append('      int32_t b = static_cast<int32_t>((raw1 >> (i * 4)) & 0xF);')
        L.append('      if (b & 0x8) b |= ~0xF;')
        L.append('      sum += a * b;')
        L.append('    }')
        L.append('    if (inst_.clamp) sum = std::clamp(sum, static_cast<int32_t>(0), std::numeric_limits<int32_t>::max());')
        L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(sum));')
    else:  # dot8_u32_u4
        L.append(f'    uint32_t acc = {s2}.read_lane(wf, lane);')
        L.append('    uint32_t sum = acc;')
        L.append('    for (int i = 0; i < 8; ++i) {')
        L.append('      uint32_t a = (raw0 >> (i * 4)) & 0xF;')
        L.append('      uint32_t b = (raw1 >> (i * 4)) & 0xF;')
        L.append('      sum += a * b;')
        L.append('    }')
        L.append(f'    {d}.write_lane(wf, lane, sum);')

    L.append('  }')
    return '\n'.join(L)

