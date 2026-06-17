# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vector ALU execute body generators.

Free functions that emit C++ execute_impl bodies for vector ALU
instructions: unary, binary, and ternary operations across float,
integer, and conversion types. Handles VOP3 source modifiers
(neg/abs) and destination modifiers (omod/clamp) via the extracted
vop3_modifiers helpers.
"""

from __future__ import annotations

from amdisa.codegen.execute.vop3_modifiers import (
    vop3_src_mod,
    vop3_dst_mod,
    vop3_dst_mod_f64,
)


def gen_vector_unary(
    dst: list[str],
    src: list[str],
    op: str | None,
    dtype: str | None,
    is_vop3: bool = False,
    has_abs: bool = False,
) -> str:
    """Generate vector unary operation body."""
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')

    if op == 'cvt':
        cvt_map = {
            'f32_i32': (
                f'    int32_t s = static_cast<int32_t>({src[0]}.read_lane(wf, lane));\n'
                f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>(s)));'
            ),
            'f32_u32': (
                f'    uint32_t s = {src[0]}.read_lane(wf, lane);\n'
                f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>(s)));'
            ),
            'i32_f32': (
                f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));\n'
                f'    int32_t r;\n'
                f'    if (std::isnan(s)) r = 0;\n'
                f'    else if (s >= 2147483648.0f) r = INT32_MAX;\n'
                f'    else if (s < -2147483648.0f) r = INT32_MIN;\n'
                f'    else r = static_cast<int32_t>(s);\n'
                f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(r));'
            ),
            'u32_f32': (
                f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));\n'
                f'    uint32_t r;\n'
                f'    if (std::isnan(s) || s < 0.0f) r = 0;\n'
                f'    else if (s >= 4294967296.0f) r = UINT32_MAX;\n'
                f'    else r = static_cast<uint32_t>(s);\n'
                f'    {dst[0]}.write_lane(wf, lane, r);'
            ),
            'rpi_i32_f32': (
                f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));\n'
                f'    float rounded = std::ceil(s - 0.5f);\n'
                f'    int32_t r;\n'
                f'    if (std::isnan(rounded)) r = 0;\n'
                f'    else if (rounded >= 2147483648.0f) r = INT32_MAX;\n'
                f'    else if (rounded < -2147483648.0f) r = INT32_MIN;\n'
                f'    else r = static_cast<int32_t>(rounded);\n'
                f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(r));'
            ),
            'flr_i32_f32': (
                f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));\n'
                f'    float rounded = std::floor(s);\n'
                f'    int32_t r;\n'
                f'    if (std::isnan(rounded)) r = 0;\n'
                f'    else if (rounded >= 2147483648.0f) r = INT32_MAX;\n'
                f'    else if (rounded < -2147483648.0f) r = INT32_MIN;\n'
                f'    else r = static_cast<int32_t>(rounded);\n'
                f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(r));'
            ),
            'f32_ubyte0': (
                f'    uint32_t s = {src[0]}.read_lane(wf, lane);\n'
                f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>(s & 0xFFu)));'
            ),
            'f32_ubyte1': (
                f'    uint32_t s = {src[0]}.read_lane(wf, lane);\n'
                f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>((s >> 8) & 0xFFu)));'
            ),
            'f32_ubyte2': (
                f'    uint32_t s = {src[0]}.read_lane(wf, lane);\n'
                f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>((s >> 16) & 0xFFu)));'
            ),
            'f32_ubyte3': (
                f'    uint32_t s = {src[0]}.read_lane(wf, lane);\n'
                f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>((s >> 24) & 0xFFu)));'
            ),
            # F64 conversions
            'f64_i32': (
                f'    int32_t s = static_cast<int32_t>({src[0]}.read_lane(wf, lane));\n'
                f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(static_cast<double>(s)));'
            ),
            'i32_f64': (
                f'    double s = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));\n'
                f'    int32_t r;\n'
                f'    if (std::isnan(s)) r = 0;\n'
                f'    else if (s >= 2147483648.0) r = INT32_MAX;\n'
                f'    else if (s < -2147483648.0) r = INT32_MIN;\n'
                f'    else r = static_cast<int32_t>(s);\n'
                f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(r));'
            ),
            'f64_u32': (
                f'    uint32_t s = {src[0]}.read_lane(wf, lane);\n'
                f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(static_cast<double>(s)));'
            ),
            'u32_f64': (
                f'    double s = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));\n'
                f'    uint32_t r;\n'
                f'    if (std::isnan(s) || s < 0.0) r = 0;\n'
                f'    else if (s >= 4294967296.0) r = UINT32_MAX;\n'
                f'    else r = static_cast<uint32_t>(s);\n'
                f'    {dst[0]}.write_lane(wf, lane, r);'
            ),
            'f64_f32': (
                f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));\n'
                f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(static_cast<double>(s)));'
            ),
            'f32_f64': (
                f'    double s = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));\n'
                f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>(s)));'
            ),
            # F16 conversions
            'f16_f32': (
                f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));\n'
                f'    {dst[0]}.write_lane(wf, lane, util::f32_to_f16(s));'
            ),
            'f32_f16': (
                f'    uint32_t raw = {src[0]}.read_lane(wf, lane);\n'
                f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(util::f16_to_f32(static_cast<uint16_t>(raw))));'
            ),
            'f16_u16': (
                f'    uint16_t s = static_cast<uint16_t>({src[0]}.read_lane(wf, lane));\n'
                f'    {dst[0]}.write_lane(wf, lane, util::f32_to_f16(static_cast<float>(s)));'
            ),
            'f16_i16': (
                f'    int16_t s = static_cast<int16_t>({src[0]}.read_lane(wf, lane) & 0xFFFF);\n'
                f'    {dst[0]}.write_lane(wf, lane, util::f32_to_f16(static_cast<float>(s)));'
            ),
            'u16_f16': (
                f'    float s = util::f16_to_f32(static_cast<uint16_t>({src[0]}.read_lane(wf, lane)));\n'
                f'    uint16_t r;\n'
                f'    if (std::isnan(s) || s < 0.0f) r = 0;\n'
                f'    else if (s >= 65536.0f) r = UINT16_MAX;\n'
                f'    else r = static_cast<uint16_t>(s);\n'
                f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(r));'
            ),
            'i16_f16': (
                f'    float s = util::f16_to_f32(static_cast<uint16_t>({src[0]}.read_lane(wf, lane)));\n'
                f'    int16_t r;\n'
                f'    if (std::isnan(s)) r = 0;\n'
                f'    else if (s >= 32768.0f) r = INT16_MAX;\n'
                f'    else if (s < -32768.0f) r = INT16_MIN;\n'
                f'    else r = static_cast<int16_t>(s);\n'
                f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(r)));'
            ),
            # 16-bit integer conversions
            'i32_i16': (
                f'    int32_t s = static_cast<int32_t>({src[0]}.read_lane(wf, lane));\n'
                f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(s & 0xFFFF))));'
            ),
            'u32_u16': (
                f'    uint32_t s = {src[0]}.read_lane(wf, lane);\n'
                f'    {dst[0]}.write_lane(wf, lane, s & 0xFFFFu);'
            ),
        }
        if dtype in cvt_map:
            L.append(cvt_map[dtype])
        else:
            L.append(f'    // TODO: cvt {dtype}')
            L.append(
                f'    {dst[0]}.write_lane(wf, lane, {src[0]}.read_lane(wf, lane));'
            )
    elif op == 'cvt_f32_bf16':
        L.append(
            f'    float r = util::bf16_to_f32(static_cast<uint16_t>({src[0]}.read_lane(wf, lane) & 0xFFFF));'
        )
        L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(r));')
    elif op == 'cvt_f32_fp8':
        L.append(
            f'    float r = util::fp8_e4m3_to_f32(static_cast<uint8_t>({src[0]}.read_lane(wf, lane) & 0xFF));'
        )
        L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(r));')
    elif op == 'cvt_f32_bf8':
        L.append(
            f'    float r = util::bf8_e5m2_to_f32(static_cast<uint8_t>({src[0]}.read_lane(wf, lane) & 0xFF));'
        )
        L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(r));')
    elif op == 'cvt_pk_f32_fp8':
        # Unpack two FP8 values into two F32s in dst[0] and dst[0]+1
        L.append(f'    uint32_t raw = {src[0]}.read_lane(wf, lane);')
        L.append(
            f'    float lo = util::fp8_e4m3_to_f32(static_cast<uint8_t>(raw & 0xFF));'
        )
        L.append(
            f'    float hi = util::fp8_e4m3_to_f32(static_cast<uint8_t>((raw >> 8) & 0xFF));'
        )
        L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(lo));')
        L.append(
            f'    wf.cu().write_vgpr(wf.vgpr_alloc().base + {dst[0]}.encoding_value_ + 1, lane, std::bit_cast<uint32_t>(hi));'
        )
    elif op == 'cvt_pk_f32_bf8':
        L.append(f'    uint32_t raw = {src[0]}.read_lane(wf, lane);')
        L.append(
            f'    float lo = util::bf8_e5m2_to_f32(static_cast<uint8_t>(raw & 0xFF));'
        )
        L.append(
            f'    float hi = util::bf8_e5m2_to_f32(static_cast<uint8_t>((raw >> 8) & 0xFF));'
        )
        L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(lo));')
        L.append(
            f'    wf.cu().write_vgpr(wf.vgpr_alloc().base + {dst[0]}.encoding_value_ + 1, lane, std::bit_cast<uint32_t>(hi));'
        )
    elif op in (
        'not',
        'bfrev',
        'ffbh_u32',
        'ffbl',
        'ffbh_i32',
        'cls_i32',
        'bcnt',
        'mbcnt_lo',
        'mbcnt_hi',
    ):
        L.append(f'    uint32_t s = {src[0]}.read_lane(wf, lane);')
        int_op_map = {
            'not': '~s',
            'bcnt': 'static_cast<uint32_t>(std::popcount(s))',
            'ffbh_u32': 's == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(s))',
            'ffbl': 's == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countr_zero(s))',
        }
        if op == 'bfrev':
            L.append('    uint32_t result = 0;')
            L.append(
                '    for (int i = 0; i < 32; ++i) result |= ((s >> i) & 1) << (31 - i);'
            )
            L.append(f'    {dst[0]}.write_lane(wf, lane, result);')
        elif op == 'ffbh_i32':
            L.append('    int32_t sv = static_cast<int32_t>(s);')
            L.append('    uint32_t abs_val = sv < 0 ? ~s : s;')
            L.append(
                f'    {dst[0]}.write_lane(wf, lane, abs_val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(abs_val)));'
            )
        elif op == 'cls_i32':
            L.append('    int32_t sv = static_cast<int32_t>(s);')
            L.append('    uint32_t abs_val = sv < 0 ? ~s : s;')
            L.append(
                f'    {dst[0]}.write_lane(wf, lane, abs_val == 0 ? 31u : static_cast<uint32_t>(std::countl_zero(abs_val)) - 1);'
            )
        elif op in int_op_map:
            L.append(f'    {dst[0]}.write_lane(wf, lane, {int_op_map[op]});')
        else:
            L.append(f'    {dst[0]}.write_lane(wf, lane, s);')
    elif op == 'frexp_exp_f32' and dtype == 'f64':
        # V_FREXP_EXP_I32_F64: extract exponent from f64, write as i32
        L.append(
            f'    double s = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));'
        )
        if is_vop3:
            L.extend(vop3_src_mod('s', 0, has_abs))
        L.append('    int exp = 0;')
        L.append(
            '    if (s != 0.0 && !std::isnan(s) && !std::isinf(s)) std::frexp(s, &exp);'
        )
        L.append(f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(exp));')
    elif op == 'frexp_mant_f32' and dtype == 'f64':
        # V_FREXP_MANT_F64: extract mantissa from f64, write as f64
        L.append(
            f'    double s = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));'
        )
        if is_vop3:
            L.extend(vop3_src_mod('s', 0, has_abs))
        L.append('    int exp = 0;')
        L.append('    double result = std::frexp(s, &exp);')
        if is_vop3:
            L.extend(vop3_dst_mod_f64('result'))
        L.append(
            f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));'
        )
    elif op == 'frexp_exp_f32':
        L.append(f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
        if is_vop3:
            L.extend(vop3_src_mod('s', 0, has_abs))
        L.append('    int exp = 0;')
        L.append(
            '    if (s != 0.0f && !std::isnan(s) && !std::isinf(s)) std::frexp(s, &exp);'
        )
        L.append(f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(exp));')
    elif op == 'frexp_exp_f16':
        L.append(
            f'    float s = util::f16_to_f32(static_cast<uint16_t>({src[0]}.read_lane(wf, lane)));'
        )
        if is_vop3:
            L.extend(vop3_src_mod('s', 0, has_abs))
        L.append('    int exp = 0;')
        L.append(
            '    if (s != 0.0f && !std::isnan(s) && !std::isinf(s)) std::frexp(s, &exp);'
        )
        L.append(f'    {dst[0]}.write_lane(wf, lane, static_cast<uint32_t>(exp));')
    elif op == 'frexp_mant_f32':
        L.append(f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
        if is_vop3:
            L.extend(vop3_src_mod('s', 0, has_abs))
        L.append('    int exp = 0;')
        L.append(f'    float result = std::frexp(s, &exp);')
        if is_vop3:
            L.extend(vop3_dst_mod('result'))
        L.append(f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
    elif op == 'clrexcp':
        L.append(f'    (void){src[0]};')
        L.append(f'    (void){dst[0]};')
    elif dtype == 'f64':
        L.append(
            f'    double s = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));'
        )
        if is_vop3:
            L.extend(vop3_src_mod('s', 0, has_abs))
        math_map_f64 = {
            'rcp': 'amdgpu::transcendental::rcp_f64(s)',
            'sqrt': 'amdgpu::transcendental::sqrt_f64(s)',
            'rsq': 'amdgpu::transcendental::rsq_f64(s)',
            'floor': 'std::floor(s)',
            'ceil': 'std::ceil(s)',
            'trunc': 'std::trunc(s)',
            'rndne': 'std::nearbyint(s)',
            'fract': 's - std::floor(s)',
            'abs': 'std::fabs(s)',
            'neg': '-s',
        }
        expr = math_map_f64.get(op, f's /* TODO: {op} */')
        if is_vop3:
            L.append(f'    double result = {expr};')
            L.extend(vop3_dst_mod_f64('result'))
            L.append(
                f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));'
            )
        else:
            L.append(
                f'    {dst[0]}.write_lane64(wf, lane, std::bit_cast<uint64_t>({expr}));'
            )
    elif dtype == 'f16':
        L.append(
            f'    float s = util::f16_to_f32(static_cast<uint16_t>({src[0]}.read_lane(wf, lane)));'
        )
        if is_vop3:
            L.extend(vop3_src_mod('s', 0, has_abs))
        math_map_f16 = {
            'rcp': '1.0f / s',
            'sqrt': 'std::sqrt(s)',
            'rsq': '1.0f / std::sqrt(s)',
            'floor': 'std::floor(s)',
            'ceil': 'std::ceil(s)',
            'trunc': 'std::trunc(s)',
            'rndne': 'std::nearbyint(s)',
            'fract': 's - std::floor(s)',
            'exp2': 'std::exp2(s)',
            'log2': 'std::log2(s)',
            'sin': 'std::sin(s * 6.2831853071795864f)',
            'cos': 'std::cos(s * 6.2831853071795864f)',
            'abs': 'std::fabs(s)',
            'neg': '-s',
        }
        expr = math_map_f16.get(op, f's /* TODO: {op} */')
        if is_vop3:
            L.append(f'    float result = {expr};')
            L.extend(vop3_dst_mod('result'))
            L.append(f'    {dst[0]}.write_lane(wf, lane, util::f32_to_f16(result));')
        else:
            L.append(f'    {dst[0]}.write_lane(wf, lane, util::f32_to_f16({expr}));')
    else:
        L.append(f'    float s = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
        if is_vop3:
            L.extend(vop3_src_mod('s', 0, has_abs))
        math_map = {
            'rcp': 'amdgpu::transcendental::rcp_f32(s)',
            'rcp_iflag': 'amdgpu::transcendental::rcp_f32(s)',
            'sqrt': 'amdgpu::transcendental::sqrt_f32(s)',
            'rsq': 'amdgpu::transcendental::rsq_f32(s)',
            'floor': 'std::floor(s)',
            'ceil': 'std::ceil(s)',
            'trunc': 'std::trunc(s)',
            'rndne': 'std::nearbyint(s)',
            'fract': 's - std::floor(s)',
            'exp2': 'amdgpu::transcendental::exp_f32(s)',
            'log2': 'amdgpu::transcendental::log_f32(s)',
            'sin': 'amdgpu::transcendental::sin_f32(s)',
            'cos': 'amdgpu::transcendental::cos_f32(s)',
            'abs': 'std::fabs(s)',
            'neg': '-s',
        }
        expr = math_map.get(op, f's /* TODO: {op} */')
        if is_vop3:
            L.append(f'    float result = {expr};')
            L.extend(vop3_dst_mod('result'))
            L.append(
                f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));'
            )
        else:
            L.append(
                f'    {dst[0]}.write_lane(wf, lane, std::bit_cast<uint32_t>({expr}));'
            )

    L.append('  }')
    return '\n'.join(L)


def gen_vector_binop(
    dst: list[str],
    src: list[str],
    op: str | None,
    dtype: str | None,
    is_vop3: bool = False,
    has_abs: bool = False,
) -> str:
    """Generate vector binary operation body."""
    if dst:
        d = dst[0]
        s0, s1 = src[0], src[1]
    else:
        d = src[0]
        s0, s1 = src[1], src[2]

    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')

    if dtype == 'f64':
        L.append(f'    double sv0 = std::bit_cast<double>({s0}.read_lane64(wf, lane));')
        if op == 'ldexp':
            # src1 is a 32-bit integer exponent, not a double.
            L.append(
                f'    int32_t sv1_i = static_cast<int32_t>({s1}.read_lane(wf, lane));'
            )
        else:
            L.append(
                f'    double sv1 = std::bit_cast<double>({s1}.read_lane64(wf, lane));'
            )
        if is_vop3:
            L.extend(vop3_src_mod('sv0', 0, has_abs))
            if op != 'ldexp':
                L.extend(vop3_src_mod('sv1', 1, has_abs))
        f_op_map = {
            'add': 'sv0 + sv1',
            'sub': 'sv0 - sv1',
            'rsub': 'sv1 - sv0',
            'mul': 'sv0 * sv1',
            'min': 'std::fmin(sv0, sv1)',
            'max': 'std::fmax(sv0, sv1)',
            'fmin': 'std::fmin(sv0, sv1)',
            'fmax': 'std::fmax(sv0, sv1)',
            'fmac': f'std::fma(sv0, sv1, std::bit_cast<double>({d}.read_lane64(wf, lane)))',
            'ldexp': 'std::ldexp(sv0, static_cast<int>(sv1_i))',
        }
        expr = f_op_map.get(op, f'sv0 /* TODO: {op} */')
        if is_vop3:
            L.append(f'    double result = {expr};')
            L.extend(vop3_dst_mod_f64('result'))
            L.append(
                f'    {d}.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));'
            )
        else:
            L.append(
                f'    {d}.write_lane64(wf, lane, std::bit_cast<uint64_t>({expr}));'
            )
    elif dtype == 'f32':
        L.append(f'    float sv0 = std::bit_cast<float>({s0}.read_lane(wf, lane));')
        if op == 'ldexp':
            # src1 is a 32-bit integer exponent, not a float.
            L.append(
                f'    int32_t sv1_i = static_cast<int32_t>({s1}.read_lane(wf, lane));'
            )
        else:
            L.append(f'    float sv1 = std::bit_cast<float>({s1}.read_lane(wf, lane));')
        if is_vop3:
            L.extend(vop3_src_mod('sv0', 0, has_abs))
            if op != 'ldexp':
                L.extend(vop3_src_mod('sv1', 1, has_abs))
        f_op_map = {
            'add': 'sv0 + sv1',
            'sub': 'sv0 - sv1',
            'rsub': 'sv1 - sv0',
            'mul': 'sv0 * sv1',
            'mul_legacy': 'sv0 == 0.0f || sv1 == 0.0f ? 0.0f : sv0 * sv1',
            'min': 'std::fmin(sv0, sv1)',
            'max': 'std::fmax(sv0, sv1)',
            'fmin': 'std::fmin(sv0, sv1)',
            'fmax': 'std::fmax(sv0, sv1)',
            'fmac': f'std::fma(sv0, sv1, std::bit_cast<float>({d}.read_lane(wf, lane)))',
            'ldexp': 'std::ldexp(sv0, static_cast<int>(sv1_i))',
        }
        expr = f_op_map.get(op, f'sv0 /* TODO: {op} */')
        if is_vop3:
            L.append(f'    float result = {expr};')
            L.extend(vop3_dst_mod('result'))
            L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));')
        else:
            L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>({expr}));')
    elif dtype == 'f16':
        L.append(
            f'    float sv0 = util::f16_to_f32(static_cast<uint16_t>({s0}.read_lane(wf, lane)));'
        )
        if op == 'ldexp':
            # src1 is a 16-bit integer exponent, not an f16 value.
            L.append(
                f'    int32_t sv1_i = static_cast<int32_t>(static_cast<int16_t>(static_cast<uint16_t>({s1}.read_lane(wf, lane))));'
            )
        else:
            L.append(
                f'    float sv1 = util::f16_to_f32(static_cast<uint16_t>({s1}.read_lane(wf, lane)));'
            )
        if is_vop3:
            L.extend(vop3_src_mod('sv0', 0, has_abs))
            if op != 'ldexp':
                L.extend(vop3_src_mod('sv1', 1, has_abs))
        f_op_map = {
            'add': 'sv0 + sv1',
            'sub': 'sv0 - sv1',
            'rsub': 'sv1 - sv0',
            'mul': 'sv0 * sv1',
            'min': 'std::fmin(sv0, sv1)',
            'max': 'std::fmax(sv0, sv1)',
            'fmin': 'std::fmin(sv0, sv1)',
            'fmax': 'std::fmax(sv0, sv1)',
            'fmac': f'std::fma(sv0, sv1, util::f16_to_f32(static_cast<uint16_t>({d}.read_lane(wf, lane))))',
            'ldexp': 'std::ldexp(sv0, static_cast<int>(sv1_i))',
        }
        expr = f_op_map.get(op, f'sv0 /* TODO: {op} */')
        if is_vop3:
            L.append(f'    float result = {expr};')
            L.extend(vop3_dst_mod('result'))
            L.append(f'    {d}.write_lane(wf, lane, util::f32_to_f16(result));')
        else:
            L.append(f'    {d}.write_lane(wf, lane, util::f32_to_f16({expr}));')
    elif dtype == 'i24':
        L.append(
            f'    int32_t sv0 = static_cast<int32_t>({s0}.read_lane(wf, lane) << 8) >> 8;'
        )
        L.append(
            f'    int32_t sv1 = static_cast<int32_t>({s1}.read_lane(wf, lane) << 8) >> 8;'
        )
        if op == 'mulhi':
            L.append(
                f'    {d}.write_lane(wf, lane, static_cast<uint32_t>((static_cast<int64_t>(sv0) * sv1) >> 32));'
            )
        else:
            L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(sv0 * sv1));')
    elif dtype == 'u24':
        L.append(f'    uint32_t sv0 = {s0}.read_lane(wf, lane) & 0x00FFFFFFu;')
        L.append(f'    uint32_t sv1 = {s1}.read_lane(wf, lane) & 0x00FFFFFFu;')
        if op == 'mulhi':
            L.append(
                f'    {d}.write_lane(wf, lane, static_cast<uint32_t>((static_cast<uint64_t>(sv0) * sv1) >> 32));'
            )
        else:
            L.append(f'    {d}.write_lane(wf, lane, sv0 * sv1);')
    elif dtype == 'i16':
        L.append(
            f'    int16_t sv0 = static_cast<int16_t>({s0}.read_lane(wf, lane) & 0xFFFF);'
        )
        L.append(
            f'    int16_t sv1 = static_cast<int16_t>({s1}.read_lane(wf, lane) & 0xFFFF);'
        )
        i_op_map = {
            'add': 'static_cast<int16_t>(sv0 + sv1)',
            'sub': 'static_cast<int16_t>(sv0 - sv1)',
            'mul': 'static_cast<int16_t>(sv0 * sv1)',
            'min': 'sv0 < sv1 ? sv0 : sv1',
            'max': 'sv0 > sv1 ? sv0 : sv1',
            'ashr': 'static_cast<int16_t>(sv1 >> (sv0 & 15))',
        }
        expr = i_op_map.get(op, f'sv0 /* TODO: {op} */')
        L.append(
            f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>({expr})));'
        )
    elif dtype == 'u16':
        L.append(f'    uint16_t sv0 = static_cast<uint16_t>({s0}.read_lane(wf, lane));')
        L.append(f'    uint16_t sv1 = static_cast<uint16_t>({s1}.read_lane(wf, lane));')
        u_op_map = {
            'add': 'static_cast<uint16_t>(sv0 + sv1)',
            'sub': 'static_cast<uint16_t>(sv0 - sv1)',
            'rsub': 'static_cast<uint16_t>(sv1 - sv0)',
            'mul': 'static_cast<uint16_t>(sv0 * sv1)',
            'min': 'sv0 < sv1 ? sv0 : sv1',
            'max': 'sv0 > sv1 ? sv0 : sv1',
            'shl': 'static_cast<uint16_t>(sv1 << (sv0 & 15u))',
            'shr': 'static_cast<uint16_t>(sv1 >> (sv0 & 15u))',
            'and': 'static_cast<uint16_t>(sv0 & sv1)',
            'or': 'static_cast<uint16_t>(sv0 | sv1)',
            'xor': 'static_cast<uint16_t>(sv0 ^ sv1)',
        }
        expr = u_op_map.get(op, f'sv0 /* TODO: {op} */')
        L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>({expr}));')
    elif dtype in ('i64',):
        if op == 'ashr':
            L.append(
                f'    int64_t val = static_cast<int64_t>({s1}.read_lane64(wf, lane));'
            )
            L.append(f'    uint32_t shift = {s0}.read_lane(wf, lane) & 63u;')
            L.append(
                f'    {d}.write_lane64(wf, lane, static_cast<uint64_t>(val >> shift));'
            )
        else:
            L.append(
                f'    int64_t sv0 = static_cast<int64_t>({s0}.read_lane64(wf, lane));'
            )
            L.append(
                f'    int64_t sv1 = static_cast<int64_t>({s1}.read_lane64(wf, lane));'
            )
            i_op_map = {
                'add': 'static_cast<uint64_t>(sv0 + sv1)',
                'sub': 'static_cast<uint64_t>(sv0 - sv1)',
                'min': 'static_cast<uint64_t>(sv0 < sv1 ? sv0 : sv1)',
                'max': 'static_cast<uint64_t>(sv0 > sv1 ? sv0 : sv1)',
            }
            expr = i_op_map.get(op, f'static_cast<uint64_t>(sv0) /* TODO: {op} */')
            L.append(f'    {d}.write_lane64(wf, lane, {expr});')
    elif dtype in ('u64', 'b64'):
        if op == 'shl':
            L.append(f'    uint64_t val = {s1}.read_lane64(wf, lane);')
            L.append(f'    uint32_t shift = {s0}.read_lane(wf, lane) & 63u;')
            L.append(f'    {d}.write_lane64(wf, lane, val << shift);')
        elif op == 'shr':
            L.append(f'    uint64_t val = {s1}.read_lane64(wf, lane);')
            L.append(f'    uint32_t shift = {s0}.read_lane(wf, lane) & 63u;')
            L.append(f'    {d}.write_lane64(wf, lane, val >> shift);')
        elif op == 'add':
            L.append(f'    uint64_t sv0 = {s0}.read_lane64(wf, lane);')
            L.append(f'    uint64_t sv1 = {s1}.read_lane64(wf, lane);')
            L.append(f'    {d}.write_lane64(wf, lane, sv0 + sv1);')
        else:
            L.append(f'    uint64_t sv0 = {s0}.read_lane64(wf, lane);')
            L.append(f'    uint64_t sv1 = {s1}.read_lane64(wf, lane);')
            u_op_map = {
                'sub': 'sv0 - sv1',
                'and': 'sv0 & sv1',
                'or': 'sv0 | sv1',
                'xor': 'sv0 ^ sv1',
                'min': 'sv0 < sv1 ? sv0 : sv1',
                'max': 'sv0 > sv1 ? sv0 : sv1',
            }
            expr = u_op_map.get(op, f'sv0 /* TODO: {op} */')
            L.append(f'    {d}.write_lane64(wf, lane, {expr});')
    elif dtype in ('i32',):
        L.append(f'    int32_t sv0 = static_cast<int32_t>({s0}.read_lane(wf, lane));')
        L.append(f'    int32_t sv1 = static_cast<int32_t>({s1}.read_lane(wf, lane));')
        i_op_map = {
            'add': 'static_cast<uint32_t>(sv0 + sv1)',
            'sub': 'static_cast<uint32_t>(sv0 - sv1)',
            'rsub': 'static_cast<uint32_t>(sv1 - sv0)',
            'min': 'static_cast<uint32_t>(sv0 < sv1 ? sv0 : sv1)',
            'max': 'static_cast<uint32_t>(sv0 > sv1 ? sv0 : sv1)',
            'ashr': 'static_cast<uint32_t>(static_cast<int32_t>(sv1) >> (sv0 & 31))',
            'mul': 'static_cast<uint32_t>(sv0 * sv1)',
            'mulhi': 'static_cast<uint32_t>(static_cast<uint64_t>(static_cast<int64_t>(sv0) * sv1) >> 32)',
        }
        expr = i_op_map.get(op, f'static_cast<uint32_t>(sv0) /* TODO: {op} */')
        L.append(f'    {d}.write_lane(wf, lane, {expr});')
    else:
        L.append(f'    uint32_t sv0 = {s0}.read_lane(wf, lane);')
        L.append(f'    uint32_t sv1 = {s1}.read_lane(wf, lane);')
        u_op_map = {
            'add': 'sv0 + sv1',
            'sub': 'sv0 - sv1',
            'rsub': 'sv1 - sv0',
            'mul': 'sv0 * sv1',
            'mulhi': 'static_cast<uint32_t>((static_cast<uint64_t>(sv0) * sv1) >> 32)',
            'and': 'sv0 & sv1',
            'or': 'sv0 | sv1',
            'xor': 'sv0 ^ sv1',
            'xnor': '~(sv0 ^ sv1)',
            'shl': 'sv1 << (sv0 & 31u)',
            'shr': 'sv1 >> (sv0 & 31u)',
            'min': 'sv0 < sv1 ? sv0 : sv1',
            'max': 'sv0 > sv1 ? sv0 : sv1',
            'bfm': '(sv0 & 31u) == 0 ? 0u : ((1u << (sv0 & 31u)) - 1u) << (sv1 & 31u)',
        }
        expr = u_op_map.get(op, f'sv0 /* TODO: {op} */')
        L.append(f'    {d}.write_lane(wf, lane, {expr});')

    L.append('  }')
    return '\n'.join(L)


def gen_vector_ternary(
    dst: list[str],
    src: list[str],
    op: str | None,
    dtype: str | None,
    is_vop3: bool = False,
    has_abs: bool = False,
) -> str:
    """Generate vector ternary (3-operand) operation body."""
    d = dst[0]
    s0, s1, s2 = src[0], src[1], src[2]

    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')

    # BFE/BFI/PERM always use unsigned raw lane reads regardless of dtype
    if op in ('bfe_u', 'bfe_i', 'perm'):
        L.append(f'    uint32_t a = {s0}.read_lane(wf, lane);')
        L.append(f'    uint32_t b = {s1}.read_lane(wf, lane);')
        L.append(f'    uint32_t c = {s2}.read_lane(wf, lane);')
        if op == 'bfe_u':
            L.append('    uint32_t offset = b & 31;')
            L.append('    uint32_t width = c & 31;')
            L.append(
                '    uint32_t mask = (width == 0 || width >= 32) ? 0u : ((1u << width) - 1);'
            )
            L.append(
                f'    {d}.write_lane(wf, lane, width == 0 ? 0u : (a >> offset) & mask);'
            )
        elif op == 'bfe_i':
            L.append('    uint32_t offset = b & 31;')
            L.append('    uint32_t width = c & 31;')
            L.append('    int32_t sv = static_cast<int32_t>(a);')
            L.append('    int32_t result_val;')
            L.append('    if (width == 0) result_val = 0;')
            # When offset + width >= 32, the extraction window extends
            # past bit 31. Arithmetic right shift of sv by offset gives
            # the correct sign-extended result without shift UB.
            L.append('    else if (offset + width >= 32) result_val = sv >> offset;')
            L.append(
                '    else result_val = (sv << (32 - offset - width)) >> (32 - width);'
            )
            L.append(
                f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(result_val));'
            )
        else:  # perm
            L.append(f'    uint32_t result = 0;')
            L.append('    uint64_t src = (static_cast<uint64_t>(a) << 32) | b;')
            L.append('    for (int i = 0; i < 4; ++i) {')
            L.append('      uint32_t sel = (c >> (i * 8)) & 0xFF;')
            L.append('      uint32_t byte;')
            L.append('      if (sel <= 7) byte = (src >> (sel * 8)) & 0xFF;')
            # Selector 0x08: constant 0x00
            # Selectors 0x09-0x0C: replicate the MSB (sign bit) of
            # byte 0-3 respectively to all 8 bits (0x00 or 0xFF).
            L.append('      else if (sel >= 0x09 && sel <= 0x0C) {')
            L.append('        uint32_t bi = sel - 0x09;')
            L.append('        byte = ((src >> (bi * 8 + 7)) & 1) ? 0xFF : 0x00;')
            L.append('      }')
            L.append('      else if (sel == 0x0D) byte = 0xFF;')
            # Selectors 0x08, 0x0E-0xFF: constant 0x00
            L.append('      else byte = 0;')
            L.append('      result |= byte << (i * 8);')
            L.append('    }')
            L.append(f'    {d}.write_lane(wf, lane, result);')
        L.append('  }')
        return '\n'.join(L)

    if dtype == 'f32':
        L.append(f'    float a = std::bit_cast<float>({s0}.read_lane(wf, lane));')
        L.append(f'    float b = std::bit_cast<float>({s1}.read_lane(wf, lane));')
        L.append(f'    float c = std::bit_cast<float>({s2}.read_lane(wf, lane));')
        if is_vop3:
            L.extend(vop3_src_mod('a', 0, has_abs))
            L.extend(vop3_src_mod('b', 1, has_abs))
            L.extend(vop3_src_mod('c', 2, has_abs))
        f_map = {
            'mad': 'a * b + c',
            'fma': 'std::fma(a, b, c)',
            'min3': 'std::fmin(std::fmin(a, b), c)',
            'max3': 'std::fmax(std::fmax(a, b), c)',
            'minimum3': '[&]() { if (std::isnan(a) || std::isnan(b) || std::isnan(c)) return std::numeric_limits<float>::quiet_NaN(); auto ab = (a == b) ? (std::signbit(a) ? a : b) : (a < b ? a : b); return (ab == c) ? (std::signbit(ab) ? ab : c) : (ab < c ? ab : c); }()',
            'maximum3': '[&]() { if (std::isnan(a) || std::isnan(b) || std::isnan(c)) return std::numeric_limits<float>::quiet_NaN(); auto ab = (a == b) ? (std::signbit(a) ? b : a) : (a > b ? a : b); return (ab == c) ? (std::signbit(ab) ? c : ab) : (ab > c ? ab : c); }()',
            'minmax': 'std::fmin(a, std::fmax(b, c))',
            'maxmin': 'std::fmax(a, std::fmin(b, c))',
            'minimummaximum': '[&]() { if (std::isnan(a) || std::isnan(b) || std::isnan(c)) return std::numeric_limits<float>::quiet_NaN(); auto ab = (a == b) ? (std::signbit(a) ? a : b) : (a < b ? a : b); return (ab == c) ? (std::signbit(ab) ? c : ab) : (ab > c ? ab : c); }()',
            'maximumminimum': '[&]() { if (std::isnan(a) || std::isnan(b) || std::isnan(c)) return std::numeric_limits<float>::quiet_NaN(); auto ab = (a == b) ? (std::signbit(a) ? b : a) : (a > b ? a : b); return (ab == c) ? (std::signbit(ab) ? ab : c) : (ab < c ? ab : c); }()',
            'minmax_num': 'std::fmin(a, std::fmax(b, c))',
            'maxmin_num': 'std::fmax(a, std::fmin(b, c))',
            'med3': 'std::fmax(std::fmin(std::fmax(a, b), c), std::fmin(a, b))',
        }
        # Cube map operations: inputs are (x, y, z)
        if op == 'cubeid':
            L.append(
                '    float ax = std::fabs(a), ay = std::fabs(b), az = std::fabs(c);'
            )
            L.append('    float face;')
            L.append('    if (az >= ax && az >= ay) face = c >= 0 ? 4.0f : 5.0f;')
            L.append('    else if (ay >= ax) face = b >= 0 ? 2.0f : 3.0f;')
            L.append('    else face = a >= 0 ? 0.0f : 1.0f;')
            if is_vop3:
                L.extend(vop3_dst_mod('face'))
            L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(face));')
        elif op == 'cubesc':
            L.append(
                '    float ax = std::fabs(a), ay = std::fabs(b), az = std::fabs(c);'
            )
            L.append('    float sc;')
            L.append('    if (az >= ax && az >= ay) sc = c >= 0 ? a : -a;')
            L.append('    else if (ay >= ax) sc = a;')
            L.append('    else sc = a >= 0 ? -c : c;')
            if is_vop3:
                L.extend(vop3_dst_mod('sc'))
            L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(sc));')
        elif op == 'cubetc':
            L.append(
                '    float ax = std::fabs(a), ay = std::fabs(b), az = std::fabs(c);'
            )
            L.append('    float tc;')
            L.append('    if (az >= ax && az >= ay) tc = -b;')
            L.append('    else if (ay >= ax) tc = b >= 0 ? c : -c;')
            L.append('    else tc = -b;')
            if is_vop3:
                L.extend(vop3_dst_mod('tc'))
            L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(tc));')
        elif op == 'cubema':
            L.append(
                '    float ax = std::fabs(a), ay = std::fabs(b), az = std::fabs(c);'
            )
            L.append('    float ma;')
            L.append('    if (az >= ax && az >= ay) ma = 2.0f * az;')
            L.append('    else if (ay >= ax) ma = 2.0f * ay;')
            L.append('    else ma = 2.0f * ax;')
            if is_vop3:
                L.extend(vop3_dst_mod('ma'))
            L.append(f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(ma));')
        elif op in f_map:
            expr = f_map[op]
            if is_vop3:
                L.append(f'    float result = {expr};')
                L.extend(vop3_dst_mod('result'))
                L.append(
                    f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(result));'
                )
            else:
                L.append(
                    f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>({expr}));'
                )
        else:
            L.append(
                f'    {d}.write_lane(wf, lane, std::bit_cast<uint32_t>(a)); // unhandled: {op}'
            )
    elif dtype == 'f64':
        L.append(f'    double a = std::bit_cast<double>({s0}.read_lane64(wf, lane));')
        L.append(f'    double b = std::bit_cast<double>({s1}.read_lane64(wf, lane));')
        L.append(f'    double c = std::bit_cast<double>({s2}.read_lane64(wf, lane));')
        if is_vop3:
            L.extend(vop3_src_mod('a', 0, has_abs))
            L.extend(vop3_src_mod('b', 1, has_abs))
            L.extend(vop3_src_mod('c', 2, has_abs))
        f_map = {
            'mad': 'a * b + c',
            'fma': 'std::fma(a, b, c)',
            'min3': 'std::fmin(std::fmin(a, b), c)',
            'max3': 'std::fmax(std::fmax(a, b), c)',
            'minimum3': '[&]() { if (std::isnan(a) || std::isnan(b) || std::isnan(c)) return std::numeric_limits<double>::quiet_NaN(); auto ab = (a == b) ? (std::signbit(a) ? a : b) : (a < b ? a : b); return (ab == c) ? (std::signbit(ab) ? ab : c) : (ab < c ? ab : c); }()',
            'maximum3': '[&]() { if (std::isnan(a) || std::isnan(b) || std::isnan(c)) return std::numeric_limits<double>::quiet_NaN(); auto ab = (a == b) ? (std::signbit(a) ? b : a) : (a > b ? a : b); return (ab == c) ? (std::signbit(ab) ? c : ab) : (ab > c ? ab : c); }()',
            'minmax': 'std::fmin(a, std::fmax(b, c))',
            'maxmin': 'std::fmax(a, std::fmin(b, c))',
            'minimummaximum': '[&]() { if (std::isnan(a) || std::isnan(b) || std::isnan(c)) return std::numeric_limits<double>::quiet_NaN(); auto ab = (a == b) ? (std::signbit(a) ? a : b) : (a < b ? a : b); return (ab == c) ? (std::signbit(ab) ? c : ab) : (ab > c ? ab : c); }()',
            'maximumminimum': '[&]() { if (std::isnan(a) || std::isnan(b) || std::isnan(c)) return std::numeric_limits<double>::quiet_NaN(); auto ab = (a == b) ? (std::signbit(a) ? b : a) : (a > b ? a : b); return (ab == c) ? (std::signbit(ab) ? ab : c) : (ab < c ? ab : c); }()',
            'minmax_num': 'std::fmin(a, std::fmax(b, c))',
            'maxmin_num': 'std::fmax(a, std::fmin(b, c))',
            'med3': 'std::fmax(std::fmin(std::fmax(a, b), c), std::fmin(a, b))',
        }
        expr = f_map.get(op, f'a /* unhandled: {op} */')
        if is_vop3:
            L.append(f'    double result = {expr};')
            L.extend(vop3_dst_mod_f64('result'))
            L.append(
                f'    {d}.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));'
            )
        else:
            L.append(
                f'    {d}.write_lane64(wf, lane, std::bit_cast<uint64_t>({expr}));'
            )
    elif dtype == 'f16':
        L.append(
            f'    float a = util::f16_to_f32(static_cast<uint16_t>({s0}.read_lane(wf, lane)));'
        )
        L.append(
            f'    float b = util::f16_to_f32(static_cast<uint16_t>({s1}.read_lane(wf, lane)));'
        )
        L.append(
            f'    float c = util::f16_to_f32(static_cast<uint16_t>({s2}.read_lane(wf, lane)));'
        )
        if is_vop3:
            L.extend(vop3_src_mod('a', 0, has_abs))
            L.extend(vop3_src_mod('b', 1, has_abs))
            L.extend(vop3_src_mod('c', 2, has_abs))
        f_map = {
            'mad': 'a * b + c',
            'fma': 'std::fma(a, b, c)',
            'min3': 'std::fmin(std::fmin(a, b), c)',
            'max3': 'std::fmax(std::fmax(a, b), c)',
            'minimum3': '[&]() { if (std::isnan(a) || std::isnan(b) || std::isnan(c)) return std::numeric_limits<float>::quiet_NaN(); auto ab = (a == b) ? (std::signbit(a) ? a : b) : (a < b ? a : b); return (ab == c) ? (std::signbit(ab) ? ab : c) : (ab < c ? ab : c); }()',
            'maximum3': '[&]() { if (std::isnan(a) || std::isnan(b) || std::isnan(c)) return std::numeric_limits<float>::quiet_NaN(); auto ab = (a == b) ? (std::signbit(a) ? b : a) : (a > b ? a : b); return (ab == c) ? (std::signbit(ab) ? c : ab) : (ab > c ? ab : c); }()',
            'minmax': 'std::fmin(a, std::fmax(b, c))',
            'maxmin': 'std::fmax(a, std::fmin(b, c))',
            'minimummaximum': '[&]() { if (std::isnan(a) || std::isnan(b) || std::isnan(c)) return std::numeric_limits<float>::quiet_NaN(); auto ab = (a == b) ? (std::signbit(a) ? a : b) : (a < b ? a : b); return (ab == c) ? (std::signbit(ab) ? c : ab) : (ab > c ? ab : c); }()',
            'maximumminimum': '[&]() { if (std::isnan(a) || std::isnan(b) || std::isnan(c)) return std::numeric_limits<float>::quiet_NaN(); auto ab = (a == b) ? (std::signbit(a) ? b : a) : (a > b ? a : b); return (ab == c) ? (std::signbit(ab) ? ab : c) : (ab < c ? ab : c); }()',
            'minmax_num': 'std::fmin(a, std::fmax(b, c))',
            'maxmin_num': 'std::fmax(a, std::fmin(b, c))',
            'med3': 'std::fmax(std::fmin(std::fmax(a, b), c), std::fmin(a, b))',
        }
        expr = f_map.get(op, f'a /* unhandled: {op} */')
        if is_vop3:
            L.append(f'    float result = {expr};')
            L.extend(vop3_dst_mod('result'))
            L.append(f'    {d}.write_lane(wf, lane, util::f32_to_f16(result));')
        else:
            L.append(f'    {d}.write_lane(wf, lane, util::f32_to_f16({expr}));')
    elif dtype in ('i16',):
        L.append(
            f'    int16_t a = static_cast<int16_t>({s0}.read_lane(wf, lane) & 0xFFFF);'
        )
        L.append(
            f'    int16_t b = static_cast<int16_t>({s1}.read_lane(wf, lane) & 0xFFFF);'
        )
        L.append(
            f'    int16_t c = static_cast<int16_t>({s2}.read_lane(wf, lane) & 0xFFFF);'
        )
        i_map = {
            'min3': 'std::min(std::min(a, b), c)',
            'max3': 'std::max(std::max(a, b), c)',
            'med3': 'std::max(std::min(std::max(a, b), c), std::min(a, b))',
            'mad': 'static_cast<int16_t>(a * b + c)',
        }
        expr = i_map.get(op, f'a /* TODO: {op} */')
        L.append(
            f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>({expr})));'
        )
    elif dtype in ('u16',):
        L.append(f'    uint16_t a = static_cast<uint16_t>({s0}.read_lane(wf, lane));')
        L.append(f'    uint16_t b = static_cast<uint16_t>({s1}.read_lane(wf, lane));')
        L.append(f'    uint16_t c = static_cast<uint16_t>({s2}.read_lane(wf, lane));')
        u_map = {
            'min3': 'std::min(std::min(a, b), c)',
            'max3': 'std::max(std::max(a, b), c)',
            'med3': 'std::max(std::min(std::max(a, b), c), std::min(a, b))',
            'mad': 'static_cast<uint16_t>(a * b + c)',
        }
        expr = u_map.get(op, f'a /* TODO: {op} */')
        L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>({expr}));')
    elif dtype in ('u64',):
        L.append(f'    uint64_t a = {s0}.read_lane64(wf, lane);')
        L.append(f'    uint64_t b = {s1}.read_lane64(wf, lane);')
        L.append(f'    uint64_t c = {s2}.read_lane64(wf, lane);')
        u_map = {
            'lshl_add': '(a << (b & 63u)) + c',
            'add3': 'a + b + c',
        }
        expr = u_map.get(op, f'a /* unhandled: {op} */')
        L.append(f'    {d}.write_lane64(wf, lane, {expr});')
    elif dtype in ('i24',):
        L.append(
            f'    int32_t a = static_cast<int32_t>({s0}.read_lane(wf, lane) << 8) >> 8;'
        )
        L.append(
            f'    int32_t b = static_cast<int32_t>({s1}.read_lane(wf, lane) << 8) >> 8;'
        )
        L.append(f'    int32_t c = static_cast<int32_t>({s2}.read_lane(wf, lane));')
        L.append(
            f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(static_cast<int64_t>(a) * b + c));'
        )
    elif dtype in ('u24',):
        L.append(f'    uint32_t a = {s0}.read_lane(wf, lane) & 0x00FFFFFFu;')
        L.append(f'    uint32_t b = {s1}.read_lane(wf, lane) & 0x00FFFFFFu;')
        L.append(f'    uint32_t c = {s2}.read_lane(wf, lane);')
        L.append(f'    {d}.write_lane(wf, lane, a * b + c);')
    elif dtype in ('i32',):
        L.append(f'    int32_t a = static_cast<int32_t>({s0}.read_lane(wf, lane));')
        L.append(f'    int32_t b = static_cast<int32_t>({s1}.read_lane(wf, lane));')
        L.append(f'    int32_t c = static_cast<int32_t>({s2}.read_lane(wf, lane));')
        i_map = {
            'min3': 'std::min(std::min(a, b), c)',
            'max3': 'std::max(std::max(a, b), c)',
            'med3': 'std::max(std::min(std::max(a, b), c), std::min(a, b))',
            'minmax': 'std::min(a, std::max(b, c))',
            'maxmin': 'std::max(a, std::min(b, c))',
            'minmax_num': 'std::min(a, std::max(b, c))',
            'maxmin_num': 'std::max(a, std::min(b, c))',
            'add_lshl': '(a + b) << (c & 31)',
            'lshl_add': '(a << (b & 31)) + c',
        }
        expr = i_map.get(op, f'a /* TODO: {op} */')
        L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>({expr}));')
    else:
        L.append(f'    uint32_t a = {s0}.read_lane(wf, lane);')
        L.append(f'    uint32_t b = {s1}.read_lane(wf, lane);')
        L.append(f'    uint32_t c = {s2}.read_lane(wf, lane);')
        # SAD and LERP need multi-line code
        if op == 'sad_u8':
            L.append('    uint32_t sum = 0;')
            L.append('    for (int i = 0; i < 4; ++i) {')
            L.append(
                '      uint32_t ba = (a >> (i * 8)) & 0xFF, bb = (b >> (i * 8)) & 0xFF;'
            )
            L.append('      sum += ba > bb ? ba - bb : bb - ba;')
            L.append('    }')
            L.append(f'    {d}.write_lane(wf, lane, sum + c);')
        elif op == 'sad_hi_u8':
            L.append('    uint32_t sum = 0;')
            L.append('    for (int i = 0; i < 4; ++i) {')
            L.append(
                '      uint32_t ba = (a >> (i * 8)) & 0xFF, bb = (b >> (i * 8)) & 0xFF;'
            )
            L.append('      sum += ba > bb ? ba - bb : bb - ba;')
            L.append('    }')
            L.append(f'    {d}.write_lane(wf, lane, (sum << 16) + c);')
        elif op == 'sad_u16':
            L.append('    uint32_t lo_a = a & 0xFFFF, hi_a = a >> 16;')
            L.append('    uint32_t lo_b = b & 0xFFFF, hi_b = b >> 16;')
            L.append(
                '    uint32_t sum = (lo_a > lo_b ? lo_a - lo_b : lo_b - lo_a) + (hi_a > hi_b ? hi_a - hi_b : hi_b - hi_a);'
            )
            L.append(f'    {d}.write_lane(wf, lane, sum + c);')
        elif op == 'sad_u32':
            L.append(f'    {d}.write_lane(wf, lane, (a > b ? a - b : b - a) + c);')
        elif op == 'msad_u8':
            L.append('    uint32_t sum = 0;')
            L.append('    for (int i = 0; i < 4; ++i) {')
            L.append(
                '      uint32_t ba = (a >> (i * 8)) & 0xFF, bb = (b >> (i * 8)) & 0xFF;'
            )
            L.append('      if (ba != 0) sum += ba > bb ? ba - bb : bb - ba;')
            L.append('    }')
            L.append(f'    {d}.write_lane(wf, lane, sum + c);')
        elif op == 'lerp_u8':
            L.append('    uint32_t result = 0;')
            L.append('    for (int i = 0; i < 4; ++i) {')
            L.append(
                '      uint32_t ba = (a >> (i * 8)) & 0xFF, bb = (b >> (i * 8)) & 0xFF;'
            )
            L.append('      uint32_t bc = (c >> (i * 8)) & 1;')
            L.append('      result |= (((ba + bb + bc) >> 1) & 0xFF) << (i * 8);')
            L.append('    }')
            L.append(f'    {d}.write_lane(wf, lane, result);')
        else:
            u_map = {
                'add3': 'a + b + c',
                'lshl_or': '(a << (b & 31)) | c',
                'and_or': '(a & b) | c',
                'or3': 'a | b | c',
                'lshl_add': '(a << (b & 31)) + c',
                'add_lshl': '(a + b) << (c & 31)',
                'xad': '(a ^ b) + c',
                'xor3': 'a ^ b ^ c',
                'min3': 'std::min(std::min(a, b), c)',
                'max3': 'std::max(std::max(a, b), c)',
                'med3': 'std::max(std::min(std::max(a, b), c), std::min(a, b))',
                'minmax': 'std::min(a, std::max(b, c))',
                'maxmin': 'std::max(a, std::min(b, c))',
                'minmax_num': 'std::min(a, std::max(b, c))',
                'maxmin_num': 'std::max(a, std::min(b, c))',
                'bfi': '(a & b) | (~a & c)',
                'alignbit': 'static_cast<uint32_t>(((static_cast<uint64_t>(a) << 32) | b) >> (c & 31))',
                'alignbyte': 'static_cast<uint32_t>(((static_cast<uint64_t>(a) << 32) | b) >> ((c & 3) * 8))',
            }
            expr = u_map.get(op, f'a /* unhandled: {op} */')
            L.append(f'    {d}.write_lane(wf, lane, {expr});')

    L.append('  }')
    return '\n'.join(L)
