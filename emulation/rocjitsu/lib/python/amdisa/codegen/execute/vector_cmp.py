# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vector compare execute body generators.

Free functions that emit C++ execute_impl bodies for vector compare
instructions: V_CMP_*, V_CMPX_*, V_CMP_CLASS_*, and V_ADD_CO_U32.
"""

from __future__ import annotations

from amdisa.codegen.execute.vop3_modifiers import (
    vop3_src_mod,
)


def _write_explicit_lane_mask(dst: str, value: str) -> list[str]:
    """Emit a write to an explicit SGPR lane-mask destination."""
    return [
        '  if (wf.wf_size() <= 32)',
        f'    {dst}.write_scalar(wf, static_cast<uint32_t>({value}));',
        '  else',
        f'    {dst}.write_scalar64(wf, {value});',
    ]


def gen_vector_cmp_class(
    dst: list[str],
    src: list[str],
    dtype: str | None,
    is_cmpx: bool,
    cmpx_writes_vcc: bool = False,
    is_vop3: bool = False,
    has_abs: bool = False,
) -> str:
    """Generate V_CMP_CLASS / V_CMPX_CLASS body."""
    L = []
    L.append('  uint64_t exec = wf.exec();')
    if is_cmpx:
        L.append('  uint64_t result = 0;')
    else:
        # All v_cmp variants zero inactive lanes regardless of encoding.
        L.append('  uint64_t vcc = 0;')
    if is_vop3 and dtype == 'f16':
        L.append('  uint32_t opsel = amdgpu::vop3_opsel(inst_);')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    if dtype == 'f64':
        L.append(
            f'    double s0 = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));'
        )
        if is_vop3:
            L.extend(vop3_src_mod('s0', 0, has_abs))
        L.append(f'    uint32_t mask = {src[1]}.read_lane(wf, lane);')
        L.append('    bool match = false;')
        L.append(
            '    if ((mask & 0x001) && std::isnan(s0) && (std::bit_cast<uint64_t>(s0) & 0x0008000000000000ULL) == 0) match = true;'
        )
        L.append(
            '    if ((mask & 0x002) && std::isnan(s0) && (std::bit_cast<uint64_t>(s0) & 0x0008000000000000ULL) != 0) match = true;'
        )
        L.append('    if ((mask & 0x004) && std::isinf(s0) && s0 < 0) match = true;')
        L.append('    if ((mask & 0x008) && std::isnormal(s0) && s0 < 0) match = true;')
        L.append(
            '    if ((mask & 0x010) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0 && std::signbit(s0)) match = true;'
        )
        L.append(
            '    if ((mask & 0x020) && s0 == 0.0 && std::signbit(s0)) match = true;'
        )
        L.append(
            '    if ((mask & 0x040) && s0 == 0.0 && !std::signbit(s0)) match = true;'
        )
        L.append(
            '    if ((mask & 0x080) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0 && !std::signbit(s0)) match = true;'
        )
        L.append('    if ((mask & 0x100) && std::isnormal(s0) && s0 > 0) match = true;')
        L.append('    if ((mask & 0x200) && std::isinf(s0) && s0 > 0) match = true;')
    elif dtype == 'f16':
        # Classify directly from the raw f16 bit fields. Converting to f32
        # first is lossy for this purpose: an f16 denormal widens to an f32
        # *normal*, so std::isnormal would misclassify every f16 denormal as
        # normal. The exponent/mantissa/sign decode below distinguishes all ten
        # classes on the true f16 value (bit 9 of the mantissa is the quiet-NaN
        # bit in IEEE 754 binary16).
        if is_vop3:
            L.append(
                f'    uint16_t s0_raw = static_cast<uint16_t>(::rocjitsu::amdgpu::read_vop3_true16_src({src[0]}, wf, lane, opsel, 0));'
            )
        else:
            L.append(
                f'    uint16_t s0_raw = static_cast<uint16_t>({src[0]}.read_lane(wf, lane));'
            )
        if is_vop3:
            # VOP3 abs/neg apply to the f16 value: abs clears the sign bit, neg
            # flips it (abs before neg, matching neg(abs(x))).
            if has_abs:
                L.append('    if (inst_.abs & (1u << 0)) s0_raw &= 0x7FFFu;')
            L.append('    if (inst_.neg & (1u << 0)) s0_raw ^= 0x8000u;')
        if is_vop3:
            L.append(
                f'    uint32_t mask = ::rocjitsu::amdgpu::read_vop3_true16_src({src[1]}, wf, lane, opsel, 1);'
            )
        else:
            L.append(f'    uint32_t mask = {src[1]}.read_lane(wf, lane);')
        L.append('    bool match = false;')
        L.append('    uint16_t f16_exp = (s0_raw >> 10) & 0x1F;')
        L.append('    uint16_t f16_mant = s0_raw & 0x3FF;')
        L.append('    bool f16_sign = (s0_raw & 0x8000) != 0;')
        L.append('    bool is_nan = (f16_exp == 0x1F) && (f16_mant != 0);')
        L.append('    bool is_inf = (f16_exp == 0x1F) && (f16_mant == 0);')
        L.append('    bool is_zero = (f16_exp == 0) && (f16_mant == 0);')
        L.append('    bool is_denorm = (f16_exp == 0) && (f16_mant != 0);')
        L.append('    bool is_normal = (f16_exp >= 1) && (f16_exp <= 30);')
        L.append(
            '    if ((mask & 0x001) && is_nan && (s0_raw & 0x0200) == 0) match = true;'
        )
        L.append(
            '    if ((mask & 0x002) && is_nan && (s0_raw & 0x0200) != 0) match = true;'
        )
        L.append('    if ((mask & 0x004) && is_inf && f16_sign) match = true;')
        L.append('    if ((mask & 0x008) && is_normal && f16_sign) match = true;')
        L.append('    if ((mask & 0x010) && is_denorm && f16_sign) match = true;')
        L.append('    if ((mask & 0x020) && is_zero && f16_sign) match = true;')
        L.append('    if ((mask & 0x040) && is_zero && !f16_sign) match = true;')
        L.append('    if ((mask & 0x080) && is_denorm && !f16_sign) match = true;')
        L.append('    if ((mask & 0x100) && is_normal && !f16_sign) match = true;')
        L.append('    if ((mask & 0x200) && is_inf && !f16_sign) match = true;')
    else:
        L.append(f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));')
        if is_vop3:
            L.extend(vop3_src_mod('s0', 0, has_abs))
        L.append(f'    uint32_t mask = {src[1]}.read_lane(wf, lane);')
        L.append('    bool match = false;')
        L.append(
            '    if ((mask & 0x001) && std::isnan(s0) && (std::bit_cast<uint32_t>(s0) & 0x00400000) == 0) match = true;'
        )
        L.append(
            '    if ((mask & 0x002) && std::isnan(s0) && (std::bit_cast<uint32_t>(s0) & 0x00400000) != 0) match = true;'
        )
        L.append('    if ((mask & 0x004) && std::isinf(s0) && s0 < 0) match = true;')
        L.append('    if ((mask & 0x008) && std::isnormal(s0) && s0 < 0) match = true;')
        L.append(
            '    if ((mask & 0x010) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0f && std::signbit(s0)) match = true;'
        )
        L.append(
            '    if ((mask & 0x020) && s0 == 0.0f && std::signbit(s0)) match = true;'
        )
        L.append(
            '    if ((mask & 0x040) && s0 == 0.0f && !std::signbit(s0)) match = true;'
        )
        L.append(
            '    if ((mask & 0x080) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0f && !std::signbit(s0)) match = true;'
        )
        L.append('    if ((mask & 0x100) && std::isnormal(s0) && s0 > 0) match = true;')
        L.append('    if ((mask & 0x200) && std::isinf(s0) && s0 > 0) match = true;')
    if is_cmpx:
        L.append('    if (match) result |= (1ULL << lane);')
    else:
        L.append('    if (match) vcc |= (1ULL << lane);')
    L.append('  }')
    if is_cmpx:
        if cmpx_writes_vcc:
            L.append('  wf.set_vcc(result);')
        L.append('  wf.set_exec(result);')
    elif dst:
        L.extend(_write_explicit_lane_mask(dst[0], 'vcc'))
    else:
        L.append('  wf.set_vcc(vcc);')
    return '\n'.join(L)


def _cmp_condition(
    src: list[str],
    op: str | None,
    dtype: str | None,
    is_vop3: bool,
    L: list[str],
    has_abs: bool = False,
) -> str:
    """Emit source reads and return the C++ condition expression.

    For FP types, handles ordered comparisons (eq, lt, le, gt, ge, lg),
    unordered comparisons (neq, nge, ngt, nle, nlt, nlg), and
    ordered/unordered predicates (o, u) per IEEE-754.
    """
    is_fp = dtype in ('f32', 'f64', 'f16')
    if is_fp:
        if dtype == 'f64':
            L.append(
                f'    double s0 = std::bit_cast<double>({src[0]}.read_lane64(wf, lane));'
            )
            L.append(
                f'    double s1 = std::bit_cast<double>({src[1]}.read_lane64(wf, lane));'
            )
        elif dtype == 'f16':
            if is_vop3:
                L.append(
                    f'    float s0 = util::f16_to_f32(static_cast<uint16_t>(::rocjitsu::amdgpu::read_vop3_true16_src({src[0]}, wf, lane, opsel, 0)));'
                )
                L.append(
                    f'    float s1 = util::f16_to_f32(static_cast<uint16_t>(::rocjitsu::amdgpu::read_vop3_true16_src({src[1]}, wf, lane, opsel, 1)));'
                )
            else:
                L.append(
                    f'    float s0 = util::f16_to_f32(static_cast<uint16_t>({src[0]}.read_lane(wf, lane)));'
                )
                L.append(
                    f'    float s1 = util::f16_to_f32(static_cast<uint16_t>({src[1]}.read_lane(wf, lane)));'
                )
        else:
            L.append(
                f'    float s0 = std::bit_cast<float>({src[0]}.read_lane(wf, lane));'
            )
            L.append(
                f'    float s1 = std::bit_cast<float>({src[1]}.read_lane(wf, lane));'
            )
        if is_vop3:
            L.extend(vop3_src_mod('s0', 0, has_abs))
            L.extend(vop3_src_mod('s1', 1, has_abs))
        # Ordered comparisons (false if NaN)
        ordered_map = {
            'eq': 's0 == s1',
            'ne': 's0 != s1',
            'lg': 's0 < s1 || s0 > s1',
            'lt': 's0 < s1',
            'le': 's0 <= s1',
            'gt': 's0 > s1',
            'ge': 's0 >= s1',
        }
        # Unordered comparisons (true if NaN)
        unordered_map = {
            'neq': 's0 != s1 || std::isnan(s0) || std::isnan(s1)',
            'nge': '!(s0 >= s1)',  # true when NaN (IEEE)
            'ngt': '!(s0 > s1)',
            'nle': '!(s0 <= s1)',
            'nlt': '!(s0 < s1)',
            'nlg': '!(s0 < s1 || s0 > s1)',
        }
        if op in ordered_map:
            return ordered_map[op]
        if op in unordered_map:
            return unordered_map[op]
        if op == 'o':
            return '!std::isnan(s0) && !std::isnan(s1)'
        if op == 'u':
            return 'std::isnan(s0) || std::isnan(s1)'
        return f's0 == s1 /* TODO: {op} */'
    elif dtype in ('i64',):
        L.append(
            f'    int64_t s0 = static_cast<int64_t>({src[0]}.read_lane64(wf, lane));'
        )
        L.append(
            f'    int64_t s1 = static_cast<int64_t>({src[1]}.read_lane64(wf, lane));'
        )
    elif dtype in ('u64',):
        L.append(f'    uint64_t s0 = {src[0]}.read_lane64(wf, lane);')
        L.append(f'    uint64_t s1 = {src[1]}.read_lane64(wf, lane);')
    elif dtype in ('i16',):
        if is_vop3:
            L.append(f'    uint32_t s0_raw = {src[0]}.read_lane(wf, lane);')
            L.append(f'    uint32_t s1_raw = {src[1]}.read_lane(wf, lane);')
            L.append('    if (opsel & (1u << 0)) s0_raw >>= 16;')
            L.append('    if (opsel & (1u << 1)) s1_raw >>= 16;')
            L.append(
                '    int16_t s0 = static_cast<int16_t>(static_cast<uint16_t>(s0_raw));'
            )
            L.append(
                '    int16_t s1 = static_cast<int16_t>(static_cast<uint16_t>(s1_raw));'
            )
        else:
            L.append(
                f'    int16_t s0 = static_cast<int16_t>({src[0]}.read_lane(wf, lane) & 0xFFFF);'
            )
            L.append(
                f'    int16_t s1 = static_cast<int16_t>({src[1]}.read_lane(wf, lane) & 0xFFFF);'
            )
    elif dtype in ('u16',):
        if is_vop3:
            L.append(f'    uint32_t s0_raw = {src[0]}.read_lane(wf, lane);')
            L.append(f'    uint32_t s1_raw = {src[1]}.read_lane(wf, lane);')
            L.append('    if (opsel & (1u << 0)) s0_raw >>= 16;')
            L.append('    if (opsel & (1u << 1)) s1_raw >>= 16;')
            L.append('    uint16_t s0 = static_cast<uint16_t>(s0_raw);')
            L.append('    uint16_t s1 = static_cast<uint16_t>(s1_raw);')
        else:
            L.append(
                f'    uint16_t s0 = static_cast<uint16_t>({src[0]}.read_lane(wf, lane));'
            )
            L.append(
                f'    uint16_t s1 = static_cast<uint16_t>({src[1]}.read_lane(wf, lane));'
            )
    elif dtype in ('i32',):
        L.append(
            f'    int32_t s0 = static_cast<int32_t>({src[0]}.read_lane(wf, lane));'
        )
        L.append(
            f'    int32_t s1 = static_cast<int32_t>({src[1]}.read_lane(wf, lane));'
        )
    else:
        L.append(f'    uint32_t s0 = {src[0]}.read_lane(wf, lane);')
        L.append(f'    uint32_t s1 = {src[1]}.read_lane(wf, lane);')
    cmp_map = {
        'eq': '==',
        'ne': '!=',
        'lg': '!=',
        'gt': '>',
        'ge': '>=',
        'lt': '<',
        'le': '<=',
    }
    cmp_op = cmp_map.get(op, f'== /* TODO: {op} */')
    return f's0 {cmp_op} s1'


def _uses_vop3_true16_opsel(op: str | None, dtype: str | None, is_vop3: bool) -> bool:
    return is_vop3 and dtype in ('f16', 'i16', 'u16') and op not in ('f', 't')


def gen_vector_cmp(
    dst: list[str],
    src: list[str],
    op: str | None,
    dtype: str | None,
    is_vop3: bool = False,
    has_abs: bool = False,
) -> str:
    """Generate vector compare body.

    VOPC (VOP2-like): result always goes to VCC.
    VOP3: result goes to dst[0] (explicit SGPR pair, may be VCC or any SGPR).
    Inactive lanes are zeroed in the result regardless of encoding.
    """
    L = []
    L.append('  uint64_t exec = wf.exec();')
    # All v_cmp variants zero inactive lanes regardless of encoding.
    L.append('  uint64_t vcc = 0;')
    if _uses_vop3_true16_opsel(op, dtype, is_vop3):
        L.append('  uint32_t opsel = amdgpu::vop3_opsel(inst_);')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')

    if op == 't':
        L.append('    vcc |= (1ULL << lane);')
    elif op != 'f':
        # vcc starts as the cleared mask, so false lanes can leave their bit
        # clear instead of emitting a redundant clear operation.
        cond = _cmp_condition(src, op, dtype, is_vop3, L, has_abs)
        L.append(f'    if ({cond})')
        L.append('      vcc |= (1ULL << lane);')
    L.append('  }')
    if dst:
        # VOP3: write to explicit destination (sdst/vdst SGPR pair).
        L.extend(_write_explicit_lane_mask(dst[0], 'vcc'))
    else:
        # VOPC: write to VCC.
        L.append('  wf.set_vcc(vcc);')
    return '\n'.join(L)


def gen_vector_cmpx(
    src: list[str],
    op: str | None,
    dtype: str | None,
    cmpx_writes_vcc: bool = False,
    is_vop3: bool = False,
    dst: list[str] | None = None,
    has_abs: bool = False,
) -> str:
    """Generate vector compare-and-write-EXEC body.

    On CDNA (GFX9), V_CMPX writes both EXEC and the SDST operand.
    For VOP3 encoding, SDST is the vdst field (which may be VCC or
    another SGPR pair). On RDNA, V_CMPX writes only EXEC.
    """
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  uint64_t result = 0;')
    if _uses_vop3_true16_opsel(op, dtype, is_vop3):
        L.append('  uint32_t opsel = amdgpu::vop3_opsel(inst_);')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')

    if op == 'f':
        L.append('    (void)lane;')
    elif op == 't':
        L.append('    result |= (1ULL << lane);')
    else:
        cond = _cmp_condition(src, op, dtype, is_vop3, L, has_abs)
        L.append(f'    if ({cond})')
        L.append('      result |= (1ULL << lane);')
    L.append('  }')
    if cmpx_writes_vcc:
        if dst and is_vop3:
            L.extend(_write_explicit_lane_mask(dst[0], 'result'))
        else:
            L.append('  wf.set_vcc(result);')
    L.append('  wf.set_exec(result);')
    return '\n'.join(L)


def gen_vector_add_co(
    dst: list[str], src: list[str], op: str | None, dtype: str | None
) -> str:
    """Generate vector add/sub with carry in/out.

    VOP2: carry in/out via VCC (implicit).
    VOP3/VOP3_SDST_ENC: carry-in from src[2] (explicit SGPR pair),
    carry-out to dst[1] (explicit SGPR pair).
    """
    L = []
    d = dst[0]
    s0, s1 = src[0], src[1]
    _is_vop3 = len(src) > 2 or len(dst) > 1

    L.append('  uint64_t exec = wf.exec();')
    if _is_vop3 and op in ('addc', 'subbc', 'subbrevco') and len(src) > 2:
        # VOP3: carry-in from explicit src2 SGPR pair.
        L.append(f'  uint64_t old_vcc = {src[2]}.read_scalar64(wf);')
    elif op in ('addc', 'subbc', 'subbrevco'):
        # VOP2: carry-in from VCC.
        L.append('  uint64_t old_vcc = wf.vcc();')
    L.append('  uint64_t vcc = wf.vcc();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    L.append(f'    uint32_t sv0 = {s0}.read_lane(wf, lane);')
    L.append(f'    uint32_t sv1 = {s1}.read_lane(wf, lane);')

    if op == 'add':
        L.append(
            '    uint64_t wide = static_cast<uint64_t>(sv0) + static_cast<uint64_t>(sv1);'
        )
    elif op == 'sub':
        L.append(
            '    uint64_t wide = static_cast<uint64_t>(sv0) - static_cast<uint64_t>(sv1);'
        )
        L.append('    bool borrow = sv0 < sv1;')
    elif op == 'rsub':
        L.append(
            '    uint64_t wide = static_cast<uint64_t>(sv1) - static_cast<uint64_t>(sv0);'
        )
        L.append('    bool borrow = sv1 < sv0;')
    elif op == 'addc':
        L.append('    uint32_t cin = (old_vcc & (1ULL << lane)) ? 1u : 0u;')
        L.append(
            '    uint64_t wide = static_cast<uint64_t>(sv0) + static_cast<uint64_t>(sv1) + cin;'
        )
    elif op == 'subbc':
        L.append('    uint32_t cin = (old_vcc & (1ULL << lane)) ? 1u : 0u;')
        L.append(
            '    uint64_t wide = static_cast<uint64_t>(sv0) - static_cast<uint64_t>(sv1) - cin;'
        )
        L.append(
            '    bool borrow = static_cast<uint64_t>(sv0) < static_cast<uint64_t>(sv1) + cin;'
        )
    elif op == 'subbrevco':
        L.append('    uint32_t cin = (old_vcc & (1ULL << lane)) ? 1u : 0u;')
        L.append(
            '    uint64_t wide = static_cast<uint64_t>(sv1) - static_cast<uint64_t>(sv0) - cin;'
        )
        L.append(
            '    bool borrow = static_cast<uint64_t>(sv1) < static_cast<uint64_t>(sv0) + cin;'
        )

    L.append(f'    {d}.write_lane(wf, lane, static_cast<uint32_t>(wide));')

    if op in ('add', 'addc'):
        L.append(
            '    if (wide > 0xFFFFFFFFULL) vcc |= (1ULL << lane); else vcc &= ~(1ULL << lane);'
        )
    elif op in ('sub', 'rsub', 'subbc', 'subbrevco'):
        L.append('    if (borrow) vcc |= (1ULL << lane); else vcc &= ~(1ULL << lane);')

    L.append('  }')
    if len(dst) > 1:
        # VOP3_SDST_ENC: carry-out goes to sdst (any SGPR pair).
        L.extend(_write_explicit_lane_mask(dst[1], 'vcc'))
    else:
        L.append('  wf.set_vcc(vcc);')
    return '\n'.join(L)
