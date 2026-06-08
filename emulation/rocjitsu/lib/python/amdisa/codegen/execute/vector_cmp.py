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
    elif dst:
        # VOP3: initialize from destination register for inactive lanes.
        L.append(f'  uint64_t vcc = {dst[0]}.read_scalar64(wf);')
    else:
        L.append('  uint64_t vcc = wf.vcc();')
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
        # Read raw f16 bits first for sNaN/qNaN detection (bit 9 is the
        # quiet NaN bit in IEEE 754 binary16), then convert to f32 for
        # the remaining class checks. The f16→f32 conversion may turn
        # sNaN into qNaN, so we cannot rely on the converted value.
        L.append(
            f'    uint16_t s0_raw = static_cast<uint16_t>({src[0]}.read_lane(wf, lane));'
        )
        L.append(f'    float s0 = util::f16_to_f32(s0_raw);')
        if is_vop3:
            L.extend(vop3_src_mod('s0', 0, has_abs))
        L.append(f'    uint32_t mask = {src[1]}.read_lane(wf, lane);')
        L.append('    bool match = false;')
        L.append(
            '    bool is_f16_nan = ((s0_raw & 0x7C00) == 0x7C00) && ((s0_raw & 0x03FF) != 0);'
        )
        L.append(
            '    if ((mask & 0x001) && is_f16_nan && (s0_raw & 0x0200) == 0) match = true;'
        )
        L.append(
            '    if ((mask & 0x002) && is_f16_nan && (s0_raw & 0x0200) != 0) match = true;'
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
        L.append('    else vcc &= ~(1ULL << lane);')
    L.append('  }')
    if is_cmpx:
        if cmpx_writes_vcc:
            L.append('  wf.set_vcc(result);')
        L.append('  wf.set_exec(result);')
    elif dst:
        L.append(f'  {dst[0]}.write_scalar64(wf, vcc);')
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
        L.append(
            f'    int16_t s0 = static_cast<int16_t>({src[0]}.read_lane(wf, lane) & 0xFFFF);'
        )
        L.append(
            f'    int16_t s1 = static_cast<int16_t>({src[1]}.read_lane(wf, lane) & 0xFFFF);'
        )
    elif dtype in ('u16',):
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
    Inactive lanes preserve the destination register's existing bits.
    """
    L = []
    L.append('  uint64_t exec = wf.exec();')
    if dst:
        # VOP3: initialize from the destination register so inactive
        # lanes preserve its existing bits (not VCC).
        L.append(f'  uint64_t vcc = {dst[0]}.read_scalar64(wf);')
    else:
        # VOPC: destination is VCC.
        L.append('  uint64_t vcc = wf.vcc();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')

    if op == 'f':
        L.append('    vcc &= ~(1ULL << lane);')
    elif op == 't':
        L.append('    vcc |= (1ULL << lane);')
    else:
        cond = _cmp_condition(src, op, dtype, is_vop3, L, has_abs)
        L.append(f'    if ({cond})')
        L.append('      vcc |= (1ULL << lane);')
        L.append('    else')
        L.append('      vcc &= ~(1ULL << lane);')
    L.append('  }')
    if dst:
        # VOP3: write to explicit destination (sdst/vdst SGPR pair).
        L.append(f'  {dst[0]}.write_scalar64(wf, vcc);')
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
            L.append(f'  {dst[0]}.write_scalar64(wf, result);')
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
        L.append(f'  {dst[1]}.write_scalar64(wf, vcc);')
    else:
        L.append('  wf.set_vcc(vcc);')
    return '\n'.join(L)
