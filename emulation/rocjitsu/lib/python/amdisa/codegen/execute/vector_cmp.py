# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vector compare execute body generators.

Free functions that emit C++ execute_impl bodies for vector compare
instructions: V_CMP_* and V_CMPX_* relations.
"""

from __future__ import annotations

from amdisa.codegen.execute import float_compare, input_policy
from amdisa.codegen.execute.vop3_modifiers import (
    vop3_src_mod,
)


def _write_explicit_lane_mask(
    dst: str, value: str, result_writer: str | None = None
) -> list[str]:
    """Emit a write to an explicit SGPR lane-mask destination."""
    if result_writer is not None:
        return [f'  {result_writer}({value});']
    return [f'  amdgpu::write_wave_mask_scalar({dst}, wf, {value});']


def _write_cmpx_result(
    dst: list[str], is_vop3: bool, cmpx_writes_vcc: bool, result_writer: str | None
) -> list[str]:
    writes = []
    if cmpx_writes_vcc:
        if dst and is_vop3:
            writes.extend(_write_explicit_lane_mask(dst[0], 'result'))
        else:
            writes.append('  wf.set_vcc_mask(result);')
    writes.append(f'  {result_writer or "wf.set_exec"}(result);')
    return writes


def _cmp_condition(
    src: list[str],
    op: str | None,
    dtype: str | None,
    is_vop3: bool,
    L: list[str],
    has_abs: bool = False,
) -> str:
    """Emit source reads and return the C++ condition expression.

    Floating relations read the raw encodings and return a
    comparison::evaluate call, which applies VOP3 ABS/NEG, MODE input
    flushing and NaN ordering; the caller declares its policy.
    """
    if dtype in float_compare.FORMATS:
        if not float_compare.is_float_relation(dtype, op):
            raise ValueError(f'unsupported floating compare relation: {op}')
        for i in range(2):
            if dtype == 'f64':
                read = f'amdgpu::RegisterAccess(wf).read_lane64({src[i]}, lane)'
            elif dtype == 'f16' and is_vop3:
                read = f'::rocjitsu::amdgpu::read_vop3_true16_src({src[i]}, wf, lane, opsel, {i})'
            else:
                read = f'amdgpu::RegisterAccess(wf).read_lane({src[i]}, lane)'
            L.append(f'    const {float_compare.lane_type(dtype)} s{i} = {read};')
        modifiers = None
        if is_vop3:
            modifiers = ('inst_.abs' if has_abs else '0u', 'inst_.neg')
        return float_compare.evaluate_expr(dtype, op, 's0', 's1', modifiers=modifiers)
    elif dtype in ('i16',):
        if is_vop3:
            L.append(
                f'    uint32_t s0_raw = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
            )
            L.append(
                f'    uint32_t s1_raw = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
            )
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
                f'    int16_t s0 = static_cast<int16_t>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane) & 0xFFFF);'
            )
            L.append(
                f'    int16_t s1 = static_cast<int16_t>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane) & 0xFFFF);'
            )
    elif dtype in ('u16',):
        if is_vop3:
            L.append(
                f'    uint32_t s0_raw = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
            )
            L.append(
                f'    uint32_t s1_raw = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
            )
            L.append('    if (opsel & (1u << 0)) s0_raw >>= 16;')
            L.append('    if (opsel & (1u << 1)) s1_raw >>= 16;')
            L.append('    uint16_t s0 = static_cast<uint16_t>(s0_raw);')
            L.append('    uint16_t s1 = static_cast<uint16_t>(s1_raw);')
        else:
            L.append(
                f'    uint16_t s0 = static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
            )
            L.append(
                f'    uint16_t s1 = static_cast<uint16_t>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
            )
    elif dtype in ('i32',):
        L.append(
            f'    int32_t s0 = static_cast<int32_t>(amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane));'
        )
        L.append(
            f'    int32_t s1 = static_cast<int32_t>(amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane));'
        )
    elif dtype == 'u32':
        L.append(
            f'    uint32_t s0 = amdgpu::RegisterAccess(wf).read_lane({src[0]}, lane);'
        )
        L.append(
            f'    uint32_t s1 = amdgpu::RegisterAccess(wf).read_lane({src[1]}, lane);'
        )
    else:
        raise ValueError(f'unsupported legacy integer compare type: {dtype}')
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
    if float_compare.is_float_relation(dtype, op):
        L.append(input_policy.policy_decl(dtype))
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
        L.append('  wf.set_vcc_mask(vcc);')
    return '\n'.join(L)


def gen_vector_cmpx(
    src: list[str],
    op: str | None,
    dtype: str | None,
    cmpx_writes_vcc: bool = False,
    is_vop3: bool = False,
    dst: list[str] | None = None,
    has_abs: bool = False,
    result_writer: str | None = None,
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
    if float_compare.is_float_relation(dtype, op):
        L.append(input_policy.policy_decl(dtype))
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
    L.extend(_write_cmpx_result(dst, is_vop3, cmpx_writes_vcc, result_writer))
    return '\n'.join(L)
