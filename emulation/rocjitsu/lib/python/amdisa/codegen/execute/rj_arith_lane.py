# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""rj-arith lane transfers, swaps and permutations with rocjitsu register access."""

from . import ExecuteContext
from .rj_arith_names import wave_api

RELATIVE_NAMES = {
    'V_MOVRELS_B32',
    'V_MOVRELD_B32',
    'V_MOVRELSD_B32',
    'V_MOVRELSD_2_B32',
    'V_SWAPREL_B32',
}


LANE_NAMES = {
    'V_PERMLANE_IDX_GEN_B32',
    'V_READFIRSTLANE_B32',
    'V_READLANE_B32',
    'V_WRITELANE_B32',
    'V_SWAP_B16',
    'V_SWAP_B32',
    'V_PERMLANE64_B32',
    'V_PERMLANE16_SWAP_B32',
    'V_PERMLANE32_SWAP_B32',
    'V_PERMLANE_BCAST_B32',
    'V_PERMLANE_UP_B32',
    'V_PERMLANE_DOWN_B32',
    'V_PERMLANE_XOR_B32',
    'V_PERM_PK16_B4_U4',
    'V_PERM_PK16_B6_U4',
    'V_PERM_PK16_B8_U4',
    'V_PERMLANE16_B32',
    'V_PERMLANEX16_B32',
    'V_PERMLANE16_VAR_B32',
    'V_PERMLANEX16_VAR_B32',
}


# Relative operations are emitted by vector_special; SWAPREL shares the adapter.
LANE_HEADER_NAMES = LANE_NAMES | RELATIVE_NAMES


def gen_lane(ctx: ExecuteContext) -> str | None:
    arch, mnemonic = ctx.arch_name, ctx.inst.name
    dst, src = ctx.dst_ops, ctx.src_ops
    if (
        arch not in ('rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5')
        or mnemonic not in LANE_NAMES
    ):
        return None
    flags = 'wf.cu().arithmetic_flags()'
    if mnemonic in ('V_PERMLANE16_SWAP_B32', 'V_PERMLANE32_SWAP_B32'):
        if len(dst) != 2:
            return None
        stride = 16 if '16' in mnemonic else 32
        return f'  amdgpu::arith::execute_permlane_swap<{stride}>(wf, {flags}, {dst[0]}, {dst[1]});'
    if len(dst) != 1:
        return None
    if (
        mnemonic
        in (
            'V_PERMLANE_BCAST_B32',
            'V_PERMLANE_UP_B32',
            'V_PERMLANE_DOWN_B32',
            'V_PERMLANE_XOR_B32',
        )
        and len(src) == 3
    ):
        kind = mnemonic.split('_')[2].title()
        api = 'rj_arith_' + mnemonic.lower()
        return (
            f'  const uint32_t selector = amdgpu::RegisterAccess(wf).read_scalar({src[1]});\n'
            f'  const uint32_t group_width = amdgpu::RegisterAccess(wf).read_scalar({src[2]});\n'
            '  if (wf.exec() && inst_.src2 < 256 && (!group_width || (group_width & (group_width - 1))))\n'
            '    wf.report_undefined_behavior("permlane group width is not a power of two");\n'
            f'  amdgpu::arith::execute_group_permutation<amdgpu::ValuPermutation::{kind}>(\n'
            f'      wf, {wave_api(api, 32)}, {wave_api(api, 64)}, {flags},\n'
            f'      {dst[0]}, {src[0]}, selector, group_width);'
        )

    if (
        mnemonic in ('V_PERM_PK16_B4_U4', 'V_PERM_PK16_B6_U4', 'V_PERM_PK16_B8_U4')
        and len(src) == 3
    ):
        bits = int(mnemonic.split('_')[3][1:])
        api = 'rj_arith_' + mnemonic.lower()
        return (
            f'  amdgpu::arith::execute_packed_permutation<{bits}>(\n'
            f'      wf, {wave_api(api, 32)}, {wave_api(api, 64)}, {flags},\n'
            f'      {dst[0]}, {src[0]}, {src[1]}, {src[2]});'
        )
    if mnemonic == 'V_PERMLANE_IDX_GEN_B32' and arch == 'cdna5' and len(src) == 2:
        return f'  amdgpu::arith::execute_permlane_index(wf, {flags}, {dst[0]}, {src[0]}, {src[1]});'

    if mnemonic == 'V_READFIRSTLANE_B32' and len(src) == 1:
        return f'  amdgpu::arith::execute_read_lane<true>(wf, {flags}, {dst[0]}, {src[0]});'
    if mnemonic == 'V_READLANE_B32' and len(src) == 2:
        return (
            f'  const uint32_t lane = amdgpu::RegisterAccess(wf).read_scalar({src[1]});\n'
            f'  amdgpu::arith::execute_read_lane<false>(wf, {flags}, {dst[0]}, {src[0]}, lane);'
        )
    if mnemonic == 'V_WRITELANE_B32' and len(src) == 2:
        return f'  amdgpu::arith::execute_write_lane(wf, {flags}, {dst[0]}, {src[0]}, {src[1]});'
    if mnemonic in ('V_SWAP_B16', 'V_SWAP_B32') and len(src) == 1:
        return f'  amdgpu::arith::execute_swap<{str(mnemonic.endswith("B16")).lower()}>(wf, {flags}, {dst[0]}, {src[0]});'
    if mnemonic == 'V_PERMLANE64_B32' and len(src) == 1:
        return f'  amdgpu::arith::execute_permlane64(wf, {flags}, {dst[0]}, {src[0]});'
    if mnemonic.startswith(('V_PERMLANE16_', 'V_PERMLANEX16_')):
        variable = '_VAR_' in mnemonic
        if len(src) != (2 if variable else 3):
            return None
        cross = mnemonic.startswith('V_PERMLANEX')
        api = 'rj_arith_' + mnemonic.lower()
        return (
            '  const uint64_t modifiers = amdgpu::vop3_opsel(inst_) & 3;\n'
            f'  amdgpu::arith::execute_permlane16<{str(cross).lower()}, {str(variable).lower()}>(\n'
            f'      wf, {wave_api(api, 32)}, {wave_api(api, 64)}, {flags}, modifiers,\n'
            f'      {dst[0]}, {src[0]}, {src[1]}'
            + ('' if variable else f', &{src[2]}')
            + ');'
        )
    return None
