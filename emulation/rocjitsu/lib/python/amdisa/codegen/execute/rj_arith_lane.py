# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""rj-arith lane transfers, swaps and permutations with rocjitsu register access."""

from .rj_arith_names import wave_api

RELATIVE_NAMES = {
    'V_MOVRELS_B32',
    'V_MOVRELD_B32',
    'V_MOVRELSD_B32',
    'V_MOVRELSD_2_B32',
    'V_SWAPREL_B32',
}


LANE_NAMES = {
    'V_READFIRSTLANE_B32',
    'V_READLANE_B32',
    'V_WRITELANE_B32',
    'V_SWAP_B16',
    'V_SWAP_B32',
    'V_PERMLANE64_B32',
    'V_PERMLANE16_B32',
    'V_PERMLANEX16_B32',
    'V_PERMLANE16_VAR_B32',
    'V_PERMLANEX16_VAR_B32',
} | RELATIVE_NAMES


def gen_lane(arch, mnemonic, dst, src):
    if arch not in ('rdna3', 'rdna4') or mnemonic not in LANE_NAMES or len(dst) != 1:
        return None
    flags = 'wf.cu().arithmetic_flags()'
    if mnemonic == 'V_READFIRSTLANE_B32' and len(src) == 1:
        return (
            f'  amdgpu::arith::execute_read_lane<true>(wf, {flags}, {dst[0]}, {src[0]});'
        )
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
