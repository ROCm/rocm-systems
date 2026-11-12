# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Register-only DS routing; memory scheduling remains in rocjitsu."""

from . import ExecuteContext
from .rj_arith_names import wave_api

DS_NAMES = {'DS_PERMUTE_B32', 'DS_BPERMUTE_B32', 'DS_BPERMUTE_FI_B32', 'DS_SWIZZLE_B32'}


def gen_ds(ctx: ExecuteContext) -> str | None:
    arch, mnemonic = ctx.arch_name, ctx.inst.name
    dst, src = ctx.dst_ops, ctx.src_ops
    if arch not in ('rdna3', 'rdna4') or mnemonic not in DS_NAMES or len(dst) != 1:
        return None
    if arch == 'rdna3' and mnemonic == 'DS_BPERMUTE_FI_B32':
        return None
    swizzle = mnemonic == 'DS_SWIZZLE_B32'
    if len(src) != (1 if swizzle else 2):
        return None
    offset = '  const uint16_t offset = inst_.offset0 | (inst_.offset1 << 8);\n'
    flags = 'wf.cu().arithmetic_flags()'
    if swizzle:
        return (
            offset
            + f'  amdgpu::arith::execute_ds_swizzle(wf, {flags}, {dst[0]}, {src[0]}, offset);'
        )
    api = 'rj_arith_' + mnemonic.lower()
    wave64 = api + (
        '_gfx12' if arch == 'rdna4' and mnemonic != 'DS_BPERMUTE_FI_B32' else ''
    )
    backward = 'BPERMUTE' in mnemonic
    inactive = '_FI_' in mnemonic
    return offset + (
        f'  amdgpu::arith::execute_ds_permute<{str(backward).lower()}, {str(inactive).lower()}>(\n'
        f'      wf, {wave_api(api, 32)}, {wave_api(wave64, 64)}, {flags},\n'
        f'      {dst[0]}, {src[0]}, {src[1]}, offset);'
    )
