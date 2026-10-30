# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Architecture-shared rj-arith vector arithmetic bindings."""

from .rj_arith_names import wave_api

INTEGER_DOT_NAMES = {
    'V_DOT4_I32_IU8',
    'V_DOT4_U32_U8',
    'V_DOT8_I32_IU4',
    'V_DOT8_U32_U4',
}


def gen_integer_dot(arch, mnemonic, dst, src, enc_name, opsel_exprs):
    if arch not in ('rdna3', 'rdna4') or mnemonic not in INTEGER_DOT_NAMES:
        return None
    if enc_name != 'ENC_VOP3P' or len(dst) != 1 or len(src) != 3:
        return None
    low, high = opsel_exprs
    modifiers = (
        '(inst_.clamp ? RJ_ARITH_DOT_CLAMP : 0u)'
        f' | (({low} & 1) ? RJ_ARITH_DOT_LO_A_HIGH : 0u)'
        f' | (({low} & 2) ? RJ_ARITH_DOT_LO_B_HIGH : 0u)'
        f' | (({high} & 1) ? 0u : RJ_ARITH_DOT_HI_A_LOW)'
        f' | (({high} & 2) ? 0u : RJ_ARITH_DOT_HI_B_LOW)'
    )
    if '_I32_' in mnemonic:
        modifiers += (
            ' |\n      ((inst_.neg & 1) ? RJ_ARITH_DOT_SIGNED_A : 0u)'
            ' |\n      ((inst_.neg & 2) ? RJ_ARITH_DOT_SIGNED_B : 0u)'
        )
    api = 'rj_arith_' + mnemonic.lower()
    return (
        f'  const uint64_t modifiers = {modifiers};\n'
        f'  amdgpu::arith::execute_ternary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
        '      wf.cu().arithmetic_flags(),\n'
        f'      modifiers, {dst[0]}, {", ".join(src)});'
    )
