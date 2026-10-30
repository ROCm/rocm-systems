# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .rj_arith_names import wave_api

DIVISION_NAMES = {
    'V_DIV_FIXUP_F16',
    'V_DIV_SCALE_F32',
    'V_DIV_SCALE_F64',
    'V_DIV_FMAS_F32',
    'V_DIV_FMAS_F64',
    'V_DIV_FIXUP_F32',
    'V_DIV_FIXUP_F64',
}


def gen_division(arch, mnemonic, dst, src, is_vop3, result_writer=None):
    if arch not in ('rdna3', 'rdna4') or mnemonic not in DIVISION_NAMES or not is_vop3:
        return None
    scale = '_SCALE_' in mnemonic
    fixup = '_FIXUP_' in mnemonic
    if len(src) != 3 or len(dst) != (2 if scale else 1):
        return None
    if not scale and result_writer is not None:
        return None
    if mnemonic.endswith('F16'):
        api = 'rj_arith_v_div_fixup_f16' + ('_gfx12' if arch == 'rdna4' else '')
        terms = []
        for i, letter in enumerate('ABC'):
            terms += [
                f'((inst_.neg & {1 << i}) ? RJ_ARITH_ALU_NEG_{letter} : 0u)',
                f'((inst_.abs & {1 << i}) ? RJ_ARITH_ALU_ABS_{letter} : 0u)',
                f'((opsel & {1 << i}) ? RJ_ARITH_ALU_HIGH_{letter} : 0u)',
            ]
        terms += [
            '((opsel & 8) ? RJ_ARITH_ALU_HIGH_D : 0u)',
            '(uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2)',
            '(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)',
        ]
        return '\n'.join(
            [
                '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
                '  const uint64_t modifiers = ' + ' | '.join(terms) + ';',
                f'  amdgpu::arith::execute_float_ternary16(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
                '      wf.cu().arithmetic_flags(), modifiers,',
                f'      {dst[0]}, {", ".join(src)}, {str(arch == "rdna3").lower()},',
                '      wf.cu().arithmetic_reports_fp_status());',
            ]
        )
    width = 64 if mnemonic.endswith('F64') else 32
    api = 'rj_arith_' + mnemonic.lower() + ('_gfx12' if arch == 'rdna4' else '')
    terms = [
        f'((inst_.neg & {1 << i}) ? RJ_ARITH_ALU_NEG_{chr(65 + i)} : 0u)'
        for i in range(3)
    ]
    if not scale:
        terms += [
            f'((inst_.abs & {1 << i}) ? RJ_ARITH_ALU_ABS_{chr(65 + i)} : 0u)'
            for i in range(3)
        ]
    terms += [
        '(uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2)',
        '(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)',
    ]
    writer = ''
    if scale:
        if result_writer is None:
            result_writer = f'[&](uint64_t result) {{ amdgpu::write_wave_mask_scalar({dst[1]}, wf, result); }}'
        writer = ', ' + result_writer
    return '\n'.join(
        [
            '  const uint64_t modifiers = ' + ' | '.join(terms) + ';',
            f'  amdgpu::arith::execute_division<{width}, amdgpu::arith::DivisionOperation::{"Scale" if scale else "Fixup" if fixup else "Fmas"}>(',
            f'      wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            f'      wf.cu().arithmetic_flags(), modifiers,',
            f'      {dst[0]}, {", ".join(src)},',
            (
                '      false'
                if scale
                else f'      wf.cu().arithmetic_reports_fp_status()'
            )
            + writer
            + ');',
        ]
    )
