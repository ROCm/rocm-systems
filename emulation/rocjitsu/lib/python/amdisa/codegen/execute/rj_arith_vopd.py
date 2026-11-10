# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RDNA dual-slot execution: rj-arith evaluates slots, rocjitsu commits the pair."""

import textwrap

from amdisa.isa_profile import VopdSlotOp

from .rj_arith_names import wave_api


def gen_vopd_execution(
    arch: str, slot_ops: tuple[VopdSlotOp, ...], wave64_guard: str
) -> str | None:
    if arch not in ('rdna3', 'rdna4'):
        return None
    cases = []
    for op in slot_ops:
        name = op.mnemonic.lower()
        dot = 'dot2acc' in name
        mov = name == 'v_dual_mov_b32'
        cnd = name == 'v_dual_cndmask_b32'
        integer = mov or cnd or name.endswith(('_u32', '_b32'))
        selection = name in (
            'v_dual_min_f32',
            'v_dual_max_f32',
            'v_dual_min_num_f32',
            'v_dual_max_num_f32',
        )
        api = 'rj_arith_' + name
        if arch == 'rdna4' and not integer and not selection:
            api += '_gfx12'
        lines = [
            f'case k{op.enum_name}: {{',
            '  uint64_t flags = wf.cu().arithmetic_flags();',
        ]
        # VOPD has no CLAMP modifier, so do not forward its architecture-specific control.
        if not integer and not dot:
            lines += [
                f'  flags = amdgpu::arith::float32_flags(wf, flags, {str(arch == "rdna3").lower()})',
                '      & ~RJ_ARITH_FP_DX10_CLAMP_OFF;',
            ]
        if dot:
            lines += [
                '  const bool replicate = slot.src0->opr_type_ == OperandType::OPR_SRC &&',
                '      amdgpu::dot2_src_needs_half_replication(slot.src0->encoding_value());',
                '  const bool narrow = replicate &&',
                '      amdgpu::is_inline_float_src(slot.src0->encoding_value());',
                '  amdgpu::arith::PackedFloatInput a(registers, wf, *slot.src0, wf.exec(),',
                f'      narrow, replicate, {str("bf16" in name).lower()});',
                '  amdgpu::arith::VectorInput32 b(registers, wf, *slot.src1, wf.exec());',
            ]
        else:
            # RDNA3/RDNA4 VOPDXY has no source-negation modifier fields.
            lines += [
                '  amdgpu::arith::VectorInput32 a(registers, wf, *slot.src0, wf.exec());',
            ]
            if not mov:
                lines += [
                    '  amdgpu::arith::VectorInput32 b(registers, wf, *slot.src1, wf.exec());',
                ]
        if dot or name == 'v_dual_fmac_f32':
            lines += [
                '  amdgpu::arith::VectorInput32 acc(registers, wf, *slot.dst, wf.exec());',
                '  std::copy_n(acc.data, 32, result);',
            ]
        args = 'flags, exec_mask, 0, &result, &a.data'
        if not mov:
            if name == 'v_dual_fmamk_f32':
                args += ', slot.src2_imm'
            args += ', &b.data'
        if name == 'v_dual_fmaak_f32':
            args += ', slot.src2_imm'
        if cnd:
            lines += [
                '  const uint32_t condition = slot.uses_vcc ? wf.vcc_mask() :',
                '      amdgpu::read_wave_mask_scalar(*slot.src2, wf);',
            ]
            args += ', condition'
        if not integer:
            args += ', state'
        lines += [f'  return {wave_api(api, 32)}({args});', '}']
        cases.append('\n'.join(lines))
    return (
        '''void Vopd::execute_impl(amdgpu::Wavefront &wf) {
  if (wf.wf_size() != 32) {
@WAVE64_GUARD@
  }
  amdgpu::arith::execute_dual_pair(wf, x_, y_,
      [&](const Slot &slot, amdgpu::RegisterAccess &registers, uint32_t exec_mask,
          uint32_t *result, uint32_t *state) -> int {
    switch (slot.op) {
'''
        + '\n'.join(cases)
        + '''
    default:
      wf.report_instruction_execution_error(
          amdgpu::InstructionExecutionError::UnimplementedInstruction);
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  });
}
'''
    ).replace('@WAVE64_GUARD@', textwrap.indent(wave64_guard, '    '))
