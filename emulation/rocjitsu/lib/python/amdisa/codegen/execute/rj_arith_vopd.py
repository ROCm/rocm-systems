# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RDNA dual-slot execution: rj-arith evaluates slots, rocjitsu commits the pair."""

import textwrap

from amdisa.isa_profile import VopdSlotOp

from .rj_arith_names import wave_api


def gen_vopd_execution(
    arch: str, slot_ops: tuple[VopdSlotOp, ...], wave64_guard: str
) -> str | None:
    if arch not in ('rdna3', 'rdna4', 'cdna5'):
        return None
    cases = []
    for op in slot_ops:
        name = op.mnemonic.lower()
        dot = 'dot2acc' in name
        mov = name == 'v_dual_mov_b32'
        cnd = name == 'v_dual_cndmask_b32'
        integer = mov or cnd or name.endswith(('_u32', '_b32', '_i32'))
        wide = name.endswith('_f64')
        ternary = name in ('v_dual_fma_f32', 'v_dual_fma_f64')
        literal_fma = name in ('v_dual_fmaak_f32', 'v_dual_fmamk_f32')
        selection = name in (
            'v_dual_min_f32',
            'v_dual_max_f32',
            'v_dual_min_num_f32',
            'v_dual_max_num_f32',
        )
        api = 'rj_arith_' + name
        if arch == 'cdna5' and (not integer or cnd):
            api += '_gfx12' if literal_fma else '_gfx1250'
        if arch == 'rdna4' and not integer and not selection:
            api += '_gfx12'
        lines = [
            f'case k{op.enum_name}: {{',
            '  uint64_t flags = wf.cu().arithmetic_flags();',
        ]
        # VOPD has no CLAMP modifier, so do not forward its architecture-specific control.
        if not integer and not dot:
            lines += [
                f'  flags = amdgpu::arith::{"float16_f64_flags" if wide else "float32_flags"}(wf, flags, {str(arch == "rdna3").lower()})',
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
                f'  amdgpu::arith::VectorInput{64 if wide else 32} b(registers, wf, *slot.src1, wf.exec());',
            ]
        else:
            # gfx1250 adds FP64 pairs and source negation in VOPD3.
            lines += [
                f'  amdgpu::arith::VectorInput{64 if wide else 32} a(registers, wf, *slot.src0, wf.exec());',
            ]
            if not mov:
                lines += [
                    f'  amdgpu::arith::VectorInput{64 if wide else 32} b(registers, wf, *slot.src1, wf.exec());',
                ]
        if dot or name == 'v_dual_fmac_f32':
            lines += [
                '  amdgpu::arith::VectorInput32 acc(registers, wf, *slot.dst, wf.exec());',
                '  std::copy_n(acc.data, 32, result);',
            ]
        modifiers = '0'
        if arch == 'cdna5' and (not integer or cnd) and not literal_fma:
            terms = [
                '((slot.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u)',
                '((slot.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u)',
            ]
            if ternary or name in ('v_dual_fmaak_f32', 'v_dual_fmamk_f32'):
                terms += ['((slot.neg & 4) ? RJ_ARITH_ALU_NEG_C : 0u)']
            lines += ['  const uint64_t modifiers = ' + ' | '.join(terms) + ';']
            modifiers = 'modifiers'
        if name == 'v_dual_bitop2_b32':
            modifiers = 'slot.src2_imm'
        if wide:
            lines += ['  uint32_t *d[] = {result, result + 32};']
        if ternary:
            lines += [
                f'  amdgpu::arith::VectorInput{64 if wide else 32} c(registers, wf, *slot.src2, wf.exec());'
            ]
        destination = 'd' if wide else '&result'
        source_a = 'a.data.data()' if wide else '&a.data'
        args = f'flags, exec_mask, {modifiers}, {destination}, {source_a}'
        if not mov:
            if name == 'v_dual_fmamk_f32':
                args += ', slot.src2_imm'
            args += ', b.data.data()' if wide else ', &b.data'
        if name == 'v_dual_fmaak_f32':
            args += ', slot.src2_imm'
        if ternary:
            args += ', c.data.data()' if wide else ', &c.data'
        if cnd:
            lines += [
                '  const uint32_t condition = slot.uses_vcc ? wf.vcc_mask() :',
                '      amdgpu::read_wave_mask_scalar(*slot.src2, wf);',
            ]
            args += ', condition'
        if not integer:
            args += ', state'
        if arch == 'rdna4' and name == 'v_dual_mul_f32':
            lines += [
                '  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY)',
                '    return amdgpu::arith::gfx12_compat_dual_mul(',
                '        wf, flags, exec_mask, result, a.data, b.data, state);',
            ]
        lines += [f'  return {wave_api(api, 32)}({args});', '}']
        cases.append('\n'.join(lines))
    return (
        (
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
        )
        .replace('@WAVE64_GUARD@', textwrap.indent(wave64_guard, '    '))
        .replace(
            'execute_dual_pair(wf',
            'execute_dual_pair<2>(wf' if arch == 'cdna5' else 'execute_dual_pair(wf',
        )
    )
