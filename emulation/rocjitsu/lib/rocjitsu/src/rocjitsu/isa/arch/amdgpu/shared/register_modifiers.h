// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_REGISTER_MODIFIERS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_REGISTER_MODIFIERS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"

namespace rocjitsu::amdgpu {

// Keep decoded metadata rules shared while retaining each instruction's class
// layout. Inlining resolves operand addresses to constant instruction offsets.
template <typename Inst>
void ds_wordwise_register_modifiers(const Inst &inst, RegisterModifiers &modifiers) {
  modifiers.wordwise_source0 = &inst.data0;
  if constexpr (requires { inst.data1; })
    modifiers.wordwise_source1 = &inst.data1;
}

template <bool Wordwise, uint8_t ResultBytes = 0xf, uint8_t LastResultBytes = ResultBytes,
          typename Inst>
void buffer_register_modifiers(const Inst &inst, RegisterModifiers &modifiers) {
  if constexpr (Wordwise)
    modifiers.wordwise_source0 = &inst.vdata;
  if constexpr (requires { inst.rsrc; })
    modifiers.buffer_resource = &inst.rsrc;
  else
    modifiers.buffer_resource = &inst.srsrc;
  if constexpr (requires { inst.vaddr; })
    modifiers.buffer_address = &inst.vaddr;
  if constexpr (requires { inst.soffset; })
    modifiers.buffer_offset = &inst.soffset;
  if constexpr (ResultBytes != 0xf)
    modifiers.memory_result_bytes = ResultBytes;
  if constexpr (LastResultBytes != 0xf)
    modifiers.memory_result_last_bytes = LastResultBytes;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_REGISTER_MODIFIERS_H_
