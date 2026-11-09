// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_REGISTER_MODIFIERS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_REGISTER_MODIFIERS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/register_set.h"

namespace rocjitsu::amdgpu {

/// Issue and overwrite checks must agree on the bytes supplied by each result.
template <typename Visit>
void for_each_memory_result_range(RegisterRef reg, const RegisterModifiers &modifiers,
                                  bool sram_ecc, Visit visit) {
  const uint8_t data = modifiers.memory_result_bytes;
  const uint8_t tail = modifiers.memory_result_last_bytes;
  if (sram_ecc || reg.cls != RegClass::VGPR) {
    visit(reg, 0xf);
    return;
  }
  const uint16_t last = reg.index + reg.width - 1;
  if (modifiers.memory_result_has_status)
    --reg.width;
  if (reg.width && data != tail) {
    const RegisterRef partial{reg.cls, static_cast<uint16_t>(reg.index + reg.width - 1), 1};
    if (--reg.width)
      visit(reg, data);
    visit(partial, tail);
  } else if (reg.width) {
    visit(reg, data);
  }
  if (modifiers.memory_result_has_status)
    visit(RegisterRef{reg.cls, last, 1}, 0xf);
}

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
