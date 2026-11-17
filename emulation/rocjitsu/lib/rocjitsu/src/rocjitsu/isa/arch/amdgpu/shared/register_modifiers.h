// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_REGISTER_MODIFIERS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_REGISTER_MODIFIERS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/register_set.h"

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
  if constexpr (requires { inst.inst_.tfe; })
    modifiers.buffer_tfe = inst.inst_.tfe;
  if constexpr (requires { inst.inst_.format; })
    modifiers.encoded_buffer_format = inst.inst_.format;
  if constexpr (requires { inst.soffset; })
    modifiers.buffer_offset = &inst.soffset;
  if constexpr (ResultBytes != 0xf)
    modifiers.memory_result_bytes = ResultBytes;
  if constexpr (LastResultBytes != 0xf)
    modifiers.memory_result_last_bytes = LastResultBytes;
}

// Visit contiguous result words with the same byte mask. The payload may end
// in a packed or high-half result before a full-DWORD TFE/LWE status word.
template <typename Visit>
void for_each_memory_result_range(RegisterRef reg, const RegisterModifiers &modifiers,
                                  bool sram_ecc, Visit visit) {
  if (reg.width == 0)
    return;
  const uint8_t bytes = modifiers.memory_result_bytes;
  const uint8_t tail = modifiers.memory_result_last_bytes;
  if (reg.cls != RegClass::VGPR || sram_ecc || (bytes == 0xf && tail == 0xf)) {
    visit(reg, 0xf);
    return;
  }
  const uint8_t payload = modifiers.memory_result_payload_words
                              ? std::min(reg.width, modifiers.memory_result_payload_words)
                              : reg.width;
  const uint8_t suffix = reg.width - payload;
  reg.width = payload;
  if (payload && bytes == tail) {
    visit(reg, bytes);
  } else if (payload) {
    if (--reg.width)
      visit(reg, bytes);
    visit(RegisterRef{reg.cls, static_cast<uint16_t>(reg.index + reg.width), 1}, tail);
  }
  if (suffix)
    visit(RegisterRef{reg.cls, static_cast<uint16_t>(reg.index + payload), suffix}, 0xf);
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_REGISTER_MODIFIERS_H_
