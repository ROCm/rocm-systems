// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_fault_gfx12_target_ops.cpp
/// @brief GFX12 fault-classification and mutation recipes.

#include "rocjitsu/code/patch/consan/consan_fault_target_ops_internal.h"

#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_semantic_classifiers.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "util/bit.h"

#include <cstring>

namespace rocjitsu {

ConSanAtomicFaultEncoding
consan_fault_target_detail::classify_gfx12_atomic_fault_encoding(std::string_view mnemonic,
                                                                 uint32_t size) {
  if ((mnemonic.starts_with("flat_atomic") || mnemonic.starts_with("global_atomic")) &&
      size == sizeof(rdna4::VflatMachineInst)) {
    return ConSanAtomicFaultEncoding::FlatLike;
  }
  if (mnemonic.starts_with("buffer_atomic") && size == sizeof(rdna4::VbufferMachineInst))
    return ConSanAtomicFaultEncoding::Buffer;
  if (mnemonic.starts_with("ds_") && is_ds_atomic(mnemonic) && lds_width_bits(mnemonic) == 32u &&
      size == sizeof(rdna4::VdsMachineInst)) {
    return ConSanAtomicFaultEncoding::Ds;
  }
  return ConSanAtomicFaultEncoding::Unsupported;
}

std::optional<ConSanOrdinaryGlobalFaultEncoding>
decode_consan_ordinary_global_fault_encoding(std::span<const uint8_t> instruction) {
  if (instruction.size() != sizeof(rdna4::VglobalMachineInst))
    return std::nullopt;
  rdna4::VglobalMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  return ConSanOrdinaryGlobalFaultEncoding{
      .byte_offset = sign_extend_24(static_cast<uint32_t>(raw.ioffset)),
      .scope = static_cast<uint32_t>(raw.scope),
  };
}

bool rewrite_consan_ordinary_global_fault_offset(std::span<uint8_t> instruction,
                                                 int32_t byte_offset) {
  if (instruction.size() != sizeof(rdna4::VglobalMachineInst) || byte_offset < -0x800000 ||
      byte_offset > 0x7fffff) {
    return false;
  }
  rdna4::VglobalMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  raw.ioffset = static_cast<uint32_t>(byte_offset) & 0xffffffu;
  std::memcpy(instruction.data(), &raw, sizeof(raw));
  return true;
}

bool rewrite_consan_ordinary_global_fault_scope(std::span<uint8_t> instruction, uint32_t scope) {
  if (instruction.size() != sizeof(rdna4::VglobalMachineInst) || scope > 3u)
    return false;
  rdna4::VglobalMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  raw.scope = scope;
  std::memcpy(instruction.data(), &raw, sizeof(raw));
  return true;
}

} // namespace rocjitsu
