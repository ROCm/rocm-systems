// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_fault_rdna4_cdna5_target_ops.cpp
/// @brief RDNA4/CDNA5 fault-classification and mutation recipes.

#include "rocjitsu/code/patch/consan/targets/consan_fault_target_ops_internal.h"

#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_semantic_classifiers.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "util/bit.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace rocjitsu::consan {

AtomicFaultEncoding
fault_target_detail::classify_rdna4_cdna5_atomic_fault_encoding(std::string_view mnemonic,
                                                                uint32_t size) {
  if ((mnemonic.starts_with("flat_atomic") || mnemonic.starts_with("global_atomic")) &&
      size == sizeof(rdna4::VflatMachineInst)) {
    return AtomicFaultEncoding::FlatLike;
  }
  if (mnemonic.starts_with("buffer_atomic") && size == sizeof(rdna4::VbufferMachineInst))
    return AtomicFaultEncoding::Buffer;
  if (mnemonic.starts_with("ds_") && is_ds_atomic(mnemonic) && lds_width_bits(mnemonic) == 32u &&
      size == sizeof(rdna4::VdsMachineInst)) {
    return AtomicFaultEncoding::Ds;
  }
  return AtomicFaultEncoding::Unsupported;
}

std::optional<OrdinaryGlobalFaultEncoding>
decode_ordinary_global_fault_encoding(std::span<const uint8_t> instruction) {
  if (instruction.size() != sizeof(rdna4::VglobalMachineInst))
    return std::nullopt;
  rdna4::VglobalMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  return OrdinaryGlobalFaultEncoding{
      .byte_offset = sign_extend_24(static_cast<uint32_t>(raw.ioffset)),
      .scope = static_cast<uint32_t>(raw.scope),
  };
}

bool rewrite_ordinary_global_fault_offset(std::span<uint8_t> instruction, int32_t byte_offset) {
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

bool rewrite_ordinary_global_fault_scope(std::span<uint8_t> instruction, uint32_t scope) {
  if (instruction.size() != sizeof(rdna4::VglobalMachineInst) || scope > 3u)
    return false;
  rdna4::VglobalMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  raw.scope = scope;
  std::memcpy(instruction.data(), &raw, sizeof(raw));
  return true;
}

AtomicFaultRewriteResult fault_target_detail::rewrite_rdna4_cdna5_atomic_fault_address(
    std::span<uint8_t> instruction, AtomicFaultEncoding encoding, uint32_t width_bits,
    uint32_t address_delta) {
  static_assert(sizeof(rdna4::VflatMachineInst) == sizeof(rdna4::VbufferMachineInst));
  const size_t expected_size = encoding == AtomicFaultEncoding::Ds
                                   ? sizeof(rdna4::VdsMachineInst)
                                   : sizeof(rdna4::VflatMachineInst);
  if (instruction.size() != expected_size)
    return {};
  std::array<uint32_t, 3> words{};
  std::memcpy(words.data(), instruction.data(), instruction.size());
  uint32_t previous_value = 0u;
  if (encoding == AtomicFaultEncoding::Ds) {
    previous_value = words[0] & 0xffu;
    const uint32_t replacement_offset = previous_value + address_delta;
    if (replacement_offset > 0xffu)
      return {.status = AtomicFaultRewriteStatus::OffsetOverflow};
    const uint32_t natural_alignment = std::max<uint32_t>(sizeof(uint32_t), width_bits / 8u);
    if (replacement_offset % natural_alignment != 0u)
      return {.status = AtomicFaultRewriteStatus::MisalignedOffset};
    words[0] = (words[0] & ~0xffu) | replacement_offset;
  } else if (encoding == AtomicFaultEncoding::FlatLike || encoding == AtomicFaultEncoding::Buffer) {
    const int32_t original_offset = sign_extend_24(words[2] >> 8u);
    const int64_t replacement_offset = static_cast<int64_t>(original_offset) + address_delta;
    if (replacement_offset > 0x7fffff)
      return {.status = AtomicFaultRewriteStatus::OffsetOverflow};
    previous_value = static_cast<uint32_t>(original_offset);
    words[2] = (words[2] & 0xffu) | ((static_cast<uint32_t>(replacement_offset) & 0xffffffu) << 8u);
  } else {
    return {};
  }
  std::memcpy(instruction.data(), words.data(), instruction.size());
  return {.status = AtomicFaultRewriteStatus::Rewritten, .previous_value = previous_value};
}

AtomicFaultRewriteResult
fault_target_detail::rewrite_rdna4_cdna5_atomic_fault_scope_to_wave(std::span<uint8_t> instruction,
                                                                    AtomicFaultEncoding encoding) {
  if (instruction.size() != sizeof(rdna4::VflatMachineInst) ||
      (encoding != AtomicFaultEncoding::FlatLike && encoding != AtomicFaultEncoding::Buffer)) {
    return {};
  }
  std::array<uint32_t, 3> words{};
  std::memcpy(words.data(), instruction.data(), instruction.size());
  const uint32_t original_scope = (words[1] >> 18u) & 0x3u;
  if (original_scope == 0u)
    return {.status = AtomicFaultRewriteStatus::AlreadyWaveScope};
  words[1] &= ~(0x3u << 18u);
  std::memcpy(instruction.data(), words.data(), instruction.size());
  return {.status = AtomicFaultRewriteStatus::Rewritten, .previous_value = original_scope};
}

} // namespace rocjitsu::consan
