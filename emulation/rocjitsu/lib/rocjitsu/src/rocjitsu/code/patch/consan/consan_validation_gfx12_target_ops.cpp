// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_validation_gfx12_target_ops.cpp
/// @brief Independent GFX12 proofs for encoded fault mutations.

#include "rocjitsu/code/patch/consan/consan_validation_target_ops.h"

#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"

#include <array>
#include <cstring>

namespace rocjitsu::consan_validation_target_detail {
namespace {

[[nodiscard]] bool same_size(std::span<const uint8_t> before, std::span<const uint8_t> after,
                             size_t expected) {
  return before.size() == expected && after.size() == expected;
}

[[nodiscard]] std::array<uint32_t, 3> words(std::span<const uint8_t> bytes) {
  std::array<uint32_t, 3> result{};
  std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}

ConSanEncodedMutationValidation
validate_gfx12_ordinary_global_address_mutation(std::span<const uint8_t> before_bytes,
                                                std::span<const uint8_t> after_bytes) {
  if (!same_size(before_bytes, after_bytes, sizeof(rdna4::VglobalMachineInst)))
    return ConSanEncodedMutationValidation::UnexpectedInstructionSize;
  const auto before = words(before_bytes);
  const auto after = words(after_bytes);
  const int32_t before_offset = sign_extend_24(before[2] >> 8u);
  const int32_t after_offset = sign_extend_24(after[2] >> 8u);
  const int64_t delta = static_cast<int64_t>(after_offset) - before_offset;
  const bool valid = before[0] == after[0] && before[1] == after[1] &&
                     (before[2] & 0xffu) == (after[2] & 0xffu) && delta > 0 && delta <= 0x7fffff &&
                     delta % static_cast<int64_t>(sizeof(uint32_t)) == 0;
  return valid ? ConSanEncodedMutationValidation::Valid
               : ConSanEncodedMutationValidation::InvalidMutation;
}

ConSanEncodedMutationValidation
validate_gfx12_ordinary_global_scope_mutation(std::span<const uint8_t> before_bytes,
                                              std::span<const uint8_t> after_bytes) {
  if (!same_size(before_bytes, after_bytes, sizeof(rdna4::VglobalMachineInst)))
    return ConSanEncodedMutationValidation::UnexpectedInstructionSize;
  const auto before = words(before_bytes);
  const auto after = words(after_bytes);
  const uint32_t original_scope = (before[1] >> 18u) & 0x3u;
  const uint32_t expected_scope_word = before[1] & ~(0x3u << 18u);
  const bool valid = (original_scope == 2u || original_scope == 3u) && before[0] == after[0] &&
                     after[1] == expected_scope_word && before[2] == after[2];
  return valid ? ConSanEncodedMutationValidation::Valid
               : ConSanEncodedMutationValidation::InvalidMutation;
}

ConSanEncodedMutationValidation
validate_gfx12_atomic_address_mutation(std::span<const uint8_t> before_bytes,
                                       std::span<const uint8_t> after_bytes) {
  const bool ds = same_size(before_bytes, after_bytes, sizeof(rdna4::VdsMachineInst));
  const bool flat = same_size(before_bytes, after_bytes, sizeof(rdna4::VflatMachineInst));
  if (!ds && !flat)
    return ConSanEncodedMutationValidation::UnexpectedInstructionSize;
  const auto before = words(before_bytes);
  const auto after = words(after_bytes);
  bool valid = false;
  if (ds) {
    const int64_t delta = static_cast<int64_t>(after[0] & 0xffu) - (before[0] & 0xffu);
    valid = (after[0] & ~0xffu) == (before[0] & ~0xffu) && after[1] == before[1] && delta > 0 &&
            delta <= 0xff && delta % static_cast<int64_t>(sizeof(uint32_t)) == 0 &&
            (after[0] & 0xffu) % sizeof(uint32_t) == 0;
  } else {
    const int32_t before_offset = sign_extend_24(before[2] >> 8u);
    const int32_t after_offset = sign_extend_24(after[2] >> 8u);
    const int64_t delta = static_cast<int64_t>(after_offset) - before_offset;
    valid = before[0] == after[0] && before[1] == after[1] &&
            (after[2] & 0xffu) == (before[2] & 0xffu) && delta > 0 && delta <= 0x7fffff &&
            delta % static_cast<int64_t>(sizeof(uint32_t)) == 0;
  }
  return valid ? ConSanEncodedMutationValidation::Valid
               : ConSanEncodedMutationValidation::InvalidMutation;
}

ConSanEncodedMutationValidation
validate_gfx12_atomic_scope_mutation(std::span<const uint8_t> before_bytes,
                                     std::span<const uint8_t> after_bytes) {
  if (same_size(before_bytes, after_bytes, sizeof(rdna4::VdsMachineInst)))
    return ConSanEncodedMutationValidation::UnsupportedInstructionEncoding;
  if (!same_size(before_bytes, after_bytes, sizeof(rdna4::VflatMachineInst)))
    return ConSanEncodedMutationValidation::UnexpectedInstructionSize;
  const auto before = words(before_bytes);
  const auto after = words(after_bytes);
  const uint32_t expected_scope_word = before[1] & ~(0x3u << 18u);
  const bool valid =
      after[0] == before[0] && after[1] == expected_scope_word && after[2] == before[2];
  return valid ? ConSanEncodedMutationValidation::Valid
               : ConSanEncodedMutationValidation::InvalidMutation;
}

} // namespace

ConSanEncodedMutationValidation validate_gfx12_encoded_mutation(ConSanEncodedMutationKind kind,
                                                                std::span<const uint8_t> before,
                                                                std::span<const uint8_t> after) {
  switch (kind) {
  case ConSanEncodedMutationKind::OrdinaryGlobalAddress:
    return validate_gfx12_ordinary_global_address_mutation(before, after);
  case ConSanEncodedMutationKind::OrdinaryGlobalScope:
    return validate_gfx12_ordinary_global_scope_mutation(before, after);
  case ConSanEncodedMutationKind::AtomicAddress:
    return validate_gfx12_atomic_address_mutation(before, after);
  case ConSanEncodedMutationKind::AtomicScope:
    return validate_gfx12_atomic_scope_mutation(before, after);
  }
  return ConSanEncodedMutationValidation::UnsupportedInstructionEncoding;
}

} // namespace rocjitsu::consan_validation_target_detail
