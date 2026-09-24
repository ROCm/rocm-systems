// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_validation_rdna4_cdna5_target_ops.cpp
/// @brief Independent RDNA4/CDNA5 proofs for encoded fault mutations.

#include "rocjitsu/code/patch/consan/targets/rdna4/consan_atomic_observation.h"

#include "rocjitsu/code/patch/consan/targets/consan_validation_target_ops.h"

#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"

#include <array>
#include <cstring>

namespace rocjitsu::consan::validation_target_detail {
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

[[nodiscard]] EncodedMutationValidation mutation_validation(bool valid) {
  return valid ? EncodedMutationValidation::Valid : EncodedMutationValidation::InvalidMutation;
}

[[nodiscard]] bool valid_flat_address_mutation(const std::array<uint32_t, 3> &before,
                                               const std::array<uint32_t, 3> &after) {
  const int32_t before_offset = sign_extend_24(before[2] >> 8u);
  const int32_t after_offset = sign_extend_24(after[2] >> 8u);
  const int64_t delta = static_cast<int64_t>(after_offset) - before_offset;
  return before[0] == after[0] && before[1] == after[1] &&
         (before[2] & 0xffu) == (after[2] & 0xffu) && delta > 0 && delta <= 0x7fffff &&
         delta % static_cast<int64_t>(sizeof(uint32_t)) == 0;
}

[[nodiscard]] bool valid_scope_mutation(const std::array<uint32_t, 3> &before,
                                        const std::array<uint32_t, 3> &after) {
  const uint32_t expected_scope_word = before[1] & ~(0x3u << 18u);
  return before[0] == after[0] && after[1] == expected_scope_word && before[2] == after[2];
}

[[nodiscard]] bool valid_ds_address_mutation(const std::array<uint32_t, 3> &before,
                                             const std::array<uint32_t, 3> &after) {
  const int64_t delta = static_cast<int64_t>(after[0] & 0xffu) - (before[0] & 0xffu);
  return (after[0] & ~0xffu) == (before[0] & ~0xffu) && after[1] == before[1] && delta > 0 &&
         delta <= 0xff && delta % static_cast<int64_t>(sizeof(uint32_t)) == 0 &&
         (after[0] & 0xffu) % sizeof(uint32_t) == 0;
}

} // namespace

EncodedMutationValidation validate_rdna4_cdna5_encoded_mutation(EncodedMutationKind kind,
                                                                std::span<const uint8_t> before,
                                                                std::span<const uint8_t> after,
                                                                bool allow_atomic_observation) {
  std::optional<std::array<uint32_t, 3>> observed_before;
  if (allow_atomic_observation &&
      (kind == EncodedMutationKind::AtomicAddress || kind == EncodedMutationKind::AtomicScope) &&
      after.size() == sizeof(rdna4::VflatMachineInst)) {
    rdna4::VflatMachineInst relocated{};
    std::memcpy(&relocated, after.data(), sizeof(relocated));
    if (relocated.th == 1u) {
      observed_before = detail::build_rdna4_atomic_observation(before, relocated.vdst);
      if (observed_before)
        before = {reinterpret_cast<const uint8_t *>(observed_before->data()),
                  sizeof(*observed_before)};
    }
  }
  switch (kind) {
  case EncodedMutationKind::OrdinaryGlobalAddress: {
    if (!same_size(before, after, sizeof(rdna4::VglobalMachineInst)))
      return EncodedMutationValidation::UnexpectedInstructionSize;
    return mutation_validation(valid_flat_address_mutation(words(before), words(after)));
  }
  case EncodedMutationKind::OrdinaryGlobalScope: {
    if (!same_size(before, after, sizeof(rdna4::VglobalMachineInst)))
      return EncodedMutationValidation::UnexpectedInstructionSize;
    const auto original = words(before);
    const uint32_t scope = (original[1] >> 18u) & 0x3u;
    return mutation_validation((scope == 2u || scope == 3u) &&
                               valid_scope_mutation(original, words(after)));
  }
  case EncodedMutationKind::AtomicAddress: {
    const bool ds = same_size(before, after, sizeof(rdna4::VdsMachineInst));
    const bool flat = same_size(before, after, sizeof(rdna4::VflatMachineInst));
    if (!ds && !flat)
      return EncodedMutationValidation::UnexpectedInstructionSize;
    return mutation_validation(ds ? valid_ds_address_mutation(words(before), words(after))
                                  : valid_flat_address_mutation(words(before), words(after)));
  }
  case EncodedMutationKind::AtomicScope:
    if (same_size(before, after, sizeof(rdna4::VdsMachineInst)))
      return EncodedMutationValidation::UnsupportedInstructionEncoding;
    if (same_size(before, after, sizeof(rdna4::VflatMachineInst)))
      return mutation_validation(valid_scope_mutation(words(before), words(after)));
    return EncodedMutationValidation::UnexpectedInstructionSize;
  }
  return EncodedMutationValidation::UnsupportedInstructionEncoding;
}

} // namespace rocjitsu::consan::validation_target_detail
