// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_validation_target_ops.h
/// @brief Target-normalized independent validation of encoded mutations.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <span>

namespace rocjitsu {

enum class ConSanEncodedMutationValidation : uint8_t {
  Valid,
  UnexpectedInstructionSize,
  UnsupportedInstructionEncoding,
  InvalidMutation,
};

enum class ConSanEncodedMutationKind : uint8_t {
  OrdinaryGlobalAddress,
  OrdinaryGlobalScope,
  AtomicAddress,
  AtomicScope,
};

/// Target-neutral inputs needed to prove and normalize descriptor resource
/// deltas. Concrete descriptor-field encodings remain in target-owned
/// implementations.
struct ConSanDescriptorResourceDeltaInput {
  uint32_t original_rsrc1 = 0;
  uint32_t replacement_rsrc1 = 0;
  uint32_t original_rsrc3 = 0;
  uint32_t replacement_rsrc3 = 0;
  uint16_t required_vgpr_count = 0;
  bool original_accumulator_bank_proven_empty = false;
  bool allow_resource_delta = false;
  bool normalize_resource_delta = false;
};

struct ConSanDescriptorResourceDeltaValidation {
  bool valid = true;
  uint32_t normalized_rsrc3 = 0;
};

[[nodiscard]] ConSanEncodedMutationValidation
validate_consan_encoded_mutation(rj_code_arch_t arch, ConSanEncodedMutationKind kind,
                                 std::span<const uint8_t> before, std::span<const uint8_t> after);

/// Prove any target-specific register-allocation boundary movement and return
/// the RSRC3 value that common whole-descriptor comparison should use.
[[nodiscard]] ConSanDescriptorResourceDeltaValidation
validate_consan_descriptor_resource_delta(rj_code_arch_t arch,
                                          const ConSanDescriptorResourceDeltaInput &input);

} // namespace rocjitsu
