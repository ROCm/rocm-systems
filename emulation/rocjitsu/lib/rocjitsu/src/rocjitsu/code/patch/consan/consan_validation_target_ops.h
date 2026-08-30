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

[[nodiscard]] ConSanEncodedMutationValidation
validate_consan_encoded_mutation(rj_code_arch_t arch, ConSanEncodedMutationKind kind,
                                 std::span<const uint8_t> before, std::span<const uint8_t> after);

} // namespace rocjitsu
