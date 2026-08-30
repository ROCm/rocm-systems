// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_validation_target_ops.cpp
/// @brief Narrow registry for independent encoded-mutation validation.

#include "rocjitsu/code/patch/consan/consan_validation_target_ops.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_validation_target_ops_internal.h"

namespace rocjitsu {
ConSanEncodedMutationValidation validate_consan_encoded_mutation(rj_code_arch_t arch,
                                                                 ConSanEncodedMutationKind kind,
                                                                 std::span<const uint8_t> before,
                                                                 std::span<const uint8_t> after) {
  if (consan_uses_gfx12_encoding(arch))
    return consan_validation_target_detail::validate_gfx12_encoded_mutation(kind, before, after);
  return ConSanEncodedMutationValidation::UnsupportedInstructionEncoding;
}

} // namespace rocjitsu
