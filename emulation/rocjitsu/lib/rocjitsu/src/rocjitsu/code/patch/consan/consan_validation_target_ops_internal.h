// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_validation_target_ops_internal.h
/// @brief Family-owned encoded-mutation proofs behind final validation.

#pragma once

#include "rocjitsu/code/patch/consan/consan_validation_target_ops.h"

namespace rocjitsu::consan_validation_target_detail {

[[nodiscard]] ConSanEncodedMutationValidation
validate_gfx12_ordinary_global_address_mutation(std::span<const uint8_t> before,
                                                std::span<const uint8_t> after);
[[nodiscard]] ConSanEncodedMutationValidation
validate_gfx12_ordinary_global_scope_mutation(std::span<const uint8_t> before,
                                              std::span<const uint8_t> after);
[[nodiscard]] ConSanEncodedMutationValidation
validate_gfx12_atomic_address_mutation(std::span<const uint8_t> before,
                                       std::span<const uint8_t> after);
[[nodiscard]] ConSanEncodedMutationValidation
validate_gfx12_atomic_scope_mutation(std::span<const uint8_t> before,
                                     std::span<const uint8_t> after);

} // namespace rocjitsu::consan_validation_target_detail
