// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_inline_register_layout.h
/// @brief Named scratch-register ABI shared by InlineShadow emission and validation.

#pragma once

#include <cstdint>

namespace rocjitsu::consan_moi_impl::inline_register_layout {

struct ExactShadow {
  static constexpr uint16_t dispatch_id_low = 2u;
  static constexpr uint16_t dispatch_id_high = 3u;
  static constexpr uint16_t diagnostic_claim_tuple = 14u;
};

struct AtomicCausalSnapshot {
  static constexpr uint16_t count_aligned = 5u;
  static constexpr uint16_t address_aligned = 6u;
  static constexpr uint16_t address_unaligned = 5u;
  static constexpr uint16_t count_unaligned = 7u;
  static constexpr uint16_t flags = 8u;
  static constexpr uint16_t owners = 9u;
  static constexpr uint16_t epochs = 13u;
  static constexpr uint16_t token_version_before = 17u;
  static constexpr uint16_t producer = 18u;
  static constexpr uint16_t producer_epoch = 19u;
  static constexpr uint16_t temporary = 20u;
  static constexpr uint16_t loop_index = 21u;

  [[nodiscard]] static constexpr uint16_t count(bool aligned_cas_pair) {
    return aligned_cas_pair ? count_aligned : count_unaligned;
  }

  [[nodiscard]] static constexpr uint16_t address(bool aligned_cas_pair) {
    return aligned_cas_pair ? address_aligned : address_unaligned;
  }
};

struct AtomicOrdering {
  static constexpr uint16_t release_version_before = 20u;
};

} // namespace rocjitsu::consan_moi_impl::inline_register_layout
