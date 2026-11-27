// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file dot_packed16.h
/// @brief Materialize packed WMMA inline operands; numerical execution lives in rj-arith.

#include <cstdint>

namespace rocjitsu::amdgpu::dot_packed16 {

/// @brief Broadcast an inline constant's 16-bit encoding into both halves.
/// @details Integer constants use raw low bits; float constants are narrowed by
/// format. Selectors 240 through 248 contain only finite normal FP32 constants,
/// so their F16 encodings are also finite normal values.
/// This helper is not a general converter for literal or SGPR sources.
template <bool Bf16> inline uint32_t inline_word(uint32_t raw, uint16_t selector) {
  uint16_t bits = uint16_t(raw);
  if (selector >= 240 && selector <= 248) {
    if constexpr (Bf16) {
      bits = uint16_t(raw >> 16);
    } else {
      // Round the discarded 13 fraction bits to even, then rebias the exponent.
      const uint32_t rounded = (raw & 0x7fffffffU) + 0xfffU + ((raw >> 13) & 1U);
      bits = uint16_t(((raw >> 16) & 0x8000U) | ((rounded >> 13) - ((127 - 15) << 10)));
    }
  }
  return uint32_t(bits) | (uint32_t(bits) << 16);
}
} // namespace rocjitsu::amdgpu::dot_packed16
