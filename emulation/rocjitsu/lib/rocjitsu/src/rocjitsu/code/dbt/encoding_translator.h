// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file encoding_translator.h
/// @brief Shared types for the auto-generated encoding translation pipeline.
///
/// @details This hand-written header provides:
/// - Coherency model structs for each ISA generation (GFX940, GFX9, GFX12).
/// - Remap functions that translate coherency bits across ISA boundaries.
/// - TranslationResult: the output format for generated encode functions.
///
/// The auto-generated per-pair headers (e.g., encoding_cdna4_to_rdna4.h) include
/// this header and call the remap functions during the encode step. This header
/// is shared across all ISA pairs — only the remap functions specific to each
/// coherency model boundary are defined here.

#pragma once

#include <cstdint>

namespace rocjitsu {

/// @brief GFX940 (CDNA4) coherency bits: sc0, sc1, nt.
struct CoherencyGfx940 {
  uint8_t sc0; ///< System-coherent scope bit 0.
  uint8_t sc1; ///< System-coherent scope bit 1.
  uint8_t nt;  ///< Non-temporal hint.
};

/// @brief GFX9 (CDNA1-3) coherency bit: glc.
struct CoherencyGfx9 {
  uint8_t glc; ///< Globally coherent (bypass L1).
};

/// @brief GFX12 (RDNA4) coherency bits: scope, th.
struct CoherencyGfx12 {
  uint8_t scope; ///< Cache scope (0=CU, 1=SE, 2=device, 3=system).
  uint8_t th;    ///< Temporal hint (0=default, 3=non-temporal).
};

/// @brief Remap GFX940 coherency bits to GFX12.
///
/// @details Mapping: scope = (sc1 << 1) | sc0; th = nt ? 0x3 : 0x0.
///
/// @param c  GFX940 coherency bits.
/// @returns Equivalent GFX12 coherency bits.
inline constexpr CoherencyGfx12 remap_gfx940_to_gfx12(CoherencyGfx940 c) {
  return {static_cast<uint8_t>((c.sc1 << 1) | c.sc0), c.nt ? uint8_t(0x3) : uint8_t(0x0)};
}

/// @brief Remap GFX9 coherency bits to GFX12.
///
/// @details Mapping: glc=1 → scope=SCOPE_SE (0x2); glc=0 → scope=0. th=0 always.
///
/// @param c  GFX9 coherency bits.
/// @returns Equivalent GFX12 coherency bits.
inline constexpr CoherencyGfx12 remap_gfx9_to_gfx12(CoherencyGfx9 c) {
  return {c.glc ? uint8_t(0x2) : uint8_t(0x0), uint8_t(0x0)};
}

/// @brief Result of encoding translation: up to 3 instruction words.
///
/// @details Returned by the generated per-pair translate functions (e.g.,
/// translate_encoding_cdna4_to_rdna4). word_count==0 indicates translation
/// failure (the caller falls back to copying the original encoding).
struct TranslationResult {
  uint32_t words[3]{};   ///< Encoded host instruction words.
  uint8_t word_count{0}; ///< Number of valid words (0 = translation failed).
};

} // namespace rocjitsu
