// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file trampoline_builder.h
/// @brief Lowers a TrampolinePlan into patched-anchor bytes and trampoline
///        words for the DBI relocation-only path.
///
/// This is the byte emitter; it owns SOPP branch math and basic plan
/// well-formedness checks (original_size 4 or 8, original_words count
/// matches, branch ranges fit). It does not touch the ELF, does not own
/// layout assignment, and does not enforce milestone-scoped restrictions
/// (e.g. "only emit s_nop placeholder bodies" — that lives in the
/// orchestrator as `validate_inline_nop_plan` in instrumentor.h).
/// See code_object_patcher.h for the ELF mutation layer.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rocjitsu {

/// @brief Concrete instruction words placed before or after the relocated
///        original in the trampoline. Declared clobbers are intentionally
///        deferred to a later milestone.
struct InlineAsmItem {
  std::vector<uint32_t> words;
};

/// @brief Builder-facing description of one trampoline.
///
/// Coordinates are .text-relative byte offsets. The orchestrator fills this
/// after validation and layout, then hands it to TrampolineBuilder.
struct TrampolinePlan {
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;

  uint64_t anchor_offset = 0;
  uint32_t original_size = 0; // 4 or 8 for the inline-nop smoke build.
  uint64_t trampoline_offset = 0;
  uint64_t return_target = 0; // Typically anchor_offset + original_size.

  std::vector<uint32_t> original_words; // Exact bytes pulled from .text.

  std::vector<InlineAsmItem> before_items;
  std::vector<InlineAsmItem> after_items;
  bool emit_original = true;
};

/// @brief Output bytes for one trampoline.
struct TrampolineBytes {
  std::vector<uint8_t> patched_anchor_bytes; // original_size bytes.
  std::vector<uint32_t> trampoline_words;
};

class TrampolineBuilder {
public:
  /// @brief Lower @p plan to patched-anchor bytes and trampoline words.
  ///
  /// Returns std::nullopt and writes a human-readable explanation to
  /// @p error_out (if non-null) on:
  ///   - arch left at ROCJITSU_CODE_ARCH_INVALID (caller forgot to set it)
  ///   - original_size other than 4 or 8
  ///   - original_words size mismatch with original_size
  ///   - Forward or return branch outside s_branch simm16 range
  ///
  /// The builder does not enforce milestone-scoped restrictions on body
  /// shape; the orchestrator decides what kind of plan to emit and calls
  /// validate_inline_nop_plan (in instrumentor.h) when appropriate.
  [[nodiscard]] static std::optional<TrampolineBytes> build(const TrampolinePlan &plan,
                                                            std::string *error_out = nullptr);
};

} // namespace rocjitsu
