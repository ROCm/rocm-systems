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
#include <span>
#include <string>
#include <utility>
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

/// @brief Physical placement selected for one DBI patch body.
enum class DbiPatchPlacementKind : uint8_t {
  Inline,
  LocalCave,
  AppendedCave,
};

struct DbiPatchLocalCave {
  uint64_t offset = 0;
  uint64_t capacity = 0;
};

struct DbiPatchPlacementRequest {
  uint64_t anchor_offset = 0;
  uint32_t original_size = 0;
  uint64_t body_size = 0;
  uint64_t inline_capacity = 0;
  std::optional<DbiPatchLocalCave> local_cave;
  bool allow_appended_cave = true;
};

struct DbiPatchPlacement {
  DbiPatchPlacementKind kind = DbiPatchPlacementKind::Inline;
  uint64_t anchor_offset = 0;
  uint32_t original_size = 0;
  uint64_t body_offset = 0;
  uint64_t body_size = 0;
  uint64_t return_branch_offset = 0;
  uint64_t return_target = 0;
};

/// @brief Transactional placement allocator shared by DBI probe families.
///
/// The planner owns overlap accounting and appended-cave cursor movement. A
/// successful trampoline reservation includes its four-byte return branch, so
/// later placements and final emitters use the same coordinates. Failed
/// requests do not mutate the planner.
class DbiPatchPlacementPlanner {
public:
  DbiPatchPlacementPlanner(rj_code_arch_t arch, uint64_t original_text_size);

  [[nodiscard]] std::optional<DbiPatchPlacement> plan(const DbiPatchPlacementRequest &request,
                                                      std::string *error_out = nullptr);

  [[nodiscard]] uint64_t appended_end() const { return appended_cursor_; }
  [[nodiscard]] std::span<const std::pair<uint64_t, uint64_t>> occupied_ranges() const {
    return occupied_ranges_;
  }

private:
  [[nodiscard]] bool range_is_free(uint64_t begin, uint64_t end) const;
  void reserve_range(uint64_t begin, uint64_t end);

  rj_code_arch_t arch_ = ROCJITSU_CODE_ARCH_INVALID;
  uint64_t original_text_size_ = 0;
  uint64_t appended_cursor_ = 0;
  std::vector<std::pair<uint64_t, uint64_t>> occupied_ranges_;
};

} // namespace rocjitsu
