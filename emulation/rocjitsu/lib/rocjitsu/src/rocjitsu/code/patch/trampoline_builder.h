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

#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/register_set.h"

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

  //----------------------------------------------------------------------------
  // Probe-call resources (filled by plan_probe_call(); left at defaults for the
  // inlined nop).
  //
  // These are the call-envelope resource decisions made BEFORE layout is known:
  // which registers the envelope uses and how many words it spans. Emission
  // materializes `before_items` from these decisions plus the assigned layout,
  // so it must not re-pick registers or recount words. Folded into
  // TrampolinePlan for now since this is the builder's one input;
  // lift back out into a dedicated resource-plan type if it grows unwieldy.
  //----------------------------------------------------------------------------
  bool is_probe_call = false;     ///< True once plan_probe_call() populated these.
  uint16_t link_pair_base = 30;   ///< Return-link pair, derived from the probe cc.
  uint16_t target_pair_base = 0;  ///< Dead even SGPR pair holding the probe address.
  bool preserve_scc = true;       ///< v0 preserves SCC across target materialization.
  uint16_t scc_temp = 0;          ///< Dead SGPR holding saved SCC across the call.
  RegisterSet builder_clobbers;   ///< {link} | {target pair} | {scc_temp}; feeds the spill formula.
  uint32_t before_word_count = 0; ///< Envelope words emitted before the relocated original.
  uint64_t probe_target_offset = 0; ///< .text-relative byte offset of the copied probe body.
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

  /// @brief Select the probe-call envelope resources and record them on @p plan.
  ///
  /// Picks the call-envelope registers and computes the envelope word count
  /// without choosing layout or emitting bytes. On success, fills
  /// `plan.is_probe_call`, `link_pair_base`, `target_pair_base`, `preserve_scc`,
  /// `scc_temp`, `builder_clobbers`, and `before_word_count`, then returns true.
  ///
  /// Policy:
  ///   - Link pair is derived from @p cc via link_pair_for(); an unknown
  ///     convention fails. If either lane of the derived pair is live at the
  ///     anchor, fail. Extending the supported conventions is deferred.
  ///   - Target-address pair is a dead, even-aligned SGPR pair (excluding the
  ///     link pair). It is consumed by s_swappc before the probe body runs, so it
  ///     may overlap @p probe_body_clobbers.
  ///   - SCC is preserved with one dead SGPR temp. The temp lives across the call
  ///     (saved before materialization, restored after), so it must avoid both
  ///     the live set and @p probe_body_clobbers. Extending this is deferred.
  ///
  /// Returns false and writes a diagnostic naming the unavailable resource to
  /// @p error_out (if non-null) when @p cc is unknown, the link pair is live, or
  /// no dead target pair / SCC temp can be found. The plan is left unmodified on
  /// failure.
  ///
  /// @param plan                Trampoline plan whose resource fields are filled.
  /// @param cc                  Probe calling convention; sets the link pair.
  /// @param live_at_anchor      Registers live immediately before the anchor.
  /// @param probe_body_clobbers Ordinary registers the copied probe body writes.
  [[nodiscard]] static bool plan_probe_call(TrampolinePlan &plan, ProbeCallingConvention cc,
                                            const RegisterSet &live_at_anchor,
                                            const RegisterSet &probe_body_clobbers,
                                            std::string *error_out = nullptr);

  /// @brief Emit a planned probe call: envelope, relocated original, return
  ///        branch.
  ///
  /// Requires @p plan.is_probe_call (i.e. plan_probe_call() succeeded). Builds
  /// the call envelope from the planned resources and @p plan.probe_target_offset
  /// then delegates to build() for layout and branch math.
  ///
  /// Returns std::nullopt and writes a diagnostic to @p error_out (if non-null)
  /// when the plan is not a probe call, the synthesized envelope size disagrees
  /// with the planned before_word_count (plan/emit drift), or build() reports a
  /// branch-range failure.
  [[nodiscard]] static std::optional<TrampolineBytes>
  emit_probe_call(const TrampolinePlan &plan, std::string *error_out = nullptr);
};

} // namespace rocjitsu
