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

/// @brief One saved register live at the anchor and clobbered by instrumentation,
///        with its stable per-lane byte offset in the DBI spill zone (assigned by
///        SpillManager). @ref cls selects the emit path in build_spill_bracket: a
///        VGPR spills straight to scratch; an AccVGPR likewise spills straight to
///        scratch via the CDNA `acc` bit; an SGPR is bridged through a VGPR lane
///        (writelane/readlane) because SGPRs cannot reach scratch directly.
///
/// VGPR, SGPR, and AccVGPR slots are held in separate vectors (TrampolinePlan::
/// vgpr_spills / sgpr_spills / acc_spills), so @ref cls is redundant with the owning
/// vector but records the register file explicitly for the emitter.
struct SpillSlot {
  RegClass cls = RegClass::VGPR; ///< Register file: VGPR/AccVGPR (direct) or SGPR (bridged).
  uint16_t reg = 0;              ///< Register index to save before / restore after the call.
  uint32_t byte_offset = 0;      ///< Per-lane scratch byte offset for this slot.
};

/// @brief One special register (EXEC/VCC/M0) saved to a dead SGPR temp before the
///        call and restored after. `operand` is its scalar-operand code
///        (arch-specific for M0).
struct SpecialStateSlot {
  uint16_t operand = 0;   ///< Scalar-operand code of the special register.
  uint16_t temp_base = 0; ///< Dead SGPR (pair base when width==2) holding the save.
  uint8_t width = 1;      ///< Register lanes: 2 for EXEC/VCC, 1 for M0.
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
  /// Upper bound (exclusive) for envelope/temp SGPR selection: the kernel's own
  /// allocation. find_free_sgpr* never picks an index >= this, so a temp cannot
  /// land past the kernel's .sgpr_count. Defaults to the conservative cross-ISA
  /// allocatable bound (no kernel-specific limit); the orchestrator narrows it to
  /// the patched kernel's actual count.
  uint32_t kernel_sgpr_count = REGISTER_SET_ALLOCATABLE_SGPRS;

  bool is_probe_call = false;    ///< True once plan_probe_call() populated these.
  uint16_t link_pair_base = 30;  ///< Return-link pair, derived from the probe cc.
  uint16_t target_pair_base = 0; ///< Dead even SGPR pair holding the probe address.
  bool preserve_scc = true;      ///< v0 preserves SCC across target materialization.
  uint16_t scc_temp = 0;         ///< Dead SGPR holding saved SCC across the call.

  // Special-state preservation: set by the orchestrator when the probe body
  // clobbers the register; plan_probe_call allocates a dead SGPR temp for each
  // and records it in special_state_saves (one plan/emit loop, not a branch each).
  bool preserve_exec = false;
  bool preserve_vcc = false;
  bool preserve_m0 = false;
  std::vector<SpecialStateSlot> special_state_saves; ///< Filled by plan_probe_call.
  RegisterSet builder_clobbers;     ///< {link} | {target pair} | {scc/special temps}; feeds spill.
  uint32_t before_word_count = 0;   ///< Envelope words emitted before the relocated original.
  uint64_t probe_target_offset = 0; ///< .text-relative byte offset of the copied probe body.

  /// VGPRs to save/restore around the call; emit_probe_call brackets each with a
  /// scratch_store before and a scratch_load after. Empty when nothing spills.
  std::vector<SpillSlot> vgpr_spills;

  /// SGPRs to save/restore, each bridged through `spill_bridge_vgpr`.
  std::vector<SpillSlot> sgpr_spills;

  /// Dead-at-anchor VGPR bridging SGPR<->scratch. Valid iff sgpr_spills non-empty.
  uint16_t spill_bridge_vgpr = 0;

  /// AccVGPRs to save/restore around the call, stored/loaded directly with the
  /// CDNA scratch `acc` bit (no bridge VGPR needed). CDNA-only. Empty otherwise.
  std::vector<SpillSlot> acc_spills;
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
  /// `plan.is_probe_call`, `link_pair_base`, `target_pair_base`, `scc_temp`,
  /// `special_state_saves`, `builder_clobbers`, and `before_word_count`, then
  /// returns true. (`preserve_scc`/`preserve_*` are inputs, read but not written.)
  ///
  /// Policy:
  ///   - Link pair is read off @p abi; one that fails is_valid_probe_abi()
  ///     fails here. If either lane of the pair is live at the
  ///     anchor, fail. Extending the supported conventions is deferred.
  ///   - Target-address pair is a dead, even-aligned SGPR pair (excluding the
  ///     link pair). It is consumed by s_swappc before the probe body runs, so it
  ///     may overlap @p probe_body_clobbers.
  ///   - SCC is preserved with one dead SGPR temp. The temp lives across the call
  ///     (saved before materialization, restored after), so it must avoid both
  ///     the live set and @p probe_body_clobbers. Extending this is deferred.
  ///   - EXEC/VCC/M0 are preserved when the corresponding plan.preserve_* flag is
  ///     set. The orchestrator sets preserve_exec/preserve_vcc unconditionally as a
  ///     conservative policy (the summary detects the special-state writes the
  ///     decoder exposes as operands, but always saving keeps correctness
  ///     independent of per-opcode implicit-def coverage) and gates preserve_m0 on
  ///     the probe's clobbers; EXEC is additionally preserved here when the site
  ///     spills (forced to -1 around the store/load). Each preserved register gets
  ///     its own dead SGPR temp (a pair for EXEC/VCC, single for M0) recorded in
  ///     plan.special_state_saves, drawn from the same dead pool as the SCC temp.
  ///
  /// Returns false and writes a diagnostic naming the unavailable resource to
  /// @p error_out (if non-null) when @p abi is unusable, the link pair is live,
  /// or no dead target pair / SCC temp can be found. The plan is left unmodified
  /// on failure.
  ///
  /// @param plan                Trampoline plan whose resource fields are filled.
  /// @param abi                 Probe ABI; supplies the link pair.
  /// @param live_at_anchor      Registers live immediately before the anchor.
  /// @param probe_body_clobbers Ordinary registers the copied probe body writes.
  [[nodiscard]] static bool plan_probe_call(TrampolinePlan &plan, const ProbeAbi &abi,
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
