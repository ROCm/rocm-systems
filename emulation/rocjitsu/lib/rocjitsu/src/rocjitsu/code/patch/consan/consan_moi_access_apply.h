// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_access_apply.h
/// @brief Shared byte-application transaction for MOI access engines.

#pragma once

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_moi_common_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_shared_lowering.h"
#include "rocjitsu/code/patch/consan/consan_placement.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu::consan_moi_impl {

using consan_moi_detail::append_word_bytes;
using consan_moi_detail::append_words_bytes;

/// Bind common original-program attribution to one mode-owned runtime map.
///
/// Shared access application owns the intent/semantic-site join and committed
/// patch geometry. Each mode supplies the expected access intent and a
/// callable that creates its typed runtime evidence representation from the
/// common attribution. The callable closes over mode-local placement facts;
/// common code never needs to add them to the generic patch product.
[[nodiscard]] std::optional<ConSanStaticAccessAttribution> make_moi_access_attribution(
    const ConSanObservationPlan &observation, const ConSanMoiCandidate &candidate,
    const ConSanPatchLoweringProduct &patch, ConSanProbeIntentKind expected_intent);

/// Complete target-state and composition contract for one relocated MOI
/// access body.
///
/// The access planners decide guest placement, preservation, and target-state
/// requirements before calling `assemble_moi_appended_body`. This value then
/// gives the common assembler everything needed to establish the probe's
/// required register-bank state, preserve scalar/vector scratch, place the
/// relocated guest instruction, and restore the incoming state. It contains no
/// engine evidence policy and owns no storage; every span and output pointer is
/// valid only for the duration of the assembly call.
struct MoiAppendedBodyOptions {
  explicit MoiAppendedBodyOptions(rj_code_arch_t arch) : arch(arch) {}

  /// Target whose native instruction builders assemble this body.
  rj_code_arch_t arch;
  /// Guest words emitted after the assembled body by its caller.
  size_t deferred_guest_word_count = 0u;
  /// Scalar preservation sequences surrounding the probe body.
  std::span<const uint32_t> scalar_save_words = {};
  std::span<const uint32_t> scalar_restore_words = {};
  /// Selectable-VGPR-bank state observed on entry. Absence and zero both need
  /// no transition; a nonzero value is valid only on a target whose profile
  /// advertises selectable banks.
  std::optional<uint16_t> incoming_vgpr_bank_mode = std::nullopt;
  /// Guest address operand to copy before selecting the low instrumentation
  /// bank, and the low-bank scratch register that receives the copy. They are
  /// either both present or both absent.
  std::optional<uint16_t> high_bank_address_source = std::nullopt;
  std::optional<uint16_t> high_bank_address_destination = std::nullopt;
  /// Probe suffix that must remain after vector-spill restoration.
  size_t trailing_guest_word_count = 0u;
  /// Already-planned entry gate or preservation words prepended to the body.
  std::span<const uint32_t> entry_prefix_words = {};
  /// Guest location within `probe_words`, if the probe embeds it.
  std::optional<uint32_t> probe_guest_instruction_offset = std::nullopt;
  std::optional<size_t> guest_instruction_word_count = std::nullopt;
  /// Receives the final byte offset of the relocated guest within the body.
  uint32_t *body_guest_instruction_offset = nullptr;
  /// Whether overlapping guest operands require restoring vector spill state
  /// before the trailing probe suffix rather than at the ordinary exit.
  bool restore_vgpr_spill_before_trailing_guest = false;
  /// Whether the guest instruction itself must run in the incoming selectable
  /// bank while the surrounding instrumentation runs in the low bank.
  bool wrap_embedded_guest_vgpr_bank = false;
};

[[nodiscard]] constexpr bool
moi_appended_body_uses_selectable_vgpr_bank(rj_code_arch_t arch,
                                            std::optional<uint16_t> incoming_vgpr_bank_mode) {
  return consan_arch_has_selectable_vgpr_bank(arch) && incoming_vgpr_bank_mode.value_or(0u) != 0u;
}

[[nodiscard]] constexpr bool
moi_appended_body_vgpr_bank_mode_is_valid(rj_code_arch_t arch,
                                          std::optional<uint16_t> incoming_vgpr_bank_mode) {
  return incoming_vgpr_bank_mode.value_or(0u) == 0u || consan_arch_has_selectable_vgpr_bank(arch);
}

[[nodiscard]] constexpr size_t
moi_appended_body_vgpr_bank_transition_word_count(rj_code_arch_t arch,
                                                  std::optional<uint16_t> incoming_vgpr_bank_mode,
                                                  bool has_vgpr_spill, bool wraps_embedded_guest) {
  if (!moi_appended_body_uses_selectable_vgpr_bank(arch, incoming_vgpr_bank_mode))
    return 0u;
  return (has_vgpr_spill ? 4u : 2u) + (wraps_embedded_guest ? 2u : 0u);
}

[[nodiscard]] constexpr bool
moi_embedded_guest_vgpr_bank_plan_is_valid(bool wraps_embedded_guest, bool selects_low_vgpr_bank,
                                           bool has_guest_instruction,
                                           size_t trailing_guest_word_count) {
  return !wraps_embedded_guest ||
         (selects_low_vgpr_bank && has_guest_instruction && trailing_guest_word_count == 0u);
}

struct MoiAppendedBodyPatchPlan {
  const VgprSpillSequence *spill = nullptr;
  std::span<const uint32_t> displaced_tail_words;
  /// Zero for position-independent assembly before whole-object placement.
  uint64_t body_size = 0;
  uint32_t guest_instruction_size = 0;
};

[[nodiscard]] std::optional<std::vector<uint32_t>>
assemble_moi_appended_body(const MoiAppendedBodyPatchPlan &plan,
                           std::span<const uint32_t> probe_words, std::string_view probe_name,
                           std::vector<std::string> &errors, const MoiAppendedBodyOptions &options);

[[nodiscard]] bool
moi_scalar_spill_requires_dynamic_vgpr_frame(const ProgramInventory &inventory,
                                             const ResolvedMoiScratchPlan &resources,
                                             const ConSanMoiOperatingPoint &point);

[[nodiscard]] std::optional<SgprSpillSequence> build_moi_sgpr_spill_sequence(
    const ProgramInventory &inventory, const ResolvedMoiScratchPlan &resources,
    const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
    const ConSanMoiOperatingPoint &point, const MoiObjectModeSemantics &mode_semantics,
    MoiSpillManagers &managers, rj_code_arch_t arch, std::vector<std::string> &warnings,
    std::optional<uint32_t> private_layout_base, const VgprSpillSequence *vgpr_spill = nullptr);

} // namespace rocjitsu::consan_moi_impl
