// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_access_emission.h"
#include "rocjitsu/code/patch/consan/consan_evidence_planning.h"
#include "rocjitsu/code/patch/consan/consan_internal.h"
#include "rocjitsu/code/patch/consan/consan_lowering_plan.h"
#include "rocjitsu/code/patch/consan/consan_report.h"

#include <optional>
#include <span>
#include <vector>

namespace rocjitsu::consan::detail {

/// Fully encoded semantic identity for one ConSan atomic synchronization
/// candidate. Physical aliases may fold only when every field matches.
struct AtomicSemantics {
  SyncRole role = SyncRole::None;
  SyncScope scope = SyncScope::None;
  SyncOutcome outcome = SyncOutcome::NotApplicable;
  uint32_t byte_count = 0;
  uint32_t descriptor = 0;
  std::optional<uint32_t> cas_failure_descriptor;

  bool operator==(const AtomicSemantics &) const = default;
};

} // namespace rocjitsu::consan::detail

namespace rocjitsu::consan::detail {

enum class AtomicSemanticsReason : uint8_t {
  None,
  UnqualifiedSharedSyncSequence,
  UnsupportedQualifiedMemoryRole,
  MissingQualifiedScope,
  UnsupportedQualifiedScope,
  UnsupportedQualifiedByteRange,
  CompareExchangeDynamicOutcomeUnavailable,
  UnsupportedQualifiedRmwOutcome,
  SyncAbiRejectedQualifiedSequence,
  SyncAbiRejectedCasFailure,
  Count,
};

struct AtomicSemanticsResult {
  std::optional<detail::AtomicSemantics> semantics;
  AtomicSemanticsReason reason = AtomicSemanticsReason::None;
};

[[nodiscard]] AtomicSemanticsResult
atomic_semantics_for_source(const AtomicEvidenceSourceView &source);

[[nodiscard]] std::string_view atomic_semantics_reason_name(AtomicSemanticsReason reason);

/// Exact owner-local state shared by ConSan barrier and atomic emission.
/// Planning resolves every broad request, resource, and operating-point fact
/// before any cave is sized; every direct, dense, or ordinary route consumes
/// this immutable product.
struct SyncEmissionPlan {
  uint64_t supercollider_report_buffer_address = 0;
  uint64_t report_generation = 0;
  std::optional<uint16_t> exec_save_sgpr;
  bool automatic_private_epoch = false;
  bool workitem_owner = false;
  PersistentSgprState persistent_sgprs;
  std::optional<SpecialStateSgprs> special_state;
  detail::ReportDispatchIdSource dispatch_id;
  WorkgroupSources workgroup_sources;
  OwnerEpochVgprSources owner_epoch_vgprs;
  uint16_t scratch_vgpr = 0;
  /// Selectable-bank state at a relocated polling-loop header. Such a loop is
  /// admitted only when its complete relocated span is proven to exit in bank
  /// zero, which is the state required by the following instrumentation.
  std::optional<uint16_t> polling_loop_entry_vgpr_bank_mode;
};

/// ConSan atomic scratch ABI owned jointly by its planner and emitter.
struct AtomicScratchLayout {
  static constexpr uint16_t kValue = 2u;
  static constexpr uint16_t kExpected = 3u;
  static constexpr uint16_t kCasFieldWords = 1u;
  static_assert(kCasFieldWords == 1u);
  static constexpr uint16_t kCasCompare = 4u;
  static constexpr uint16_t kCasResult = kCasCompare + kCasFieldWords;
  static constexpr uint16_t kOwner = kCasResult + kCasFieldWords;
  static constexpr uint16_t kEpoch = kOwner + 1u;
  static constexpr uint16_t kBank = kEpoch + 1u;
  static constexpr uint16_t kSavedAddress = kBank + 1u;
  static constexpr uint16_t kSavedAddressCount = 2u;
  static constexpr uint16_t kCount = kSavedAddress + kSavedAddressCount;
};

[[nodiscard]] constexpr uint16_t atomic_scratch_count() { return AtomicScratchLayout::kCount; }

[[nodiscard]] std::optional<SyncRole> atomic_role(AtomicEventKind kind, bool is_rmw);
[[nodiscard]] bool atomic_guest_preserves_address(const AtomicLoweringForm &form);
[[nodiscard]] bool atomic_spill_overlaps_guest_operands(const VgprSpillSequence &spill,
                                                        const AtomicLoweringForm &form);

[[nodiscard]] std::optional<std::vector<uint32_t>> build_pending_acquire_cave_words(
    std::span<const uint8_t> bytes, const AtomicEvidenceSourceView &source,
    const AtomicLoweringForm &lowering_form, uint64_t owner_descriptor_file_offset,
    const AtomicAddressPlan &address_plan, const SyncEmissionPlan &plan,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    const PrivateStateLayout *private_layout, rj_code_arch_t arch, uint32_t selected_slot,
    uint32_t bank_count, std::optional<uint32_t> release_selected_slot,
    const ReportBufferLayout &layout, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr,
    std::span<const uint32_t> leading_guest_words = {},
    std::span<const uint32_t> trailing_guest_words = {}, uint32_t *emitted_guest_size = nullptr);

[[nodiscard]] std::optional<std::vector<uint32_t>> build_atomic_sync_cave_words(
    std::span<const uint8_t> bytes, const AtomicEvidenceSourceView &source,
    const AtomicLoweringForm &lowering_form, uint64_t owner_descriptor_file_offset,
    const AtomicAddressPlan &address_plan, const SyncEmissionPlan &plan,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    const PrivateStateLayout *private_layout, rj_code_arch_t arch, uint32_t selected_slot,
    uint32_t bank_count, const ReportBufferLayout &layout, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr,
    std::span<const uint32_t> leading_guest_words = {},
    std::span<const uint32_t> trailing_guest_words = {}, uint32_t *emitted_guest_size = nullptr);

} // namespace rocjitsu::consan::detail
