// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_sampled_access_emission.h"

#include <optional>
#include <span>
#include <vector>

namespace rocjitsu::consan_moi_impl {

/// Exact owner-local state shared by Sampled barrier and atomic emission.
/// Planning resolves every broad request, resource, and operating-point fact
/// before any cave is sized; every direct, dense, or ordinary route consumes
/// this immutable product.
struct MoiSampledSyncEmissionPlan {
  uint64_t report_buffer_address = 0;
  uint64_t report_generation = 0;
  std::optional<uint16_t> exec_save_sgpr;
  bool automatic_private_epoch = false;
  bool workitem_owner = false;
  ConSanMoiPersistentSgprState persistent_sgprs;
  MoiScalarAbiPlan scalar_abi;
  consan_moi_detail::ConSanMoiReportDispatchIdSource dispatch_id;
  ConSanMoiWorkgroupSources workgroup_sources;
  ConSanMoiOwnerEpochVgprSources owner_epoch_vgprs;
  uint16_t scratch_vgpr = 0;
};

/// Sampled atomic scratch ABI owned jointly by its planner and emitter.
struct SampledAtomicScratchLayout {
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

[[nodiscard]] constexpr uint16_t sampled_atomic_scratch_count() {
  return SampledAtomicScratchLayout::kCount;
}

[[nodiscard]] std::optional<ConSanMoiSampledSyncRole>
sampled_atomic_role(ConSanMoiAtomicEventKind kind, bool is_rmw);
[[nodiscard]] bool sampled_atomic_guest_preserves_address(const ConSanAtomicLoweringForm &form);
[[nodiscard]] bool
sampled_atomic_spill_overlaps_guest_operands(const VgprSpillSequence &spill,
                                             const ConSanAtomicLoweringForm &form);

[[nodiscard]] std::optional<std::vector<uint32_t>> build_sampled_pending_acquire_cave_words(
    std::span<const uint8_t> bytes, const consan_detail::MoiAtomicEvidenceSitePlan &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const MoiSampledSyncEmissionPlan &plan,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    const MoiPrivateEpochLayout *private_layout, rj_code_arch_t arch, uint32_t selected_slot,
    uint32_t bank_count, std::optional<uint32_t> release_selected_slot,
    const ConSanMoiReportBufferLayout &layout, uint64_t cave_text_offset,
    uint64_t return_text_offset, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr,
    std::span<const uint32_t> trailing_guest_words = {});

[[nodiscard]] std::optional<std::vector<uint32_t>> build_sampled_atomic_sync_cave_words(
    std::span<const uint8_t> bytes, const consan_detail::MoiAtomicEvidenceSitePlan &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const MoiSampledSyncEmissionPlan &plan,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    const MoiPrivateEpochLayout *private_layout, rj_code_arch_t arch, uint32_t selected_slot,
    uint32_t bank_count, const ConSanMoiReportBufferLayout &layout, uint64_t cave_text_offset,
    uint64_t return_text_offset, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr,
    std::span<const uint32_t> trailing_guest_words = {}, bool preserve_guest_at_anchor = false,
    std::span<const uint32_t> leading_guest_words = {});

} // namespace rocjitsu::consan_moi_impl
