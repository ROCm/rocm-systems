// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"

#include <optional>
#include <span>
#include <vector>

namespace rocjitsu::consan_moi_impl {

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
[[nodiscard]] std::optional<ConSanMoiSampledSyncScope> sampled_atomic_scope(uint32_t raw_scope);
[[nodiscard]] bool sampled_atomic_guest_preserves_address(const ConSanAtomicLoweringForm &form);
[[nodiscard]] bool
sampled_atomic_spill_overlaps_guest_operands(const VgprSpillSequence &spill,
                                             const ConSanAtomicLoweringForm &form);

[[nodiscard]] bool append_sampled_indexed_address(std::vector<uint32_t> &words,
                                                  uint64_t first_address, uint32_t element_size,
                                                  uint16_t index_vgpr, uint16_t address_vgpr,
                                                  uint16_t temporary_vgpr, rj_code_arch_t arch);

[[nodiscard]] bool
append_sampled_window_bank_index(std::vector<uint32_t> &words, const ConSanMoiOperatingPoint &point,
                                 const BoundRuntimeResources &resources,
                                 const ConSanMoiWorkgroupSources &workgroup_sources,
                                 uint32_t bank_count, uint16_t bank_vgpr, uint16_t temporary_vgpr,
                                 uint16_t owner_vgpr, rj_code_arch_t arch);

[[nodiscard]] std::optional<std::vector<uint32_t>> build_sampled_pending_acquire_cave_words(
    std::span<const uint8_t> bytes, const consan_detail::MoiAtomicEvidenceSitePlan &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const ConSanRequest &request,
    const BoundRuntimeResources &bound_resources, const ConSanMoiOperatingPoint &point,
    uint16_t scratch_vgpr, const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    const MoiPrivateEpochLayout *private_layout, rj_code_arch_t arch, uint32_t selected_slot,
    uint32_t bank_count, std::optional<uint32_t> release_selected_slot,
    const ConSanMoiReportBufferLayout &layout, uint64_t cave_text_offset,
    uint64_t return_text_offset, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr,
    std::span<const uint32_t> trailing_guest_words = {});

[[nodiscard]] std::optional<std::vector<uint32_t>> build_sampled_atomic_sync_cave_words(
    std::span<const uint8_t> bytes, const consan_detail::MoiAtomicEvidenceSitePlan &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const ConSanRequest &request,
    const BoundRuntimeResources &bound_resources, const ConSanMoiOperatingPoint &point,
    uint16_t scratch_vgpr, const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    const MoiPrivateEpochLayout *private_layout, rj_code_arch_t arch, uint32_t selected_slot,
    uint32_t bank_count, const ConSanMoiReportBufferLayout &layout, uint64_t cave_text_offset,
    uint64_t return_text_offset, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr,
    std::span<const uint32_t> trailing_guest_words = {}, bool preserve_guest_at_anchor = false,
    std::span<const uint32_t> leading_guest_words = {});

} // namespace rocjitsu::consan_moi_impl
