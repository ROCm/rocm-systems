// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"

namespace rocjitsu::consan_moi_impl {

/// Address-free private-memory layout for owner-local persistent MOI identity.
struct MoiPrivateEpochLayout {
  uint32_t epoch_offset = 0;
  std::optional<uint32_t> owner_offset;
  std::optional<uint32_t> workgroup_key_offset;
  std::optional<uint32_t> dispatch_id_offset;
  ConSanMoiPersistentWorkgroupPrivateOffsets record_replay_workgroup_offsets;
  uint32_t persistent_state_end = 0;
  uint32_t ephemeral_base = 0;
};

[[nodiscard]] std::optional<uint16_t> moi_descriptor_owner_shift(std::span<const uint8_t> image,
                                                                 uint64_t descriptor_file_offset,
                                                                 rj_code_arch_t arch,
                                                                 std::vector<std::string> &errors);

[[nodiscard]] std::optional<consan_detail::MoiSpecialStateSgprs>
moi_special_state_sgprs(const ConSanRequest &request, const ConSanMoiOperatingPoint &point);

[[nodiscard]] std::optional<ConSanMoiWorkgroupSources>
moi_persistent_or_descriptor_workgroup_sources(
    std::span<const uint8_t> image, uint64_t descriptor_file_offset, ConSanMoiEngine engine,
    const ConSanMoiOperatingPoint &point, rj_code_arch_t arch, std::vector<std::string> &errors,
    bool uses_cluster_workgroup_id = false);

[[nodiscard]] bool append_sampled_private_owner_epoch_load(
    std::vector<uint32_t> &words, std::span<const uint8_t> bytes, uint64_t descriptor_file_offset,
    const ConSanMoiOperatingPoint &point, const ConSanMoiOwnerEpochVgprSources &owner_epoch_vgprs,
    const MoiPrivateEpochLayout &layout, rj_code_arch_t arch, std::vector<std::string> &errors);

} // namespace rocjitsu::consan_moi_impl
