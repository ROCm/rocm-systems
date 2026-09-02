// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] std::optional<uint16_t> moi_descriptor_owner_shift(std::span<const uint8_t> image,
                                                                 uint64_t descriptor_file_offset,
                                                                 rj_code_arch_t arch,
                                                                 std::vector<std::string> &errors);

[[nodiscard]] std::optional<ConSanMoiWorkgroupSources>
moi_persistent_or_descriptor_workgroup_sources(
    std::span<const uint8_t> image, uint64_t descriptor_file_offset, ConSanMoiEngine engine,
    const ConSanMoiOperatingPoint &point, rj_code_arch_t arch, std::vector<std::string> &errors,
    bool uses_cluster_workgroup_id = false,
    const ConSanMoiPersistentWorkgroupPrivateOffsets *private_offsets = nullptr);

[[nodiscard]] bool append_sampled_private_owner_epoch_load(
    std::vector<uint32_t> &words, std::span<const uint8_t> bytes, uint64_t descriptor_file_offset,
    bool automatic_private_epoch, const ConSanMoiOwnerEpochVgprSources &owner_epoch_vgprs,
    const ConSanMoiPrivateStateLayout &layout, rj_code_arch_t arch,
    std::vector<std::string> &errors);

} // namespace rocjitsu::consan_moi_impl
