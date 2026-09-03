// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_common_emission.h
/// @brief Shared compiled routing emission for MOI access engines.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] bool emit_moi_dense_access_group(
    std::span<const MoiPlannedAccessPatch *const> group, uint64_t dispatcher_offset,
    const std::map<uint64_t, MoiDenseEntryHost> &entry_hosts, const MoiDenseRouterPlan &router,
    rj_code_arch_t arch, uint64_t dispatcher_words_per_site, std::string_view diagnostic_subject,
    std::vector<uint8_t> &new_text, std::vector<ConSanPatchInfo> &patches,
    std::vector<std::string> &errors);

} // namespace rocjitsu::consan_moi_impl
