// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_common_emission.h
/// @brief Shared compiled routing emission for MOI access engines.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"

namespace rocjitsu::consan_moi_impl {

struct MoiDenseRelayRoute {
  uint32_t identity = 0;
  uint64_t caller_return = 0;
  uint64_t target = 0;
};

/// Build the target-normalized dispatcher shared by dense MOI access and
/// synchronization routes. Semantic owners supply only caller identities and
/// body destinations; route-key recovery, SCC restoration, and indirect
/// transfer remain one mechanical authority.
[[nodiscard]] std::optional<std::vector<uint32_t>> build_moi_dense_relay_dispatcher(
    std::span<const MoiDenseRelayRoute> routes, uint16_t jump_pc_sgpr,
    uint16_t saved_scc_sgpr, uint16_t key_sgpr, uint16_t call_return_sgpr,
    bool derive_key_at_entry, bool key_encodes_scc, bool match_call_return,
    bool clone_local_routes, uint64_t dispatcher_offset, uint64_t island_offset,
    const ConSanTargetProfile &target, rj_code_arch_t arch);

[[nodiscard]] bool emit_moi_dense_access_group(
    std::span<const MoiPlannedAccessPatch *const> group, uint64_t dispatcher_offset,
    const std::map<uint64_t, MoiDenseEntryHost> &entry_hosts, const MoiDenseRouterPlan &router,
    rj_code_arch_t arch, uint64_t dispatcher_words_per_site, std::string_view diagnostic_subject,
    std::vector<uint8_t> &new_text, std::vector<ConSanPatchInfo> &patches,
    std::vector<std::string> &errors);

} // namespace rocjitsu::consan_moi_impl
