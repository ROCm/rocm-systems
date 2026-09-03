// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_common_emission.h
/// @brief Shared compiled routing emission for MOI access engines.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"

namespace rocjitsu::consan_moi_impl {

struct MoiDenseRelayRoute {
  uint32_t identity = 0;
  uint64_t anchor = 0;
  uint32_t anchor_size = 0;
  uint64_t caller_return = 0;
  uint32_t call_return_adjustment = 0;
  uint64_t target = 0;
};

struct MoiDenseRelayPlan {
  ConSanDirectCallForm call_form = ConSanDirectCallForm::SCallB64;
  ConSanIndirectJumpSgprs indirect_jump;
  uint16_t key_sgpr = 0;
  uint16_t call_return_sgpr = 0;
  bool derive_key_at_entry = false;
  bool key_encodes_scc = false;
  bool match_call_return = false;
  bool clone_local_routes = false;
  bool explicit_key = false;
  bool restore_scc_before_route = false;
  bool normalize_encoded_scc = true;
  bool requires_indirect_pc_wait = false;
  uint64_t entry_island_words = 0;
  uint64_t host_offset = 0;
  uint64_t host_body_offset = 0;
  uint64_t island_offset = 0;
  uint64_t dispatcher_offset = 0;
  uint64_t dispatcher_capacity_bytes = 0;
  std::span<const uint32_t> displaced_host_words;
  std::span<const MoiDenseRelayRoute> routes;
  std::span<const uint64_t> owner_descriptors;
};

[[nodiscard]] uint64_t moi_dense_call_relay_island_words(size_t route_count,
                                                          bool derive_key_at_entry,
                                                          bool key_encodes_scc,
                                                          bool clone_local_routes);

/// Commit the common host, entry-island, dispatcher, and call-anchor mechanics
/// for an already planned set of dense semantic bodies.
[[nodiscard]] bool emit_moi_dense_relay(const MoiDenseRelayPlan &plan, rj_code_arch_t arch,
                                        std::string_view diagnostic_subject,
                                        std::vector<uint8_t> &new_text,
                                        std::vector<ConSanPatchInfo> &patches,
                                        std::vector<std::string> &errors);

/// Build the target-normalized dispatcher shared by dense MOI access and
/// synchronization routes. Semantic owners supply only caller identities and
/// body destinations; route-key recovery, SCC restoration, and indirect
/// transfer remain one mechanical authority.
[[nodiscard]] std::optional<std::vector<uint32_t>> build_moi_dense_relay_dispatcher(
    std::span<const MoiDenseRelayRoute> routes, uint16_t jump_pc_sgpr,
    uint16_t saved_scc_sgpr, uint16_t key_sgpr, uint16_t call_return_sgpr,
    bool derive_key_at_entry, bool key_encodes_scc, bool match_call_return,
    bool clone_local_routes, bool explicit_key, bool restore_scc_before_route,
    bool normalize_encoded_scc, uint64_t dispatcher_offset, uint64_t island_offset,
    const ConSanTargetProfile &target, rj_code_arch_t arch);

[[nodiscard]] std::optional<std::vector<uint32_t>> build_moi_dense_relay_anchor(
    const MoiDenseRelayRoute &route, ConSanDirectCallForm call_form, uint16_t key_sgpr,
    uint16_t call_return_sgpr, bool explicit_key, bool key_encodes_scc,
    bool clone_local_routes, uint64_t island_offset, rj_code_arch_t arch);

[[nodiscard]] bool emit_moi_dense_access_group(
    std::span<const MoiPlannedAccessPatch *const> group, uint64_t dispatcher_offset,
    const std::map<uint64_t, MoiDenseEntryHost> &entry_hosts, const MoiDenseRouterPlan &router,
    rj_code_arch_t arch, uint64_t dispatcher_words_per_site, std::string_view diagnostic_subject,
    std::vector<uint8_t> &new_text, std::vector<ConSanPatchInfo> &patches,
    std::vector<std::string> &errors);

} // namespace rocjitsu::consan_moi_impl
