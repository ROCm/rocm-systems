// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_internal.h
/// @brief Private, directly testable invariants shared by MOI lowering paths.
///
/// Keep pure release-active checks here when multiple lowering `.inc` files
/// depend on them and direct unit coverage is more precise than a test hook in
/// the full patching pipeline.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

struct ConSanOptions;

namespace consan_detail {

struct ScalarOwnerContextSummary {
  std::optional<uint64_t> descriptor_file_offset;
  uint16_t current_sgpr_count = 0;
  uint16_t max_referenced_sgpr_count = 0;
  bool descriptor_valid = false;
};

struct ScalarOwnerContextResolution {
  std::vector<size_t> context_indices;
  uint32_t tail_floor = 0;
};

/// Return the inline-shadow transaction scratch size shared by placement and
/// emission. Atomic tracking retains additional publication state.
[[nodiscard]] constexpr uint16_t inline_shadow_transaction_scratch_count(bool has_exec_save,
                                                                         bool track_atomics) {
  return has_exec_save ? (track_atomics ? 24u : 16u) : 11u;
}

/// Return the first scratch VGPR reserved for a wide-access cell loop.
[[nodiscard]] constexpr uint16_t
inline_shadow_loop_counter_vgpr(uint16_t scratch_vgpr, bool has_exec_save, bool track_atomics) {
  return static_cast<uint16_t>(
      scratch_vgpr + inline_shadow_transaction_scratch_count(has_exec_save, track_atomics));
}

/// Return the offset/counter scratch reserved when one access spans multiple
/// exact-shadow cells.
[[nodiscard]] constexpr uint16_t inline_shadow_loop_scratch_count(uint32_t width_bits,
                                                                  uint32_t granule_bytes) {
  return width_bits > granule_bytes * 8u ? 2u : 0u;
}

/// Resolve every requested owner to one valid context and compute the scalar
/// tail beyond all original allocations and statically referenced registers.
/// Empty owner sets and every inconsistent planning state fail closed.
[[nodiscard]] std::optional<ScalarOwnerContextResolution>
resolve_scalar_owner_contexts(bool planning_state_valid,
                              std::span<const ScalarOwnerContextSummary> contexts,
                              std::span<const uint64_t> owners);

/// Validate the site-local VGPR half of scalar-persistent MOI state before
/// emission. This remains release-active because ConSan rewrites untrusted
/// code objects and must fail cleanly if placement and emission ever diverge.
[[nodiscard]] bool validate_scalar_state_temporaries(const ConSanOptions &options,
                                                     std::string_view consumer,
                                                     std::vector<std::string> &errors);

} // namespace consan_detail
} // namespace rocjitsu
