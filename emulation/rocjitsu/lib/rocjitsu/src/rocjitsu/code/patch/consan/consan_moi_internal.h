// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_internal.h
/// @brief Private, directly testable invariants shared by MOI lowering paths.
///
/// Keep pure release-active checks here when multiple lowering `.inc` files
/// depend on them and direct unit coverage is more precise than a test hook in
/// the full patching pipeline.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

struct ConSanOptions;

/// One scalar, vector, or entry-captured private-state source for a
/// workgroup-coordinate component.
///
/// The none-set state represents an absent dimension. More than one set source
/// is malformed. Keeping this release-active invariant beside the
/// representation prevents individual emitters from interpreting ambiguous
/// sources differently.
struct ConSanMoiWorkgroupSource {
  std::optional<uint16_t> scalar_src;
  std::optional<uint16_t> vector_src;
  std::optional<uint32_t> private_offset;
  bool shift_right_16 = false;
  bool mask_low_16 = false;

  [[nodiscard]] uint8_t source_count() const {
    return static_cast<uint8_t>(scalar_src.has_value()) +
           static_cast<uint8_t>(vector_src.has_value()) +
           static_cast<uint8_t>(private_offset.has_value());
  }
  [[nodiscard]] bool has_value() const { return source_count() == 1u; }
  [[nodiscard]] bool is_well_formed() const { return source_count() <= 1u; }
  [[nodiscard]] std::optional<uint16_t> operand() const;

  [[nodiscard]] static ConSanMoiWorkgroupSource scalar(uint16_t source, bool shift_right_16 = false,
                                                       bool mask_low_16 = false) {
    ConSanMoiWorkgroupSource result;
    result.scalar_src = source;
    result.shift_right_16 = shift_right_16;
    result.mask_low_16 = mask_low_16;
    return result;
  }
  [[nodiscard]] static ConSanMoiWorkgroupSource vector(uint16_t source, bool shift_right_16 = false,
                                                       bool mask_low_16 = false) {
    ConSanMoiWorkgroupSource result;
    result.vector_src = source;
    result.shift_right_16 = shift_right_16;
    result.mask_low_16 = mask_low_16;
    return result;
  }
  [[nodiscard]] static ConSanMoiWorkgroupSource private_state(uint32_t offset) {
    ConSanMoiWorkgroupSource result;
    result.private_offset = offset;
    return result;
  }

  auto operator<=>(const ConSanMoiWorkgroupSource &) const = default;
};

namespace consan_detail {

struct ScalarOwnerContextSummary {
  std::optional<uint64_t> descriptor_file_offset;
  uint16_t current_sgpr_count = 0;
  uint16_t max_referenced_sgpr_count = 0;
  /// Scalar-relative access can reach registers absent from explicit def/use
  /// sets, so a static maximum alone does not bound this owner.
  bool has_indirect_sgpr_access = false;
  /// True only when every executable control-flow destination is represented
  /// by the owner reference scan.
  bool sgpr_reference_coverage_complete = false;
  bool descriptor_valid = false;
};

struct ScalarOwnerContextResolution {
  std::vector<size_t> context_indices;
  /// Hard bound that dominates every SGPR any resolved owner may reach.
  uint32_t tail_floor = 0;
};

[[nodiscard]] uint16_t scalar_owner_tail_floor(const ScalarOwnerContextSummary &context);

/// Fully encoded semantic identity for one sampled atomic synchronization
/// candidate. Physical aliases may fold only when every field matches.
struct SampledAtomicSemantics {
  ConSanMoiSampledSyncRole role = ConSanMoiSampledSyncRole::None;
  ConSanMoiSampledSyncScope scope = ConSanMoiSampledSyncScope::None;
  ConSanMoiSampledSyncOutcome outcome = ConSanMoiSampledSyncOutcome::NotApplicable;
  uint32_t byte_count = 0;
  uint32_t descriptor = 0;
  std::optional<uint32_t> cas_failure_descriptor;

  bool operator==(const SampledAtomicSemantics &) const = default;
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

/// Materialize one persistent workgroup-coordinate source, including any
/// ABI-specific extraction applied after a scalar, vector, or private load.
[[nodiscard]] bool append_workgroup_source_value(std::vector<uint32_t> &words,
                                                 const ConSanMoiWorkgroupSource &source,
                                                 uint16_t value_vgpr, rj_code_arch_t arch);

} // namespace consan_detail
} // namespace rocjitsu
