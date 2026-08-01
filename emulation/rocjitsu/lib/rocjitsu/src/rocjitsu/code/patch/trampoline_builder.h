// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file trampoline_builder.h
/// @brief Lowers a TrampolinePlan into patched-anchor bytes and trampoline
///        words for the DBI relocation-only path.
///
/// This is the byte emitter; it owns SOPP branch math and basic plan
/// well-formedness checks (original_size 4 or 8, original_words count
/// matches, branch ranges fit). It does not touch the ELF, does not own
/// layout assignment, and does not enforce milestone-scoped restrictions
/// (e.g. "only emit s_nop placeholder bodies" — that lives in the
/// orchestrator as `validate_inline_nop_plan` in instrumentor.h).
/// See code_object_patcher.h for the ELF mutation layer.

#pragma once

#include "rocjitsu/code/patch/planning_work.h"
#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/register_set.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {

/// @brief Concrete instruction words placed before or after the relocated
///        original in the trampoline. Declared clobbers are intentionally
///        deferred to a later milestone.
struct InlineAsmItem {
  std::vector<uint32_t> words;
};

/// @brief Builder-facing description of one trampoline.
///
/// Coordinates are .text-relative byte offsets. The orchestrator fills this
/// after validation and layout, then hands it to TrampolineBuilder.
struct TrampolinePlan {
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;

  uint64_t anchor_offset = 0;
  uint32_t original_size = 0; // 4 or 8 for the inline-nop smoke build.
  uint64_t trampoline_offset = 0;
  uint64_t return_target = 0; // Typically anchor_offset + original_size.

  std::vector<uint32_t> original_words; // Exact bytes pulled from .text.

  std::vector<InlineAsmItem> before_items;
  std::vector<InlineAsmItem> after_items;
  bool emit_original = true;

  //----------------------------------------------------------------------------
  // Probe-call resources (filled by plan_probe_call(); left at defaults for the
  // inlined nop).
  //
  // These are the call-envelope resource decisions made BEFORE layout is known:
  // which registers the envelope uses and how many words it spans. Emission
  // materializes `before_items` from these decisions plus the assigned layout,
  // so it must not re-pick registers or recount words. Folded into
  // TrampolinePlan for now since this is the builder's one input;
  // lift back out into a dedicated resource-plan type if it grows unwieldy.
  //----------------------------------------------------------------------------
  bool is_probe_call = false;     ///< True once plan_probe_call() populated these.
  uint16_t link_pair_base = 30;   ///< Return-link pair, derived from the probe cc.
  uint16_t target_pair_base = 0;  ///< Dead even SGPR pair holding the probe address.
  bool preserve_scc = true;       ///< v0 preserves SCC across target materialization.
  uint16_t scc_temp = 0;          ///< Dead SGPR holding saved SCC across the call.
  RegisterSet builder_clobbers;   ///< {link} | {target pair} | {scc_temp}; feeds the spill formula.
  uint32_t before_word_count = 0; ///< Envelope words emitted before the relocated original.
  uint64_t probe_target_offset = 0; ///< .text-relative byte offset of the copied probe body.
};

/// @brief Output bytes for one trampoline.
struct TrampolineBytes {
  std::vector<uint8_t> patched_anchor_bytes; // original_size bytes.
  std::vector<uint32_t> trampoline_words;
};

class TrampolineBuilder {
public:
  /// @brief Lower @p plan to patched-anchor bytes and trampoline words.
  ///
  /// Returns std::nullopt and writes a human-readable explanation to
  /// @p error_out (if non-null) on:
  ///   - arch left at ROCJITSU_CODE_ARCH_INVALID (caller forgot to set it)
  ///   - original_size other than 4 or 8
  ///   - original_words size mismatch with original_size
  ///   - Forward or return branch outside s_branch simm16 range
  ///
  /// The builder does not enforce milestone-scoped restrictions on body
  /// shape; the orchestrator decides what kind of plan to emit and calls
  /// validate_inline_nop_plan (in instrumentor.h) when appropriate.
  [[nodiscard]] static std::optional<TrampolineBytes> build(const TrampolinePlan &plan,
                                                            std::string *error_out = nullptr);

  /// @brief Select the probe-call envelope resources and record them on @p plan.
  ///
  /// Picks the call-envelope registers and computes the envelope word count
  /// without choosing layout or emitting bytes. On success, fills
  /// `plan.is_probe_call`, `link_pair_base`, `target_pair_base`, `preserve_scc`,
  /// `scc_temp`, `builder_clobbers`, and `before_word_count`, then returns true.
  ///
  /// Policy:
  ///   - Link pair is derived from @p cc via link_pair_for(); an unknown
  ///     convention fails. If either lane of the derived pair is live at the
  ///     anchor, fail. Extending the supported conventions is deferred.
  ///   - Target-address pair is a dead, even-aligned SGPR pair (excluding the
  ///     link pair). It is consumed by s_swappc before the probe body runs, so it
  ///     may overlap @p probe_body_clobbers.
  ///   - SCC is preserved with one dead SGPR temp. The temp lives across the call
  ///     (saved before materialization, restored after), so it must avoid both
  ///     the live set and @p probe_body_clobbers. Extending this is deferred.
  ///
  /// Returns false and writes a diagnostic naming the unavailable resource to
  /// @p error_out (if non-null) when @p cc is unknown, the link pair is live, or
  /// no dead target pair / SCC temp can be found. The plan is left unmodified on
  /// failure.
  ///
  /// @param plan                Trampoline plan whose resource fields are filled.
  /// @param cc                  Probe calling convention; sets the link pair.
  /// @param live_at_anchor      Registers live immediately before the anchor.
  /// @param probe_body_clobbers Ordinary registers the copied probe body writes.
  [[nodiscard]] static bool plan_probe_call(TrampolinePlan &plan, ProbeCallingConvention cc,
                                            const RegisterSet &live_at_anchor,
                                            const RegisterSet &probe_body_clobbers,
                                            std::string *error_out = nullptr);

  /// @brief Emit a planned probe call: envelope, relocated original, return
  ///        branch.
  ///
  /// Requires @p plan.is_probe_call (i.e. plan_probe_call() succeeded). Builds
  /// the call envelope from the planned resources and @p plan.probe_target_offset
  /// then delegates to build() for layout and branch math.
  ///
  /// Returns std::nullopt and writes a diagnostic to @p error_out (if non-null)
  /// when the plan is not a probe call, the synthesized envelope size disagrees
  /// with the planned before_word_count (plan/emit drift), or build() reports a
  /// branch-range failure.
  [[nodiscard]] static std::optional<TrampolineBytes>
  emit_probe_call(const TrampolinePlan &plan, std::string *error_out = nullptr);
};

/// @brief One source admitted by the forward SOPP relay planner.
///
/// `relay_offsets` excludes the source and assigned island. Every adjacent
/// pair in source -> relays -> island is a valid forward s_branch hop.
struct SoppBranchRelayRoute {
  size_t source_index = 0;
  uint64_t island_offset = 0;
  std::vector<uint64_t> relay_offsets;
};

/// @brief Maximum-cardinality assignment of sources to interchangeable
///        islands through capacity-one relay slots.
struct SoppBranchRelayPlan {
  std::vector<SoppBranchRelayRoute> routes;
  std::vector<size_t> rejected_source_indices;
  /// Saturated residual min-cut hops, expressed as original text coordinates.
  /// These identify address bands where another relay vertex can increase
  /// maximum route cardinality.
  std::vector<std::pair<uint64_t, uint64_t>> min_cut_hops;
  std::vector<uint64_t> min_cut_relay_offsets;
};

struct SoppBranchRelayPlanningWorkLimits {
  /// Input units are the total source, relay, and island coordinates. The
  /// base and per-input headroom bound incrementally measured graph building,
  /// residual traversal, reconstruction, and the scalable greedy path. The
  /// exact-versus-greedy policy remains the explicit coordinate threshold.
  PlanningWorkLimit total = kDefaultSoppRelayPlanningWorkLimit;
};

struct SoppBranchRelayPlanningWorkTelemetry {
  /// Callers may reuse one telemetry object across calls; counters accumulate
  /// with saturation rather than being reset by the planner.
  size_t work_count = 0u;
  size_t exhaustion_count = 0u;

  bool operator==(const SoppBranchRelayPlanningWorkTelemetry &) const = default;
};

/// @brief Plan forward-only s_branch routes through one-word relay slots.
///
/// Sources and islands are interchangeable only on the island side: every
/// admitted source is assigned exactly one island, while every relay slot and
/// island may appear in at most one route. Coordinates are byte offsets and
/// must be distinct, dword-aligned instruction addresses. An edge exists only
/// when `compute_sopp_branch_simm16(from, to)` accepts the hop and `to > from`.
///
/// The result has maximum possible route cardinality. Ties are deterministic:
/// coordinates are considered in ascending order, then original input order.
/// Invalid coordinate sets return std::nullopt without a partial plan.
/// Reaching the configured work allowance also returns std::nullopt and marks
/// the optional work telemetry, so callers never mistake a bounded prefix for
/// a maximum-cardinality assignment.
///
/// Island offsets are interchangeable destinations. Use
/// `BranchOnlyRelayRouter::plan_pairs` when each source has one fixed target.
[[nodiscard]] std::optional<SoppBranchRelayPlan> plan_forward_sopp_branch_relays(
    std::span<const uint64_t> source_offsets, std::span<const uint64_t> relay_offsets,
    std::span<const uint64_t> island_offsets, std::string *error_out = nullptr,
    SoppBranchRelayPlanningWorkTelemetry *work_telemetry = nullptr,
    const SoppBranchRelayPlanningWorkLimits &work_limits = {});

/// @brief Plan backward-only s_branch routes through one-word relay slots.
///
/// This is the direction-reversed counterpart of
/// `plan_forward_sopp_branch_relays`. Every adjacent pair in source -> relays
/// -> island is a valid backward `s_branch` hop and every relay/island still
/// has capacity one. Returned relay offsets are in execution order, from the
/// higher source toward the lower island.
/// Work exhaustion follows the same fail-closed contract as the forward
/// planner.
///
/// Island offsets are interchangeable destinations. Use
/// `BranchOnlyRelayRouter::plan_pairs` when each source has one fixed target.
[[nodiscard]] std::optional<SoppBranchRelayPlan> plan_backward_sopp_branch_relays(
    std::span<const uint64_t> source_offsets, std::span<const uint64_t> relay_offsets,
    std::span<const uint64_t> island_offsets, std::string *error_out = nullptr,
    SoppBranchRelayPlanningWorkTelemetry *work_telemetry = nullptr,
    const SoppBranchRelayPlanningWorkLimits &work_limits = {});

/// @brief Physical placement selected for one DBI patch body.
enum class DbiPatchPlacementKind : uint8_t {
  Inline,
  LocalCave,
  AppendedCave,
};

struct DbiPatchLocalCave {
  uint64_t offset = 0;
  uint64_t capacity = 0;
};

struct DbiPatchPlacementRequest {
  uint64_t anchor_offset = 0;
  uint32_t original_size = 0;
  uint64_t body_size = 0;
  uint64_t inline_capacity = 0;
  std::optional<DbiPatchLocalCave> local_cave;
  bool allow_appended_cave = true;
};

struct DbiPatchPlacement {
  DbiPatchPlacementKind kind = DbiPatchPlacementKind::Inline;
  uint64_t anchor_offset = 0;
  uint32_t original_size = 0;
  uint64_t body_offset = 0;
  uint64_t body_size = 0;
  uint64_t return_branch_offset = 0;
  uint64_t return_target = 0;
};

/// @brief Transactional placement allocator shared by DBI probe families.
///
/// The planner owns overlap accounting and appended-cave cursor movement.
/// Every successful reservation remains in `occupied_ranges()`, including
/// appended bodies beyond the original image, so composition and audit users
/// see the complete owned address space. A successful trampoline reservation
/// includes its four-byte return branch, so later placements and final emitters
/// use the same coordinates. Failed requests do not mutate the planner.
class DbiPatchPlacementPlanner {
public:
  DbiPatchPlacementPlanner(rj_code_arch_t arch, uint64_t original_text_size);

  [[nodiscard]] std::optional<DbiPatchPlacement> plan(const DbiPatchPlacementRequest &request,
                                                      std::string *error_out = nullptr);

  /// Reserve an appended body whose entry and return are implemented by
  /// caller-supplied indirect control flow. This retains the planner's overlap
  /// and cursor guarantees without imposing SOPP reachability.
  [[nodiscard]] std::optional<DbiPatchPlacement>
  plan_indirect_appended(uint64_t anchor_offset, uint32_t original_size, uint64_t body_size,
                         std::string *error_out = nullptr);

  /// Seed a range already owned by an earlier probe family in a composed
  /// transaction. The range must fit the current text image and not overlap a
  /// prior reservation.
  [[nodiscard]] bool reserve_existing_range(uint64_t begin, uint64_t size,
                                            std::string *error_out = nullptr);

  /// Check an existing-image range without changing planner state.
  [[nodiscard]] bool can_reserve_existing_range(uint64_t begin, uint64_t size) const;

  /// Check a half-open text-coordinate range for overlap with any retained
  /// reservation. Invalid empty or reversed ranges are conservatively treated
  /// as overlapping so placement callers fail closed.
  [[nodiscard]] bool overlaps_reserved_range(uint64_t begin, uint64_t end) const;

  /// Reserve a generated prefix at the current appended cursor. This is used
  /// for fixed-size veneer banks whose stable addresses must be known before
  /// any variable-size bodies are placed.
  [[nodiscard]] bool reserve_appended_prefix(uint64_t size, std::string *error_out = nullptr);

  [[nodiscard]] uint64_t appended_end() const { return appended_cursor_; }
  /// Reservations are retained as pairwise-disjoint ranges in increasing
  /// begin-coordinate order so read-only overlap queries are logarithmic.
  /// Insertion remains linear in the number of retained ranges.
  [[nodiscard]] std::span<const std::pair<uint64_t, uint64_t>> occupied_ranges() const {
    return occupied_ranges_;
  }

private:
  [[nodiscard]] bool range_is_free(uint64_t begin, uint64_t end) const;
  /// The sorted, disjoint inventory places every appended reservation after
  /// every original-text reservation, so checking the last range covers the
  /// entire appended suffix in constant time.
  [[nodiscard]] bool highest_appended_reservation_within_cursor() const;
  void insert_validated_range(uint64_t begin, uint64_t end);
  [[nodiscard]] bool reserve_ranges(std::span<const std::pair<uint64_t, uint64_t>> ranges);
  [[nodiscard]] bool reserve_range(uint64_t begin, uint64_t end);

  rj_code_arch_t arch_ = ROCJITSU_CODE_ARCH_INVALID;
  uint64_t original_text_size_ = 0;
  uint64_t appended_cursor_ = 0;
  std::vector<std::pair<uint64_t, uint64_t>> occupied_ranges_;
};

} // namespace rocjitsu
