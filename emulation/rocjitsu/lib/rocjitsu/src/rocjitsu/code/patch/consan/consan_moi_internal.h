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

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu {

struct MoiOptions;
struct VgprSpillSequence;

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

/// Semantic request to materialize the current resident wave's owner identity
/// in one scalar register.
///
/// `destination_sgpr` names the register selected by resource planning.
/// `one_based` requests the representation used by packed ConSan shadows,
/// where zero denotes an empty cell and architectural identity N is stored as
/// N+1. The request deliberately contains no architecture or HWREG numbers:
/// those belong to `ConSanTargetProfile` and are lowered by the target
/// operation below.
struct MoiResidentWaveOwnerRequest {
  uint16_t destination_sgpr = 0;
  bool one_based = false;

  bool operator==(const MoiResidentWaveOwnerRequest &) const = default;
};

/// Append the target instruction sequence implementing one resident-wave
/// owner request.
///
/// The operation is transactional: an invalid destination or target encoding
/// returns false without changing `words`. Successful output ends with the
/// target's required scalar-to-vector dependency wait, so later vector
/// consumers observe the completed identity. The target profile supplies all
/// architectural selection; callers supply only semantic intent.
[[nodiscard]] bool append_moi_resident_wave_owner(std::vector<uint32_t> &words,
                                                  const MoiResidentWaveOwnerRequest &request,
                                                  const ConSanTargetProfile &target);

/// Complete semantic request to replay one displaced guest LDS access.
///
/// `image` is the pristine code-object image from which an unchanged guest
/// instruction can be copied. `candidate` supplies the normalized access
/// semantics and original file range. `target` binds the request to the
/// selected architectural contract. `replay_address_vgpr` is the address base
/// that is valid at the relocation site; `adjusted_address_vgpr` is optional
/// scratch reserved by planning for a split address whose static offset does
/// not fit the target instruction. No engine policy or report state belongs in
/// this request.
struct MoiGuestAccessRelocationRequest {
  std::span<const uint8_t> image;
  const ConSanMoiCandidate *candidate = nullptr;
  const ConSanTargetProfile *target = nullptr;
  uint16_t replay_address_vgpr = 0;
  std::optional<uint16_t> adjusted_address_vgpr;
};

/// Return whether relocation of this candidate requires an extra address
/// VGPR. The answer is shared by resource planning and target emission so they
/// cannot disagree about the split-instruction scratch contract.
[[nodiscard]] bool
moi_guest_access_relocation_requires_adjusted_address(const ConSanMoiCandidate &candidate,
                                                      const ConSanTargetProfile &target);

/// Build the target instruction words that replay one displaced guest access.
///
/// Ordinary targets copy the pristine instruction exactly. A target whose
/// profile requires a two-address split receives semantically equivalent
/// single-address instructions using the request's replay address. Malformed
/// input returns no words and appends a diagnostic; the pristine image is
/// never modified.
[[nodiscard]] std::optional<std::vector<uint32_t>>
build_moi_relocated_guest_access_words(const MoiGuestAccessRelocationRequest &request,
                                       std::vector<std::string> &errors);

/// Selected scalar-register plan for preserving the guest's VCC and SCC
/// across one injected operation sequence.
///
/// `vcc_save_sgpr` is the first register of an aligned scalar pair that holds
/// the complete guest VCC value. `scc_save_sgpr` holds zero or one, produced by
/// materializing the incoming SCC predicate. The two roles are intentionally
/// named rather than represented as offsets from a broad EXEC-save window:
/// placement may choose different layouts for Record/Replay, Sampled, and
/// InlineShadow, while the target preservation operation is identical.
struct MoiSpecialStateSgprs {
  uint16_t vcc_save_sgpr = 0;
  uint16_t scc_save_sgpr = 0;

  bool operator==(const MoiSpecialStateSgprs &) const = default;
};

/// Append a transactional snapshot of guest SCC followed by guest VCC.
///
/// SCC is captured first because every later scalar instruction is permitted
/// to overwrite it. Invalid register assignments or unsupported target
/// encodings return false without appending either instruction.
[[nodiscard]] bool append_save_moi_special_state(std::vector<uint32_t> &words,
                                                 const MoiSpecialStateSgprs &registers,
                                                 const ConSanTargetProfile &target);

/// Append a transactional restoration of guest VCC followed by guest SCC.
///
/// SCC restoration is deliberately last so no injected scalar comparison can
/// overwrite the value that the displaced guest instruction observes.
/// Invalid assignments return false without changing `words`.
[[nodiscard]] bool append_restore_moi_special_state(std::vector<uint32_t> &words,
                                                    const MoiSpecialStateSgprs &registers,
                                                    const ConSanTargetProfile &target);

/// Semantic request to restore SCC from bit zero of a scalar route key.
///
/// Dense routers may pack the caller's boolean SCC value into the otherwise
/// aligned low bit of a route key. `encoded_sgpr` names that key; the target
/// operation owns the generation-specific bit-test opcode and normalizes the
/// resulting SCC back through a scalar conditional select. The request does
/// not expose those encoding details to routing policy.
struct MoiEncodedSccRestoreRequest {
  uint16_t encoded_sgpr = 0;

  bool operator==(const MoiEncodedSccRestoreRequest &) const = default;
};

/// Append the target sequence that restores SCC from an encoded route key.
///
/// Only target profiles with the qualified gfx9 CDNA bit-test forms are
/// admitted. Unsupported targets or invalid register assignments return false
/// without changing `words`.
[[nodiscard]] bool append_restore_moi_scc_from_route_key(std::vector<uint32_t> &words,
                                                         const MoiEncodedSccRestoreRequest &request,
                                                         const ConSanTargetProfile &target);

/// Append the target's device-scope cache refresh before retrying a contended
/// global publication.
///
/// Qualified gfx9 CDNA targets require an explicit buffer invalidate sequence;
/// targets whose coherent atomic-load path needs no extra instruction succeed
/// without appending words. Encoding failure returns false without partial
/// output.
[[nodiscard]] bool append_moi_device_cache_refresh(std::vector<uint32_t> &words,
                                                   const ConSanTargetProfile &target);

/// Canonical set of occupied half-open ranges in pristine executable text.
///
/// Dense relay placement must reject original instructions already owned by
/// an access or synchronization site. Callers may supply overlapping,
/// adjacent, empty, and unsorted ranges; construction removes empty ranges and
/// coalesces the rest. Keeping normalization and overlap queries in this type
/// prevents each instrumentation engine from implementing subtly different
/// interval arithmetic around the same original code.
class MoiOccupiedTextRanges {
public:
  /// Build a canonical occupied-range set from half-open `[begin, end)` pairs.
  explicit MoiOccupiedTextRanges(std::vector<std::pair<uint64_t, uint64_t>> ranges) {
    std::erase_if(ranges, [](const auto &range) { return range.first >= range.second; });
    std::ranges::sort(ranges);
    for (const auto &range : ranges) {
      if (!ranges_.empty() && ranges_.back().second >= range.first) {
        ranges_.back().second = std::max(ranges_.back().second, range.second);
      } else {
        ranges_.push_back(range);
      }
    }
  }

  /// Return whether `[begin, end)` intersects any nonempty occupied range.
  /// Empty or reversed queries do not overlap.
  [[nodiscard]] bool overlaps(uint64_t begin, uint64_t end) const {
    if (begin >= end)
      return false;
    const auto after =
        std::ranges::lower_bound(ranges_, end, {}, [](const auto &range) { return range.first; });
    return after != ranges_.begin() && std::prev(after)->second > begin;
  }

  /// Return the sorted, disjoint ranges retained by this set. This observation
  /// API exists for precise unit tests and diagnostics; placement needs only
  /// `overlaps`.
  [[nodiscard]] std::span<const std::pair<uint64_t, uint64_t>> ranges() const { return ranges_; }

private:
  /// Sorted, nonempty, pairwise-disjoint half-open ranges.
  std::vector<std::pair<uint64_t, uint64_t>> ranges_;
};

/// Stable identity of one kernel or helper function during dense access-route
/// planning.
///
/// The kind is retained explicitly because a kernel and helper may share a
/// symbol name without sharing an execution owner or legal relocation range.
/// Descriptor identity is deliberately absent: one helper group may execute
/// for several descriptors while still needing one physical entry relay.
struct MoiDenseRouteOwner {
  ConSanProgramContainerKind kind = ConSanProgramContainerKind::Count;
  std::string name;

  auto operator<=>(const MoiDenseRouteOwner &) const = default;
};

/// Minimal target-independent site presented to the common dense-route
/// partitioner.
///
/// `source_index` remains in the caller's domain, allowing access, barrier,
/// and atomic planners to share grouping without sharing their semantic
/// candidate types.
struct MoiDenseRouteSite {
  MoiDenseRouteOwner owner;
  uint64_t anchor = 0;
  size_t source_index = 0;
};

/// One owner-local, branch-reachable set of caller indices that may share a
/// dense dispatcher.
///
/// Indices are ordered by their corresponding pristine-text anchor.
/// `first_anchor` makes the common branch-span invariant explicit without
/// requiring callers to reconstruct it from engine-private candidate values.
struct MoiDenseRouteIndexGroup {
  MoiDenseRouteOwner owner;
  uint64_t first_anchor = 0;
  std::vector<size_t> source_indices;
};

/// Partition heterogeneous lowering sites by typed owner, dispatcher
/// capacity, and the maximum span reachable from one SOPP relay.
///
/// A zero capacity means unlimited sites per group. Every returned group is
/// nonempty; input order does not affect the result except as a deterministic
/// tie-break for sites with the same owner and anchor.
[[nodiscard]] inline std::vector<MoiDenseRouteIndexGroup>
partition_moi_dense_route_sites(std::span<const MoiDenseRouteSite> sites,
                                size_t max_sites_per_group) {
  std::vector<MoiDenseRouteSite> ordered(sites.begin(), sites.end());
  std::ranges::sort(ordered, [](const MoiDenseRouteSite &lhs, const MoiDenseRouteSite &rhs) {
    return std::tie(lhs.owner, lhs.anchor, lhs.source_index) <
           std::tie(rhs.owner, rhs.anchor, rhs.source_index);
  });
  constexpr uint64_t kMaxCommonRelayAnchorSpan =
      static_cast<uint64_t>(static_cast<int64_t>(std::numeric_limits<int16_t>::max()) -
                            static_cast<int64_t>(std::numeric_limits<int16_t>::min())) *
      sizeof(uint32_t);
  std::vector<MoiDenseRouteIndexGroup> groups;
  for (const MoiDenseRouteSite &site : ordered) {
    const bool capacity_reached = max_sites_per_group != 0u && !groups.empty() &&
                                  groups.back().source_indices.size() >= max_sites_per_group;
    if (groups.empty() || groups.back().owner != site.owner || capacity_reached ||
        site.anchor - groups.back().first_anchor > kMaxCommonRelayAnchorSpan) {
      groups.push_back({.owner = site.owner, .first_anchor = site.anchor, .source_indices = {}});
    }
    groups.back().source_indices.push_back(site.source_index);
  }
  return groups;
}

/// One bounded, branch-reachable group of access candidates that may share a
/// dense entry relay and dispatcher.
///
/// All candidates belong to `owner`, are ordered by original-text anchor, and
/// fit within the common SOPP branch span. `first_candidate_index` refers to
/// the caller's unmodified candidate sequence and therefore identifies the
/// appended relay-bank slot owned by this group.
struct MoiDenseCandidateGroup {
  MoiDenseRouteOwner owner;
  std::vector<const ConSanMoiCandidate *> candidates;
  size_t first_candidate_index = 0;

  /// Return the original-text anchors in their canonical group order.
  [[nodiscard]] std::vector<uint64_t> anchors() const {
    std::vector<uint64_t> result;
    result.reserve(candidates.size());
    for (const ConSanMoiCandidate *candidate : candidates)
      result.push_back(candidate->anchor());
    return result;
  }

  /// Return whether adjacent sites are too close to own independent entry
  /// regions of `entry_bytes` each.
  [[nodiscard]] bool has_overlapping_entries(uint64_t entry_bytes) const {
    for (size_t index = 1; index < candidates.size(); ++index) {
      if (candidates[index]->anchor() < candidates[index - 1u]->anchor() + entry_bytes)
        return true;
    }
    return false;
  }

  /// Return whether every guest instruction provides at least `minimum_bytes`
  /// of explicit anchor storage for a keyed dense route.
  [[nodiscard]] bool every_site_has_size(uint32_t minimum_bytes) const {
    return std::ranges::all_of(candidates, [&](const ConSanMoiCandidate *candidate) {
      return candidate->size() >= minimum_bytes;
    });
  }
};

/// Shared, immutable partition of access candidates used by dense route
/// planners in Record/Replay, Sampled, and InlineShadow.
///
/// `candidates_by_owner` protects every access anchor in an owner even when
/// that owner is split across several relay groups. `index_by_candidate`
/// preserves the caller's relay-bank assignment. `groups` is the only place
/// where owner boundaries, anchor ordering, maximum group size, and common
/// branch reach are combined.
struct MoiDenseCandidatePartition {
  std::map<MoiDenseRouteOwner, std::vector<const ConSanMoiCandidate *>> candidates_by_owner;
  std::unordered_map<const ConSanMoiCandidate *, size_t> index_by_candidate;
  std::vector<MoiDenseCandidateGroup> groups;
};

/// Partition candidates for shared dense entry routing.
///
/// A zero `max_candidates_per_group` means that capacity does not split an
/// owner; SOPP reach still does. Nonzero limits encode engine/target dispatcher
/// capacity and are policy supplied by the caller rather than hidden in the
/// shared mechanism.
[[nodiscard]] inline MoiDenseCandidatePartition
partition_moi_dense_candidates(std::span<const ConSanMoiCandidate *const> candidates,
                               size_t max_candidates_per_group) {
  MoiDenseCandidatePartition partition;
  partition.index_by_candidate.reserve(candidates.size());
  std::vector<MoiDenseRouteSite> sites;
  sites.reserve(candidates.size());
  for (size_t index = 0; index < candidates.size(); ++index) {
    const ConSanMoiCandidate *candidate = candidates[index];
    partition.index_by_candidate.emplace(candidate, index);
    const MoiDenseRouteOwner owner{candidate->container.kind, candidate->container.name};
    partition.candidates_by_owner[owner].push_back(candidate);
    sites.push_back({.owner = owner, .anchor = candidate->anchor(), .source_index = index});
  }

  for (auto &[owner, owner_candidates] : partition.candidates_by_owner) {
    (void)owner;
    std::ranges::sort(owner_candidates, {},
                      [](const ConSanMoiCandidate *candidate) { return candidate->anchor(); });
  }
  for (const MoiDenseRouteIndexGroup &group :
       partition_moi_dense_route_sites(sites, max_candidates_per_group)) {
    MoiDenseCandidateGroup candidates_group{
        .owner = group.owner,
        .candidates = {},
        .first_candidate_index = group.source_indices.front(),
    };
    candidates_group.candidates.reserve(group.source_indices.size());
    for (size_t index : group.source_indices)
      candidates_group.candidates.push_back(candidates[index]);
    partition.groups.push_back(std::move(candidates_group));
  }
  return partition;
}

/// Return whether one emitted MOI patch makes its owning kernel consume all
/// three launch workgroup coordinates. This shared descriptor-mutation and
/// validation predicate distinguishes MOI patches from independently composed
/// mutation and malformed-barrier patches. CDNA consumes the tuple only
/// through an entry capture with complete persistent storage; RDNA and CDNA5
/// observation bodies consume the firmware payload directly.
[[nodiscard]] inline bool patch_requires_full_workgroup_id_payload(ConSanCapabilityEngine engine,
                                                                   rj_code_arch_t arch,
                                                                   const ConSanPatchInfo &patch) {
  if (!consan_is_capability_arch(arch) || patch.owner_descriptor_file_offsets.empty() ||
      engine == ConSanCapabilityEngine::SuperCollider || engine == ConSanCapabilityEngine::Count) {
    return false;
  }
  if (consan_uses_gfx9_cdna_encoding(arch)) {
    const bool entry_capture = patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue ||
                               patch.kind == ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
    return entry_capture && (patch.persistent_sgpr_state.record_replay_workgroup.complete() ||
                             patch.persistent_record_replay_workgroup_vgprs.complete() ||
                             patch.persistent_record_replay_workgroup_private_offsets.complete());
  }
  return (patch.kind >= ConSanPatchKind::InlineMoiAccessRecordStore &&
          patch.kind <= ConSanPatchKind::TrampolineMoiFenceRecord) ||
         patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
}

/// Release-active outcome from recovering one guest VGPR out of an
/// instrumentation spill window.
enum class MoiSpilledVgprReloadResult : uint8_t {
  Appended,
  SourceOutsideWindow,
  IncompleteSlotMetadata,
  UnsupportedEncoding,
};

[[nodiscard]] const char *moi_spilled_vgpr_reload_result_name(MoiSpilledVgprReloadResult result);

/// Append one private-memory reload without leaving partial output on failure.
///
/// Dynamic-stack spills select the target-native scalar-addressed encoding.
/// Fixed-frame spills use the address-free private-load encoding.
[[nodiscard]] MoiSpilledVgprReloadResult
append_reload_moi_spilled_vgpr(std::vector<uint32_t> &words, const VgprSpillSequence &spill,
                               uint16_t destination, uint16_t source, rj_code_arch_t arch);

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

struct ScalarOwnerSgprRange {
  uint16_t base = 0;
  uint16_t width = 0;
};

/// Return whether any owner's original or grown CDNA physical VCC aliases one
/// of the requested ordinary-SGPR ranges. Empty or invalid owner sets fail
/// closed.
[[nodiscard]] bool scalar_owner_contexts_conflict_with_physical_vcc(
    std::span<const ScalarOwnerContextSummary> contexts,
    std::span<const ScalarOwnerSgprRange> ranges);

/// Return whether every owner admits a persistent ordinary-SGPR window above
/// its complete scalar tail. CDNA callers additionally request physical-VCC
/// qualification.
[[nodiscard]] bool
scalar_owner_contexts_admit_reserved_window(std::span<const ScalarOwnerContextSummary> contexts,
                                            uint16_t base, uint16_t width,
                                            bool protect_physical_vcc);

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

/// Return the transaction register that holds the expected version for CAS.
///
/// The address-formation phase completes before this register is initialized,
/// so a narrow spill-backed probe may use it temporarily to recover one guest
/// LDS address component.
[[nodiscard]] constexpr uint16_t inline_shadow_cas_expected_vgpr(uint16_t old_value_vgpr) {
  return static_cast<uint16_t>(old_value_vgpr + 10u);
}

/// Return the offset/counter scratch reserved when a wide access spans enough
/// exact-shadow cells to make a compact runtime loop preferable to unrolling.
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
[[nodiscard]] bool validate_scalar_state_temporaries(const MoiOptions &options,
                                                     std::string_view consumer,
                                                     std::vector<std::string> &errors);

/// Materialize one persistent workgroup-coordinate source, including any
/// ABI-specific extraction applied after a scalar, vector, or private load.
[[nodiscard]] bool append_workgroup_source_value(std::vector<uint32_t> &words,
                                                 const ConSanMoiWorkgroupSource &source,
                                                 uint16_t value_vgpr, rj_code_arch_t arch);

/// Append a dynamic-record address materialization for
/// `field_address + slot * stride_bytes`.
///
/// The address occupies `address_vgpr:address_vgpr+1`. The slot and offset
/// must be distinct from that pair. The report-layout validator guarantees
/// that `slot * stride_bytes` fits in 32 bits. The emitted plan uses only the
/// address pair as temporary storage, preserves the slot, and clobbers VCC;
/// production callers save and restore VCC around dynamic publication.
[[nodiscard]] bool append_dynamic_record_address(std::vector<uint32_t> &words,
                                                 uint64_t field_address, uint32_t stride_bytes,
                                                 uint16_t address_vgpr, uint16_t slot_vgpr,
                                                 rj_code_arch_t arch);

/// Return the nearest emitted trampoline body strictly after `offset` across
/// both already committed and current-pass patch inventories. Empty bodies do
/// not reserve bytes. Incremental lowering must use both inventories when it
/// grows a shared dispatcher, or it can overwrite a body emitted earlier in
/// the current pass.
[[nodiscard]] std::optional<uint64_t>
next_moi_trampoline_boundary(uint64_t offset, std::span<const ConSanPatchInfo> committed,
                             std::span<const ConSanPatchInfo> current_pass);

} // namespace consan_detail
} // namespace rocjitsu
