// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file contract.h
/// @brief The verification object for one region pair.
///
/// @details The relation tested by P0 is
///   C(s_old, s_new) AND R_in(s_old, s_new)
///     => both reach the logical exit AND R_out(final_old, final_new) AND frame preserved
/// with
///   - R_in: equality on `entry_equal`, at the same physical location on both sides
///     (identity mapping at a common cutpoint). Locations outside it get independent
///     per-side canaries, which turn a missed dependency into a visible mismatch.
///   - C: `constraints`, each carrying its own source.
///   - R_out: equality on `observable`, the union of both continuations' live-ins.
///   - frame: fixed control state that P0 instructions must not change.
/// Every set records where it came from, and a hand-written contract is never upgraded
/// from ASSUMED.
#pragma once

#include "json.h"
#include "location.h"
#include "target.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rocjitsu::isa_diff {

enum class ContractStatus { Established, Assumed, Unresolved };
[[nodiscard]] const char *contract_status_name(ContractStatus s);
[[nodiscard]] std::optional<ContractStatus> parse_contract_status(std::string_view s);

/// One entry constraint on a single location's value.
struct Constraint {
  unsigned loc = 0;
  uint32_t known_zero = 0; ///< Bits that must be 0.
  uint32_t known_one = 0;  ///< Bits that must be 1.
  std::optional<uint32_t> umin, umax;
  std::optional<unsigned> equal_to; ///< Must equal another location's value.
  /// One of: "common_prefix_analysis", "relation_summary", "reachability_witness",
  /// "user_assumption".
  std::string source = "user_assumption";
};

/// One way out of a region with control flow: the exit each side takes (0: falling off the
/// end of the region, else the PC a branch leaves to) and what is compared there.
struct RegionExit {
  uint64_t old_pc = 0, new_pc = 0;
  LocSet observable; ///< Live at the exit target (either side).
  VSet vobservable;  ///< VGPRs live there.
  VSet vactive_only; ///< Compared on active lanes only (see RegionContract::vactive_only).
  bool operator==(const RegionExit &) const = default;
};

struct Assumption {
  std::string what;
  std::string source;
};

struct RegionContract {
  std::string target = "gfx950";
  std::string wave = "wave64";
  /// The GPU target and wave size these name (nullopt for an unsupported pair).
  [[nodiscard]] std::optional<Target> gpu() const;
  void set_gpu(const Target &t) {
    target = t.name;
    wave = "wave" + std::to_string(t.wave_size);
  }
  std::string arithmetic = "default";
  std::string mapping = "identity_at_common_cutpoint";
  LocSet entry_equal;
  LocSet canary;
  LocSet observable;
  LocSet declared_writes_old; ///< Effect model; execution may change nothing else.
  LocSet declared_writes_new;
  LocSet wide_reads; ///< Input-generation feature: low halves of 64-bit operands.
  /// VALU support. VGPRs are whole-register locations whose 64 lanes are all compared.
  /// The tracked range is v0..v{vgpr_count-1}; it is exactly the wave's allocation in the
  /// runner, so any access beyond it is an invalid register access, never silent.
  unsigned vgpr_count = 0;
  VSet ventry_equal; ///< Live-in VGPRs (either side): equal on every lane.
  VSet vobservable;  ///< Live-out VGPRs (either side): compared on every lane.
  /// Live-in VGPRs that hold no defined value on that side (never written since the function
  /// entry, not ABI-defined): LLVM undef. Old's outputs that depend on them are don't care,
  /// and new may not depend on its own beyond that (README "Undefined values").
  VSet vundef_old, vundef_new;
  /// The same for scalar locations (SGPRs, VCC, M0, SCC).
  LocSet undef_old, undef_new;
  /// VGPRs observable only through a return value narrower than the register: just these
  /// bits are compared (e.g. 0x0000ffff for a half returned in v0).
  std::vector<std::pair<unsigned, uint32_t>> vexit_masks;
  /// The same for scalar locations (a shader's i8/i16 return in an SGPR).
  std::vector<std::pair<unsigned, uint32_t>> exit_masks;
  /// Regions with branches: one entry per exit pair (empty: one fallthrough exit, whose
  /// R_out is observable/vobservable/vactive_only). observable/vobservable then hold the
  /// union over the exits.
  std::vector<RegionExit> exits;
  /// EXEC at the entry is not zero (its wave-sized mask): it is still the dispatch or call
  /// EXEC, and a wave runs with at least one active lane. Inputs with EXEC 0 are invalid.
  bool exec_nonzero = false;
  /// Observable VGPRs compared on the lanes active in EXEC at the exit only: EXEC never
  /// changes after the exit and nothing reads an inactive lane again (both sides), and the
  /// callee need not preserve them (not callee-saved).
  VSet vactive_only;
  VSet vwrites_old, vwrites_new; ///< Effect model for VGPRs.
  VSet vrange;                   ///< v0..v{vgpr_count-1} (planner only; not serialised).
  VSet vwide_reads;              ///< First register of 64-bit VGPR sources.
  /// MODE at the region entry on each side (FP rounding/denormals, DX10_CLAMP, IEEE),
  /// from each kernel's descriptor. VALU float results depend on it.
  uint32_t mode_old = kDefaultInitialMode, mode_new = kDefaultInitialMode;
  /// Memory (handoff P2 item, global and scalar): the region loads (memory is an input,
  /// equal on both sides) and/or stores (the bytes either side stores are compared).
  bool memory_reads = false, memory_writes = false;
  /// Changed code downstream reads memory: a memory mismatch may be compensated there.
  bool downstream_memory_reads = false;
  /// The region contains comment-only inline-asm blocks: a `def` block writes values
  /// drawn per case, equal for the old and new block with the same index; what a `use`
  /// block observes is compared between the two sides.
  bool inline_asm = false;
  std::vector<Constraint> constraints;
  std::string exit_mapping = "single fall-through exit after the last region instruction";
  /// Hand-written register correspondence (handoff 3.1), old location -> new location,
  /// at the region entry and at its exit: after a register-allocation change the two can
  /// differ (an input in s[4:5] vs s[8:9], the result in s4 on both sides). Empty means
  /// identity. entry_equal/canary/observable use old naming; the new side receives inputs
  /// at entry_map(l) and is compared at exit_map(l). Never upgraded beyond ASSUMED.
  std::vector<std::pair<unsigned, unsigned>> entry_map, exit_map;
  [[nodiscard]] bool mapped() const { return !entry_map.empty() || !exit_map.empty(); }
  /// Permutations (forward old->new, inverse new->old) completed from the maps.
  [[nodiscard]] std::array<unsigned, kNumLocs> entry_forward() const;
  [[nodiscard]] std::array<unsigned, kNumLocs> entry_inverse() const;
  [[nodiscard]] std::array<unsigned, kNumLocs> exit_forward() const;
  [[nodiscard]] std::array<unsigned, kNumLocs> exit_inverse() const;
  /// Memory operations outstanding at the cutpoint (handoff 6.1): "NONE"; "DISJOINT"
  /// (outstanding, but their destinations are not read or written by the region);
  /// "CONFLICT" (an outstanding write targets a location the region reads or writes);
  /// "UNKNOWN" (an outstanding op has unknown effects). CONFLICT and UNKNOWN leave the
  /// context unresolved: the isolated run starts with every input ready, real hardware may not.
  std::string pending_effects = "NONE";
  /// Composition check (handoff 5.2): "HOLDS" when must-equality propagated from the
  /// kernel entry through common code and upstream regions' R_out covers entry_equal,
  /// "FAILS" otherwise (then the contract is UNRESOLVED), "NOT_CHECKED" for synthetic or
  /// hand-built contracts.
  std::string composition = "NOT_CHECKED";
  /// Changed code downstream of the region exit (other regions, control-flow changes)
  /// and what it reads on either side. A local mismatch on one of these locations may be
  /// compensated there, so it is a witness, not a verdict (handoff 5.2/13). A pass is not
  /// affected: identity R_out is the strongest exit relation.
  std::vector<std::string> downstream_changed;
  LocSet downstream_reads;
  VSet vdownstream_reads;
  ContractStatus status = ContractStatus::Unresolved;
  std::vector<std::string> depends_on;
  std::vector<Assumption> sources;

  [[nodiscard]] Json to_json() const;
  static std::optional<RegionContract> from_json(const Json &j, std::string &error);
};

/// @brief Parse "s4=s8,s5=s9" into a register map; nullopt (and @p error) when invalid
/// (unknown names, a location mapped twice, or two locations mapped to one).
[[nodiscard]] std::optional<std::vector<std::pair<unsigned, unsigned>>>
parse_register_map(std::string_view text, std::string &error);
[[nodiscard]] std::string format_register_map(const std::vector<std::pair<unsigned, unsigned>> &m);
/// @brief Complete a partial injective map to a permutation (forward old->new): a target
/// that is not itself mapped takes the start of its chain (s4=s8,s8=s9 gives s9 -> s4).
[[nodiscard]] std::array<unsigned, kNumLocs>
complete_register_map(const std::vector<std::pair<unsigned, unsigned>> &m);
[[nodiscard]] std::array<unsigned, kNumLocs>
invert_permutation(const std::array<unsigned, kNumLocs> &forward);

/// Fixed frame fields checked for preservation (names match MachineState::Frame).
[[nodiscard]] const std::vector<std::string> &frame_field_names();

} // namespace rocjitsu::isa_diff
