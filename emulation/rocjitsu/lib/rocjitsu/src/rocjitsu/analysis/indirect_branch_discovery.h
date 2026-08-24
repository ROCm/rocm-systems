// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file indirect_branch_discovery.h
/// @brief Dataflow discovery of statically-built indirect branch targets.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu {

class Instruction;

/// @brief How indirect-target discovery identifies externally reachable blocks.
enum class ExternalEntryPolicy : uint8_t {
  /// Treat every predecessorless block as a possible external function entry.
  /// This preserves conservative recovery when callers do not have a complete
  /// list of entries for all functions sharing one .text section.
  InferPredecessorless,
  /// Treat only section entry and caller-supplied leaders as external entries.
  /// Callers may use this when their supplied leader list contains every
  /// externally reachable entry; other predecessorless blocks remain unreachable.
  ExplicitOnly,
};

/// @brief Recovered indirect PC-relative branch through a statically-built PC register.
///
/// @details BasicBlock construction uses this metadata in two ways. A recovered
/// setpc, or a swappc that does not validate as a returning call, becomes an
/// ordinary CFG successor from the consumer block. A swappc whose callee returns
/// through the recorded destination SGPR is modeled as a context-sensitive call
/// edge instead, because its continuation depends on the call site. DBT keeps
/// the same metadata so relocation can rewrite the original getpc-relative
/// address-builder range in place after final target offsets are known.
struct IndirectCallFixup {
  uint64_t source_getpc_offset = 0;          ///< Source offset of the s_getpc_b64 producer.
  uint64_t source_recovery_begin_offset = 0; ///< First source byte of replaceable builder code.
  uint64_t source_recovery_end_offset = 0;   ///< One-past-end source byte of builder code.
  uint64_t source_call_offset = 0;           ///< Source offset of the setpc/swappc consumer.
  uint64_t source_target_offset = 0;         ///< Recovered source branch target offset.
  uint16_t source_call_sreg = 0;             ///< Low SGPR the consumer reads the PC pair from.
  /// @brief Low SGPR of the pair the builder range itself writes.
  ///
  /// @details Usually the same pair the consumer reads, and for a plain getpc/add/swappc chain
  /// it always is. A lane-banked dispatcher breaks the two apart: it builds the address in a
  /// scratch pair, stashes it with `v_writelane_b32`, and restores it into the consumer's pair
  /// with `v_readlane_b32` much later. patch_recovered_builder_fixups regenerates only the add
  /// half of the builder and leaves the original `s_getpc_b64` where it is, so the replacement
  /// has to name the pair that getpc writes. Naming the consumer's pair instead emits an add
  /// against a getpc that wrote a different register: it corrupts the consumer's pair, breaks
  /// the stash the dispatcher still reads back, and leaves a getpc/add pairing that the next
  /// translation's relocation lattice reads as a code address naming no body at all.
  uint16_t source_builder_sreg = 0;
  bool source_is_call = false; ///< Whether the consumer is a call-like swappc.
  /// @brief True when the recovered fact for this consumer was incomplete: at least
  /// one predecessor path left the PC pair at an unconstrained value. The concrete
  /// targets are still valid for relocation and liveness, but the consumer must NOT
  /// be replaced with a direct transfer window — an unconstrained path would be
  /// redirected to a concrete target it never dynamically reaches.
  bool source_incomplete = false;
  /// @brief True when the recovery range contains an `s_wait_xcnt` the replacement
  /// must reproduce. The canonical builder overwrites the whole range and writes
  /// the same SGPR pair the drain was ordering, so dropping the wait would let the
  /// pair be rewritten while an operation still has the old value in flight.
  bool source_requires_xcnt_drain = false;
  /// @brief True when this builder's consumer was replaced by a direct transfer window.
  ///
  /// @details The window recomputes the address, so the builder survives only as dead code -- and
  /// with no consumer left, a later translation pass can account for it only if the rewrite
  /// lattice can see it. That obliges the rewrite to use the literal64 add form, which is the only
  /// one the lattice models. A builder whose consumer remains needs no such constraint: the
  /// consumer is recovered again on the next pass and accounts for the builder through that.
  bool consumer_replaced_by_window = false;
  uint16_t source_return_sreg = 0;           ///< Low SGPR receiving the return PC for calls.
  uint64_t target_getpc_offset = 0;          ///< Relocated offset of the s_getpc_b64 producer.
  uint64_t target_recovery_begin_offset = 0; ///< Relocated first byte of replaceable builder code.
  uint64_t target_recovery_end_offset = 0;   ///< Relocated one-past-end byte of builder code.
};

/// @brief One statically discovered `s_getpc_b64`-rooted PC-relative address producer.
///
/// @details A recovered indirect branch is only one consumer of a getpc builder.
/// The same construct materializes function pointers that are copied, spilled,
/// or passed as arguments before they reach a dynamic transfer. DBT relocates
/// `.text`, so every such builder that is copied verbatim keeps its original
/// delta and therefore computes `new_pc + old_delta` — a stale address. Reporting
/// every builder lets the translator prove the complementary property: that no
/// stale PC-derived value can exist in a kernel scope at all.
///
/// One record is produced per `s_getpc_b64` instruction, whether or not the pass
/// could follow it. @ref resolved distinguishes the two cases.
struct PcAddressBuilder {
  uint64_t source_getpc_offset = 0;          ///< Source offset of the s_getpc_b64 producer.
  uint64_t source_recovery_begin_offset = 0; ///< First source byte of replaceable builder code.
  uint64_t source_recovery_end_offset = 0;   ///< One-past-end source byte of builder code.
  /// Text-relative byte offset the builder leaves in its SGPR pair at
  /// @ref source_recovery_end_offset. Signed because an unrelocatable data
  /// reference can compute an address below the section.
  int64_t source_target_offset = 0;
  uint16_t source_sreg = 0; ///< Low SGPR of the pair the builder writes.
  /// @brief True when the pass followed this getpc to a single concrete offset.
  ///
  /// False means an unmodeled write reached the pair before any stable point,
  /// or two incompatible values were observed for the same producer. Such a
  /// producer cannot be made relocation-correct and must clear any whole-scope
  /// relocation invariant that depends on it.
  bool resolved = false;
  /// @brief True when [source_recovery_begin_offset, source_recovery_end_offset)
  /// holds only the builder's own arithmetic, with no unrelated instruction
  /// between steps. The relocation patcher rewrites that interval as one
  /// contiguous run and NOPs the remainder, so a non-contiguous range would
  /// erase an intervening instruction. A non-contiguous producer cannot back a
  /// whole-scope relocation invariant even though its final value is known.
  bool contiguous = true;
  /// @brief True when two observations of this producer disagreed on its value.
  ///
  /// Distinct from a cleared @ref resolved, which also covers a producer this
  /// pass simply never followed to a setpc. A poisoned producer is one no single
  /// delta rewrite can satisfy, so a caller reasoning about whether every code
  /// address is relocated must fail closed on it rather than defer to another
  /// analysis that happened to track the same getpc.
  bool poisoned = false;

  friend bool operator==(const PcAddressBuilder &, const PcAddressBuilder &) = default;
};

/// @brief Whether an AMDGPU physical VGPR is callee-saved by the ABI.
///
/// @details A callee-saved VGPR keeps its value across a call, so a value stashed
/// there is still the caller's after the callee returns. @p phys_vgpr is the
/// resolved physical index; gfx1250 VGPR_MSB banking can push it past 255, and the
/// ABI table only covers v0-v255, so a banked register above that range is not
/// proven callee-saved and reports false. The default device and graphics
/// conventions share this VGPR set, so unlike the SGPR rule below no explicit
/// intersection is necessary.
[[nodiscard]] bool is_callee_saved_vgpr(uint16_t phys_vgpr);

/// @brief Whether an SGPR is preserved by every AMDGPU calling convention the
/// translator may encounter for a device function.
///
/// @details Code objects do not record whether a particular helper uses the
/// default device convention or the graphics convention. This predicate uses
/// their intersection, so an unsummarized call cannot preserve a stale value
/// merely because one convention saves the register.
[[nodiscard]] bool is_callee_saved_sgpr(uint16_t sgpr);

/// @brief Discover concrete targets for statically-built setpc/swappc consumers.
///
/// @details This pass runs before BasicBlock storage is finalized because any
/// recovered target must become a block leader. The pass is deliberately
/// conservative: it only records a target when an s_setpc_b64/s_swappc_b64
/// source SGPR pair can be proven to hold one or more bounded, concrete
/// s_getpc_b64-relative text offsets. If the target set reaches the cap, the
/// consumer is left unresolved rather than creating a partial edge set. If
/// path-insensitive joins leave the lattice incomplete but still expose a small
/// concrete target set, those concrete targets are returned; BasicBlock decides
/// whether each target is an ordinary CFG successor or a context-sensitive call
/// edge.
///
/// The implementation first builds a direct-CFG block skeleton, scans each
/// block once to summarize writes to PC-builder SGPR pairs, runs bounded
/// forward dataflow over those block summaries, and finally emits fixups for
/// direct intra-block consumers plus deferred inter-block consumers with bounded
/// concrete entry values. Newly recovered edges are fed back into the temporary
/// graph for a bounded number of rounds so nested helper-return patterns can be
/// discovered without making the initial analysis depend on guessed edges.
///
/// @param insts Decoded instructions with Instruction::src_loc() populated.
/// @param text Raw .text bytes matching @p insts.
/// @param arch ISA architecture used for scalar instruction matching.
/// @param extra_leaders Additional known block starts that are also EXTERNAL ENTRIES. Under
///        ExplicitOnly every one of these is treated as externally entered, which discards the
///        incoming SGPR-pair facts a caller would establish, so pass only offsets that really are
///        entered from outside the decoded graph. Offsets that merely need to start a block belong
///        in @p extra_split_points.
/// @param entry_policy Whether predecessorless blocks are inferred to be external entries.
/// @param pc_builders Optional sink for every discovered PC-relative address
///        producer, sorted by `source_getpc_offset`. Populated only when the
///        section actually contains a recoverable indirect consumer, because a
///        section with no dynamic transfer has no stale-PC branch hazard to
///        prove anything about.
/// @returns Recovered indirect branch/call metadata.
/// @param extra_split_points Offsets that must start a block without being treated as external
///        entries. Function-entry symbols and stored-pointer targets belong here: they are real
///        boundaries, but most of them are ordinary helpers their callers reach by a decoded edge.
[[nodiscard]] std::vector<IndirectCallFixup> discover_indirect_branch_edges(
    std::span<const Instruction *const> insts, std::span<const uint8_t> text, rj_code_arch_t arch,
    std::span<const uint64_t> extra_leaders = {},
    ExternalEntryPolicy entry_policy = ExternalEntryPolicy::InferPredecessorless,
    std::vector<PcAddressBuilder> *pc_builders = nullptr,
    std::span<const uint64_t> extra_split_points = {},
    rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID);

} // namespace rocjitsu
