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
  uint16_t source_call_sreg = 0;             ///< Low SGPR of the recovered PC pair.
  bool source_is_call = false;               ///< Whether the consumer is a call-like swappc.
  uint16_t source_return_sreg = 0;           ///< Low SGPR receiving the return PC for calls.
  uint64_t target_getpc_offset = 0;          ///< Relocated offset of the s_getpc_b64 producer.
  uint64_t target_recovery_begin_offset = 0; ///< Relocated first byte of replaceable builder code.
  uint64_t target_recovery_end_offset = 0;   ///< Relocated one-past-end byte of builder code.
};

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
/// @param extra_leaders Additional known block starts, usually kernel entries.
/// @returns Recovered indirect branch/call metadata.
[[nodiscard]] std::vector<IndirectCallFixup>
discover_indirect_branch_edges(std::span<const Instruction *const> insts,
                               std::span<const uint8_t> text, rj_code_arch_t arch,
                               std::span<const uint64_t> extra_leaders = {});

} // namespace rocjitsu
