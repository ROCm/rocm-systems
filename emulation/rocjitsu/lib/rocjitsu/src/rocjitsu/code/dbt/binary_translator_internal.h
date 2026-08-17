// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file binary_translator_internal.h
/// @brief Internal BinaryTranslator helpers exposed only for unit testing.
///
/// @details These declarations are implementation details of
/// binary_translator.cpp. They are surfaced in a header solely so focused unit
/// tests can exercise soundness gates that are otherwise unreachable through the
/// public translate() entry point without a large end-to-end fixture. Do not use
/// them from production code.

#pragma once

#include "rocjitsu/code/dbt/translation_diagnostic.h"
#include "rocjitsu/code/patch/code_object_patcher.h"

#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocjitsu {

class BasicBlock;

namespace internal {

/// @brief Decide whether every external entry into an incomplete-consumer scope
///        is an entry-state root that cannot carry an original `.text` pointer.
///
/// @details See the definition in binary_translator.cpp for the full soundness
/// argument. A block with an in-scope ordinary predecessor is reachable within
/// the scope and needs no check. A predecessor-less block is an external entry;
/// it is safe only as a hardware kernel entry or a getpc-recovered in-scope call
/// target. A relocation-table-dispatched callee is never safe: the dispatch
/// delivers unconstrained caller-supplied SGPR arguments, so it is rejected even
/// when it also carries an in-scope CallEdge.
///
/// @param blocks Every block in the kernel-local scope, in any order.
/// @param hardware_entry_offsets Start offsets of ABI-initialized hardware
///        entries (the scope entry and any kernarg-preload firmware entry).
/// @param table_callee_offsets Start offsets of blocks that are reachable as a
///        relocation-table dispatch callee anywhere in the object.
/// @returns true when the scope may keep an incomplete dynamic transfer; false
///          (fail closed) when any external root is unconstrained.
[[nodiscard]] bool
scope_roots_are_entry_state(std::span<BasicBlock *const> blocks,
                            const std::unordered_set<uint64_t> &hardware_entry_offsets,
                            const std::unordered_set<uint64_t> &table_callee_offsets);

/// @brief A code-address builder awaiting the final placement of the body it names.
///
/// @details Every builder waits, including one whose target this scope emitted itself. The
/// canonical placement and the variant-conflict set are both still being written while scopes are
/// placed, so a decision taken in the loop is a decision taken against a partial answer. Carrying
/// the emitting scope's own copy here keeps the deferred resolution able to prefer it without
/// having to re-derive which scope asked.
struct PendingCodeRelocation {
  uint64_t target_getpc_offset = 0;
  uint64_t target_literal_offset = 0;
  uint64_t source_target_text_offset = 0;
  /// @brief This scope's own placement of the target, when it emitted one.
  std::optional<uint64_t> local_target_text_offset;
};

/// @brief Point every deferred code-address builder at the placement its target received.
///
/// @details Runs after every scope is placed, which is the first moment a target's final offset
/// is known. A source offset emitted more than once has no single answer -- a runtime-dereferenced
/// code address cannot choose between clones -- so that fails closed rather than picking one,
/// matching how relocate_relative_text_addends treats a conflicting relocation addend.
///
/// Surfaced here because the two inputs that decide a refusal cannot be produced together by any
/// buildable image: the variant-conflict set is populated only by virtual-LDS sidecar variants,
/// which exist for the CDNA4-to-CDNA3 pair alone, while a PC-relative address builder is recovered
/// only from a getpc plus `s_add_nc_u64`, an encoding CDNA4 does not have. A focused test is the
/// only way to drive both refusal orderings.
///
/// @param pending Deferred builders in the order the scope walk queued them, so an entry queued
///        before the conflict was discovered comes first.
/// @param text_relocations Every scope's source-to-target block placement, used only for a target
///        no scope nominated canonical.
/// @param canonical_placement Final placement of each address-taken body, keyed by source offset.
/// @param canonical_placement_variant_conflict Source offsets whose clones disagreed on variant.
/// @param code_relocations Receives one entry per resolved builder.
/// @param diagnostics Receives the refusal reason when this returns false.
/// @returns false when the object must be left unchanged.
[[nodiscard]] bool resolve_pending_code_relocations(
    const std::vector<PendingCodeRelocation> &pending,
    const std::vector<TextOffsetRelocation> &text_relocations,
    const std::unordered_map<uint64_t, uint64_t> &canonical_placement,
    const std::unordered_set<uint64_t> &canonical_placement_variant_conflict,
    std::vector<PcRelativeTextRelocation> &code_relocations,
    std::vector<TranslationDiagnostic> &diagnostics);

} // namespace internal
} // namespace rocjitsu
