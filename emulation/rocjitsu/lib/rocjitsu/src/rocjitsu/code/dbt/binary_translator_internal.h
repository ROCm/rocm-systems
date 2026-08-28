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

#include <cstdint>
#include <span>
#include <unordered_set>

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

} // namespace internal
} // namespace rocjitsu
