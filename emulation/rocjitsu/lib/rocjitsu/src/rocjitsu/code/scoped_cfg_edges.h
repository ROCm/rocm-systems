// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file scoped_cfg_edges.h
/// @brief Context-sensitive call/return edges for kernel-scoped CFG analyses.
///
/// @details BasicBlock deliberately separates call edges from ordinary CFG
/// successors, because a shared callee can return to different continuations
/// depending on the call site. Liveness and EXEC-state analyses still need to
/// see those flows, so these helpers turn each scoped call edge into temporary
/// analysis edges (`caller -> callee`, `return -> continuation`) without
/// mutating the CFG or creating cross-kernel return edges. Shared by the DBT
/// binary translator and the instrumentor so both analyze the same graph.

#pragma once

#include "rocjitsu/code/analysis/liveness.h" // ScopedCfgEdge

#include <cstdint>
#include <span>
#include <unordered_set>
#include <vector>

namespace rocjitsu {

class BasicBlock;

/// @brief Materialize context-sensitive call/return edges for a block scope.
///
/// @details Each in-scope call edge yields `caller -> callee` plus, for every
/// validated return block, `return -> continuation`. @p text is the raw .text
/// image, used to confirm a terminator reads the call edge's saved return SGPR.
[[nodiscard]] std::vector<ScopedCfgEdge>
scoped_call_liveness_edges(std::span<BasicBlock *const> blocks, std::span<const uint8_t> text);

/// @brief Collect validated return-like terminator offsets for a block scope.
///
/// @details A call-return `s_setpc_b64` is left indirect in emitted code; this
/// marks only the return offsets reachable from a `BasicBlock::CallEdge` whose
/// callee and continuation both belong to @p blocks.
[[nodiscard]] std::unordered_set<uint64_t>
scoped_call_return_offsets(std::span<BasicBlock *const> blocks, std::span<const uint8_t> text);

} // namespace rocjitsu
