// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <atomic>
#include <charconv>
#include <cstdint>
#include <string>

namespace torch_trace_collector
{

inline constexpr const char* kAtenTopLevelLeaf     = "aten:0";
inline constexpr const char* kAtenNestedLeaf       = "aten.nested:0";
inline constexpr const char* kAutogradEngineLeaf   = "autograd.engine:0";
inline constexpr const char* kAutogradBackwardLeaf = "autograd.bwd:0";

// Returns the default leaf-context label emitted by the producer for a
// given scope. The Python coverage parser matches these tokens exactly.
inline const char* default_leaf_label(bool is_backward_scope, std::int64_t seq_nr, bool stack_was_empty)
{
    if (is_backward_scope)
    {
        return (seq_nr >= 0) ? kAutogradBackwardLeaf : kAutogradEngineLeaf;
    }
    return stack_was_empty ? kAtenTopLevelLeaf : kAtenNestedLeaf;
}

// Returns the leaf context for one operator invocation as "#<counter>@<label>".
// The counter is process-wide, so every invocation is distinct.
inline std::string default_leaf_context(bool is_backward_scope, std::int64_t seq_nr, bool stack_was_empty)
{
    static std::atomic<std::uint64_t> invocations{0};
    const std::uint64_t invocation = invocations.fetch_add(1, std::memory_order_relaxed) + 1;

    char       digits[20];
    const auto end = std::to_chars(digits, digits + sizeof(digits), invocation).ptr;

    std::string context;
    context += '#';
    context.append(digits, end);
    context += '@';
    context += default_leaf_label(is_backward_scope, seq_nr, stack_was_empty);
    return context;
}

}  // namespace torch_trace_collector
