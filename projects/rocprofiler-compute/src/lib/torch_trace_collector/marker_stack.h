// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "stack_entry.h"

#include <c10/util/ThreadLocalDebugInfo.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace torch_trace_collector::detail
{

// Per-thread marker state. guards holds one slot per push_user_scope frame,
// null when the guard could not be built, so every such frame has a slot for
// pop_user_scope() to pop. RecordFunction frames are pushed onto stack without
// a guard.
struct ThreadState
{
    std::vector<StackEntry>                           stack;
    std::vector<std::unique_ptr<c10::DebugInfoGuard>> guards;
};

// The calling thread's state. Each thread walks its own marker stack, so this
// state is per-thread where process_state() is per-process.
inline ThreadState& thread_state()
{
    static thread_local ThreadState state;
    return state;
}

// Pushes `chain` onto the thread stack, skipping any leading prefix that is
// already present. Returns the number of frames pushed.
inline std::size_t push_with_prefix_dedup(const std::vector<StackEntry>& chain)
{
    std::vector<StackEntry>& stack  = thread_state().stack;
    const std::size_t        maxc   = std::min(chain.size(), stack.size());
    std::size_t              common = 0;
    for (; common < maxc; ++common)
    {
        if (chain[common].marker != stack[common].marker || chain[common].context != stack[common].context)
        {
            break;
        }
    }
    std::size_t pushed = 0;
    for (std::size_t i = common; i < chain.size(); ++i)
    {
        stack.push_back(chain[i]);
        ++pushed;
    }
    return pushed;
}

}  // namespace torch_trace_collector::detail
