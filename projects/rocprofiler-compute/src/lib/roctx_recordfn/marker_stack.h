// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "stack_entry.h"

#include <c10/util/ThreadLocalDebugInfo.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace roctx_recordfn::detail
{

// Per-thread marker state. guards holds one DebugInfoGuard per push_user_scope
// frame; RecordFunction frames are pushed onto stack without a guard.
struct ThreadState
{
    std::vector<StackEntry>                           stack;
    std::vector<std::unique_ptr<c10::DebugInfoGuard>> guards;
};

inline thread_local ThreadState g_thread;

// Pushes `chain` onto the thread stack, skipping any leading prefix that is
// already present. Returns the number of frames pushed.
inline std::size_t push_with_prefix_dedup(const std::vector<StackEntry>& chain)
{
    const std::size_t maxc   = std::min(chain.size(), g_thread.stack.size());
    std::size_t       common = 0;
    for (; common < maxc; ++common)
    {
        if (chain[common].marker != g_thread.stack[common].marker ||
            chain[common].context != g_thread.stack[common].context)
        {
            break;
        }
    }
    std::size_t pushed = 0;
    for (std::size_t i = common; i < chain.size(); ++i)
    {
        g_thread.stack.push_back(chain[i]);
        ++pushed;
    }
    return pushed;
}

}  // namespace roctx_recordfn::detail
