// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "stack_entry.h"

#include <c10/util/ThreadLocalDebugInfo.h>

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

// The calling thread's state. Each thread walks its own marker stack.
ThreadState& thread_state();

// Pushes `chain` onto the thread stack, skipping any leading prefix that is
// already present. Returns the number of frames pushed.
std::size_t push_with_prefix_dedup(const std::vector<StackEntry>& chain);

}  // namespace torch_trace_collector::detail
