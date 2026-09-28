// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "stack_entry.h"

#include <c10/util/ThreadLocalDebugInfo.h>

#include <memory>
#include <vector>

namespace torch_trace_collector::detail
{

// Per-thread marker stack. guards has one entry per push_user_scope frame.
struct ThreadState
{
    std::vector<StackEntry>                           stack;
    std::vector<std::unique_ptr<c10::DebugInfoGuard>> guards;
};

ThreadState& thread_state();

}  // namespace torch_trace_collector::detail
