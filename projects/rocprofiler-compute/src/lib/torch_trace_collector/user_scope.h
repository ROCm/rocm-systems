// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "stack_entry.h"

#include <c10/util/ThreadLocalDebugInfo.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace torch_trace_collector::detail
{

// Carries the main-thread USER_SCOPE chain to autograd workers.
class RoctxUserScopeChain : public c10::DebugInfoBase
{
public:
    explicit RoctxUserScopeChain(std::vector<StackEntry> c)
        : chain(std::move(c))
    {
    }

    std::vector<StackEntry> chain;
};

inline const c10::DebugInfoKind kRoctxDbgKind = c10::DebugInfoKind::TEST_INFO_2;

// Overlays the published USER_SCOPE chain onto the thread stack.
std::size_t apply_userscope_overlay();

// Pushes a USER_SCOPE frame and emits a ROCTX range. When non-empty,
// backend is appended to the range as "|<backend>".
void push_user_scope(const std::string& marker, const std::string& context, const std::string& backend);

void pop_user_scope();

}  // namespace torch_trace_collector::detail
