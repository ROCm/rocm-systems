// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

namespace torch_trace_collector::detail
{

// Pushes a marker frame, publishes the thread stack to ThreadLocalDebugInfo,
// and emits a ROCTX range. A non-empty backend is appended as "|<backend>".
void push_user_scope(const std::string& marker, const std::string& context, const std::string& backend);

// Pops the matching marker frame and ROCTX range.
void pop_user_scope();

}  // namespace torch_trace_collector::detail
