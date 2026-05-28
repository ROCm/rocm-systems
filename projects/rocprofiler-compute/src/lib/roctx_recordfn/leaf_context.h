// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

namespace roctx_recordfn
{

inline constexpr const char* kAtenTopLevelLeaf     = "#1@aten:0";
inline constexpr const char* kAtenNestedLeaf       = "#1@aten.nested:0";
inline constexpr const char* kAutogradEngineLeaf   = "#1@autograd.engine:0";
inline constexpr const char* kAutogradBackwardLeaf = "#1@autograd.bwd:0";

// Producer-side source of truth for default leaf-context labels. The Python
// coverage parser consumes these exact tokens to infer whether C++ tier ran.
inline const char* default_leaf_context(bool is_backward_scope, std::int64_t seq_nr, bool stack_was_empty)
{
    if (is_backward_scope)
    {
        return (seq_nr >= 0) ? kAutogradBackwardLeaf : kAutogradEngineLeaf;
    }
    return stack_was_empty ? kAtenTopLevelLeaf : kAtenNestedLeaf;
}

}  // namespace roctx_recordfn
