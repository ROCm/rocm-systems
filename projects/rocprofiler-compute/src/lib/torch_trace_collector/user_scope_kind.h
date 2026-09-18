// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Picks the ThreadLocalDebugInfo slot the collector publishes its wrap stack
// to. Which payload c10 expects depends on the loaded libtorch, so it is
// resolved on first use.

#pragma once

#include <c10/util/ThreadLocalDebugInfo.h>

namespace torch_trace_collector::detail
{

// True when the loaded libc10 lets callers own a debug info slot, which is
// PyTorch 2.13 and later.
bool c10_has_custom_debug_info_slots();

c10::DebugInfoKind user_scope_kind();

}  // namespace torch_trace_collector::detail
