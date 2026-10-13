// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <atomic>
#include <cstdint>

namespace torch_trace_collector::detail
{

// Runtime counters exposed through dump_stats().
struct Stats
{
    std::atomic<std::uint64_t> pushes{0};
    std::atomic<std::uint64_t> pops{0};
    std::atomic<std::uint64_t> snapshots_saved{0};
    std::atomic<std::uint64_t> snapshots_consumed{0};
    std::atomic<std::uint64_t> snapshots_dropped{0};
    std::atomic<std::uint64_t> snapshots_overwritten{0};
    std::atomic<std::uint64_t> callback_errors{0};
    std::atomic<std::uint64_t> user_scope_pushes{0};
    std::atomic<std::uint64_t> user_scope_pops{0};
    std::atomic<std::uint64_t> user_scope_inherits{0};
};

}  // namespace torch_trace_collector::detail
