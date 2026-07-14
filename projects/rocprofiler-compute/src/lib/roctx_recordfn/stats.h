// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <atomic>
#include <cstdint>

namespace roctx_recordfn::detail
{

// Runtime counters exposed through dump_stats().
struct Stats
{
    std::atomic<std::uint64_t> pushes{0};
    std::atomic<std::uint64_t> pops{0};
    std::atomic<std::uint64_t> snapshots_saved{0};
    std::atomic<std::uint64_t> snapshots_consumed{0};
    std::atomic<std::uint64_t> snapshots_dropped{0};
    std::atomic<std::uint64_t> callback_errors{0};
    std::atomic<std::uint64_t> user_scope_pushes{0};
    std::atomic<std::uint64_t> user_scope_pops{0};
    std::atomic<std::uint64_t> user_scope_inherits{0};
};

inline Stats g_stats;

inline void inc(std::atomic<std::uint64_t>& counter)
{
    counter.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace roctx_recordfn::detail
