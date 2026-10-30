// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "kineto_loader.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace profiler_hub::pytorch
{
inline constexpr std::uint16_t k_not_in_tree = 0xFFFF;

struct tree_options_t
{
    std::int64_t bleed_ns                     = 1000;
    bool         equal_start_longer_is_parent = true;
};

struct tree_stats_t
{
    std::uint64_t placed, orphans_kept_as_child, small_overlap_pops,
        children_ending_after_parent;
    std::uint16_t max_level;
};

struct tree_result_t
{
    std::vector<std::uint16_t> depth;
    tree_stats_t               stats;
};

// Per-thread host nesting from TraceLens's TraceToTree.build_tree, except that an event
// TraceLens would leave without a parent is kept as a child of the stack top, and on an
// equal start the longer event becomes the parent.
tree_result_t
build_host_tree(std::span<const event_t> events_of_one_thread,
                const loaded_trace_t&    t,
                const tree_options_t&    o);
}  // namespace profiler_hub::pytorch
