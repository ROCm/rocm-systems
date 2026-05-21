// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output/process_metadata.hpp"
#include "core/output_file_registry.hpp"

#include <sys/types.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace rocprofsys::output
{

enum class role_hint : std::uint8_t
{
    main,
    gpu,
    engine,
};

struct helper_range
{
    pid_t       min_pid{ -1 };
    pid_t       max_pid{ -1 };
    std::size_t count{ 0 };
};

struct process_node
{
    process_metadata            meta;
    std::vector<output_file>    rows;
    std::optional<role_hint>    role;
    std::optional<helper_range> collapsed;
    std::vector<process_node>   children;
    // Sort-unique union over self + descendants, filled by classify().
    std::vector<int> effective_gpu_ids;
};

struct build_diagnostics
{
    std::vector<pid_t> missing_metadata_pids;
};

struct build_result
{
    // Orphan PIDs (PPID absent from the input set) attach as
    // additional roots alongside the main process.
    std::vector<process_node> roots;
    build_diagnostics         diagnostics;
};

// Pure: no I/O, no globals. Per-node rows are sorted descending by
// size_bytes.
[[nodiscard]] build_result
build_tree(const std::vector<output_file>&      rows,
           const std::vector<process_metadata>& processes);

// Post-order visitor. `fn` is taken by lvalue so a stateful functor
// is not moved-from between siblings.
template <typename F>
void
for_each_post(process_node& node, F&& fn)
{
    for(auto& c : node.children)
        for_each_post(c, fn);
    fn(node);
}

}  // namespace rocprofsys::output
