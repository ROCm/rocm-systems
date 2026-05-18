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

enum class role_hint
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
    // Sort-unique union of own gpu_ids and all descendants' gpu_ids.
    // Computed by classify() in its post-order pass. Same shape as
    // `role` and `collapsed` — these are classifier outputs, not
    // input tree data; a future refactor may group them into a
    // node_classification sub-struct.
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

// Post-order traversal: visit each child subtree before visiting the
// node itself. Used when a node's value depends on its children (the
// engine role rule, the effective gpu-id union).
//
// `fn` is passed by lvalue across siblings — std::forward<F>(fn)
// would move from a stateful functor on the first child and leave
// the rest dangling. Idiomatic for visitor templates.
template <typename F>
void
for_each_post(process_node& node, F&& fn)
{
    for(auto& c : node.children)
        for_each_post(c, fn);
    fn(node);
}

template <typename F>
void
for_each_post(const process_node& node, F&& fn)
{
    for(const auto& c : node.children)
        for_each_post(c, fn);
    fn(node);
}

// Pre-order traversal: visit the node before descending. Used when
// the node's value is independent of children's outputs.
template <typename F>
void
for_each_pre(process_node& node, F&& fn)
{
    fn(node);
    for(auto& c : node.children)
        for_each_pre(c, fn);
}

template <typename F>
void
for_each_pre(const process_node& node, F&& fn)
{
    fn(node);
    for(const auto& c : node.children)
        for_each_pre(c, fn);
}

}  // namespace rocprofsys::output
