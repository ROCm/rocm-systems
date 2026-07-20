// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/output/process_tree.hpp"

#include <algorithm>
#include <iterator>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rocprofsys::output
{

namespace
{
void
sort_rows_desc_by_size(process_node& node)
{
    std::ranges::sort(node.rows, [](const artifact& a, const artifact& b) {
        return a.size_bytes > b.size_bytes;
    });
}

process_node
make_node(const process_metadata& meta, std::vector<artifact> rows)
{
    process_node node{};
    node.meta = meta;
    node.rows = std::move(rows);
    sort_rows_desc_by_size(node);
    node.own_size_bytes        = sum_sizes(node.rows);
    node.cumulative_size_bytes = node.own_size_bytes;
    return node;
}

struct subtree_walk
{
    std::vector<pid_t>               order;
    std::unordered_map<pid_t, pid_t> parent_of;
};

// Stack-based pre-order enumeration. Iterative so deep parent chains
// (MPI fork generations) do not blow the call stack.
[[nodiscard]] subtree_walk
collect_subtree_order(
    pid_t root_pid, const std::unordered_map<pid_t, std::vector<pid_t>>& children_by_ppid)
{
    subtree_walk       walk{};
    std::vector<pid_t> stack{ root_pid };
    while(!stack.empty())
    {
        const pid_t pid = stack.back();
        stack.pop_back();
        walk.order.push_back(pid);
        auto it = children_by_ppid.find(pid);
        if(it == children_by_ppid.end()) continue;
        for(pid_t cp : it->second)
        {
            walk.parent_of[cp] = pid;
            stack.push_back(cp);
        }
    }
    return walk;
}

void
attach_children_bottom_up(const subtree_walk&                      walk,
                          std::unordered_map<pid_t, process_node>& built, pid_t root_pid)
{
    for(auto rit = walk.order.rbegin(); rit != walk.order.rend(); ++rit)
    {
        const pid_t pid = *rit;
        if(pid == root_pid) continue;
        const pid_t ppid = walk.parent_of.at(pid);
        auto&       dst  = built.at(ppid);
        auto&       src  = built.at(pid);
        dst.cumulative_size_bytes += src.cumulative_size_bytes;
        dst.children.push_back(std::move(src));
    }
}

process_node
extract_subtree(std::unordered_map<pid_t, process_node>&             nodes,
                const std::unordered_map<pid_t, std::vector<pid_t>>& children_by_ppid,
                pid_t                                                root_pid)
{
    const auto walk = collect_subtree_order(root_pid, children_by_ppid);

    std::unordered_map<pid_t, process_node> built;
    built.reserve(walk.order.size());
    for(pid_t pid : walk.order)
        built.insert(nodes.extract(pid));

    attach_children_bottom_up(walk, built, root_pid);
    return std::move(built.at(root_pid));
}

[[nodiscard]] std::unordered_map<pid_t, process_metadata>
build_metadata_index(std::span<const process_metadata> processes)
{
    std::unordered_map<pid_t, process_metadata> meta_by_pid;
    meta_by_pid.reserve(processes.size());
    for(const auto& p : processes)
        meta_by_pid.emplace(p.pid, p);
    return meta_by_pid;
}

[[nodiscard]] std::unordered_map<pid_t, std::vector<artifact>>
build_rows_index(std::span<const artifact> rows)
{
    std::unordered_map<pid_t, std::vector<artifact>> rows_by_pid;
    for(const auto& r : rows)
        rows_by_pid[r.pid].push_back(r);
    return rows_by_pid;
}

[[nodiscard]] std::unordered_map<pid_t, process_node>
build_all_nodes(std::span<const artifact>                          rows,
                const std::unordered_map<pid_t, process_metadata>& meta_by_pid,
                std::unordered_map<pid_t, std::vector<artifact>>&  rows_by_pid,
                process_tree_diagnostics&                          diagnostics)
{
    std::unordered_set<pid_t> pids_in_rows;
    for(const auto& r : rows)
        pids_in_rows.insert(r.pid);

    std::unordered_map<pid_t, process_node> nodes;
    nodes.reserve(pids_in_rows.size());
    for(pid_t pid : pids_in_rows)
    {
        auto meta_it      = meta_by_pid.find(pid);
        auto rows_it      = rows_by_pid.find(pid);
        auto rows_for_pid = (rows_it != rows_by_pid.end()) ? std::move(rows_it->second)
                                                           : std::vector<artifact>{};

        if(meta_it == meta_by_pid.end())
        {
            process_metadata stub{};
            stub.pid = pid;
            nodes.emplace(pid, make_node(stub, std::move(rows_for_pid)));
            diagnostics.missing_metadata_pids.push_back(pid);
        }
        else
        {
            nodes.emplace(pid, make_node(meta_it->second, std::move(rows_for_pid)));
        }
    }
    return nodes;
}

[[nodiscard]] std::vector<pid_t>
sorted_node_pids(const std::unordered_map<pid_t, process_node>& nodes)
{
    std::vector<pid_t> sorted_pids;
    sorted_pids.reserve(nodes.size());
    std::ranges::copy(nodes | std::views::keys, std::back_inserter(sorted_pids));
    std::ranges::sort(sorted_pids);
    return sorted_pids;
}

[[nodiscard]] std::unordered_map<pid_t, std::vector<pid_t>>
build_children_index(std::span<const pid_t>                         sorted_pids,
                     const std::unordered_map<pid_t, process_node>& nodes)
{
    std::unordered_map<pid_t, std::vector<pid_t>> children_by_ppid;
    children_by_ppid.reserve(nodes.size());
    for(pid_t pid : sorted_pids)
    {
        const auto& meta = nodes.at(pid).meta;
        if(meta.ppid != NO_PID && nodes.contains(meta.ppid))
            children_by_ppid[meta.ppid].push_back(pid);
    }
    for(auto& [_, vec] : children_by_ppid)
        std::ranges::sort(vec);
    return children_by_ppid;
}

[[nodiscard]] std::vector<pid_t>
find_root_pids(std::span<const pid_t>                         sorted_pids,
               const std::unordered_map<pid_t, process_node>& nodes)
{
    std::vector<pid_t> root_pids;
    for(pid_t pid : sorted_pids)
    {
        const auto& meta = nodes.at(pid).meta;
        if(meta.ppid == NO_PID || !nodes.contains(meta.ppid)) root_pids.push_back(pid);
    }
    return root_pids;
}

// Anything still left in `nodes` after every root's subtree has been
// extracted was never reachable from a root — the only way that happens is
// a ppid cycle
[[nodiscard]] std::vector<pid_t>
collect_unreachable_pids(const std::unordered_map<pid_t, process_node>& nodes)
{
    std::vector<pid_t> unreachable;
    std::ranges::copy(nodes | std::views::keys, std::back_inserter(unreachable));
    return unreachable;
}
}  // namespace

process_tree::process_tree(std::span<const artifact>         rows,
                           std::span<const process_metadata> processes)
{
    const auto meta_by_pid = build_metadata_index(processes);
    auto       rows_by_pid = build_rows_index(rows);
    auto       nodes = build_all_nodes(rows, meta_by_pid, rows_by_pid, m_diagnostics);

    const auto sorted_pids      = sorted_node_pids(nodes);
    const auto children_by_ppid = build_children_index(sorted_pids, nodes);
    const auto root_pids        = find_root_pids(sorted_pids, nodes);

    m_roots.reserve(root_pids.size());
    for(pid_t pid : root_pids)
        m_roots.push_back(extract_subtree(nodes, children_by_ppid, pid));

    m_diagnostics.cyclic_ppid_pids = collect_unreachable_pids(nodes);

    std::ranges::sort(m_diagnostics.missing_metadata_pids);
    std::ranges::sort(m_diagnostics.cyclic_ppid_pids);
}

}  // namespace rocprofsys::output
