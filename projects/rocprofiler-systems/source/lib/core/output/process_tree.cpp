// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/output/output_summary.hpp"

#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rocprofsys::output
{

namespace
{
process_node
make_node(process_metadata meta, std::vector<artifact> rows)
{
    process_node node{};
    node.meta = std::move(meta);
    node.rows = std::move(rows);
    std::ranges::sort(node.rows, [](const artifact& a, const artifact& b) {
        return a.size_bytes > b.size_bytes;
    });
    return node;
}

// One ordered map, filled directly from rows/metadata — no separate
// meta/rows index structures, no explicit pid sort (the map is already
// ordered by pid).
[[nodiscard]] std::map<pid_t, process_node>
build_nodes(std::span<const artifact> rows, std::span<const process_metadata> processes,
            process_tree_diagnostics& diagnostics)
{
    std::unordered_map<pid_t, process_metadata> meta_by_pid;
    for(const auto& p : processes)
        meta_by_pid.emplace(p.pid, p);

    std::map<pid_t, std::vector<artifact>> rows_by_pid;
    for(const auto& r : rows)
        rows_by_pid[r.pid].push_back(r);

    std::map<pid_t, process_node> nodes;
    for(auto& [pid, pid_rows] : rows_by_pid)
    {
        auto meta_it = meta_by_pid.find(pid);
        if(meta_it == meta_by_pid.end())
        {
            diagnostics.missing_metadata_pids.push_back(pid);
            nodes.emplace(pid,
                          make_node(process_metadata{ .pid = pid }, std::move(pid_rows)));
        }
        else
        {
            nodes.emplace(pid, make_node(meta_it->second, std::move(pid_rows)));
        }
    }
    return nodes;
}

// A node and everything nested under it is unreachable from any real root
// (it only got there via a ppid chain that cycles back on itself); record
// every pid in the subtree as excluded rather than just the one we happened
// to stop iterating on.
void
mark_cyclic(const process_node& node, process_tree_diagnostics& diagnostics)
{
    diagnostics.cyclic_ppid_pids.push_back(node.meta.pid);
    for(const auto& child : node.children)
        mark_cyclic(child, diagnostics);
}
}  // namespace

process_tree::process_tree(std::span<const artifact>         rows,
                           std::span<const process_metadata> processes)
{
    auto nodes = build_nodes(rows, processes, m_diagnostics);

    std::unordered_set<pid_t> known_pids;
    for(const auto& [pid, node] : nodes)
    {
        known_pids.insert(pid);
    }

    std::unordered_set<pid_t> attached_pids;
    for(auto it = nodes.rbegin(); it != nodes.rend(); ++it)
    {
        const pid_t pid  = it->first;
        const pid_t ppid = it->second.meta.ppid;
        if(ppid == NO_PID)
        {
            continue;
        }

        auto parent_it = nodes.find(ppid);
        if(parent_it == nodes.end() || attached_pids.contains(ppid))
        {
            continue;
        }

        parent_it->second.children.insert(parent_it->second.children.begin(),
                                          std::move(it->second));
        attached_pids.insert(pid);
    }

    for(auto& [pid, node] : nodes)
    {
        if(attached_pids.contains(pid))
        {
            continue;
        }  // moved into a parent above

        if(node.meta.ppid != NO_PID && known_pids.contains(node.meta.ppid))
        {
            mark_cyclic(node, m_diagnostics);
        }
        else
        {
            m_roots.push_back(std::move(node));
        }
    }

    std::ranges::sort(m_diagnostics.missing_metadata_pids);
    std::ranges::sort(m_diagnostics.cyclic_ppid_pids);
}

}  // namespace rocprofsys::output
