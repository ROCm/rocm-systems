// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/process_tree_builder.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace rocprofsys::output
{

namespace
{
process_node
make_node(const process_metadata& meta)
{
    process_node node{};
    node.meta = meta;
    return node;
}

void
sort_rows_desc_by_size(process_node& node)
{
    std::sort(node.rows.begin(), node.rows.end(),
              [](const output_file& a, const output_file& b) {
                  if(a.size_bytes && b.size_bytes) return *a.size_bytes > *b.size_bytes;
                  return a.size_bytes.has_value() && !b.size_bytes.has_value();
              });
}

struct subtree_walk
{
    std::vector<pid_t>               order;
    std::unordered_map<pid_t, pid_t> parent_of;
};

// Stack-based pre-order enumeration. Iterative so deep parent chains
// (MPI fork generations) do not blow the call stack.
subtree_walk
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

// Bottom-up child attach. Reverse-walks the pre-order so each child
// is moved into its parent before the parent itself is moved out.
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
        dst.children.push_back(std::move(src));
    }
}
}  // namespace

build_result
build_tree(const std::vector<output_file>&      rows,
           const std::vector<process_metadata>& processes)
{
    build_result result{};

    std::unordered_map<pid_t, process_metadata> meta_by_pid;
    meta_by_pid.reserve(processes.size());
    for(const auto& p : processes)
        meta_by_pid.emplace(p.pid, p);

    std::unordered_set<pid_t> pids_in_rows;
    for(const auto& r : rows)
        pids_in_rows.insert(r.pid);

    std::unordered_map<pid_t, process_node> nodes;
    nodes.reserve(pids_in_rows.size());

    for(pid_t pid : pids_in_rows)
    {
        auto it = meta_by_pid.find(pid);
        if(it == meta_by_pid.end())
        {
            process_metadata stub{};
            stub.pid = pid;
            nodes.emplace(pid, make_node(stub));
            result.diagnostics.missing_metadata_pids.push_back(pid);
        }
        else
        {
            nodes.emplace(pid, make_node(it->second));
        }
    }

    for(const auto& r : rows)
    {
        auto it = nodes.find(r.pid);
        if(it != nodes.end()) it->second.rows.push_back(r);
    }

    for(auto& [pid, node] : nodes)
        sort_rows_desc_by_size(node);

    std::vector<pid_t> sorted_pids;
    sorted_pids.reserve(nodes.size());
    for(const auto& kv : nodes)
        sorted_pids.push_back(kv.first);
    std::sort(sorted_pids.begin(), sorted_pids.end());

    // O(N) precompute keeps subtree construction O(N) instead of O(N^2).
    std::unordered_map<pid_t, std::vector<pid_t>> children_by_ppid;
    children_by_ppid.reserve(nodes.size());
    for(pid_t pid : sorted_pids)
    {
        const auto& meta = nodes.at(pid).meta;
        if(meta.ppid != -1 && nodes.find(meta.ppid) != nodes.end())
            children_by_ppid[meta.ppid].push_back(pid);
    }
    for(auto& [_, vec] : children_by_ppid)
        std::sort(vec.begin(), vec.end());

    std::vector<pid_t> root_pids;
    for(pid_t pid : sorted_pids)
    {
        const auto& meta = nodes.at(pid).meta;
        if(meta.ppid == -1 || nodes.find(meta.ppid) == nodes.end())
            root_pids.push_back(pid);
    }

    auto take_subtree = [&](pid_t root_pid) -> process_node {
        auto walk = collect_subtree_order(root_pid, children_by_ppid);

        std::unordered_map<pid_t, process_node> built;
        built.reserve(walk.order.size());
        for(pid_t pid : walk.order)
            built.insert(nodes.extract(pid));

        attach_children_bottom_up(walk, built, root_pid);
        return std::move(built.at(root_pid));
    };

    result.roots.reserve(root_pids.size());
    for(pid_t pid : root_pids)
        result.roots.push_back(take_subtree(pid));

    std::sort(result.diagnostics.missing_metadata_pids.begin(),
              result.diagnostics.missing_metadata_pids.end());

    return result;
}

}  // namespace rocprofsys::output
