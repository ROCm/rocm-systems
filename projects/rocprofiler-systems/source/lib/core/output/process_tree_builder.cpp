// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/process_tree_builder.hpp"

#include <algorithm>
#include <ranges>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>

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
    std::ranges::sort(node.rows, [](const output_file& a, const output_file& b) {
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

// Moves the root's subtree out of `nodes` (consuming those entries) and
// returns it assembled. `nodes` is passed explicitly so the extraction
// is visible at the call site rather than a captured side effect.
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
}  // namespace

build_result
build_tree(std::span<const output_file> rows, std::span<const process_metadata> processes)
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
    std::ranges::sort(sorted_pids);

    // O(N) precompute keeps subtree construction O(N) instead of O(N^2).
    std::unordered_map<pid_t, std::vector<pid_t>> children_by_ppid;
    children_by_ppid.reserve(nodes.size());
    for(pid_t pid : sorted_pids)
    {
        const auto& meta = nodes.at(pid).meta;
        if(meta.ppid != -1 && nodes.contains(meta.ppid))
            children_by_ppid[meta.ppid].push_back(pid);
    }
    for(auto& [_, vec] : children_by_ppid)
        std::ranges::sort(vec);

    std::vector<pid_t> root_pids;
    for(pid_t pid : sorted_pids)
    {
        const auto& meta = nodes.at(pid).meta;
        if(meta.ppid == -1 || !nodes.contains(meta.ppid)) root_pids.push_back(pid);
    }

    result.roots.reserve(root_pids.size());
    for(pid_t pid : root_pids)
        result.roots.push_back(extract_subtree(nodes, children_by_ppid, pid));

    std::ranges::sort(result.diagnostics.missing_metadata_pids);

    return result;
}

namespace
{
std::uintmax_t
sum_known_row_sizes(std::span<const output_file> rows)
{
    std::uintmax_t total = 0;
    for(const auto& file : rows)
        if(file.size_bytes) total += *file.size_bytes;
    return total;
}

bool
is_small_process(const process_node& node)
{
    if(!node.meta.gpu_ids.empty()) return false;
    if(node.collapsed.has_value()) return false;
    if(node.rows.empty()) return true;

    std::uintmax_t max_size  = 0;
    bool           any_known = false;
    for(const auto& r : node.rows)
    {
        if(r.size_bytes)
        {
            any_known = true;
            if(*r.size_bytes > max_size) max_size = *r.size_bytes;
        }
    }
    // Unknown size -> not a small process, so a transient try_file_size
    // failure can't silently demote a large output into a small process.
    if(!any_known) return false;
    return max_size < SMALL_PROCESS_MAX_SIZE_BYTES;
}

process_node
make_range_node(const std::vector<process_node>& group)
{
    const auto pids =
        group | std::views::transform([](const process_node& g) { return g.meta.pid; });
    const auto [min_pid, max_pid] = std::ranges::minmax(pids);

    process_node node{};
    node.collapsed = collapsed_process_range{ .min_pid = min_pid,
                                              .max_pid = max_pid,
                                              .count   = group.size() };
    return node;
}

// Restructures one level of children: identifies small-process siblings,
// removes them in-place, and (if >= 2) appends a single synthetic
// range node representing the collapsed group.
void
collapse_small_process_siblings(std::vector<process_node>& siblings)
{
    std::vector<process_node> kept_larger_processes;
    std::vector<process_node> small_process_pool;
    kept_larger_processes.reserve(siblings.size());

    for(auto& s : siblings)
    {
        if(is_small_process(s))
            small_process_pool.push_back(std::move(s));
        else
            kept_larger_processes.push_back(std::move(s));
    }

    std::vector<process_node> out;
    out.reserve(kept_larger_processes.size() + 1);
    for(auto& k : kept_larger_processes)
        out.push_back(std::move(k));

    if(small_process_pool.size() >= 2)
        out.push_back(make_range_node(small_process_pool));
    else if(small_process_pool.size() == 1)
        out.push_back(std::move(small_process_pool.front()));

    siblings = std::move(out);
}
}  // namespace

std::vector<process_node>
collapse_small_processes(std::vector<process_node> roots)
{
    // Bottom-up: fold each subtree's children first, then fold the
    // root level. for_each_post visits each node after its children,
    // so when we reach a node its children are already folded and
    // we apply the sibling-group folding to that node's now-final
    // children vector.
    for(auto& r : roots)
    {
        for_each_post(r, [](process_node& node) {
            collapse_small_process_siblings(node.children);
        });
    }
    collapse_small_process_siblings(roots);
    return roots;
}

void
compute_subtree_sizes(std::vector<process_node>& roots)
{
    for(auto& root : roots)
    {
        for_each_post(root, [](process_node& node) {
            node.own_size_bytes        = sum_known_row_sizes(node.rows);
            node.cumulative_size_bytes = node.own_size_bytes;
            for(const auto& child : node.children)
                node.cumulative_size_bytes += child.cumulative_size_bytes;
        });
    }
}

}  // namespace rocprofsys::output
