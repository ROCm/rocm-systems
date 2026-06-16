// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/helper_collapser.hpp"

#include <algorithm>
#include <ranges>
#include <utility>

namespace rocprofsys::output
{

namespace
{
bool
is_helper(const process_node& node)
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
    // Unknown size -> not a helper, so a transient try_file_size
    // failure can't silently demote a large output into a helper.
    if(!any_known) return false;
    return max_size < HELPER_MAX_SIZE_BYTES;
}

process_node
make_range_node(const std::vector<process_node>& group)
{
    const auto pids =
        group | std::views::transform([](const process_node& g) { return g.meta.pid; });
    const auto [min_pid, max_pid] = std::ranges::minmax(pids);

    process_node node{};
    node.collapsed =
        helper_range{ .min_pid = min_pid, .max_pid = max_pid, .count = group.size() };
    return node;
}

// Restructures one level of children: identifies helper siblings,
// removes them in-place, and (if >= 2) appends a single synthetic
// range node representing the collapsed group.
void
fold_helper_siblings(std::vector<process_node>& siblings)
{
    std::vector<process_node> kept_non_helpers;
    std::vector<process_node> helper_pool;
    kept_non_helpers.reserve(siblings.size());

    for(auto& s : siblings)
    {
        if(is_helper(s))
            helper_pool.push_back(std::move(s));
        else
            kept_non_helpers.push_back(std::move(s));
    }

    std::vector<process_node> out;
    out.reserve(kept_non_helpers.size() + 1);
    for(auto& k : kept_non_helpers)
        out.push_back(std::move(k));

    if(helper_pool.size() >= 2)
        out.push_back(make_range_node(helper_pool));
    else if(helper_pool.size() == 1)
        out.push_back(std::move(helper_pool.front()));

    siblings = std::move(out);
}
}  // namespace

std::vector<process_node>
collapse_helpers(std::vector<process_node> roots)
{
    // Bottom-up: fold each subtree's children first, then fold the
    // root level. for_each_post visits each node after its children,
    // so when we reach a node its children are already folded and
    // we apply the sibling-group folding to that node's now-final
    // children vector.
    for(auto& r : roots)
    {
        for_each_post(r, [](process_node& node) { fold_helper_siblings(node.children); });
    }
    fold_helper_siblings(roots);
    return roots;
}

}  // namespace rocprofsys::output
