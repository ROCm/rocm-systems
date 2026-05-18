// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/role_classifier.hpp"

#include <algorithm>

namespace rocprofsys::output
{

namespace
{
// Sort-unique union of `dst` and `src`, in place into `dst`.
void
merge_unique(std::vector<int>& dst, const std::vector<int>& src)
{
    dst.insert(dst.end(), src.begin(), src.end());
    std::sort(dst.begin(), dst.end());
    dst.erase(std::unique(dst.begin(), dst.end()), dst.end());
}

bool
any_child_is_gpu(const process_node& node)
{
    for(const auto& c : node.children)
    {
        if(c.role.has_value() && *c.role == role_hint::gpu) return true;
    }
    return false;
}
}  // namespace

// Pipeline order: build_tree -> collapse_helpers -> classify -> render.
// classify() runs AFTER collapse_helpers() so effective_gpu_ids is
// computed over the final post-collapse tree shape. Helper-range
// nodes carry no gpu_ids (collapser rejects any gpu-bearing node),
// so the union over surviving descendants stays correct.
void
classify(std::vector<process_node>& roots, pid_t main_pid)
{
    // Post-order so the engine rule can see each child's classified
    // role, and so effective_gpu_ids can roll up bottom-up.
    for(auto& r : roots)
    {
        for_each_post(r, [main_pid](process_node& node) {
            node.effective_gpu_ids = node.meta.gpu_ids;
            for(const auto& c : node.children)
                merge_unique(node.effective_gpu_ids, c.effective_gpu_ids);

            if(node.meta.pid == main_pid)
                node.role = role_hint::main;
            else if(!node.meta.gpu_ids.empty())
                node.role = role_hint::gpu;
            else if(any_child_is_gpu(node))
                node.role = role_hint::engine;
        });
    }
}

}  // namespace rocprofsys::output
