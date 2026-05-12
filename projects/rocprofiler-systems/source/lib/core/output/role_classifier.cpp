// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/role_classifier.hpp"

namespace rocprofsys::output
{

namespace
{
bool
any_child_is_gpu(const process_node& node)
{
    for(const auto& c : node.children)
    {
        if(c.role.has_value() && *c.role == role_hint::gpu) return true;
    }
    return false;
}

void
classify_node(process_node& node, pid_t main_pid)
{
    // Bottom-up: classify children first so the engine rule can see
    // their assigned role.
    for(auto& c : node.children)
        classify_node(c, main_pid);

    if(node.meta.pid == main_pid)
        node.role = role_hint::main;
    else if(!node.meta.gpu_ids.empty())
        node.role = role_hint::gpu;
    else if(any_child_is_gpu(node))
        node.role = role_hint::engine;
    // else leave nullopt
}
}  // namespace

void
classify(std::vector<process_node>& roots, pid_t main_pid)
{
    for(auto& r : roots)
        classify_node(r, main_pid);
}

}  // namespace rocprofsys::output
