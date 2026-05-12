// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output/process_tree_builder.hpp"

#include <sys/types.h>

#include <vector>

namespace rocprofsys::output
{

// Numbered precedence:
//   1. *main*   if node.pid == main_pid
//   2. *gpu*    if !node.meta.gpu_ids.empty()
//   3. *engine* if a child has role_hint::gpu (else nullopt)
// Evaluated bottom-up so the engine rule can see its children's roles.
void
classify(std::vector<process_node>& roots, pid_t main_pid);

}  // namespace rocprofsys::output
