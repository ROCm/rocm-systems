// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <sys/types.h>

#include <string>
#include <vector>

namespace rocprofsys::output
{

// Per-process metadata carried into the Output Summary pipeline.
// Lives in its own tiny header so the registry can value-store it
// without pulling in the rest of the tree-builder API.
struct process_metadata
{
    pid_t            pid{ -1 };
    pid_t            ppid{ -1 };
    std::string      command;
    std::vector<int> gpu_ids;
};

}  // namespace rocprofsys::output
