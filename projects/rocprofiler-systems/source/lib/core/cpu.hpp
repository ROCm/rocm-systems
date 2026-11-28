// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include <string>
#include <vector>

namespace rocprofsys::cpu
{

struct cpu_info
{
    long        processor   = -1;
    long        family      = -1;
    long        model       = -1;
    long        physical_id = -1;
    long        core_id     = -1;
    long        apicid      = -1;
    std::string vendor_id;
    std::string model_name;
};

std::vector<cpu_info>
process_cpu_info_data();

std::vector<cpu_info>
get_cpu_info();

size_t
device_count();

/// @brief Registers one CPU agent per socket, parsed from /proc/cpuinfo, in the agent
///        manager. Used when rocprofiler-sdk did not report any CPU agent.
void
query_cpu_agents();

}  // namespace rocprofsys::cpu
