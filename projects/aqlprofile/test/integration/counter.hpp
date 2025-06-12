// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <assert.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include "aql_profile_v2.h"
#include <stdlib.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <vector>
#include <map>
#include "agent.hpp"

class AQLPacket
{
    using desc_t = aqlprofile_buffer_desc_flags_t;
public:
    AQLPacket(AgentInfo& _agent, const std::vector<std::string>& counters);
    ~AQLPacket() { aqlprofile_pmc_delete_packets(this->handle); };

    void iterate();

    static hsa_status_t Alloc(void** ptr, size_t size, desc_t flags, void* data);
    static void         Free(void* ptr, void* data);
    static hsa_status_t Copy(void* dst, const void* src, size_t size, void* data);

    std::map<std::string, int64_t> get()
    {
        std::map<std::string, int64_t> ret;
        for (auto& [event, counter] : counter_names)
            ret.emplace(counter, results.at(event));
        return ret;
    }

    std::map<aqlprofile_pmc_event_t, std::string> counter_names;
    std::map<aqlprofile_pmc_event_t, int64_t> results;
    std::map<aqlprofile_pmc_event_t, int64_t> prev_results;

    aqlprofile_handle_t handle{0};
    hsa_agent_t hsa_agent;
    aqlprofile_pmc_aql_packets_t packets;

    bool delta = false;
};
