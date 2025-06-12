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
#include <atomic>
#include <future>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
//#include "/opt/rocm/include/aqlprofile-sdk/aql_profile_v2.h"
#include "aql_profile_v2.h"

#define CHECK_HSA(x) if ((x) != HSA_STATUS_SUCCESS) { std::cerr << __FILE__ << " error at " << __LINE__ << std::endl; exit(-1); }

inline bool operator==(const aqlprofile_pmc_event_t& a, const aqlprofile_pmc_event_t& b) {
    return a.event_id == b.event_id && a.block_name == b.block_name;
}

inline bool operator<(const aqlprofile_pmc_event_t& a, const aqlprofile_pmc_event_t& b) {
    return (a.block_name == b.block_name) ? (a.event_id < b.event_id) : (a.block_name < b.block_name);
}

struct AgentInfo
{
    AgentInfo() { gfxip.resize(64); }

    std::string gfxip;
    hsa_agent_t hsa_agent;
    aqlprofile_agent_info_t info;
    aqlprofile_agent_handle_t handle;
    std::map<std::string, std::vector<aqlprofile_pmc_event_t>> counters;

    static void iterate_agents();

    static std::vector<std::shared_ptr<AgentInfo>> gpu_agents;
    static hsa_agent_t cpu_agent;
    static hsa_amd_memory_pool_t cpu_pool;
    static hsa_amd_memory_pool_t kernarg_pool;

private:
    void add_event(aqlprofile_pmc_event_t block, const std::string& counter, int block_cnt, int event_id);

    static hsa_status_t get_agent_handle_cb(hsa_agent_t agent, void* userdata);
};

class Queue
{
public:
    Queue(std::shared_ptr<AgentInfo>& _agent);
    ~Queue() { if (queue) hsa_queue_destroy(queue); }

    bool Submit(hsa_ext_amd_aql_pm4_packet_t* packet);
    void flush();

    hsa_queue_t* queue = nullptr;
    std::shared_ptr<AgentInfo> agent;
};