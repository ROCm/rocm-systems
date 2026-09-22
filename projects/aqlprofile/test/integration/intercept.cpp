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

#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <atomic>
#include <cstring>
#include <hsa/hsa.h>
#include <hsa_ext_amd.h>
#include <hsa_api_trace.h>
#include <hsa_ven_amd_aqlprofile.h>

#define CHECK_HSA(x) { auto _status = (x); if (_status != HSA_STATUS_SUCCESS) { std::cerr << __FILE__ << ':' << __LINE__ << std::endl; abort(); } }

extern "C" const uint32_t HSA_AMD_TOOL_PRIORITY = 25;
decltype(hsa_amd_profiling_set_profiler_enabled)* hsa_amd_profiling_set_profiler_enabled_fn = nullptr;
decltype(hsa_amd_memory_pool_allocate)* hsa_amd_memory_pool_allocate_fn = nullptr;
decltype(hsa_amd_agents_allow_access)* hsa_amd_agents_allow_access_fn = nullptr;
decltype(hsa_amd_memory_pool_free)* hsa_amd_memory_pool_free_fn = nullptr;
decltype(hsa_signal_store_screlease)* hsa_signal_store_screlease_fn = nullptr;
decltype(hsa_queue_load_read_index_relaxed)* hsa_queue_load_read_index_relaxed_fn = nullptr;
decltype(hsa_queue_add_write_index_relaxed)* hsa_queue_add_write_index_relaxed_fn = nullptr;

decltype(hsa_amd_memory_pool_get_info)* hsa_amd_memory_pool_get_info_fn = nullptr;
decltype(hsa_agent_get_info)* hsa_agent_get_info_fn = nullptr;
decltype(hsa_amd_agent_iterate_memory_pools)* hsa_amd_agent_iterate_memory_pools_fn = nullptr;
decltype(hsa_queue_create)* hsa_queue_create_fn = nullptr;

hsa_amd_memory_pool_t cpu_pool;

hsa_status_t FindGlobalPool(hsa_amd_memory_pool_t pool, void* data)
{
    hsa_amd_segment_t segment;
    CHECK_HSA(hsa_amd_memory_pool_get_info_fn(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment));

    if(HSA_AMD_SEGMENT_GLOBAL != segment) return HSA_STATUS_SUCCESS;

    uint32_t flag;
    CHECK_HSA(hsa_amd_memory_pool_get_info_fn(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag));
    uint32_t karg_st = flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;

    if (karg_st == 0) cpu_pool = pool;

    return HSA_STATUS_SUCCESS;
}

hsa_status_t iterate_agent_cb(hsa_agent_t agent, void* userdata)
{
    hsa_device_type_t type;

    CHECK_HSA(hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_DEVICE, &type));
    if (type != HSA_DEVICE_TYPE_CPU) return HSA_STATUS_SUCCESS;
    
    CHECK_HSA(hsa_amd_agent_iterate_memory_pools_fn(agent, FindGlobalPool, nullptr));
    return HSA_STATUS_SUCCESS;
}

bool queue_submit(hsa_queue_t* queue, hsa_ext_amd_aql_pm4_packet_t* packet)
{
    const uint64_t write_idx = hsa_queue_add_write_index_relaxed_fn(queue, 1);

    size_t index      = (write_idx % queue->size) * sizeof(hsa_ext_amd_aql_pm4_packet_t);
    auto*  queue_slot = reinterpret_cast<uint32_t*>(size_t(queue->base_address) + index);  // NOLINT

    const auto* slot_data = reinterpret_cast<const uint32_t*>(packet);

    std::memcpy(&queue_slot[1], &slot_data[1], sizeof(hsa_ext_amd_aql_pm4_packet_t) - sizeof(uint32_t));
    auto* header = reinterpret_cast<std::atomic<uint32_t>*>(queue_slot);

    header->store(slot_data[0], std::memory_order_release);
    hsa_signal_store_screlease_fn(queue->doorbell_signal, write_idx);

    int loops = 0;
    while(hsa_queue_load_read_index_relaxed_fn(queue) <= write_idx)
    {
        loops++;
        usleep(1);
        if(loops > 10000)
        {
            std::cerr << "Packet submission failed!" << std::endl;
            return false;
        }
    }
    return true;
}

void set_profiler_active_on_queue(
    hsa_agent_t  hsa_agent,
    hsa_queue_t* queue
) {
    hsa_ext_amd_aql_pm4_packet_t packet{};
    hsa_ven_amd_aqlprofile_profile_t profile{};
    profile.agent = hsa_agent;

    // Query for cmd buffer size
    CHECK_HSA(hsa_ven_amd_aqlprofile_get_info(&profile, HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD, nullptr));

    // Allocate cmd buffer
    const size_t mask = 0x1000 - 1;
    auto         size = (profile.command_buffer.size + mask) & ~mask;

    CHECK_HSA(hsa_amd_memory_pool_allocate_fn(cpu_pool, size, 
                                              HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, &profile.command_buffer.ptr));
    CHECK_HSA(hsa_amd_agents_allow_access_fn(1, &hsa_agent, nullptr, profile.command_buffer.ptr));

    CHECK_HSA(hsa_ven_amd_aqlprofile_get_info(&profile, HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD, &packet));

    queue_submit(queue, &packet);
    hsa_amd_memory_pool_free_fn(profile.command_buffer.ptr);
}


hsa_status_t QueueCreateInterceptor(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
                                    void* data, uint32_t private_segment_size,
                                    uint32_t group_segment_size, hsa_queue_t** queue)
{
    CHECK_HSA(hsa_queue_create_fn(agent, size, type, callback, data, private_segment_size, group_segment_size, queue));
    //CHECK_HSA(hsa_amd_profiling_set_profiler_enabled_fn(*queue, true));
    set_profiler_active_on_queue(agent, *queue);
    return HSA_STATUS_SUCCESS;
}

extern "C" __attribute__((visibility("default"))) bool
OnLoad(HsaApiTable* table, uint64_t, uint64_t, const char* const*)
{
    hsa_queue_create_fn = table->core_->hsa_queue_create_fn;
    // Install the Queue intercept
    table->core_->hsa_queue_create_fn = QueueCreateInterceptor;
    hsa_amd_profiling_set_profiler_enabled_fn = table->amd_ext_->hsa_amd_profiling_set_profiler_enabled_fn;
    hsa_amd_memory_pool_allocate_fn = table->amd_ext_->hsa_amd_memory_pool_allocate_fn;
    hsa_amd_agents_allow_access_fn = table->amd_ext_->hsa_amd_agents_allow_access_fn;
    hsa_amd_memory_pool_free_fn = table->amd_ext_->hsa_amd_memory_pool_free_fn;
    hsa_signal_store_screlease_fn = table->core_->hsa_signal_store_screlease_fn;
    hsa_queue_load_read_index_relaxed_fn = table->core_->hsa_queue_load_read_index_relaxed_fn;
    hsa_queue_add_write_index_relaxed_fn = table->core_->hsa_queue_add_write_index_relaxed_fn;
    hsa_amd_memory_pool_get_info_fn = table->amd_ext_->hsa_amd_memory_pool_get_info_fn;
    hsa_amd_agent_iterate_memory_pools_fn = table->amd_ext_->hsa_amd_agent_iterate_memory_pools_fn;
    hsa_agent_get_info_fn = table->core_->hsa_agent_get_info_fn;

    CHECK_HSA(table->core_->hsa_iterate_agents_fn(iterate_agent_cb, nullptr));

    return true;
}
