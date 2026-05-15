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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/common/logging.hpp"

#include <fmt/core.h>
#include <stdexcept>

#include "lib/rocprofiler-sdk/context/context.hpp"

namespace
{
// This function checks to see if the provided
// pool has the HSA_AMD_SEGMENT_GLOBAL property. If the kern_arg flag is true,
// the function adds an additional requirement that the pool have the
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT property. If kern_arg is false,
// pools must NOT have this property.
// Upon finding a pool that meets these conditions, HSA_STATUS_INFO_BREAK is
// returned. HSA_STATUS_SUCCESS is returned if no errors were encountered, but
// no pool was found meeting the requirements. If an error is encountered, we
// return that error.
hsa_status_t
FindGlobalPool(hsa_amd_memory_pool_t pool, void* data, bool kern_arg)
{
    if(!data) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    auto [api_ptr, pool_ptr] =
        *static_cast<std::pair<const AmdExtTable*, hsa_amd_memory_pool_t*>*>(data);
    hsa_amd_segment_t segment;
    ROCP_FATAL_IF(api_ptr->hsa_amd_memory_pool_get_info_fn(
                      pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment) == HSA_STATUS_ERROR)
        << "Could not get pool segment";
    if(HSA_AMD_SEGMENT_GLOBAL != segment) return HSA_STATUS_SUCCESS;

    uint32_t flag;
    ROCP_FATAL_IF(api_ptr->hsa_amd_memory_pool_get_info_fn(
                      pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag) == HSA_STATUS_ERROR)
        << "Could not get flag value";
    uint32_t karg_st = flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
    if((karg_st == 0 && kern_arg) || (karg_st != 0 && !kern_arg))
    {
        return HSA_STATUS_SUCCESS;
    }
    *(pool_ptr) = pool;
    return HSA_STATUS_INFO_BREAK;
}

// This is the call-back function for hsa_amd_agent_iterate_memory_pools() that
// finds a pool with the properties of HSA_AMD_SEGMENT_GLOBAL and that is NOT
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
hsa_status_t
FindStandardPool(hsa_amd_memory_pool_t pool, void* data)
{
    return FindGlobalPool(pool, data, false);
}

// This is the call-back function for hsa_amd_agent_iterate_memory_pools() that
// finds a pool with the properties of HSA_AMD_SEGMENT_GLOBAL and that IS
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
hsa_status_t
FindKernArgPool(hsa_amd_memory_pool_t pool, void* data)
{
    return FindGlobalPool(pool, data, true);
}

void
init_cpu_pool(const AmdExtTable& api, rocprofiler::hsa::AgentCache& agent)
{
    std::pair<const AmdExtTable*, hsa_amd_memory_pool_t*> params =
        std::make_pair(&api, &agent.cpu_pool());

    auto status =
        api.hsa_amd_agent_iterate_memory_pools_fn(agent.near_cpu(), FindStandardPool, &params);
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "HSA Command Buffer Pool is not initialized";

    params.second = &agent.kernarg_pool();
    status =
        api.hsa_amd_agent_iterate_memory_pools_fn(agent.near_cpu(), FindKernArgPool, &(params));
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "HSA Output Buffer Pool is not initialized";
}

void
init_gpu_pool(const AmdExtTable& api, rocprofiler::hsa::AgentCache& agent)
{
    std::pair<const AmdExtTable*, hsa_amd_memory_pool_t*> params =
        std::make_pair(&api, &agent.gpu_pool());
    auto status =
        api.hsa_amd_agent_iterate_memory_pools_fn(agent.get_hsa_agent(), FindStandardPool, &params);

    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "HSA GPU Pool is not initialized";
}

}  // namespace

namespace rocprofiler
{
namespace hsa
{
void
AgentCache::init_device_counting_service_queue(const CoreApiTable& api,
                                               const AmdExtTable&  ext) const
{
    static std::mutex           m_mutex;
    std::lock_guard<std::mutex> lock(m_mutex);

    using context         = rocprofiler::context::context;
    const auto* agent_ctx = []() -> const context* {
        for(auto& ctx : rocprofiler::context::get_registered_contexts())
        {
            if(ctx->device_counter_collection) return ctx;
        }
        return nullptr;
    }();

    if(!agent_ctx || m_profile_queue) return;

    // create the queue and set it to high_priority
    CHECK(api.hsa_queue_create_fn) << "no hsa_queue_create_fn in api table";
    auto status = api.hsa_queue_create_fn(get_hsa_agent(),
                                          64,
                                          HSA_QUEUE_TYPE_SINGLE,
                                          nullptr,
                                          nullptr,
                                          UINT32_MAX,
                                          UINT32_MAX,
                                          &m_profile_queue);
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "HSA Queue is not initialized";

    CHECK(ext.hsa_amd_queue_set_priority_fn) << "no hsa_amd_queue_set_priority_fn in api table";
    ext.hsa_amd_queue_set_priority_fn(m_profile_queue, HSA_AMD_QUEUE_PRIORITY_HIGH);
}

AgentCache::AgentCache(const rocprofiler_agent_t* rocp_agent,
                       hsa_agent_t                hsa_agent,
                       size_t                     index,
                       hsa_agent_t                nearest_cpu,
                       const AmdExtTable&         ext_table,
                       const CoreApiTable&        api)
: m_rocp_agent{rocp_agent}
, m_index{index}
, m_hsa_agent{hsa_agent}
, m_nearest_cpu{nearest_cpu}
, m_name{rocp_agent->name}
{
    // Construct CPU/GPU pools
    try
    {
        init_cpu_pool(ext_table, *this);
        init_gpu_pool(ext_table, *this);
        init_device_counting_service_queue(api, ext_table);
    } catch(std::runtime_error& e)
    {
        ROCP_WARNING << fmt::format(
            "Buffer creation for Agent {} failed ({}), Some profiling options will be unavailable.",
            rocp_agent->node_id,
            e.what());
    }
}

}  // namespace hsa
}  // namespace rocprofiler
