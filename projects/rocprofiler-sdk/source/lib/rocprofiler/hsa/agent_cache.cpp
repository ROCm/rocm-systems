// Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
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

#include "agent_cache.hpp"

#include <glog/logging.h>
#include <filesystem>
#include <fstream>
#include <optional>

#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"

// For Pre-ROCm 6.0 releases
#if ROCPROFILER_HSA_RUNTIME_VERSION <= 100900
#    define HSA_AMD_AGENT_INFO_NEAREST_CPU 0xA113
#endif

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
    LOG_IF(FATAL,
           api_ptr->hsa_amd_memory_pool_get_info_fn(
               pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment) == HSA_STATUS_ERROR)
        << "Could not get pool segment";
    if(HSA_AMD_SEGMENT_GLOBAL != segment) return HSA_STATUS_SUCCESS;

    uint32_t flag;
    LOG_IF(FATAL,
           api_ptr->hsa_amd_memory_pool_get_info_fn(
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
    if(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
    {
        throw std::runtime_error("Error: Command Buffer Pool is not initialized");
    }

    params.second = &agent.kernarg_pool();
    status =
        api.hsa_amd_agent_iterate_memory_pools_fn(agent.near_cpu(), FindKernArgPool, &(params));
    if(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
    {
        throw std::runtime_error("Error: Output Buffer Pool is not initialized");
    }
}

void
init_gpu_pool(const AmdExtTable& api, rocprofiler::hsa::AgentCache& agent)
{
    std::pair<const AmdExtTable*, hsa_amd_memory_pool_t*> params =
        std::make_pair(&api, &agent.gpu_pool());
    auto status =
        api.hsa_amd_agent_iterate_memory_pools_fn(agent.get_agent(), FindStandardPool, &params);

    if(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
    {
        throw std::runtime_error("Error: GPU Pool is not initialized");
    }
}

}  // namespace

namespace rocprofiler
{
namespace hsa
{
AgentCache::AgentCache(rocprofiler_agent_t   agent_t,
                       size_t                index,
                       const ::CoreApiTable& table,
                       const AmdExtTable&    ext)
: _agent_t(agent_t)
, _index(index)
, _name(agent_t.name)
{
    // Get HSA Agents
    std::vector<hsa_agent_t> agents;
    table.hsa_iterate_agents_fn(
        [](hsa_agent_t agent, void* data) {
            CHECK_NOTNULL(static_cast<std::vector<hsa_agent_t>*>(data))->emplace_back(agent);
            return HSA_STATUS_SUCCESS;
        },
        &agents);

    // In case HSA_AMD_AGENT_INFO_NEAREST_CPU is non-functional, default to original v1 behavior
    // of last CPU agent being nearest.
    std::optional<hsa_agent_t> last_cpu;

    bool found = false;
    // Find the HSA agent that is represented by rocprofiler_agent_t
    for(const auto& agent : agents)
    {
        hsa_device_type_t type = HSA_DEVICE_TYPE_CPU;
        if(table.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
        {
            throw std::runtime_error("hsa_agent_get_info failed to find device");
        }

        if(type != HSA_DEVICE_TYPE_GPU)
        {
            if(type == HSA_DEVICE_TYPE_CPU && !last_cpu) last_cpu = agent;
            continue;
        }

        uint32_t node_id = 0;
        if(table.hsa_agent_get_info_fn(
               agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID), &node_id) !=
           HSA_STATUS_SUCCESS)
        {
            throw std::runtime_error("hsa_agent_get_info failed to find driver id");
        }

        // Match rocprofiler_agent_t to hsa_agent for GPU agents
        if(_index != node_id) continue;

        if(table.hsa_agent_get_info_fn(
               agent,
               static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NEAREST_CPU),
               &_nearest_cpu) != HSA_STATUS_SUCCESS)
        {
            _nearest_cpu = last_cpu ? *last_cpu : hsa_agent_t{.handle = 0};
        }

        found  = true;
        _agent = agent;
    }

    if(!found)
    {
        throw std::runtime_error(fmt::format("Could not find GPU id = {}", agent_t.id.handle));
    }

    // Construct CPU/GPU pools

    try
    {
        init_cpu_pool(ext, *this);
        init_gpu_pool(ext, *this);
    } catch(std::runtime_error& e)
    {
        LOG(WARNING) << fmt::format(
            "Buffer creation for Agent {} failed ({}), Some profiling options will be unavialable.",
            agent_t.id.handle,
            e.what());
    }
}

}  // namespace hsa
}  // namespace rocprofiler
