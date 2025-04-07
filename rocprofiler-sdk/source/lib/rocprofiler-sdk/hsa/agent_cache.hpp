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

#pragma once

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

#include <string_view>

#include <rocprofiler-sdk/agent.h>

// Construct const and non-const accessor functions
#define CONST_NONCONST_ACCESSOR(RTYPE, NAME, VAL)                                                  \
    const RTYPE& NAME() const { return VAL; }                                                      \
    RTYPE&       NAME() { return VAL; }

namespace rocprofiler
{
namespace hsa
{
static const uint32_t LDS_BLOCK_SIZE = 128 * 4;

// Stores per-agent HSA information such as GPU and Kernel pools
// along with nearest CPU agent and its pool. Links rocprofiler_agent_t
// to its HSA agent. Note this class is only valid when HSA is
// init'd
class AgentCache
{
public:
    AgentCache(const rocprofiler_agent_t* rocp_agent,
               hsa_agent_t                hsa_agent,
               size_t                     index,
               hsa_agent_t                nearest_cpu,
               const AmdExtTable&         ext_table,
               const CoreApiTable&        api);
    ~AgentCache()                     = default;
    AgentCache(const AgentCache&)     = default;
    AgentCache(AgentCache&&) noexcept = default;

    AgentCache& operator=(const AgentCache&) = default;
    AgentCache& operator=(AgentCache&&) noexcept = default;

    // Provides const and a non-const accessor functions.
    CONST_NONCONST_ACCESSOR(hsa_amd_memory_pool_t, cpu_pool, m_cpu_pool);
    CONST_NONCONST_ACCESSOR(hsa_amd_memory_pool_t, kernarg_pool, m_kernarg_pool);
    CONST_NONCONST_ACCESSOR(hsa_amd_memory_pool_t, gpu_pool, m_gpu_pool);
    CONST_NONCONST_ACCESSOR(hsa_agent_t, get_hsa_agent, m_hsa_agent);
    CONST_NONCONST_ACCESSOR(hsa_agent_t, near_cpu, m_nearest_cpu);

    hsa_queue_t*               profile_queue() const { return m_profile_queue; }
    const rocprofiler_agent_t* get_rocp_agent() const { return m_rocp_agent; }
    std::string_view           name() const { return m_name; }
    size_t                     index() const { return m_index; }

    void init_device_counting_service_queue(const CoreApiTable& api, const AmdExtTable& ext) const;
    bool operator==(const rocprofiler_agent_t*) const;
    bool operator==(hsa_agent_t) const;

private:
    // Agent info
    const rocprofiler_agent_t* m_rocp_agent = nullptr;
    size_t                     m_index{0};  // rocprofiler_agent index

    // GPU Agent
    hsa_agent_t m_hsa_agent{.handle = 0};
    hsa_agent_t m_nearest_cpu{.handle = 0};

    // memory pools
    hsa_amd_memory_pool_t m_cpu_pool{.handle = 0};
    hsa_amd_memory_pool_t m_kernarg_pool{.handle = 0};
    hsa_amd_memory_pool_t m_gpu_pool{.handle = 0};

    std::string_view     m_name          = {};
    mutable hsa_queue_t* m_profile_queue = {};
};

inline bool
AgentCache::operator==(const rocprofiler_agent_t* agent) const
{
    return (agent == m_rocp_agent);
}

inline bool
AgentCache::operator==(hsa_agent_t agent) const
{
    return (agent.handle == m_hsa_agent.handle);
}
}  // namespace hsa
}  // namespace rocprofiler
