// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/thread_trace/shared_trace_resources.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/thread_trace/hsa_util.hpp"

#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace thread_trace
{
namespace
{
// Over-allocate, then round the returned pointer up to a 4K page. The round-up consumes
// up to one page of the padding; the second padding page is a tail guard so cache-flush
// operations at the end of the buffer cannot spill into the next allocation.
constexpr size_t PAGE_ALIGN_PADDING = 0x2000;
constexpr size_t PAGE_ALIGN_MASK    = 0xFFFul;

struct shared_buffer_t
{
    void*                               raw     = nullptr;
    void*                               aligned = nullptr;
    decltype(hsa_amd_memory_pool_free)* free_fn = nullptr;
};

struct manager_entry_t
{
    std::optional<hsa_agent_t>    hsa_agent    = std::nullopt;
    trace_resource_requirements_t requirements = {};
    agent_trace_resources_ptr_t   resources    = {};
};

class SharedTraceResourceManager
{
public:
    void register_requirements(rocprofiler_agent_id_t agent_id,
                               hsa_agent_t            hsa_agent,
                               uint64_t               buffer_size,
                               uint64_t               num_buffers)
    {
        const uint64_t staging_size  = num_buffers > 1 ? buffer_size : 0;
        const uint64_t staging_count = num_buffers > 1 ? num_buffers : 0;

        auto  lk    = std::lock_guard{m_mutex};
        auto& entry = m_agents[agent_id.handle];

        ROCP_FATAL_IF(entry.hsa_agent && entry.hsa_agent->handle != hsa_agent.handle)
            << "ATT agent identity changed for agent " << agent_id.handle;

        entry.hsa_agent = hsa_agent;

        if(entry.resources)
        {
            const auto& registered = entry.requirements;
            ROCP_FATAL_IF(buffer_size > registered.output_buffer_size ||
                          staging_size > registered.staging_buffer_size ||
                          staging_count > registered.staging_buffer_count)
                << "ATT requirements grew after resources were acquired for agent "
                << agent_id.handle;
            return;
        }

        entry.requirements.output_buffer_size =
            std::max(entry.requirements.output_buffer_size, buffer_size);
        entry.requirements.staging_buffer_size =
            std::max(entry.requirements.staging_buffer_size, staging_size);
        entry.requirements.staging_buffer_count =
            std::max(entry.requirements.staging_buffer_count, staging_count);
    }

    agent_trace_resources_ptr_t acquire(const hsa::AgentCache& agent)
    {
        const auto* rocp_agent = CHECK_NOTNULL(agent.get_rocp_agent());
        auto        lk         = std::lock_guard{m_mutex};
        auto        it         = m_agents.find(rocp_agent->id.handle);

        ROCP_FATAL_IF(it == m_agents.end())
            << "ATT resources acquired before requirements registration for agent "
            << rocp_agent->id.handle;

        auto& entry = it->second;
        ROCP_FATAL_IF(!entry.hsa_agent || entry.hsa_agent->handle != agent.get_hsa_agent().handle)
            << "ATT HSA agent does not match registered requirements for agent "
            << rocp_agent->id.handle;
        ROCP_FATAL_IF(entry.requirements.output_buffer_size == 0)
            << "ATT output buffer size was not registered for agent " << rocp_agent->id.handle;

        if(!entry.resources)
            entry.resources = std::make_shared<AgentTraceResources>(agent, entry.requirements);
        return entry.resources;
    }

    void shutdown()
    {
        auto lk = std::lock_guard{m_mutex};
        for(const auto& [agent, entry] : m_agents)
        {
            ROCP_FATAL_IF(entry.resources && entry.resources.use_count() != 1)
                << "ATT resources for agent " << agent << " still have "
                << entry.resources.use_count() - 1 << " borrower(s) during finalization";
        }
        m_agents.clear();
    }

private:
    std::mutex                          m_mutex  = {};
    std::map<uint64_t, manager_entry_t> m_agents = {};
};

SharedTraceResourceManager&
get_manager()
{
    static auto* manager = common::static_object<SharedTraceResourceManager>::construct();
    return *CHECK_NOTNULL(manager);
}
}  // namespace

struct AgentTraceResources::impl
{
    impl(const hsa::AgentCache& _agent, trace_resource_requirements_t _requirements)
    : agent_id(CHECK_NOTNULL(_agent.get_rocp_agent())->id)
    , requirements(std::move(_requirements))
    , queue(att_queue_create(_agent,
                             requirements.staging_buffer_size,
                             requirements.staging_buffer_count))
    {}

    ~impl()
    {
        ROCP_FATAL_IF(active_traces != 0)
            << "Destroying ATT resources for agent " << agent_id.handle << " with " << active_traces
            << " active trace(s)";
        for(auto& buffer : output_buffers)
            if(buffer.raw && buffer.free_fn) buffer.free_fn(buffer.raw);
        att_queue_destroy(queue);
    }

    rocprofiler_agent_id_t        agent_id     = {};
    trace_resource_requirements_t requirements = {};
    att_queue_t                   queue        = {};

    mutable std::mutex                      mutex          = {};
    std::vector<shared_buffer_t>            output_buffers = {};
    std::optional<rocprofiler_context_id_t> active_context = std::nullopt;
    uint64_t                                active_traces  = 0;
};

AgentTraceResources::AgentTraceResources(const hsa::AgentCache&        agent,
                                         trace_resource_requirements_t requirements)
: m_impl(std::make_unique<impl>(agent, std::move(requirements)))
{}

AgentTraceResources::~AgentTraceResources() = default;

att_queue_t&
AgentTraceResources::queue()
{
    return m_impl->queue;
}

void*
AgentTraceResources::acquire_output_buffer(const hsa::TraceMemoryPool& pool,
                                           size_t                      slot,
                                           uint64_t                    requested_size)
{
    auto lk = std::lock_guard{m_impl->mutex};
    ROCP_FATAL_IF(requested_size > m_impl->requirements.output_buffer_size)
        << "ATT output request of " << requested_size << " bytes exceeds registered maximum "
        << m_impl->requirements.output_buffer_size << " for agent " << m_impl->agent_id.handle;

    if(m_impl->output_buffers.size() <= slot) m_impl->output_buffers.resize(slot + 1);

    auto& buffer = m_impl->output_buffers.at(slot);
    if(buffer.aligned) return buffer.aligned;
    if(!pool.allocate_fn) return nullptr;

    void* raw    = nullptr;
    auto  status = pool.allocate_fn(pool.gpu_pool_,
                                   m_impl->requirements.output_buffer_size + PAGE_ALIGN_PADDING,
                                   hsa::hsa_amd_memory_pool_executable_flag,
                                   &raw);
    if(status != HSA_STATUS_SUCCESS || raw == nullptr)
    {
        ROCP_ERROR << "Failed to allocate shared thread trace buffer: " << status;
        return nullptr;
    }

    void* aligned = reinterpret_cast<void*>(  // NOLINT(performance-no-int-to-ptr)
        (reinterpret_cast<uintptr_t>(raw) + PAGE_ALIGN_MASK) & ~PAGE_ALIGN_MASK);

    buffer.raw     = raw;
    buffer.aligned = aligned;
    buffer.free_fn = pool.free_fn;
    return aligned;
}

bool
AgentTraceResources::owns_output_buffer(void* ptr) const
{
    if(!ptr) return false;
    auto lk = std::lock_guard{m_impl->mutex};
    return std::any_of(m_impl->output_buffers.begin(),
                       m_impl->output_buffers.end(),
                       [ptr](const auto& buffer) { return buffer.aligned == ptr; });
}

rocprofiler_agent_id_t
AgentTraceResources::agent_id() const
{
    return m_impl->agent_id;
}

void
AgentTraceResources::begin_trace(rocprofiler_context_id_t context_id)
{
    auto lk = std::lock_guard{m_impl->mutex};
    if(m_impl->active_traces == 0) m_impl->active_context = context_id;
    ROCP_FATAL_IF(!m_impl->active_context)
        << "ATT resource owner missing for agent " << m_impl->agent_id.handle;
    ROCP_FATAL_IF(m_impl->active_context->handle != context_id.handle)
        << "ATT trace overlap on agent " << m_impl->agent_id.handle << ": context "
        << context_id.handle << " attempted to use resources owned by context "
        << m_impl->active_context->handle;
    ++m_impl->active_traces;
}

void
AgentTraceResources::end_trace(rocprofiler_context_id_t context_id)
{
    auto lk = std::lock_guard{m_impl->mutex};
    ROCP_FATAL_IF(m_impl->active_traces == 0 || !m_impl->active_context ||
                  m_impl->active_context->handle != context_id.handle)
        << "ATT trace ownership underflow on agent " << m_impl->agent_id.handle;
    if(--m_impl->active_traces == 0) m_impl->active_context = std::nullopt;
}

void
register_shared_trace_requirements(rocprofiler_agent_id_t agent_id,
                                   hsa_agent_t            hsa_agent,
                                   uint64_t               buffer_size,
                                   uint64_t               num_buffers)
{
    get_manager().register_requirements(agent_id, hsa_agent, buffer_size, num_buffers);
}

agent_trace_resources_ptr_t
acquire_shared_trace_resources(const hsa::AgentCache& agent)
{
    return get_manager().acquire(agent);
}

void
free_shared_trace_resources()
{
    get_manager().shutdown();
}

}  // namespace thread_trace
}  // namespace rocprofiler
