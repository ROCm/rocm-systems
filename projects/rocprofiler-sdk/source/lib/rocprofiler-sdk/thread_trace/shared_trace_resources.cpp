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

#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/kfd/resource.hpp"
#include "lib/rocprofiler-sdk/platform/wsl/agent.hpp"
#include "lib/rocprofiler-sdk/thread_trace/hsa_util.hpp"

#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
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
// operations at the end of the buffer cannot spill into the next allocation. The KFD
// allocator pads and page-aligns internally, so this only applies to the ROCr path.
constexpr size_t PAGE_ALIGN_PADDING = 0x2000;
constexpr size_t PAGE_ALIGN_MASK    = 0xFFFul;

struct manager_entry_t
{
    trace_resource_requirements_t requirements = {};
    agent_trace_resources_ptr_t   resources    = {};
};

class SharedTraceResourceManager
{
public:
    void register_requirements(rocprofiler_agent_id_t agent_id,
                               uint64_t               buffer_size,
                               uint64_t               num_buffers)
    {
        const uint64_t output_slot_count  = std::max<uint64_t>(num_buffers, 1);
        const uint64_t staging_slot_count = num_buffers > 1 ? num_buffers : 0;

        auto  lk    = std::lock_guard{m_mutex};
        auto& entry = m_agents[agent_id.handle];

        auto& output_slot_sizes  = entry.requirements.output_slot_sizes;
        auto& staging_slot_sizes = entry.requirements.staging_slot_sizes;

        if(entry.resources)
        {
            bool fits = output_slot_count <= output_slot_sizes.size() &&
                        staging_slot_count <= staging_slot_sizes.size();
            for(uint64_t slot = 0; fits && slot < output_slot_count; ++slot)
                fits = buffer_size <= output_slot_sizes.at(slot);
            for(uint64_t slot = 0; fits && slot < staging_slot_count; ++slot)
                fits = buffer_size <= staging_slot_sizes.at(slot);

            ROCP_FATAL_IF(!fits) << "ATT requirements grew after resources were acquired for agent "
                                 << agent_id.handle;
            return;
        }

        if(output_slot_sizes.size() < output_slot_count)
            output_slot_sizes.resize(output_slot_count, 0);
        for(uint64_t slot = 0; slot < output_slot_count; ++slot)
            output_slot_sizes[slot] = std::max(output_slot_sizes[slot], buffer_size);

        if(staging_slot_sizes.size() < staging_slot_count)
            staging_slot_sizes.resize(staging_slot_count, 0);
        for(uint64_t slot = 0; slot < staging_slot_count; ++slot)
            staging_slot_sizes[slot] = std::max(staging_slot_sizes[slot], buffer_size);
    }

    agent_trace_resources_ptr_t acquire(rocprofiler_agent_id_t agent_id)
    {
        auto lk = std::lock_guard{m_mutex};
        auto it = m_agents.find(agent_id.handle);

        ROCP_FATAL_IF(it == m_agents.end())
            << "ATT resources acquired before requirements registration for agent "
            << agent_id.handle;

        auto& entry = it->second;
        ROCP_FATAL_IF(entry.requirements.output_slot_sizes.empty())
            << "ATT output buffer size was not registered for agent " << agent_id.handle;

        if(!entry.resources)
            entry.resources = std::make_shared<AgentTraceResources>(agent_id, entry.requirements);
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

SharedTraceResourceManager*
peek_manager()
{
    return common::static_object<SharedTraceResourceManager>::get();
}

// The SDMA ring and its command buffer are sized once for the agent, so they have to
// cover the largest copy any context on it can request.
uint64_t
max_slot_size(const std::vector<uint64_t>& slot_sizes)
{
    auto it = std::max_element(slot_sizes.begin(), slot_sizes.end());
    return it == slot_sizes.end() ? 0 : *it;
}
}  // namespace

AgentTraceResources::AgentTraceResources(rocprofiler_agent_id_t        agent_id,
                                         trace_resource_requirements_t requirements)
: m_agent_id(agent_id)
, m_requirements(std::move(requirements))
{
    const uint64_t max_copy_size = max_slot_size(m_requirements.output_slot_sizes);

#ifndef _WIN32
    if(!common::get_env("ROCPROFILER_SQTT_FORCE_HSA", false))
    {
        const auto& gpu = *CHECK_NOTNULL(agent::get_agent(agent_id));
        if(kfd_copy_queue_t::is_supported(gpu.gfx_target_version) &&
           common::filesystem::exists("/dev/kfd"))
        {
            m_kfd_memory = kfd_memory_pool_t::create(gpu);
            if(!m_kfd_memory)
                throw std::runtime_error{"Could not create KFD thread-trace memory pool"};
        }
        else if(!platform::wsl::is_available())
        {
            static auto once = std::once_flag{};
            std::call_once(once, []() {
                ROCP_CI_LOG(WARNING)
                    << "Direct KFD thread trace is unavailable or unsupported for this "
                       "GPU; using the ROCr/HSA thread-trace backend";
            });
        }
    }
#endif

    m_queue =
        make_att_queue(agent_id, max_copy_size, m_requirements.staging_slot_sizes, m_kfd_memory);
    if(!m_queue) throw std::runtime_error{"Could not create thread-trace queue"};
}

AgentTraceResources::~AgentTraceResources()
{
    ROCP_FATAL_IF(m_active_traces != 0)
        << "Destroying ATT resources for agent " << m_agent_id.handle << " with " << m_active_traces
        << " active trace(s)";
    for(auto& buffer : m_output_buffers)
    {
        if(!buffer.raw) continue;
        if(m_kfd_memory)
            m_kfd_memory->deallocate(buffer.raw);
        else if(buffer.free_fn)
            buffer.free_fn(buffer.raw);
    }
    // Packet/marker destructors share the pool. Close the engines while all
    // command, signal and trace allocations are still owned.
    if(m_queue && m_queue->kfd_copy_queue) m_queue->kfd_copy_queue->close();
    m_queue.reset();
}

att_queue_t&
AgentTraceResources::queue()
{
    return *CHECK_NOTNULL(m_queue.get());
}

const std::shared_ptr<kfd_memory_pool_t>&
AgentTraceResources::kfd_memory() const
{
    return m_kfd_memory;
}

const std::shared_ptr<kfd_copy_queue_t>&
AgentTraceResources::kfd_copy_queue() const
{
    return CHECK_NOTNULL(m_queue.get())->kfd_copy_queue;
}

void*
AgentTraceResources::acquire_output_buffer(const hsa::TraceMemoryPool& pool,
                                           size_t                      slot,
                                           uint64_t                    requested_size)
{
    auto lk = std::lock_guard{m_mutex};
    ROCP_FATAL_IF(slot >= m_requirements.output_slot_sizes.size())
        << "ATT output slot " << slot << " exceeds the " << m_requirements.output_slot_sizes.size()
        << " slot(s) registered for agent " << m_agent_id.handle;

    const uint64_t slot_size = m_requirements.output_slot_sizes.at(slot);
    ROCP_FATAL_IF(requested_size > slot_size)
        << "ATT output request of " << requested_size << " bytes exceeds registered maximum "
        << slot_size << " for agent " << m_agent_id.handle;

    if(m_output_buffers.size() <= slot) m_output_buffers.resize(slot + 1);

    auto& buffer = m_output_buffers.at(slot);
    if(buffer.aligned) return buffer.aligned;

    if(m_kfd_memory)
    {
        // KFD rounds the allocation up to a page and honours the page alignment
        // itself, so no manual padding is needed here.
        buffer.raw = m_kfd_memory->allocate(slot_size, kfd::kfd_memory_kind_t::device);
        if(!buffer.raw)
        {
            ROCP_ERROR << "Failed to allocate " << slot_size << " byte thread trace buffer (slot "
                       << slot << ") for agent " << m_agent_id.handle;
            buffer = shared_buffer_t{};
            return nullptr;
        }
        buffer.aligned = buffer.raw;
        return buffer.aligned;
    }

    if(!pool.allocate_fn) return nullptr;

    void* raw    = nullptr;
    auto  status = pool.allocate_fn(pool.gpu_pool_,
                                   slot_size + PAGE_ALIGN_PADDING,
                                   hsa::hsa_amd_memory_pool_executable_flag,
                                   &raw);
    if(status != HSA_STATUS_SUCCESS || raw == nullptr)
    {
        ROCP_ERROR << "Failed to allocate " << slot_size << " byte thread trace buffer (slot "
                   << slot << ") for agent " << m_agent_id.handle << ": " << status;
        return nullptr;
    }

    buffer.raw     = raw;
    buffer.aligned = reinterpret_cast<void*>(  // NOLINT(performance-no-int-to-ptr)
        (reinterpret_cast<uintptr_t>(raw) + PAGE_ALIGN_MASK) & ~PAGE_ALIGN_MASK);
    buffer.free_fn = pool.free_fn;
    return buffer.aligned;
}

bool
AgentTraceResources::owns_output_buffer(void* ptr) const
{
    if(!ptr) return false;
    auto lk = std::lock_guard{m_mutex};
    return std::any_of(m_output_buffers.begin(), m_output_buffers.end(), [ptr](const auto& buffer) {
        return buffer.aligned == ptr;
    });
}

rocprofiler_agent_id_t
AgentTraceResources::agent_id() const
{
    return m_agent_id;
}

void
AgentTraceResources::begin_trace(rocprofiler_context_id_t context_id)
{
    auto lk = std::lock_guard{m_mutex};
    if(m_active_traces == 0) m_active_context = context_id;
    ROCP_FATAL_IF(!m_active_context)
        << "ATT resource owner missing for agent " << m_agent_id.handle;
    ROCP_FATAL_IF(m_active_context->handle != context_id.handle)
        << "ATT trace overlap on agent " << m_agent_id.handle << ": context " << context_id.handle
        << " attempted to use resources owned by context " << m_active_context->handle;
    ++m_active_traces;
}

void
AgentTraceResources::end_trace(rocprofiler_context_id_t context_id)
{
    auto lk = std::lock_guard{m_mutex};
    ROCP_FATAL_IF(m_active_traces == 0 || !m_active_context ||
                  m_active_context->handle != context_id.handle)
        << "ATT trace ownership underflow on agent " << m_agent_id.handle;
    if(--m_active_traces == 0) m_active_context = std::nullopt;
}

void
register_shared_trace_requirements(rocprofiler_agent_id_t agent_id,
                                   uint64_t               buffer_size,
                                   uint64_t               num_buffers)
{
    get_manager().register_requirements(agent_id, buffer_size, num_buffers);
}

agent_trace_resources_ptr_t
acquire_shared_trace_resources(rocprofiler_agent_id_t agent_id)
{
    return get_manager().acquire(agent_id);
}

void
free_shared_trace_resources()
{
    // Finalization runs for every process, including those that never configured ATT.
    // Skip it entirely rather than constructing the manager just to clear nothing.
    if(auto* manager = peek_manager()) manager->shutdown();
}

}  // namespace thread_trace
}  // namespace rocprofiler
