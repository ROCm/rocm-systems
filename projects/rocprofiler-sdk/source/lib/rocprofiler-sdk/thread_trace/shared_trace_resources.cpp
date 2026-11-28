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
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/kfd/resource.hpp"
#ifndef _WIN32
#    include "lib/rocprofiler-sdk/platform/wsl/agent.hpp"
#endif
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/thread_trace/hsa_util.hpp"

#include <rocprofiler-sdk/cxx/operators.hpp>

#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <map>
#include <mutex>
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
    trace_resource_requirements_t           requirements = {};
    agent_trace_resources_ptr_t             resources    = {};
    std::shared_ptr<agent_activity_state_t> activity = std::make_shared<agent_activity_state_t>();
};

class SharedTraceResourceManager
{
public:
    rocprofiler_status_t register_requirements(rocprofiler_agent_id_t agent_id,
                                               uint64_t               buffer_size,
                                               uint64_t               num_buffers,
                                               uint64_t               shader_engine_mask)
    {
        const auto* agent = rocprofiler::agent::get_agent(agent_id);
        if(!agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

        // Coupled to AQLProfile's ATT allocation sequence: slot 0 is the common output buffer
        // of buffer_size, followed in multi-buffer mode by (GPU_RING_BUFFERS - 1) * active_se
        // per-SE buffers whose size AQLProfile rounds down from buffer_size / active_se.
        auto requested_output_sizes = std::vector<uint64_t>{buffer_size};
        if(num_buffers > 1)
        {
            shader_engine_mask =
                clamp_shader_engine_mask(shader_engine_mask, agent->num_shader_banks);
            auto active_se = uint64_t{0};
            for(; shader_engine_mask != 0; shader_engine_mask >>= 1)
                active_se += shader_engine_mask & 1;
            if(active_se > 0)
                requested_output_sizes.resize(
                    1 + (GPU_RING_BUFFERS - 1) * active_se,
                    buffer_size / active_se + (buffer_size % active_se != 0 ? 1 : 0));
        }
        const uint64_t output_slot_count  = requested_output_sizes.size();
        const uint64_t staging_slot_count = num_buffers > 1 ? num_buffers : 0;

        auto  lk    = std::lock_guard{m_mutex};
        auto& entry = m_agents[agent_id];

        auto& output_slot_sizes = entry.requirements.output_slot_sizes;

        if(entry.resources)
        {
            bool fits =
                output_slot_count <= output_slot_sizes.size() &&
                staging_slot_count <= entry.requirements.staging_slot_count &&
                (staging_slot_count == 0 || buffer_size <= entry.requirements.max_staging_size);
            for(uint64_t slot = 0; fits && slot < output_slot_count; ++slot)
                fits = requested_output_sizes.at(slot) <= output_slot_sizes.at(slot);

            ROCP_WARNING_IF(!fits)
                << "ATT configuration for agent " << agent_id.handle
                << " exceeds initialized shared capacity (output_slots=" << output_slot_count << "/"
                << output_slot_sizes.size() << ", staging_slots=" << staging_slot_count << "/"
                << entry.requirements.staging_slot_count << ", buffer_size=" << buffer_size
                << ", max_staging_size=" << entry.requirements.max_staging_size << ")";
            return fits ? ROCPROFILER_STATUS_SUCCESS : ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES;
        }

        if(output_slot_sizes.size() < output_slot_count)
            output_slot_sizes.resize(output_slot_count, 0);
        for(uint64_t slot = 0; slot < output_slot_count; ++slot)
            output_slot_sizes[slot] =
                std::max(output_slot_sizes[slot], requested_output_sizes.at(slot));

        if(staging_slot_count > 0)
        {
            entry.requirements.staging_slot_count =
                std::max(entry.requirements.staging_slot_count, staging_slot_count);
            entry.requirements.max_staging_size =
                std::max(entry.requirements.max_staging_size, buffer_size);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    }

    device_trace_guard_t try_begin_device_trace(rocprofiler_agent_id_t agent_id)
    {
        auto activity = std::shared_ptr<agent_activity_state_t>{};
        {
            auto lk = std::lock_guard{m_mutex};
            auto it = m_agents.find(agent_id);
            if(it == m_agents.end()) return {};
            activity = it->second.activity;
        }
        if(!activity->try_begin_device_trace()) return {};
        return device_trace_guard_t{std::move(activity)};
    }

    agent_trace_resources_ptr_t acquire(rocprofiler_agent_id_t agent_id)
    {
        auto lk = std::lock_guard{m_mutex};
        auto it = m_agents.find(agent_id);

        if(it == m_agents.end())
            throw std::runtime_error{"ATT resource requirements were not registered"};

        auto& entry = it->second;
        if(entry.requirements.output_slot_sizes.empty())
            throw std::runtime_error{"ATT output buffer size was not registered"};

        if(!entry.resources)
            entry.resources =
                std::make_shared<AgentTraceResources>(agent_id, entry.requirements, entry.activity);
        return entry.resources;
    }

    void shutdown()
    {
        auto retired = agent_resource_map_t{};
        {
            auto lock = std::lock_guard{m_mutex};
            retired.swap(m_agents);
        }
        for(auto& [_, entry] : retired)
            if(entry.resources) entry.resources->close_submission_queue();
    }

private:
    using agent_resource_map_t = std::map<rocprofiler_agent_id_t, manager_entry_t>;

    std::mutex           m_mutex  = {};
    agent_resource_map_t m_agents = {};
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

bool
agent_activity_state_t::try_acquire_dispatch_slot()
{
    auto current = state.load(std::memory_order_acquire);
    while((current & DEVICE_ACTIVE_BIT) == 0)
    {
        if(state.compare_exchange_weak(
               current, current + 1, std::memory_order_acq_rel, std::memory_order_acquire))
            return true;
    }
    return false;
}

void
agent_activity_state_t::release_dispatch_slot()
{
    auto prev = state.fetch_sub(1, std::memory_order_release);
    ROCP_FATAL_IF(prev == 0 || (prev & DEVICE_ACTIVE_BIT) != 0)
        << "ATT dispatch slot underflow (state=" << prev << ")";
}

bool
agent_activity_state_t::try_begin_device_trace()
{
    auto expected = uint64_t{0};
    return state.compare_exchange_strong(
        expected, DEVICE_ACTIVE_BIT, std::memory_order_acq_rel, std::memory_order_acquire);
}

void
agent_activity_state_t::end_device_trace()
{
    state.fetch_and(~DEVICE_ACTIVE_BIT, std::memory_order_release);
}

AgentTraceResources::AgentTraceResources(rocprofiler_agent_id_t                  agent_id,
                                         trace_resource_requirements_t           requirements,
                                         std::shared_ptr<agent_activity_state_t> activity)
: m_agent_id(agent_id)
, m_requirements(std::move(requirements))
, m_activity(std::move(activity))
{
    const uint64_t max_copy_size = max_slot_size(m_requirements.output_slot_sizes);

    auto kfd_memory = std::shared_ptr<kfd_memory_pool_t>{};
#ifndef _WIN32
    if(!common::get_env(FORCE_HSA_ENV, false))
    {
        const auto& gpu = *CHECK_NOTNULL(agent::get_agent(agent_id));
        if(kfd_copy_queue_t::is_supported(gpu.gfx_target_version) &&
           common::filesystem::exists("/dev/kfd"))
        {
            kfd_memory = kfd_memory_pool_t::create(gpu);
            if(!kfd_memory)
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

    m_queue = make_att_queue(agent_id,
                             m_requirements.max_staging_size,
                             m_requirements.staging_slot_count,
                             std::move(kfd_memory),
                             max_copy_size);
    if(!m_queue) throw std::runtime_error{"Could not create thread-trace queue"};
}

AgentTraceResources::~AgentTraceResources()
{
    // Close the queues while all command, signal and trace allocations are still owned.
    close_submission_queue();
    if(const auto& copy_queue = kfd_copy_queue()) copy_queue->close();

    const auto& kfd_mem = kfd_memory();
    for(auto& buffer : m_output_buffers)
    {
        if(!buffer.raw) continue;
        if(kfd_mem)
            kfd_mem->deallocate(buffer.raw);
        else if(buffer.free_fn && registration::get_fini_status() <= 0)
            buffer.free_fn(buffer.raw);
    }
}

att_queue_t&
AgentTraceResources::queue()
{
    return *CHECK_NOTNULL(m_queue.get());
}

void
AgentTraceResources::close_submission_queue()
{
    if(m_queue) att_queue_close(*m_queue);
}

const std::shared_ptr<kfd_memory_pool_t>&
AgentTraceResources::kfd_memory() const
{
    return CHECK_NOTNULL(m_queue.get())->kfd_memory;
}

const std::shared_ptr<kfd_copy_queue_t>&
AgentTraceResources::kfd_copy_queue() const
{
    return CHECK_NOTNULL(m_queue.get())->kfd_copy_queue;
}

void*
AgentTraceResources::acquire_output_buffer(const rocr_output_allocator_t& allocator,
                                           size_t                         slot,
                                           uint64_t                       requested_size)
{
    auto lk = std::lock_guard{m_mutex};
    if(slot >= m_requirements.output_slot_sizes.size() ||
       requested_size > m_requirements.output_slot_sizes.at(slot))
    {
        ROCP_ERROR << "ATT output request exceeds registered capacity for agent "
                   << m_agent_id.handle << " (slot=" << slot
                   << ", registered_slots=" << m_requirements.output_slot_sizes.size()
                   << ", requested_size=" << requested_size << ", slot_capacity="
                   << (slot < m_requirements.output_slot_sizes.size()
                           ? m_requirements.output_slot_sizes.at(slot)
                           : 0)
                   << ")";
        return nullptr;
    }

    const uint64_t slot_size = m_requirements.output_slot_sizes.at(slot);

    if(m_output_buffers.size() <= slot) m_output_buffers.resize(slot + 1);

    auto& buffer = m_output_buffers.at(slot);
    if(buffer.aligned) return buffer.aligned;

    if(const auto& kfd_mem = kfd_memory())
    {
        // KFD rounds the allocation up to a page and honours the page alignment
        // itself, so no manual padding is needed here.
        buffer.raw = kfd_mem->allocate(slot_size, kfd::kfd_memory_kind_t::device);
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

    if(!allocator.allocate_fn || !allocator.free_fn)
    {
        ROCP_ERROR << "Missing ROCr allocator callbacks for ATT output buffer on agent "
                   << m_agent_id.handle;
        return nullptr;
    }

    auto* raw    = static_cast<void*>(nullptr);
    auto  status = allocator.allocate_fn(allocator.gpu_pool,
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
    buffer.free_fn = allocator.free_fn;
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

dispatch_slot_guard_t
AgentTraceResources::try_acquire_dispatch_slot()
{
    if(!m_activity || !m_activity->try_acquire_dispatch_slot()) return {};
    return dispatch_slot_guard_t{m_activity};
}

rocprofiler_status_t
register_shared_trace_requirements(rocprofiler_agent_id_t agent_id,
                                   uint64_t               buffer_size,
                                   uint64_t               num_buffers,
                                   uint64_t               shader_engine_mask)
{
    return get_manager().register_requirements(
        agent_id, buffer_size, num_buffers, shader_engine_mask);
}

agent_trace_resources_ptr_t
acquire_shared_trace_resources(rocprofiler_agent_id_t agent_id)
{
    return get_manager().acquire(agent_id);
}

device_trace_guard_t
try_begin_shared_device_trace(rocprofiler_agent_id_t agent_id)
{
    return get_manager().try_begin_device_trace(agent_id);
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
