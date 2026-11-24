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

#pragma once

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace kfd
{
class kfd_copy_queue_t;
class kfd_memory_pool_t;
}  // namespace kfd

namespace thread_trace
{
struct att_queue_t;
using att_queue_ptr_t = std::unique_ptr<att_queue_t>;
using kfd::kfd_copy_queue_t;
using kfd::kfd_memory_pool_t;
inline constexpr uint64_t GPU_RING_BUFFERS = 3;
inline constexpr auto     FORCE_HSA_ENV    = "ROCPROFILER_SQTT_FORCE_HSA";

using memory_pool_allocate_fn_t = decltype(hsa_amd_memory_pool_allocate)*;
using memory_pool_free_fn_t     = decltype(hsa_amd_memory_pool_free)*;

inline uint64_t
clamp_shader_engine_mask(uint64_t mask, uint32_t num_shader_banks)
{
    if(num_shader_banks < 64) mask &= (uint64_t{1} << num_shader_banks) - 1;
    return mask;
}

struct rocr_output_allocator_t
{
    hsa_amd_memory_pool_t     gpu_pool    = {};
    memory_pool_allocate_fn_t allocate_fn = nullptr;
    memory_pool_free_fn_t     free_fn     = nullptr;
};

struct trace_resource_requirements_t
{
    // AQLProfile allocates one common GPU output buffer and, for multi-buffer traces,
    // GPU_RING_BUFFERS - 1 extra buffers per enabled shader engine. Each allocation
    // index is sized to the largest context that reaches it.
    std::vector<uint64_t> output_slot_sizes  = {};
    uint64_t              max_staging_size   = 0;
    uint64_t              staging_slot_count = 0;
};

// Tracks mutual exclusion between an active device trace (top bit) and in-flight
// dispatch packets (lower 63 bits) on one agent. Shared between the manager entry
// (for deferred CODE_OBJECT device starts) and AgentTraceResources.
struct agent_activity_state_t
{
    static constexpr uint64_t DEVICE_ACTIVE_BIT = uint64_t{1} << 63;

    std::atomic<uint64_t> state{0};

    bool try_acquire_dispatch_slot();
    void release_dispatch_slot();
    bool try_begin_device_trace();
    void end_device_trace();
};

template <void (agent_activity_state_t::*ReleaseFn)()>
class agent_activity_guard_t
{
public:
    agent_activity_guard_t() = default;
    explicit agent_activity_guard_t(std::shared_ptr<agent_activity_state_t> activity)
    : m_activity{std::move(activity)}
    {}
    ~agent_activity_guard_t()
    {
        if(m_activity) (m_activity.get()->*ReleaseFn)();
    }

    agent_activity_guard_t(const agent_activity_guard_t&) = delete;
    agent_activity_guard_t& operator=(const agent_activity_guard_t&) = delete;
    agent_activity_guard_t(agent_activity_guard_t&&) noexcept        = default;
    agent_activity_guard_t& operator=(agent_activity_guard_t&& other) noexcept
    {
        if(this != &other)
        {
            if(m_activity) (m_activity.get()->*ReleaseFn)();
            m_activity = std::move(other.m_activity);
        }
        return *this;
    }

    explicit operator bool() const { return m_activity != nullptr; }

private:
    std::shared_ptr<agent_activity_state_t> m_activity = {};
};

using device_trace_guard_t = agent_activity_guard_t<&agent_activity_state_t::end_device_trace>;
using dispatch_slot_guard_t =
    agent_activity_guard_t<&agent_activity_state_t::release_dispatch_slot>;

// Owns every heavyweight ATT resource shared by contexts targeting one agent: the KFD
// memory pool (or its ROCr fallback), the submission queue, the CPU staging buffers, and
// the GPU output slots. Callers retain a shared handle so none of it can disappear while
// packets or producer/consumer workers still reference it.
class AgentTraceResources
{
public:
    AgentTraceResources(rocprofiler_agent_id_t                  agent_id,
                        trace_resource_requirements_t           requirements,
                        std::shared_ptr<agent_activity_state_t> activity);
    ~AgentTraceResources();

    AgentTraceResources(const AgentTraceResources&) = delete;
    AgentTraceResources(AgentTraceResources&&)      = delete;
    AgentTraceResources& operator=(const AgentTraceResources&) = delete;
    AgentTraceResources& operator=(AgentTraceResources&&) = delete;

    att_queue_t& queue();
    void         close_submission_queue();

    // Null when the KFD backend was unavailable and the queue fell back to ROCr.
    const std::shared_ptr<kfd_memory_pool_t>& kfd_memory() const;
    const std::shared_ptr<kfd_copy_queue_t>&  kfd_copy_queue() const;

    void* acquire_output_buffer(const rocr_output_allocator_t& allocator,
                                size_t                         slot,
                                uint64_t                       requested_size);
    bool  owns_output_buffer(void* ptr) const;

    rocprofiler_agent_id_t agent_id() const;

    dispatch_slot_guard_t try_acquire_dispatch_slot();

private:
    struct shared_buffer_t
    {
        void*                 raw     = nullptr;
        void*                 aligned = nullptr;
        memory_pool_free_fn_t free_fn = nullptr;
    };

    rocprofiler_agent_id_t                  m_agent_id     = {};
    trace_resource_requirements_t           m_requirements = {};
    att_queue_ptr_t                         m_queue;
    std::shared_ptr<agent_activity_state_t> m_activity = {};

    mutable std::mutex           m_mutex          = {};
    std::vector<shared_buffer_t> m_output_buffers = {};
};

using agent_trace_resources_ptr_t = std::shared_ptr<AgentTraceResources>;

// Register all requirements before the first acquire. Registration is combined
// because the output slots and queue have the same per-agent lifetime.
rocprofiler_status_t
register_shared_trace_requirements(rocprofiler_agent_id_t agent_id,
                                   uint64_t               buffer_size,
                                   uint64_t               num_buffers,
                                   uint64_t               shader_engine_mask = 1);

agent_trace_resources_ptr_t
acquire_shared_trace_resources(rocprofiler_agent_id_t agent_id);

device_trace_guard_t
try_begin_shared_device_trace(rocprofiler_agent_id_t agent_id);

// Drop manager ownership. Packets and workers retain resources until they retire.
void
free_shared_trace_resources();

}  // namespace thread_trace
}  // namespace rocprofiler
