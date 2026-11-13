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

#include "lib/common/scope_destructor.hpp"
#include "lib/rocprofiler-sdk/thread_trace/hsa_util.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa_ext_amd.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
struct TraceMemoryPool;
}  // namespace hsa

namespace thread_trace
{
inline constexpr uint64_t GPU_RING_BUFFERS = 3;

struct trace_usage_t;
using trace_lease_ptr_t = std::unique_ptr<common::scope_destructor>;

std::pair<rocprofiler_status_t, trace_lease_ptr_t>
acquire_device_trace(rocprofiler_agent_id_t agent_id, rocprofiler_context_id_t context_id);

struct trace_resource_requirements_t
{
    // AQLProfile allocates one common GPU output buffer and, for multi-buffer traces,
    // GPU_RING_BUFFERS - 1 extra buffers per enabled shader engine. Each allocation
    // index is sized to the largest context that reaches it.
    std::vector<uint64_t> output_slot_sizes = {};
    // Size of each CPU staging slot, following the same per-slot rule.
    std::vector<uint64_t> staging_slot_sizes = {};
};

// Owns every heavyweight ATT resource shared by contexts targeting one agent: the KFD
// memory pool (or its ROCr fallback), the submission queue, the CPU staging buffers, and
// the GPU output slots. Callers retain a shared handle so none of it can disappear while
// packets or producer/consumer workers still reference it.
class AgentTraceResources
{
public:
    AgentTraceResources(rocprofiler_agent_id_t         agent_id,
                        trace_resource_requirements_t  requirements,
                        std::shared_ptr<trace_usage_t> usage);
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

    void* acquire_output_buffer(const hsa::TraceMemoryPool& pool,
                                size_t                      slot,
                                uint64_t                    requested_size);
    bool  owns_output_buffer(void* ptr) const;

    rocprofiler_agent_id_t agent_id() const;

    // Dispatch packets execute under the per-agent serializer. Their leases prevent
    // a device trace from starting until all queued dispatches have retired.
    trace_lease_ptr_t acquire_dispatch_trace();

private:
    struct shared_buffer_t
    {
        void*                               raw     = nullptr;
        void*                               aligned = nullptr;
        decltype(hsa_amd_memory_pool_free)* free_fn = nullptr;
    };

    rocprofiler_agent_id_t             m_agent_id     = {};
    trace_resource_requirements_t      m_requirements = {};
    std::shared_ptr<kfd_memory_pool_t> m_kfd_memory   = {};
    att_queue_ptr_t                    m_queue        = {};

    mutable std::mutex             m_mutex          = {};
    std::vector<shared_buffer_t>   m_output_buffers = {};
    std::shared_ptr<trace_usage_t> m_usage          = {};
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

// Drop manager ownership. Packets and workers retain resources until they retire.
void
free_shared_trace_resources();

}  // namespace thread_trace
}  // namespace rocprofiler
