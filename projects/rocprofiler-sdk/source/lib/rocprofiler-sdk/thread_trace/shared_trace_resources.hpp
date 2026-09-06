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

#include "lib/rocprofiler-sdk/thread_trace/hsa_util.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
class AgentCache;
struct TraceMemoryPool;
}  // namespace hsa

namespace thread_trace
{
struct trace_resource_requirements_t
{
    // Size of each GPU output slot. A context occupies one slot per requested buffer, so
    // the slot count is the largest request on the agent and each entry is sized to the
    // largest context that reaches that slot.
    std::vector<uint64_t> output_slot_sizes = {};
    // Size of each CPU staging slot, following the same per-slot rule.
    std::vector<uint64_t> staging_slot_sizes = {};
};

// Owns every heavyweight ATT resource shared by contexts targeting one agent.
// Callers retain a shared handle so the queue and output slots cannot disappear
// while packets or producer/consumer workers still reference them.
class AgentTraceResources
{
public:
    AgentTraceResources(const hsa::AgentCache& agent, trace_resource_requirements_t requirements);
    ~AgentTraceResources();

    AgentTraceResources(const AgentTraceResources&) = delete;
    AgentTraceResources(AgentTraceResources&&)      = delete;
    AgentTraceResources& operator=(const AgentTraceResources&) = delete;
    AgentTraceResources& operator=(AgentTraceResources&&) = delete;

    att_queue_t& queue();

    void* acquire_output_buffer(const hsa::TraceMemoryPool& pool,
                                size_t                      slot,
                                uint64_t                    requested_size);
    bool  owns_output_buffer(void* ptr) const;

    rocprofiler_agent_id_t agent_id() const;

    // Assert the hardware invariant directly at trace boundaries. Multiple
    // in-flight dispatches from the same context are allowed; another context is not.
    void begin_trace(rocprofiler_context_id_t context_id);
    void end_trace(rocprofiler_context_id_t context_id);

private:
    struct shared_buffer_t
    {
        void*                               raw     = nullptr;
        void*                               aligned = nullptr;
        decltype(hsa_amd_memory_pool_free)* free_fn = nullptr;
    };

    rocprofiler_agent_id_t        m_agent_id     = {};
    trace_resource_requirements_t m_requirements = {};
    att_queue_t                   m_queue        = {};

    mutable std::mutex                      m_mutex          = {};
    std::vector<shared_buffer_t>            m_output_buffers = {};
    std::optional<rocprofiler_context_id_t> m_active_context = std::nullopt;
    uint64_t                                m_active_traces  = 0;
};

using agent_trace_resources_ptr_t = std::shared_ptr<AgentTraceResources>;

// Register all requirements before the first acquire. Registration is combined
// because the output slots and queue have the same per-agent lifetime.
void
register_shared_trace_requirements(rocprofiler_agent_id_t agent_id,
                                   hsa_agent_t            hsa_agent,
                                   uint64_t               buffer_size,
                                   uint64_t               num_buffers);

agent_trace_resources_ptr_t
acquire_shared_trace_resources(const hsa::AgentCache& agent);

// Destroy all manager-owned resources after every tracer/packet/worker handle
// has gone away. The manager wrapper itself is destroyed after finalize().
void
free_shared_trace_resources();

}  // namespace thread_trace
}  // namespace rocprofiler
