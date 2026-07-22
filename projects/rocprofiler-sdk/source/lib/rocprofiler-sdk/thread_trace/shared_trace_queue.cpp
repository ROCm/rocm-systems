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

#include "lib/rocprofiler-sdk/thread_trace/shared_trace_queue.hpp"
#include "lib/rocprofiler-sdk/thread_trace/hsa_util.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <map>

namespace rocprofiler
{
namespace thread_trace
{
namespace
{
struct agent_queue_t
{
    // Max CPU staging buffer size and count requested by any context on this agent;
    // the shared queue is created with these so every context's flips fit.
    uint64_t        max_staging_size = 0;
    uint64_t        max_num_buffers  = 0;
    att_queue_ptr_t queue            = {};
};

struct shared_queue_state_t
{
    std::map<uint64_t, agent_queue_t> agents = {};  // keyed by hsa_agent_t.handle
};

// Held in a static_object (see shared_trace_lease.cpp for the rationale) so it stays alive
// through registration::finalize() on the attach path, without leaking. free_shared_queues()
// frees the contents.
common::Synchronized<shared_queue_state_t>& g_queue_state =
    *common::static_object<common::Synchronized<shared_queue_state_t>>::construct();
}  // namespace

void
register_shared_queue_size(hsa_agent_t agent, uint64_t buffer_size, uint64_t num_buffers)
{
    // Only multi-buffer contexts stage flips through CPU buffers; single-buffer
    // contexts use the synchronous path and need no staging memory.
    const uint64_t staging = num_buffers > 1 ? buffer_size : 0;
    const uint64_t nbuf    = num_buffers > 1 ? num_buffers : 0;
    g_queue_state.wlock([&](shared_queue_state_t& state) {
        auto& entry            = state.agents[agent.handle];
        entry.max_staging_size = std::max(entry.max_staging_size, staging);
        entry.max_num_buffers  = std::max(entry.max_num_buffers, nbuf);
    });
}

att_queue_t*
acquire_shared_queue(const hsa::AgentCache& agent, uint64_t buffer_size, uint64_t num_buffers)
{
    // Only multi-buffer contexts stage via CPU buffers. Recording max here keeps sizing
    // correct if the pre-pass was skipped, but it cannot grow an already-created queue --
    // correctness relies on the pre-pass running for every context before the first acquire.
    const uint64_t staging = num_buffers > 1 ? buffer_size : 0;
    const uint64_t nbuf    = num_buffers > 1 ? num_buffers : 0;
    return g_queue_state.wlock([&](shared_queue_state_t& state) -> att_queue_t* {
        auto& entry            = state.agents[agent.get_hsa_agent().handle];
        entry.max_staging_size = std::max(entry.max_staging_size, staging);
        entry.max_num_buffers  = std::max(entry.max_num_buffers, nbuf);

        // Reuse the agent's queue; the first acquire allocates it sized to the agent max
        // and the rest just reuse.
        if(entry.queue) return entry.queue.get();

        entry.queue = make_att_queue(agent, entry.max_staging_size, entry.max_num_buffers);
        return entry.queue.get();
    });
}

void
free_shared_queues()
{
    // Clearing runs each att_queue_ptr_t deleter (att_queue_destroy), so the HSA
    // queues are torn down exactly once here rather than per context.
    g_queue_state.wlock([](shared_queue_state_t& state) { state.agents.clear(); });
}

}  // namespace thread_trace
}  // namespace rocprofiler
