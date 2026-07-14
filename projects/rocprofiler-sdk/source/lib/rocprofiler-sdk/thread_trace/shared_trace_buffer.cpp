// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/thread_trace/shared_trace_buffer.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"

#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <unordered_set>
#include <vector>

namespace rocprofiler
{
namespace thread_trace
{
namespace
{
// Padding + mask to page-align the returned pointer.
constexpr size_t PAGE_ALIGN_PADDING = 0x2000;
constexpr size_t PAGE_ALIGN_MASK    = 0xFFFul;

struct shared_buffer_t
{
    void*    raw     = nullptr;  // as returned by the allocator (used to free)
    void*    aligned = nullptr;  // page-aligned pointer handed out
    uint64_t size    = 0;        // allocated size, so reuse can verify it fits
};

struct agent_buffers_t
{
    uint64_t                     max_buffer_size = 0;
    std::vector<shared_buffer_t> buffers         = {};
};

struct shared_state_t
{
    std::map<uint64_t, agent_buffers_t> agents      = {};       // keyed by hsa_agent_t.handle
    std::unordered_set<void*>           shared_ptrs = {};       // aligned pointers handed out
    decltype(hsa_amd_memory_pool_free)* free_fn     = nullptr;  // captured for shutdown free
};

using locked_state_t = common::Synchronized<shared_state_t>;

locked_state_t&
get_state()
{
    static auto*& _state = common::static_object<locked_state_t>::construct();
    return *CHECK_NOTNULL(_state);
}
}  // namespace

void
register_shared_buffer_size(hsa_agent_t agent, uint64_t buffer_size)
{
    get_state().wlock([&](shared_state_t& state) {
        auto& entry           = state.agents[agent.handle];
        entry.max_buffer_size = std::max(entry.max_buffer_size, buffer_size);
    });
}

void*
acquire_shared_buffer(const hsa::TraceMemoryPool& pool, size_t index, uint64_t size)
{
    return get_state().wlock([&](shared_state_t& state) -> void* {
        // Self-register in case the pre-pass didn't: acquire is the single buffer source.
        auto& entry           = state.agents[pool.gpu_agent.handle];
        entry.max_buffer_size = std::max(entry.max_buffer_size, size);

        if(entry.buffers.size() <= index) entry.buffers.resize(index + 1);

        // Reuse the slot's buffer; the pre-pass sizes it to the agent max up front.
        auto& buf = entry.buffers.at(index);
        if(buf.aligned != nullptr && buf.size >= size) return buf.aligned;

        if(!pool.allocate_fn) return nullptr;

        void* raw    = nullptr;
        auto  status = pool.allocate_fn(pool.gpu_pool_,
                                       entry.max_buffer_size + PAGE_ALIGN_PADDING,
                                       hsa::hsa_amd_memory_pool_executable_flag,
                                       &raw);
        if(status != HSA_STATUS_SUCCESS || raw == nullptr)
        {
            ROCP_ERROR << "Failed to allocate shared thread trace buffer: " << status;
            return nullptr;
        }

        // Page-align to avoid cache flush overlap.
        void* aligned = reinterpret_cast<void*>(  // NOLINT(performance-no-int-to-ptr)
            (reinterpret_cast<uintptr_t>(raw) + PAGE_ALIGN_MASK) & ~PAGE_ALIGN_MASK);

        buf.raw       = raw;
        buf.aligned   = aligned;
        buf.size      = entry.max_buffer_size;
        state.free_fn = pool.free_fn;
        state.shared_ptrs.insert(aligned);
        return aligned;
    });
}

bool
is_shared_buffer(void* ptr)
{
    if(ptr == nullptr) return false;
    return get_state().rlock(
        [&](const shared_state_t& state) { return state.shared_ptrs.count(ptr) != 0; });
}

void
free_shared_buffers()
{
    get_state().wlock([](shared_state_t& state) {
        if(state.free_fn != nullptr)
        {
            for(auto& [_, entry] : state.agents)
                for(auto& buf : entry.buffers)
                    if(buf.raw != nullptr) state.free_fn(buf.raw);
        }

        state.agents.clear();
        state.shared_ptrs.clear();
        state.free_fn = nullptr;
    });
}

}  // namespace thread_trace
}  // namespace rocprofiler
