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
// Over-allocate, then round the returned pointer up to a 4K page. The round-up consumes
// up to one page of the padding; the second padding page is a tail guard so cache-flush
// operations at the end of the buffer cannot spill into the next allocation.
constexpr size_t PAGE_ALIGN_PADDING = 0x2000;   // 2 pages: align slack + tail guard
constexpr size_t PAGE_ALIGN_MASK    = 0xFFFul;  // align to a 4K page

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

// Held in a static_object (see shared_trace_lease.cpp for the rationale) so it stays alive
// through registration::finalize() on the attach path, without leaking. free_shared_buffers()
// frees the contents.
common::Synchronized<shared_state_t>& g_buffer_state =
    *common::static_object<common::Synchronized<shared_state_t>>::construct();
}  // namespace

void
register_shared_buffer_size(hsa_agent_t agent, uint64_t buffer_size)
{
    g_buffer_state.wlock([&](shared_state_t& state) {
        auto& entry           = state.agents[agent.handle];
        entry.max_buffer_size = std::max(entry.max_buffer_size, buffer_size);
    });
}

void*
acquire_shared_buffer(const hsa::TraceMemoryPool& pool, size_t index, uint64_t size)
{
    return g_buffer_state.wlock([&](shared_state_t& state) -> void* {
        // Self-register in case the pre-pass didn't: acquire is the single buffer source.
        auto& entry           = state.agents[pool.gpu_agent.handle];
        entry.max_buffer_size = std::max(entry.max_buffer_size, size);

        if(entry.buffers.size() <= index) entry.buffers.resize(index + 1);

        // Reuse the slot's buffer; the pre-pass sizes it to the agent max up front.
        auto& buf = entry.buffers.at(index);
        if(buf.aligned != nullptr && buf.size >= size) return buf.aligned;

        // Re-allocating an existing (too-small) slot: release the old buffer first so we
        // neither leak it nor leave a stale aligned pointer registered as shared. In the
        // normal flow the pre-pass fixes max_buffer_size before any acquire, so this only
        // runs if a larger size ever arrives after a slot was first sized.
        if(buf.raw != nullptr)
        {
            if(pool.free_fn) pool.free_fn(buf.raw);
            state.shared_ptrs.erase(buf.aligned);
            buf = shared_buffer_t{};
        }

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
    return g_buffer_state.rlock(
        [&](const shared_state_t& state) { return state.shared_ptrs.count(ptr) != 0; });
}

void
free_shared_buffers()
{
    g_buffer_state.wlock([](shared_state_t& state) {
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
