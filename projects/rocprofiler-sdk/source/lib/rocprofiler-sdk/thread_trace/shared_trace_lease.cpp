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

#include "lib/rocprofiler-sdk/thread_trace/shared_trace_lease.hpp"

#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"

#include <cstdint>
#include <map>

namespace rocprofiler
{
namespace thread_trace
{
namespace
{
struct agent_lease_t
{
    const void* owner    = nullptr;  // the ThreadTracerAgent that currently holds it
    uint64_t    refcount = 0;        // overlapping traces from that same owner
};

struct lease_state_t
{
    std::map<uint64_t, agent_lease_t> agents = {};  // keyed by hsa_agent_t.handle
};

// Held in a static_object, not a plain namespace-scope global: on the attach path the global's
// destructor can run before finalize() frees it (via free_agent_leases()) -- a use-after-free.
// static_object is destroyed after finalize() (by destroy_static_objects()), so it outlives
// teardown without leaking. The buffer and queue managers mirror this.
common::Synchronized<lease_state_t>& g_lease_state =
    *common::static_object<common::Synchronized<lease_state_t>>::construct();
}  // namespace

bool
try_acquire_agent_lease(hsa_agent_t agent, const void* owner)
{
    return g_lease_state.wlock([&](lease_state_t& state) {
        auto& lease = state.agents[agent.handle];
        if(lease.refcount == 0)
        {
            lease.owner    = owner;
            lease.refcount = 1;
            return true;
        }
        if(lease.owner == owner)
        {
            ++lease.refcount;
            return true;
        }
        return false;  // held by a different owner
    });
}

void
release_agent_lease(hsa_agent_t agent, const void* owner)
{
    g_lease_state.wlock([&](lease_state_t& state) {
        auto it = state.agents.find(agent.handle);
        if(it == state.agents.end()) return;

        auto& lease = it->second;
        if(lease.owner != owner || lease.refcount == 0) return;

        if(--lease.refcount == 0) lease.owner = nullptr;
    });
}

void
free_agent_leases()
{
    g_lease_state.wlock([](lease_state_t& state) { state.agents.clear(); });
}

}  // namespace thread_trace
}  // namespace rocprofiler
