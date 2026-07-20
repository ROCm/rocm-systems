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

#include <hsa/hsa.h>

namespace rocprofiler
{
namespace thread_trace
{
// Process-global per-agent exclusive trace lease. The shared output buffer and
// submission queue for an agent are safe to share only because at most one trace is
// active per agent at a time. This lease enforces that invariant directly instead of
// relying on higher-level context management: while one owner (a ThreadTracerAgent)
// holds an agent's lease, another owner's acquire fails, so two contexts can never
// use the same shared buffer/queue concurrently (e.g. a dispatch trace still draining
// in-flight while a device trace starts on the same agent).
//
// Reference counted per owner so a same-owner handoff -- one trace's release interleaving
// with the next trace's acquire -- only decrements and never frees the lease out from
// under the owner. Overlapping traces are counted by ThreadTracerAgent::active_traces, not
// here. It is a common enforcement point a future CPU staging-buffer change can reuse.

/// Take or extend the lease on @p agent for @p owner. Returns true if @p owner now
/// holds it (freshly acquired, or already held and reference count incremented).
/// Returns false without side effects if a different owner currently holds it.
bool
try_acquire_agent_lease(hsa_agent_t agent, const void* owner);

/// Release one reference held by @p owner on @p agent's lease. The lease becomes free
/// once the owner's reference count reaches zero. No-op if @p owner is not the holder.
void
release_agent_lease(hsa_agent_t agent, const void* owner);

/// Drop all lease state. Called from thread_trace::finalize() once every tracer is
/// destroyed.
void
free_agent_leases();

}  // namespace thread_trace
}  // namespace rocprofiler
