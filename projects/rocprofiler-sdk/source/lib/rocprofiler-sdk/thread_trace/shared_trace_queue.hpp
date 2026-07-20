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

#include <cstdint>

namespace rocprofiler
{
namespace hsa
{
class AgentCache;
}  // namespace hsa

namespace thread_trace
{
struct att_queue_t;

// Process-global manager of per-agent SQTT submission queues. Only one trace is
// active per agent at a time, so every context on an agent shares a single HSA
// queue instead of each ThreadTracerAgent creating its own. A per-context queue
// exhausts the HSA per-agent queue limit after ~100 contexts; sharing lets hundreds
// of contexts coexist. Triple-buffer staging memory on the shared queue is sized to
// the largest requested buffer_size among triple-buffering contexts.

/// Record a context's CPU staging requirement for @p agent: @p buffer_size bytes per
/// staging buffer and @p num_buffers staging buffers (single-buffer contexts, i.e.
/// num_buffers <= 1, contribute nothing). Call for every context before the first
/// acquire so the shared queue's staging memory fits all contexts.
void
register_shared_queue_size(hsa_agent_t agent, uint64_t buffer_size, uint64_t num_buffers);

/// Return the shared queue for @p agent, allocating it (sized to the registered max
/// staging size/count, self-registering @p buffer_size / @p num_buffers first in case the
/// pre-pass was skipped) on first use and reusing it after. nullptr on failure.
att_queue_t*
acquire_shared_queue(const hsa::AgentCache& agent, uint64_t buffer_size, uint64_t num_buffers);

/// Destroy all shared per-agent queues. Called from thread_trace::finalize().
void
free_shared_queues();

}  // namespace thread_trace
}  // namespace rocprofiler
