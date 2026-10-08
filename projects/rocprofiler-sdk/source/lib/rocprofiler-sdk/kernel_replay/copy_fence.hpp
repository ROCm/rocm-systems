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

#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <hsa/hsa.h>

#include <chrono>
#include <cstdint>

namespace rocprofiler
{
namespace kernel_replay
{
// Fences async copies (hsa_amd_memory_async_copy, _on_engine and _rect; HIP issues
// hipMemcpyAsync through them) against kernel-replay windows.
//
// The replay window drains the agent's AQL queues before it snapshots, but async copies run on
// SDMA engines, or on ROCr's internal blit queues, which neither that drain nor the per-agent
// replay lock sees. A copy in flight when the window snapshots is captured half-written, and the
// restores between passes then write that half-written snapshot back over bytes the copy has
// since delivered. The application later reads stale data and nothing reports it.
//
// With kernel replay configured, every async copy counts itself in flight on each agent it
// touches until its completion signal fires, and does not start while a replay window is open on
// any of those agents. A replay window opens the fence on its agent before it snapshots: new copies
// on the agent wait, and the window waits for the copies already in flight.
namespace copy_fence
{
// Lets the copies that the previous window held off start, then blocks new async copies on
// `agent` and waits until the ones in flight complete. Returns false, with the fence closed again,
// when they do not complete within `timeout` (a copy waiting on a dependency signal that cannot
// fire while the window holds the replay lock); the caller then declines replay for the dispatch.
bool
open_window(hsa_agent_t agent, std::chrono::milliseconds timeout);

// Lets async copies on `agent` start again. Only call after a successful open_window().
void
close_window(hsa_agent_t agent);

// Async copies currently in flight on `agent` (for tests and diagnostics).
int64_t
copies_in_flight(hsa_agent_t agent);
}  // namespace copy_fence

// Wraps the async copy entries of `table`. The wrappers cost one relaxed atomic load on top of the
// chained call while kernel replay is not configured.
void
copy_fence_init(hsa::hsa_amd_ext_table_t* table, uint64_t lib_instance);
}  // namespace kernel_replay
}  // namespace rocprofiler
