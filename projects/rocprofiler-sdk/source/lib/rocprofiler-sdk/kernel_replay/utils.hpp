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

// Internal helpers shared between the kernel-replay memory tracker and its tests. These are kept
// out of the primary memory_tracker.hpp interface on purpose: the allocation classifier is an
// implementation detail of what the tracker records, and exposing it here lets tests verify the
// kernarg/host pool exclusions directly without polluting the public tracker surface.

#include <hsa/hsa.h>

namespace rocprofiler
{
namespace kernel_replay
{
namespace memory_tracker
{
// Per-allocation properties resolved from a single hsa_amd_pointer_info query.
struct alloc_query_t
{
    hsa_agent_t agent = {.handle = 0};  // owning (preferred-access) agent, for per-agent scoping
    bool        trackable = false;      // coarse-grained device VRAM, excluding kernarg
};

// Classify a pointer via hsa_amd_pointer_info. Used by the allocation interceptor to decide what to
// record, and by tests to confirm the exclusions hold. We snapshot only coarse-grained device
// (VRAM) memory:
//  - kernarg is excluded because it holds kernel pointer arguments; a torn/stale restore of it
//    faults the GPU -- this is the primary reason snap/restore must not touch it.
//  - fine-grained / host memory is out of scope (and precarious to restore).
// agentOwner additionally lets snapshots be scoped to the replaying agent.
alloc_query_t
query_alloc(void* ptr);
}  // namespace memory_tracker
}  // namespace kernel_replay
}  // namespace rocprofiler
