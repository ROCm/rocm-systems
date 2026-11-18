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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

// Kernarg memory that outlives the range it was allocated for.
//
// Every replayed pass reads its dispatches' arguments from a kernarg block the executor fills. The
// block comes from the agent's kernarg pool, so allocating it costs a pool allocation and a GPU
// mapping, and freeing it another runtime call. A tool that closes one range per iteration would
// pay both every iteration, so the executor keeps the block when its range finishes and the next
// range on the same agent reuses it.
//
// A block is retained only after its range's last pass has drained, so the GPU is no longer reading
// it when a later range refills it. The agent replay lock admits one replay window per agent at a
// time, so each agent has at most one block in use or retained, and a retained block is at most
// twice the most recent range's kernarg footprint (see kernarg_block_fits). Retained blocks are
// never freed at exit; the runtime reclaims them with the process.

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>

#include <cstddef>

namespace rocprofiler
{
namespace range_replay
{
struct kernarg_block_t
{
    void*  base                    = nullptr;
    size_t capacity                = 0;
    hsa_status_t (*free_fn)(void*) = nullptr;
};

// Whether a block of `capacity` bytes may serve a range needing `request` bytes. A smaller block
// cannot. One more than twice the request is freed rather than reused, so a single large range
// does not pin its block for every smaller range after it.
constexpr bool
kernarg_block_fits(size_t capacity, size_t request)
{
    return capacity >= request && capacity / 2 <= request;
}

// Hand over the block retained for `agent`, leaving none retained there. Empty when there is none.
kernarg_block_t
take_retained_kernarg_block(rocprofiler_agent_id_t agent);

// Keep `block` for the next range on `agent`, freeing any block already retained there.
void
retain_kernarg_block(rocprofiler_agent_id_t agent, kernarg_block_t block);

// Free `block` and leave it empty.
void
free_kernarg_block(kernarg_block_t& block);

// Capacity of the block retained for `agent`, or 0 when none is.
size_t
retained_kernarg_bytes(rocprofiler_agent_id_t agent);
}  // namespace range_replay
}  // namespace rocprofiler
