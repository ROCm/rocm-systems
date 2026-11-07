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

// Divergence checking for range replay.
//
// A range is only a meaningful unit of measurement if re-executing it from the range-entry memory
// state reproduces the application's own execution. That holds when every input the recorded
// dispatches read is inside the snapshot's coverage. When it does not -- a kernel reads unified
// memory, or host code recomputed a buffer inside the range -- the replayed passes silently measure
// different work.
//
// Checking it is cheap: hash the snapshot's regions after the application's execution of the range,
// hash them again after the final replayed pass, and count the regions that differ. Two runs of the
// same deterministic program over the same inputs end in the same state, so a difference means the
// range was not self-contained (or the kernels are non-deterministic, which is equally
// disqualifying for pass-to-pass comparison).
//
// This is opt-in (it costs one extra device->host copy of the snapshot per digest) and reported to
// the tool as rocprofiler_callback_tracing_range_replay_data_t::divergence_count.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rocprofiler
{
namespace range_replay
{
namespace digest
{
// FNV-1a over `size` bytes at `data`. Chosen for being trivially auditable and dependency-free;
// this detects divergence, it does not defend against anything.
uint64_t
hash_bytes(const void* data, size_t size);

// Per-region digests of one point in time, positionally aligned with the snapshot's block list.
using region_digests_t = std::vector<uint64_t>;

// Number of regions whose digest differs. Mismatched lengths count as fully divergent: the two
// digests do not describe the same set of regions, so nothing can be concluded region by region.
size_t
count_divergent(const region_digests_t& lhs, const region_digests_t& rhs);
}  // namespace digest
}  // namespace range_replay
}  // namespace rocprofiler
