// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

namespace rocprofiler
{
namespace tool
{
// kernel_id -> set of 1-based launch indices to profile (empty == all)
using kernel_target_map_t = std::unordered_map<uint64_t, std::unordered_set<size_t>>;
// kernel_id -> launches seen so far (1-based)
using kernel_iteration_count_map_t = std::unordered_map<uint64_t, size_t>;

// Advance kernel_id's 1-based counter and decide whether this dispatch is
// profiled. Pure core of is_targeted_kernel(), extracted for unit testing.
//   - kernel_id not targeted    -> false (not counted)
//   - empty range, PMC          -> every iteration
//   - empty range, thread trace -> first iteration only
//   - non-empty range           -> iterations in the range
bool
is_targeted_kernel_iteration(uint64_t                      kernel_id,
                             const kernel_target_map_t&    targeted_kernels,
                             kernel_iteration_count_map_t& kernel_iteration,
                             bool                          advanced_thread_trace);
}  // namespace tool
}  // namespace rocprofiler
