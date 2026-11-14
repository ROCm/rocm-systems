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
#include <optional>
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

// Remembers the kernel filter's verdict for the most recent logical dispatch seen by one thread.
//
// The iteration filter is stateful (every consultation advances the kernel's launch count), so a
// dispatch must be judged exactly once however many callbacks ask about it. Kernel replay asks
// several times per dispatch: at CONFIG (to size the replay), once per pass, and on the ordinary
// path when the dispatch is not replayed. All of them run synchronously on the enqueuing thread
// and carry the same dispatch id, so a single (id, verdict) slot per thread is enough.
class dispatch_decision_cache_t
{
public:
    std::optional<bool> find(uint64_t dispatch_id) const
    {
        if(dispatch_id != 0 && m_valid && m_dispatch_id == dispatch_id) return m_targeted;
        return std::nullopt;
    }

    void store(uint64_t dispatch_id, bool targeted)
    {
        if(dispatch_id == 0) return;
        m_valid       = true;
        m_dispatch_id = dispatch_id;
        m_targeted    = targeted;
    }

private:
    bool     m_valid       = false;
    uint64_t m_dispatch_id = 0;
    bool     m_targeted    = false;
};

// Returns the cached verdict for dispatch_id, or calls consult() once and caches the result.
// A dispatch_id of 0 means the dispatch has no stable identity, so it is never cached.
template <typename ConsultT>
bool
targeted_once_per_dispatch(dispatch_decision_cache_t& cache,
                           uint64_t                   dispatch_id,
                           ConsultT&&                 consult)
{
    if(const auto cached = cache.find(dispatch_id)) return *cached;
    const bool targeted = consult();
    cache.store(dispatch_id, targeted);
    return targeted;
}

// Number of kernel-replay passes for a dispatch. A dispatch the kernel filter does not target is
// never replayed: replay snapshots and restores device memory and re-executes the kernel, which is
// wasted work for a kernel nobody asked to profile and incorrect for one with cross-process side
// effects (a collective communication kernel in an MPI job). 1 means "run once, no replay".
inline uint64_t
replay_pass_count_for_dispatch(bool targeted, size_t counter_groups)
{
    return (!targeted || counter_groups == 0) ? uint64_t{1} : static_cast<uint64_t>(counter_groups);
}
}  // namespace tool
}  // namespace rocprofiler
