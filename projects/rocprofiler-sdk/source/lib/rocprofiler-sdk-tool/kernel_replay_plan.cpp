// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#include "kernel_replay_plan.hpp"

#include <algorithm>

namespace rocprofiler
{
namespace tool
{
uint64_t
replay_pass_layout::total_passes() const
{
    return counter_passes + (thread_trace_pass ? 1 : 0);
}

std::optional<uint64_t>
replay_pass_layout::counter_group(uint64_t pass) const
{
    if(!thread_trace_pass || pass < *thread_trace_pass) return pass;
    if(pass == *thread_trace_pass) return std::nullopt;
    return pass - 1;
}

uint64_t
thread_trace_pass_index(const std::vector<uint64_t>& agent_group_sources, uint64_t att_after_groups)
{
    return static_cast<uint64_t>(
        std::count_if(agent_group_sources.begin(),
                      agent_group_sources.end(),
                      [att_after_groups](uint64_t source) { return source < att_after_groups; }));
}
}  // namespace tool
}  // namespace rocprofiler
