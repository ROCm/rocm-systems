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

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace rocprofiler
{
namespace tool
{
// The passes kernel replay runs for one dispatch: the agent's counter groups in order, plus at
// most one dispatch thread trace pass placed among them. The thread trace never shares a pass with
// counter collection, so every pass collects exactly one of the two.
struct replay_pass_layout
{
    uint64_t                counter_passes    = 0;
    std::optional<uint64_t> thread_trace_pass = std::nullopt;

    uint64_t total_passes() const;

    // The counter group a pass collects, or nullopt for the thread trace pass. Groups keep their
    // order; those after the thread trace pass run one pass later.
    std::optional<uint64_t> counter_group(uint64_t pass) const;
};

// The pass an agent's thread trace takes: right after each of the agent's counter groups that was
// built from one of the first `att_after_groups` configured groups. `agent_group_sources[i]` is the
// position, in the configured group list, of the group behind the agent's i-th counter profile. An
// agent skips the configured groups it cannot collect, so the two lists can differ.
uint64_t
thread_trace_pass_index(const std::vector<uint64_t>& agent_group_sources,
                        uint64_t                     att_after_groups);
}  // namespace tool
}  // namespace rocprofiler
