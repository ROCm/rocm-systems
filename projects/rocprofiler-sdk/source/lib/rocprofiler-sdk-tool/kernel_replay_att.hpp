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

#pragma once

#include <cstddef>
#include <cstdint>

// Pass planning for dispatch thread trace (ATT) under kernel replay.
//
// A replayed dispatch runs once per counter group, then once more for thread trace:
// passes [0, groups) collect counters with thread trace stopped, and pass `groups` collects
// thread trace with counters stopped. Nothing here touches the SDK, so the decisions can be
// tested without a GPU; tool.cpp applies them from the replay callbacks.
namespace rocprofiler
{
namespace tool
{
namespace kernel_replay_att
{
struct pass_services_t
{
    bool counters     = false;
    bool thread_trace = false;
};

// Number of passes for a dispatch. A dispatch that is not replayed (counters paused, not selected
// by the kernel filters, or no counter groups on its agent) is one pass. Thread trace adds a pass
// after the counter groups, so a single group with thread trace is two passes and the two services
// never share an execution.
constexpr uint64_t
pass_count(size_t counter_groups, bool replayed, bool thread_trace)
{
    return (!replayed || counter_groups == 0) ? uint64_t{1}
                                              : counter_groups + (thread_trace ? 1 : 0);
}

// The one pass that collects thread trace. It follows the counter groups.
constexpr bool
is_thread_trace_pass(uint64_t pass, size_t counter_groups, bool thread_trace)
{
    return thread_trace && counter_groups > 0 && pass >= counter_groups;
}

// A pass past the counter groups has no group to collect, so it must not select a profile or
// produce a counter row.
constexpr bool
pass_has_no_counter_group(uint64_t pass, size_t counter_groups)
{
    return counter_groups > 0 && pass >= counter_groups;
}

// The services enabled during a pass. Exactly one of the two, so counter collection never observes
// thread trace's instrumentation and thread trace is captured once per dispatch.
constexpr pass_services_t
services_for_pass(uint64_t pass, size_t counter_groups, bool thread_trace)
{
    return is_thread_trace_pass(pass, counter_groups, thread_trace) ? pass_services_t{false, true}
                                                                    : pass_services_t{true, false};
}

// Counter collection and thread trace are handed the same user_data slot for one dispatch. The
// counter hook packs the enqueuing thread and replay pass into it and the counter record reads it
// back; thread trace stores its dispatch id there. Inside a replay pass only one service is
// enabled, so the slot is not shared. A dispatch outside any pass -- replay declined, a HIP graph,
// a multi-packet submission -- runs both services, and thread trace must leave the counter value
// alone. Without counter groups on the agent there is no counter record to protect.
constexpr bool
thread_trace_yields_to_counters(bool kernel_replay, bool in_replay_pass, size_t counter_groups)
{
    return kernel_replay && !in_replay_pass && counter_groups > 0;
}
}  // namespace kernel_replay_att
}  // namespace tool
}  // namespace rocprofiler
