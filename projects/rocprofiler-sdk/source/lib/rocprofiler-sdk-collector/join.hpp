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

#include "lib/rocprofiler-sdk-collector/ctf/reader.hpp"
#include "lib/rocprofiler-sdk-collector/ctf/types.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace collector
{
// The end of a call whose exit event was not recorded: the buffers were full, or the call
// was still running when tracing stopped.
constexpr uint64_t no_end = std::numeric_limits<uint64_t>::max();

// One API call: an <api>_enter event joined with its <api>_exit event on the same thread.
struct api_call
{
    std::string                   provider       = {};  // "rocm_hip"
    std::string                   name           = {};  // "hipMalloc"
    uint64_t                      pid            = 0;   // vpid
    uint64_t                      tid            = 0;   // vtid
    uint64_t                      correlation_id = 0;   // per process, from 1, in start order
    uint64_t                      parent_id = 0;  // the enclosing call on the thread, 0 if none
    uint32_t                      depth     = 0;  // number of enclosing calls on the thread
    uint64_t                      start     = 0;  // clock ns (CLOCK_MONOTONIC for LTTng-UST)
    uint64_t                      end       = no_end;
    std::vector<ctf::field_value> arguments = {};  // the enter event's fields
    std::vector<ctf::field_value> results   = {};  // the exit event's fields
};

// A range of time in which a process's buffers discarded events.
struct event_gap
{
    uint64_t pid   = 0;  // 0 when the trace does not record its process
    uint64_t count = 0;
    uint64_t begin = 0;  // clock ns
    uint64_t end   = 0;
};

struct join_statistics
{
    ctf::read_statistics read            = {};
    uint64_t             calls           = 0;
    uint64_t             lost_exits      = 0;  // an exit missing inside a recorded call
    uint64_t             open_at_stop    = 0;  // still running when the trace ended
    uint64_t             unmatched_exits = 0;  // an exit with no recorded enter
    uint64_t             ambiguous_exits = 0;  // not paired: events lost in between
    uint64_t             other_events    = 0;  // not <api>_enter / <api>_exit
};

using api_call_callback = std::function<void(const api_call&)>;

// Join the API enter and exit events of the traces into calls. A call is reported once
// complete: at its exit, or with end = no_end when its exit is missing. Requires the vpid
// and vtid contexts.
//
// An exit is paired with the innermost open call of the same API on its thread. When the
// process's buffers discarded events between the two, the exit may belong to a call whose
// enter was lost: such an exit is not paired (ambiguous_exits), and the open call is
// reported with end = no_end (lost_exits). Gaps are known from the packet headers before
// the events are read. Transport fields (_ms_*) are dropped; a ";k<bit>;" keyword suffix is
// removed from event names.
bool
join_api_calls(const std::vector<ctf::trace_ptr>& traces,
               const api_call_callback&           on_call,
               std::vector<event_gap>*            gaps,
               join_statistics*                   statistics,
               std::string*                       error);

// Splits "<provider>:<api>_enter;k0;" into its parts; false for other event names.
bool
parse_api_event_name(const std::string& event_name,
                     std::string*       provider,
                     std::string*       api,
                     bool*              is_enter);
}  // namespace collector
}  // namespace rocprofiler
