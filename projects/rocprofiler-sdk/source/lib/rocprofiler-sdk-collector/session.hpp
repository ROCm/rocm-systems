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

#include <rocprofiler-sdk-collector/types.h>

#include <cstdint>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace collector
{
// One LTTng capture session with one user-space channel.
struct trace_session_config
{
    std::string              name                = {};
    std::string              trace_path          = {};  // where the consumer daemon writes CTF
    std::vector<std::string> providers           = {};  // "<provider>:*" is enabled for each
    uint64_t                 subbuffer_size      = 0;   // 0: LTTng's per-process default
    uint64_t                 num_subbuffers      = 0;   // 0: LTTng's per-process default
    int64_t                  blocking_timeout_us = 0;   // 0: discard; -1: wait without limit
};

struct trace_session
{
    std::string name    = {};
    bool        created = false;
    bool        started = false;
};

// Create the session, its channel (per-process buffers, never overwritten: a full buffer
// makes a writer wait up to the blocking timeout, then discards and counts the event), the
// providers' events at every log level, and the vpid and vtid contexts. Tracing is not started.
rocprofiler_collector_status_t
create_trace_session(const trace_session_config& config, trace_session* session);

rocprofiler_collector_status_t
start_trace_session(trace_session* session);

// Stop tracing; returns once the buffers are written to the trace path.
rocprofiler_collector_status_t
stop_trace_session(trace_session* session);

// Stop tracing if started, then destroy the session.
rocprofiler_collector_status_t
destroy_trace_session(trace_session* session);
}  // namespace collector
}  // namespace rocprofiler
