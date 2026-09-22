// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
// <evntcons.h> is not self-contained; it must follow <windows.h>.
#include <evntcons.h>

#include <cstdint>

namespace rocprofiler
{
namespace hip
{
namespace etw
{
struct decoder_stats
{
    uint64_t events_decoded  = 0;
    uint64_t events_dropped  = 0;
    uint64_t records_emitted = 0;
    uint64_t unpaired_enters = 0;
};

// Restrict decoding to one producer process. The kernel-side EVENT_FILTER_TYPE_PID caps at
// eight pids, so this is the authoritative filter and the kernel one is only an optimization.
void
set_process_filter(uint32_t process_id);

// Decode one delivered event and, on a matching enter/exit pair, emplace a
// rocprofiler_buffer_tracing_hip_api_record_t into every subscribed buffer. Called only from
// the thread running ProcessTrace().
void
handle_event(const EVENT_RECORD* event_record);

// Drop all pending state and return the counters accumulated since the last reset. Unpaired
// enters still in flight are added to unpaired_enters.
decoder_stats
reset_decoder();
}  // namespace etw
}  // namespace hip
}  // namespace rocprofiler
