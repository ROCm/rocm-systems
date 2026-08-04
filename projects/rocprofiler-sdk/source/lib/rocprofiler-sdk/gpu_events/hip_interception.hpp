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

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>

#include <cstdint>

namespace rocprofiler
{
namespace gpu_events
{
uint64_t
get_gpu_event_id();

rocprofiler_stream_id_t
get_gpu_event_stream_id();

rocprofiler_gpu_event_operation_t
get_gpu_event_op();

struct pending_wait_info
{
    uint64_t                event_id;
    rocprofiler_stream_id_t wait_stream_id;
};

void
register_record_signal(uint64_t event_id, hsa_signal_t signal);

void
unregister_record_signal(uint64_t event_id);

bool
lookup_pending_wait(hsa_signal_t dep_signal, pending_wait_info& out);

template <typename TableT>
void
initialize(TableT* table);
}  // namespace gpu_events
}  // namespace rocprofiler
