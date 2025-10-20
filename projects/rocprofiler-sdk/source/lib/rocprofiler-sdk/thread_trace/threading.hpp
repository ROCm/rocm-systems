// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/experimental/thread_trace.h>

#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/thread_trace/hsa_util.hpp"

#include <atomic>
#include <memory>

namespace rocprofiler
{
namespace thread_trace
{
struct triple_buffer_worker_data_t
{
    rocprofiler_thread_trace_shader_data_callback_t callback_fn{};
    rocprofiler_user_data_t                         userdata{};

    std::shared_ptr<HsaATTQueue>       queue{};
    std::shared_ptr<std::atomic<bool>> running_flag{};

    std::shared_ptr<Signal>                     start_pkt_signal{};
    std::unique_ptr<hsa::TraceControlAQLPacket> control_packet{};
};

void
worker_loop(hsa::SQTTBufferingPackets packets, triple_buffer_worker_data_t parameters);

};  // namespace thread_trace
};  // namespace rocprofiler
