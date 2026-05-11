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
#include <condition_variable>
#include <memory>
#include <mutex>

namespace rocprofiler
{
namespace thread_trace
{
/// Performs a blocking async copy while honoring the supplied signal dependency.
void
copy_data_sync(void*         dst,
               const void*   src,
               hsa_agent_t   dst_agent,
               hsa_agent_t   src_agent,
               size_t        size,
               hsa_signal_t* dependency);

typedef decltype(copy_data_sync) copy_data_t;

/// Shared state used to coordinate the producer and consumer sides of the triple buffer.
struct triple_buffer_shared_data_t
{
    /// Per-buffer state: mutex for synchronization, flags for buffer status, and CPU-side memory
    struct buffer_slot_t
    {
        std::mutex mutex{};
        int        flags{};
        void*      memory{nullptr};
        size_t     size{};
    };

    att_queue_t*            queue{nullptr};  // non-owning; ThreadTracerAgent owns the queue
    std::atomic<bool>       consumer_running{true};
    std::condition_variable write_cv{};
    std::atomic<size_t>     write_index{0};
    std::atomic<size_t>     read_index{0};

    std::array<buffer_slot_t, NUM_CPU_BUFFERS> buffers{};
};

/// Parameters passed into the consumer worker thread.
struct triple_buffer_consumer_data_t
{
    rocprofiler_thread_trace_shader_data_callback_t callback_fn{};
    rocprofiler_user_data_t                         userdata{};
    std::shared_ptr<triple_buffer_shared_data_t>    shared{};
};

/// Parameters passed into the producer worker thread.
struct triple_buffer_producer_data_t
{
    copy_data_t*                                 copy_data_fn{};
    std::shared_ptr<std::atomic<int>>            producer_running{};
    std::shared_ptr<hsa_signal_t>                start_pkt_signal{};
    std::unique_ptr<hsa::TraceControlAQLPacket>  control_packet{};
    std::shared_ptr<triple_buffer_shared_data_t> shared{};
    std::unique_ptr<hsa::SQTTBufferingPackets>   buffer_packet{};
};

// Worker flags have three states: stop (either stopped or stopping), running and (global)destructor
enum worker_flag_status_t
{
    WORKER_FLAG_STOP = 0,
    WORKER_FLAG_RUNNING,
    WORKER_FLAG_DESTRUCTOR
};

void
producer_loop(triple_buffer_producer_data_t parameters);

/// Important: Only one consumer is allowed per instance of "parameters"
void
consumer_loop(triple_buffer_consumer_data_t parameters);

};  // namespace thread_trace
};  // namespace rocprofiler
