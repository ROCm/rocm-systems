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

#include <array>
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

/// Shared state coordinating the single producer and N worker threads.
///
/// Each slot is owned by exactly one consumer thread; the producer hands
/// chunks to whichever slot is currently free. A single global `stopping`
/// flag tells all consumers to exit during shutdown.
///
/// Producer flow:
///   1. Linear-scan slots for `filled == false`. If none free → CPU stall.
///   2. Populate slot fields (memory, size, flags, chunk_index, se_id).
///   3. Lock the slot's mutex, set `filled = true`, notify its cv.
///
/// Consumer flow (one thread per slot):
///   1. Wait on its own slot's cv until `filled == true || stopping`.
///   2. If `filled`, run the user callback then store `filled = false`.
///   3. If `stopping` and the slot is empty, exit.
///
/// Note: callbacks across slots run in parallel — chunk_index is provided so
/// downstream consumers can reorder if they need a strict sequence.
struct triple_buffer_shared_data_t
{
    struct buffer_slot_t
    {
        void*    memory{nullptr};
        size_t   size{};
        int      flags{};
        int64_t  se_id{0};
        uint64_t chunk_index{0};

        /// Producer sets true after writing slot fields; consumer stores
        /// false after running the callback.
        std::atomic<bool>       filled{false};
        std::mutex              mut{};
        std::condition_variable cv{};
    };

    /// Maximum number of slots. Capped at 16 to keep the per-slot thread
    /// count bounded; the public API rejects values above this.
    static constexpr size_t MAX_SLOTS = 16;

    att_queue_t* queue{nullptr};  // non-owning; ThreadTracerAgent owns the queue

    /// Global shutdown flag. Producer sets true after draining final chunks
    /// and notifies every slot's cv so consumers can exit.
    std::atomic<bool> stopping{false};

    /// Fixed-capacity storage; the active prefix has length `num_buffers`.
    /// Backed by std::array because `buffer_slot_t` holds non-movable
    /// synchronization primitives.
    std::array<buffer_slot_t, MAX_SLOTS> buffers{};
    size_t                               num_buffers{0};
};

/// Parameters passed into each worker thread (one instance per worker).
/// Each consumer is bound to a single slot index in the shared state.
struct triple_buffer_consumer_data_t
{
    rocprofiler_thread_trace_shader_data_callback_t callback_fn{};
    rocprofiler_user_data_t                         userdata{};
    std::shared_ptr<triple_buffer_shared_data_t>    shared{};
    /// Index of the slot this consumer thread owns.
    size_t slot_index{0};
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
    int64_t                                      shader_engine_id{0};
};

// Worker flags have three states: stop (either stopped or stopping), running and (global)destructor
enum worker_flag_status_t
{
    WORKER_FLAG_STOP = 0,
    WORKER_FLAG_RUNNING,
    WORKER_FLAG_DESTRUCTOR,
    WORKER_FLAG_ERROR
};

void
producer_loop(triple_buffer_producer_data_t parameters);

/// One instance per worker thread; spawn one per slot in the shared state,
/// each with a unique `slot_index`.
void
consumer_loop(triple_buffer_consumer_data_t parameters);

};  // namespace thread_trace
};  // namespace rocprofiler
