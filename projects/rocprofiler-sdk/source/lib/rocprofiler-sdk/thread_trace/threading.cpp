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

// Implements the CPU-side producer/consumer loops that service ATT triple buffering.
#include "lib/rocprofiler-sdk/thread_trace/threading.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace rocprofiler
{
namespace thread_trace
{
constexpr double SQTT_BANDWIDTH = 17E9f * 4;  // 17GB/s, times 4 for wiggle room

namespace
{
struct trace_callback_data_t
{
    void*        data{};
    uint64_t     size{};
    hsa_status_t status{};
};

trace_callback_data_t
iterate_data(aqlprofile_handle_t handle)
{
    auto thread_trace_callback = [](uint32_t, void* buffer, uint64_t size, void* userdata) {
        auto& data = *static_cast<trace_callback_data_t*>(userdata);
        data.data  = buffer;
        data.size  = size;
        return HSA_STATUS_SUCCESS;
    };
    trace_callback_data_t data{};
    data.status = aqlprofile_att_iterate_data(handle, thread_trace_callback, &data);
    return data;
}
};  // namespace

// Performs a synchronous GPU-to-CPU copy using the async engine, chaining the supplied dependency
// and reusing a thread-local completion signal to avoid allocation churn.
void
copy_data_sync(void*       dst,
               const void* src,
               hsa_agent_t dst_agent,
               hsa_agent_t src_agent,
               size_t      size,
               Signal*     dependency)
{
    ROCP_FATAL_IF(dependency == nullptr) << "Dependency must not be null";

    thread_local Signal signal{};
    auto                dep = dependency->getSignal();

    auto copy_fn = CHECK_NOTNULL(hsa::get_amd_ext_table())->hsa_amd_memory_async_copy_fn;

    signal.reset();
    auto status = copy_fn(dst, dst_agent, src, src_agent, size, 1, &dep, signal.getSignal());
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to copy: " << status;
    signal.WaitOn();
}

void
consumer_loop(triple_buffer_consumer_data_t parameters)
{
    auto& buffers     = parameters.shared->buffers;
    auto& running     = parameters.shared->consumer_running;
    auto& write_cv    = parameters.shared->write_cv;
    auto& write_index = parameters.shared->write_index;
    auto& read_index  = parameters.shared->read_index;
    auto  agent_id    = parameters.shared->queue->agent_id;
    auto  userdata    = parameters.userdata;
    auto  callback_fn = parameters.callback_fn;

    while(true)
    {
        auto& buffer = buffers.at(read_index % buffers.size());
        auto  lock   = std::unique_lock{buffer.mutex};

        // Wait until the producer signals that a new buffer is ready or the trace shuts down.
        write_cv.wait(lock, [&]() { return write_index > read_index || !running; });

        if(!running && write_index <= read_index) return;

        auto flags = static_cast<rocprofiler_thread_trace_shader_data_flags_t>(buffer.flags);
        callback_fn(agent_id, 0, buffer.memory, buffer.size, flags, userdata);
        read_index.fetch_add(1);
    }
}

// Producer loop: Polls SQTT hardware status, copies GPU trace buffers to CPU memory,
// and wakes the consumer thread when data is ready.
//
// The producer operates in three phases:
// 1. Poll: Send status query packets to check if GPU buffer is full
// 2. Copy: When buffer is full, perform async GPU->CPU memory copy
// 3. Notify: Signal the consumer via condition variable that buffer is ready to process
//
// The loop uses adaptive polling with backoff based on estimated bandwidth to minimize
// CPU overhead while ensuring timely buffer flips before GPU overflow.
void
producer_loop(triple_buffer_producer_data_t parameters)
{
    CHECK_NOTNULL(parameters.copy_data_fn);

    auto& queue = *CHECK_NOTNULL(parameters.shared->queue);
    auto& flag  = *CHECK_NOTNULL(parameters.producer_running);

    const size_t buffer_size           = queue.buffer_size;
    auto&        buffers               = parameters.shared->buffers;
    const auto   interval_microseconds = static_cast<size_t>(1E6 * buffer_size / SQTT_BANDWIDTH);

    auto& write_cv      = parameters.shared->write_cv;
    auto& write_index   = parameters.shared->write_index;
    auto& read_index    = parameters.shared->read_index;
    auto& buffer_packet = *CHECK_NOTNULL(parameters.buffer_packet);

    Signal submit_signal{};

    auto start_t0 = std::chrono::system_clock::now();
    bool do_sleep{false};
    // Wait until ATT start packets have been executed
    CHECK_NOTNULL(parameters.start_pkt_signal)->WaitOn();

    auto send_to_consumer = [&](void* src, size_t size, int flags) {
        auto t0 = std::chrono::system_clock::now();

        auto& buffer = buffers.at(write_index % buffers.size());
        auto  lock   = std::unique_lock{buffer.mutex};
        buffer.flags = flags;
        buffer.size  = size;

        // Perform the actual GPU->CPU memory copy into our triple-buffer slot
        parameters.copy_data_fn(
            buffer.memory, src, queue.near_cpu, queue.hsa_agent, size, &submit_signal);
        auto copy_time = (std::chrono::system_clock::now() - t0).count() * 1E-9f;
        ROCP_TRACE << "Copy: " << copy_time << " s. BW: " << size / float(copy_time);

        // PHASE 3: Wake up consumer thread
        // Increment write_index to signal a new buffer is available, then notify
        // the consumer via condition variable so it can process the data.
        write_index.fetch_add(1);
        write_cv.notify_all();
    };

    auto sleep_fn = [&]() {
        sched_yield();
        std::this_thread::sleep_for(std::chrono::microseconds(interval_microseconds));
    };

    auto stop_trace = [&]() {
        ROCP_INFO << "Stopping the trace";
        queue.SubmitAndSignalLast(parameters.control_packet->after_krn_pkt);
    };

    auto iterate_trace = [&]() {
        // Wait consumer to catch up before adding to next buffer
        while(read_index < write_index)
            sleep_fn();

        auto wptr = iterate_data(parameters.control_packet->GetHandle());
        ROCP_INFO << "Iterate data with size: " << wptr.size;
        send_to_consumer(wptr.data, wptr.size, ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_END);
    };

    while(flag.load())
    {
        if(do_sleep) sleep_fn();
        do_sleep = true;  // Reset value

        // PHASE 1: Poll SQTT buffer status
        // Send a query packet to the GPU asking if the trace buffer is full and ready to swap.
        // This is a non-blocking query that completes via signal.
        queue.Submit(&buffer_packet.query_status, &submit_signal);
        submit_signal.WaitOn();

        if(auto status = buffer_packet.query_buffer_status())
        {
            // PHASE 2: Copy GPU buffer to CPU memory
            // The GPU has signaled that a buffer is full. We need to:
            // a) Submit a packet to trigger GPU-side buffer swap
            // b) Copy the full buffer from GPU memory to our CPU-side triple buffer
            // Query returned buffer full: Send packet to trigger a buffer swap
            queue.Submit(&status->packet, &submit_signal);
            ROCP_FATAL_IF(status->size != buffer_size)
                << "GPU buffer overflow: " << status->size << " vs " << buffer_size;

            {
                // With triple buffering, stop when consumer lags by 2 buffers (all 3 slots full)
                // or when the GPU buffer has been tagged as full.
                const bool cpu_full = read_index + 2 < write_index;
                if(cpu_full || status->gpu_full) stop_trace();

                int flags = ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_NONE;
                if(cpu_full) flags |= ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL;
                if(status->gpu_full)
                    flags |= ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_GPU_BUFFER_FULL;

                send_to_consumer(status->data, buffer_size, flags);

                if(cpu_full || status->gpu_full)
                {
                    iterate_trace();
                    ROCP_INFO << "Restarting the trace!";

                    // We need to resend the header is applicable
                    if(buffer_packet.header)
                        send_to_consumer(&buffer_packet.header, sizeof(buffer_packet.header), ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_NONE);
                    queue.SubmitAndSignalLast(parameters.control_packet->before_krn_pkt);
                }
            }
            // The status_query test verifies we immediately poll again after consuming a
            // buffer, so skip the backoff when a flip just occurred.
            do_sleep = false;
            submit_signal.WaitOn();
        }
    }
    stop_trace();
    iterate_trace();
    parameters.shared->consumer_running.store(false);
    write_cv.notify_all();

    auto end_t0 = std::chrono::system_clock::now();
    ROCP_INFO << "Total trace size: " << (end_t0 - start_t0).count() * 1E-9f << " s.";
}
}  // namespace thread_trace
}  // namespace rocprofiler
