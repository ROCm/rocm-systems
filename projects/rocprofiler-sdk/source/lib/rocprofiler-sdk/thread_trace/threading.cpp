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
#include "lib/common/environment.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"

#include <fmt/format.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace rocprofiler
{
namespace thread_trace
{
constexpr double SQTT_BANDWIDTH_DEFAULT = 70E9;  // 70GB/s, for wiggle room

namespace
{
// RAII wrapper for hsa_signal_t used in .cpp scope
struct scoped_signal_t
{
    hsa_signal_t sig;
    scoped_signal_t()
    : sig{signal_create()}
    {}
    ~scoped_signal_t() { signal_destroy(sig); }
    scoped_signal_t(const scoped_signal_t&) = delete;
    scoped_signal_t& operator=(const scoped_signal_t&) = delete;
};

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
copy_data_sync(void*         dst,
               const void*   src,
               hsa_agent_t   dst_agent,
               hsa_agent_t   src_agent,
               size_t        size,
               hsa_signal_t* dependency)
{
    ROCP_FATAL_IF(dependency == nullptr) << "Dependency must not be null";
    ROCP_TRACE << fmt::format("Executing async copy from {} to {}", src, dst);

    thread_local auto signal = scoped_signal_t{};

    auto copy_fn = CHECK_NOTNULL(hsa::get_amd_ext_table())->hsa_amd_memory_async_copy_fn;

    signal_reset(signal.sig);
    auto status = copy_fn(dst, dst_agent, src, src_agent, size, 1, dependency, signal.sig);
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to copy: " << status;
    signal_wait(signal.sig);
}

// Consumer loop: Waits for consumer notification, then forwards to tool/user
// read_index is incremented so the consumer is aware they can try to lock the buffer mutex
void
consumer_loop(
    triple_buffer_consumer_data_t parameters)  // NOLINT(performance-unnecessary-value-param)
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

        // wait_for handles the shutdown race: if notify_all is lost (producer
        // holds a different buffer mutex), the timeout ensures we re-check.
        write_cv.wait_for(lock, std::chrono::milliseconds(1), [&]() {
            return write_index > read_index || !running;
        });

        if(write_index <= read_index)
        {
            if(!running) return;
            continue;
        }

        ROCP_TRACE << read_index.load() << "Consumer received ptr " << buffer.memory;
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
producer_loop(
    triple_buffer_producer_data_t parameters)  // NOLINT(performance-unnecessary-value-param)
{
    CHECK_NOTNULL(parameters.copy_data_fn);

    auto& queue       = *CHECK_NOTNULL(parameters.shared->queue);
    auto& worker_flag = *CHECK_NOTNULL(parameters.producer_running);

    const size_t buffer_size = queue.buffer_size;
    auto&        buffers     = parameters.shared->buffers;
    const auto   sqtt_bandwidth =
        std::max(1.0, common::get_env("ROCPROFILER_SQTT_BANDWIDTH", SQTT_BANDWIDTH_DEFAULT));
    const auto interval_microseconds = static_cast<size_t>(1E6 * buffer_size / sqtt_bandwidth);

    auto& write_cv      = parameters.shared->write_cv;
    auto& write_index   = parameters.shared->write_index;
    auto& read_index    = parameters.shared->read_index;
    auto& buffer_packet = *CHECK_NOTNULL(parameters.buffer_packet);

    auto submit_signal = scoped_signal_t{};

    auto start_t0 = std::chrono::system_clock::now();
    bool do_sleep{false};
    // Wait until ATT start packets have been executed
    signal_wait(*CHECK_NOTNULL(parameters.start_pkt_signal));

    auto sleep_fn = [&]() {
        sched_yield();
        std::this_thread::sleep_for(std::chrono::microseconds(interval_microseconds));
    };

    auto send_to_consumer = [&](void* src, size_t size, int flags, bool isHeader = false) {
        auto t0 = std::chrono::system_clock::now();

        auto&       buffer      = buffers.at(write_index % buffers.size());
        auto        lock        = std::unique_lock{buffer.mutex};
        const auto& near_cpu_v  = queue.near_cpu;
        const auto& hsa_agent_v = queue.hsa_agent;
        buffer.flags            = flags;
        buffer.size             = size;

        // Perform the actual GPU->CPU memory copy into our triple-buffer slot
        if(!isHeader)
            parameters.copy_data_fn(
                buffer.memory, src, near_cpu_v, hsa_agent_v, size, &submit_signal.sig);
        else
            std::memcpy(buffer.memory, src, size);

        auto copy_time = (std::chrono::system_clock::now() - t0).count() * 1E-9f;
        ROCP_TRACE << "Copy: " << copy_time << " s. BW: " << size / copy_time;

        write_index.fetch_add(1);
        write_cv.notify_all();
    };

    auto stop_trace = [&]() {
        ROCP_INFO << "Stopping the trace";
        att_queue_submit_and_wait_last(queue, parameters.control_packet->after_krn_pkt);
    };

    auto iterate_trace = [&]() {
        // Wait consumer to catch up before adding to next buffer
        while(read_index < write_index)
            sleep_fn();

        auto wptr = iterate_data(parameters.control_packet->GetHandle());
        buffer_packet.reset_current_buffer();
        ROCP_INFO << "Iterate data with size: " << wptr.size;
        send_to_consumer(wptr.data, wptr.size, ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_END);
    };

    while(worker_flag.load() == WORKER_FLAG_RUNNING)
    {
        if(do_sleep) sleep_fn();
        do_sleep = true;  // Reset value

        // PHASE 1: Poll SQTT buffer status
        // Send a query packet to the GPU asking if the trace buffer is full and ready to swap.
        // This is a non-blocking query that completes via signal.
        att_queue_submit(queue, &buffer_packet.query_status, &submit_signal.sig);
        signal_wait(submit_signal.sig);

        if(auto status = buffer_packet.query_buffer_status())
        {
            ROCP_TRACE << "Sending buffer swap";
            // PHASE 2: Copy GPU buffer to CPU memory
            // The GPU has signaled that a buffer is full. We need to:
            // a) Submit a packet to trigger GPU-side buffer swap
            // b) Copy the full buffer from GPU memory to our CPU-side triple buffer
            // Query returned buffer full: Send packet to trigger a buffer swap
            att_queue_submit(queue, &status->packet, &submit_signal.sig);
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

                    // We need to resend the header if applicable
                    if(buffer_packet.header != 0)
                        send_to_consumer(&buffer_packet.header,
                                         sizeof(buffer_packet.header),
                                         ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_NONE,
                                         true);

                    att_queue_submit_and_wait_last(queue,
                                                   parameters.control_packet->before_krn_pkt);
                }
            }
            // The status_query test verifies we immediately poll again after consuming a
            // buffer, so skip the backoff when a flip just occurred.
            do_sleep = false;
            signal_wait(submit_signal.sig);
        }
    }
    stop_trace();
    iterate_trace();
    parameters.shared->consumer_running.store(false);
    write_cv.notify_all();

    auto end_t0 = std::chrono::system_clock::now();
    ROCP_INFO << "Total trace time: " << (end_t0 - start_t0).count() * 1E-9f << " s.";
    ROCP_INFO << "Total flips: " << write_index.load();
}
}  // namespace thread_trace
}  // namespace rocprofiler
