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

void copy_data_sync(void* dst, const void* src, hsa_agent_t dst_agent, hsa_agent_t src_agent, size_t size, Signal* dependency)
{
    ROCP_FATAL_IF(dependency == nullptr) << "Dependency must not be null";

    thread_local Signal signal{};
    auto dep = dependency->getSignal();

    auto copy_fn = CHECK_NOTNULL(hsa::get_amd_ext_table())->hsa_amd_memory_async_copy_fn;

    signal.reset();
    auto status = copy_fn(dst, dst_agent, src, src_agent, size, 1, &dep, signal.getSignal());
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to copy: " << status;
    signal.WaitOn();
}

void
consumer_loop(triple_buffer_consumer_data_t parameters)
{
    const size_t buffer_size = parameters.shared->queue->buffer_size;
    const auto   buffer      = parameters.shared->queue->get_double_buffer_memory();
    auto&        running     = parameters.shared->consumer_running;
    auto&        write_cv    = parameters.shared->write_cv;
    auto&        mut         = parameters.shared->mut;
    auto&        write_index = parameters.shared->write_index;
    auto&        read_index  = parameters.shared->read_index;
    auto         agent_id    = parameters.shared->queue->agent_id;
    auto         userdata    = parameters.userdata;
    auto         callback_fn = parameters.callback_fn;

    while(true)
    {
        size_t parity = read_index % buffer.size();
        auto   lock   = std::unique_lock{mut.at(parity).first};

        write_cv.wait(lock, [&]() { return write_index > read_index || !running; });

        if(!running && write_index <= read_index) return;

        auto flags = static_cast<rocprofiler_thread_trace_shader_data_flags_t>(mut.at(parity).second);
        callback_fn(agent_id, 0, buffer.at(parity), buffer_size, flags, userdata);
        read_index.fetch_add(1);
    }
}

void
producer_loop(triple_buffer_producer_data_t parameters)
{
    auto& queue = *CHECK_NOTNULL(parameters.shared->queue);
    auto& flag  = *CHECK_NOTNULL(parameters.producer_running);

    const size_t buffer_size           = queue.buffer_size;
    const auto   buffer                = queue.get_double_buffer_memory();
    const auto   interval_microseconds = static_cast<size_t>(1E6 * buffer_size / SQTT_BANDWIDTH);

    auto& write_cv      = parameters.shared->write_cv;
    auto& mut           = parameters.shared->mut;
    auto& write_index   = parameters.shared->write_index;
    auto& read_index    = parameters.shared->read_index;
    auto& buffer_packet = *CHECK_NOTNULL(parameters.buffer_packet);

    auto stop_trace = [&]() {
        queue.SubmitAndSignalLast(parameters.control_packet->after_krn_pkt);
    };

    Signal submit_signal{};

    auto copy_sync = [&](void* dst, const void* src) {
        auto copy_data_fn = CHECK_NOTNULL(parameters.copy_data_fn);
        copy_data_fn(dst, src, queue.near_cpu, queue.hsa_agent, buffer_size, &submit_signal);
    };

    auto start_t0 = std::chrono::system_clock::now();
    bool do_sleep = false;
    // Wait until ATT start packets have been executed
    CHECK_NOTNULL(parameters.start_pkt_signal)->WaitOn();

    while(flag.load())
    {
        if(do_sleep) std::this_thread::sleep_for(std::chrono::microseconds(interval_microseconds));
        do_sleep = true;  // Reset value

        // Send query status packet and wait for result
        queue.Submit(&buffer_packet.query_status, &submit_signal);
        submit_signal.WaitOn();
        if(auto status = buffer_packet.query_buffer_status())
        {
            auto t0 = std::chrono::system_clock::now();
            // Query returned buffer full: Send packet to trigger a buffer swap
            queue.Submit(&status->packet, &submit_signal);
            ROCP_FATAL_IF(status->size != buffer_size)
                << "GPU buffer overflow: " << status->size << " vs " << buffer_size;

            {
                const bool should_stop = read_index + 1 < write_index;
                if(should_stop)
                {
                    ROCP_WARNING << "SQTT buffer full!";
                    stop_trace();  // Check is_running so we dont send twice
                    while(read_index + 1 < write_index)
                        std::this_thread::sleep_for(std::chrono::microseconds(10));
                }

                {
                    int flags = ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_NONE;

                    size_t parity = write_index % buffer.size();
                    auto   lock   = std::unique_lock{mut.at(parity).first};

                    if (should_stop)
                        flags = ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL;

                    mut.at(parity).second = flags;

                    copy_sync(buffer.at(parity), status->data);
                    auto copy_time = (std::chrono::system_clock::now() - t0).count() * 1E-9f;
                    ROCP_TRACE << "Copy: " << copy_time << " s. BW: " << buffer_size / float(copy_time);

                    write_index.fetch_add(1);
                    write_cv.notify_all();
                }

                if(should_stop)
                {
                    parameters.shared->consumer_running.store(false);
                    write_cv.notify_all();
                    return;
                }
            }
            // If a buffer flip has happened, we dont want to sleep due to transfer delay
            do_sleep = false;
            submit_signal.WaitOn();
        }
    }
    stop_trace();
    parameters.shared->consumer_running.store(false);
    write_cv.notify_all();

    auto end_t0 = std::chrono::system_clock::now();
    ROCP_INFO << "Total trace size: " << (end_t0 - start_t0).count() * 1E-9f << " s.";
}
}  // namespace thread_trace
}  // namespace rocprofiler
