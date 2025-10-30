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
#include <mutex>
#include <thread>
#include <vector>

namespace rocprofiler
{
namespace thread_trace
{
constexpr double SQTT_BANDIWDTH = 17E9f * 3;  // 17GB/s, times 4 for wiggle room

namespace
{
void amd_copy_async(void* dst, const void* src, hsa_agent_t dst_agent, hsa_agent_t src_agent, size_t size, Signal& dependency)
{
    thread_local Signal signal{};
    auto dep = dependency.getSignal();

    auto copy_fn = CHECK_NOTNULL(hsa::get_amd_ext_table())->hsa_amd_memory_async_copy_fn;

    signal.reset();
    auto status = copy_fn(dst, dst_agent, src, src_agent, size, 1, &dep, signal.getSignal());
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to copy: " << status;
    signal.WaitOn();
}
};

void
worker_loop(hsa::SQTTBufferingPackets packets, triple_buffer_worker_data_t parameters)
{
    auto& queue = *CHECK_NOTNULL(parameters.queue);
    auto& flag  = *CHECK_NOTNULL(parameters.running_flag);

    const size_t buffer_size           = queue.buffer_size;
    const auto   buffer                = queue.get_double_buffer_memory();
    const auto   interval_microseconds = static_cast<size_t>(1E6 * buffer_size / SQTT_BANDIWDTH);

    std::atomic<bool>       consumer_running{true};
    std::condition_variable write_cv{};
    std::atomic<size_t>     write_index{0};
    std::atomic<size_t>     read_index{0};

    std::array<std::mutex, 2> mut{};

    static_assert(mut.size() == buffer.size());

    auto consumer = std::thread{[&]() {
        while(true)
        {
            size_t parity = read_index % buffer.size();
            {
                std::unique_lock<std::mutex> lock(mut.at(parity));
                write_cv.wait(lock,
                              [&]() { return write_index > read_index || !consumer_running; });
            }
            if(!consumer_running && write_index <= read_index) return;

            parameters.callback_fn(
                queue.agent_id, 0, buffer.at(parity), buffer_size, ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_NONE, parameters.userdata);
            read_index.fetch_add(1);
        }
    }};

    auto stop_consumer = [&]() {
        consumer_running.store(false);
        write_cv.notify_all();
        consumer.join();
    };

    auto stop_trace = [&]() {
        queue.SubmitAndSignalLast(parameters.control_packet->after_krn_pkt);
    };

    Signal submit_signal{};

    auto copy_sync = [&](void* dst, const void* src) {
        amd_copy_async(dst, src, queue.near_cpu, queue.hsa_agent, buffer_size, submit_signal);
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
        queue.Submit(&packets.query_status, &submit_signal);
        submit_signal.WaitOn();
        if(auto status = packets.query_buffer_status())
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
                    size_t                       parity = write_index % buffer.size();
                    std::unique_lock<std::mutex> lock(mut.at(parity));

                    copy_sync(buffer.at(parity), status->data);
                    auto copy_time = (std::chrono::system_clock::now() - t0).count() * 1E-9f;
                    ROCP_TRACE << "Copy: " << copy_time << " s. BW: " << buffer_size / float(copy_time);

                    write_index.fetch_add(1);
                    write_cv.notify_all();
                }

                if(should_stop)
                {
                    stop_consumer();
                    return;
                }
            }
            // If a buffer flip has happened, we dont want to sleep due to transfer delay
            do_sleep = false;
            submit_signal.WaitOn();
        }
    }
    stop_trace();
    stop_consumer();

    auto end_t0 = std::chrono::system_clock::now();
    ROCP_INFO << "Total trace size: " << (end_t0 - start_t0).count() * 1E-9f << " s.";
}
}  // namespace thread_trace
}  // namespace rocprofiler
