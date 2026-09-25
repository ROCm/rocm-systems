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

// Implements the CPU-side producer/consumer loops that service ATT N-buffering.
// One producer thread + one consumer thread per slot.
#include "lib/rocprofiler-sdk/thread_trace/threading.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

namespace rocprofiler
{
namespace thread_trace
{
constexpr double SQTT_BANDWIDTH_DEFAULT = 60E9;  // 60GB/s, for wiggle room

namespace
{
struct trace_callback_data_t
{
    void*        data{};
    uint64_t     size{};
    hsa_status_t status{};
};

trace_callback_data_t
iterate_data(hsa::SQTTBufferingPackets& packets)
{
    auto thread_trace_callback = [](uint32_t, void* buffer, uint64_t size, void* userdata) {
        auto& data = *static_cast<trace_callback_data_t*>(userdata);
        data.data  = buffer;
        data.size  = size;
        return HSA_STATUS_SUCCESS;
    };
    trace_callback_data_t data{};
    data.status = packets.iterate_data(thread_trace_callback, &data);
    return data;
}
};  // namespace

// Worker thread body. One instance per slot; each owns a single slot index.
// Waits on its slot's cv until the slot is filled or the global stop flag is
// set, runs the callback lock-free, then marks the slot free again.
void
consumer_loop(
    triple_buffer_consumer_data_t parameters)  // NOLINT(performance-unnecessary-value-param)
{
    auto& shared      = *parameters.shared;
    auto& slot        = shared.buffers[parameters.slot_index];
    auto& stopping    = shared.stopping;
    auto  agent_id    = shared.queue->agent_id;
    auto  userdata    = parameters.userdata;
    auto  callback_fn = parameters.callback_fn;

    while(true)
    {
        {
            auto lk = std::unique_lock{slot.mut};
            slot.cv.wait(lk, [&]() { return slot.filled.load() || stopping.load(); });
        }

        // Drain priority: process pending data even if a stop has been signaled.
        if(!slot.filled.load())
        {
            if(stopping.load()) return;
            continue;
        }

        ROCP_TRACE << "Worker handling chunk " << slot.chunk_index << " slot "
                   << parameters.slot_index << " ptr " << slot.memory;
        auto shader_data             = rocprofiler_thread_trace_shader_data_t{};
        shader_data.size             = sizeof(shader_data);
        shader_data.data             = slot.memory;
        shader_data.data_size        = slot.size;
        shader_data.shader_engine_id = slot.se_id;
        shader_data.chunk_index      = slot.chunk_index;
        shader_data.read_offset      = slot.read_offset;
        shader_data.agent            = agent_id;
        shader_data.flags = static_cast<rocprofiler_thread_trace_shader_data_flags_t>(slot.flags);

        callback_fn(shader_data, userdata);

        // Hand the slot back to the producer.
        slot.filled.store(false);
    }
}

// Producer loop: Polls SQTT hardware status, copies GPU trace buffers to CPU memory,
// and notifies the owning consumer of each filled slot.
//
// The producer operates in three phases:
// 1. Poll: Send status query packets to check if GPU buffer is full
// 2. Copy: Wait for the buffer swap, then synchronously copy GPU data to CPU memory
// 3. Notify: Signal the consumer that owns the slot via its per-slot cv
//
// The loop uses adaptive polling with backoff based on estimated bandwidth to minimize
// CPU overhead while ensuring timely buffer flips before GPU overflow.
void
producer_loop(
    triple_buffer_producer_data_t parameters)  // NOLINT(performance-unnecessary-value-param)
{
    CHECK_NOTNULL(parameters.copy_data_fn);
    CHECK_NOTNULL(parameters.submit_signal);

    auto& queue       = *CHECK_NOTNULL(parameters.shared->queue);
    auto& worker_flag = *CHECK_NOTNULL(parameters.producer_running);

    const size_t buffer_size = queue.buffer_size;
    auto&        buffers     = parameters.shared->buffers;
    const size_t num_buffers = parameters.shared->num_buffers;
    const auto   sqtt_bandwidth =
        std::max(1.0, common::get_env("ROCPROFILER_SQTT_BANDWIDTH", SQTT_BANDWIDTH_DEFAULT));
    const auto interval_microseconds = static_cast<size_t>(1E6 * buffer_size / sqtt_bandwidth);

    auto& buffer_packet = *CHECK_NOTNULL(parameters.buffer_packet);

    auto& submit_signal = *parameters.submit_signal;

    auto     start_t0 = std::chrono::system_clock::now();
    bool     do_sleep{false};
    uint64_t next_chunk_index = 0;

    auto sleep_fn = [&]() {
        sched_yield();
        std::this_thread::sleep_for(std::chrono::microseconds(interval_microseconds));
    };

    // Linear scan for any free (unfilled) slot. Returns num_buffers if none.
    auto try_claim_slot = [&]() -> size_t {
        for(size_t i = 0; i < num_buffers; i++)
            if(!buffers[i].filled.load()) return i;
        return num_buffers;
    };

    // Block until at least one slot is freed by a worker.
    auto wait_for_free_slot = [&]() -> size_t {
        for(;;)
        {
            size_t idx = try_claim_slot();
            if(idx != num_buffers) return idx;
            sleep_fn();
        }
    };

    auto send_to_consumer = [&](void*    src,
                                size_t   size,
                                int      flags,
                                size_t   slot_idx,
                                bool     isHeader    = false,
                                uint64_t read_offset = 0) {
        auto t0 = std::chrono::system_clock::now();

        auto& buffer       = buffers[slot_idx];
        buffer.flags       = flags;
        buffer.size        = size;
        buffer.se_id       = buffer_packet.shader_engine_id;
        buffer.chunk_index = next_chunk_index++;
        buffer.read_offset = read_offset;

        if(!isHeader)
            parameters.copy_data_fn(queue, buffer.memory, src, size);
        else
            std::memcpy(buffer.memory, src, size);

        auto copy_time = (std::chrono::system_clock::now() - t0).count() * 1E-9f;
        ROCP_TRACE << "Copy: " << copy_time << " s. BW: " << size / copy_time;

        // Publish: producer's writes above happen-before the consumer's
        // observation of `filled` via the slot mutex's release/acquire.
        {
            auto lk = std::unique_lock{buffer.mut};
            buffer.filled.store(true);
        }
        buffer.cv.notify_one();
    };

    auto stop_trace = [&]() {
        ROCP_INFO << "Stopping the trace";
        if(!att_queue_submit(
               queue, &parameters.control_packet->after_krn_pkt.at(0), &submit_signal))
        {
            ROCP_CI_LOG(ERROR) << "Failed to submit thread-trace stop packet for agent "
                               << queue.agent_id.handle;
            return false;
        }
        signal_wait(submit_signal);
        return true;
    };

    // Drain remaining ATT data after a stop; waits for a free slot to land it in.
    auto iterate_trace = [&]() {
        size_t idx  = wait_for_free_slot();
        auto   wptr = iterate_data(buffer_packet);
        buffer_packet.reset_current_buffer();
        int flags = ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_END;
        if(wptr.status == HSA_STATUS_ERROR_OUT_OF_RESOURCES)
            flags |= ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_GPU_BUFFER_FULL;
        if(wptr.status != HSA_STATUS_SUCCESS || (wptr.size > 0 && !wptr.data) ||
           wptr.size > buffer_size)
        {
            if(wptr.status == HSA_STATUS_ERROR_OUT_OF_RESOURCES)
                ROCP_WARNING << "Discarding ATT drain payload after GPU buffer overflow";
            else
                ROCP_CI_LOG(ERROR) << "Discarding ATT drain payload: status " << wptr.status
                                   << ", size " << wptr.size;
            wptr.size = 0;
        }
        ROCP_INFO << "Iterate data with size: " << wptr.size;
        send_to_consumer(wptr.data, wptr.size, flags, idx);
    };

    std::array<uint64_t, 4> header_plus_zeros{};  // Used for warmup the decoder path
    header_plus_zeros.at(0) = buffer_packet.header;

    auto send_header = [&] {
        ROCP_INFO << "Restarting the trace!";
        if(buffer_packet.header == 0) return;

        size_t hidx = wait_for_free_slot();
        send_to_consumer(header_plus_zeros.data(),
                         sizeof(header_plus_zeros),
                         ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_NONE,
                         hidx,
                         true);
    };

    send_header();

    auto finish = common::scope_destructor{[&] {
        // Wake consumers on every exit; they drain filled slots before returning.
        parameters.shared->stopping.store(true);
        for(size_t i = 0; i < num_buffers; i++)
        {
            auto lk = std::unique_lock{buffers[i].mut};
            buffers[i].cv.notify_one();
        }

        auto end_t0 = std::chrono::system_clock::now();
        ROCP_INFO << "Total trace time: " << (end_t0 - start_t0).count() * 1E-9f << " s.";
    }};

    while(worker_flag.load() == WORKER_FLAG_RUNNING)
    {
        if(do_sleep) sleep_fn();
        do_sleep = true;  // Reset value

        // PHASE 1: Poll SQTT buffer status
        if(!att_queue_submit(queue, &buffer_packet.query_status, &submit_signal)) return;
        signal_wait(submit_signal);

        if(auto status = buffer_packet.query_buffer_status())
        {
            if(status->gpu_full)
            {
                auto submit_lock = std::unique_lock{queue.submit_mutex};
                queue.submit_fn  = nullptr;

                // Leave SQTT untouched after overflow: no swap, stop, restart,
                // or later code-object markers on this queue.
                ROCP_ERROR << "GPU buffer overflow: ATT tracing disabled for agent "
                           << queue.agent_id.handle
                           << ". Discarding GPU-resident trace data and rejecting ALL further "
                              "packets on this queue, including stop/restart. Tracing will not "
                              "resume on this queue; already-copied CPU data will still be "
                              "delivered.";
                return;
            }

            ROCP_TRACE << "Sending buffer swap";
            // PHASE 2: trigger GPU buffer swap and stage the data into a CPU slot
            // The copy runs on a different engine than the AQL queue, so the packet's
            // barrier bit does not order it. The retired buffer is only complete once
            // the swap has executed.
            if(!att_queue_submit(queue, &status->packet, &submit_signal)) return;
            signal_wait(submit_signal);

            ROCP_FATAL_IF(status->size != buffer_size)
                << "GPU buffer overflow: " << status->size << " vs " << buffer_size;

            // Try to claim a free CPU slot. If none free, the consumers haven't
            // kept up and we have to stop the trace.
            size_t     slot_idx = try_claim_slot();
            const bool cpu_full = (slot_idx == num_buffers);

            int flags = ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_NONE;
            if(cpu_full)
            {
                if(!stop_trace()) return;
                flags    = ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL;
                slot_idx = wait_for_free_slot();
            }

            send_to_consumer(
                status->data, buffer_size, flags, slot_idx, false, status->read_offset);

            if(cpu_full)
            {
                iterate_trace();
                send_header();

                if(!parameters.restart_trace(parameters.control_packet)) return;
            }
            // The status_query test verifies we immediately poll again after consuming a
            // buffer, so skip the backoff when a flip just occurred.
            do_sleep = false;
        }
    }

    if(stop_trace()) iterate_trace();
}
}  // namespace thread_trace
}  // namespace rocprofiler
