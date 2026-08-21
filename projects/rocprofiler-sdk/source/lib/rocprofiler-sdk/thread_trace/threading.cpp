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
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"

#include <fmt/format.h>

#include <atomic>
#include <cstdint>
#include <thread>

namespace rocprofiler
{
namespace thread_trace
{
constexpr double SQTT_BANDWIDTH_DEFAULT = 60E9;  // 60GB/s, for wiggle room

namespace
{
using buffer_slot_t = triple_buffer_shared_data_t::buffer_slot_t;

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
    ROCP_TRACE << fmt::format("Executing async copy from {} to {}", src, dst);

    thread_local auto signal = scoped_signal_t{};

    auto copy_fn = CHECK_NOTNULL(hsa::get_amd_ext_table())->hsa_amd_memory_async_copy_fn;

    // Workaround for ROCM-25606
    if(dependency) signal_wait(*dependency);

    signal_reset(signal.sig);
    auto status = copy_fn(dst, dst_agent, src, src_agent, size, 0, nullptr, signal.sig);
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to copy: " << status;
    signal_wait(signal.sig);
}

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
// 2. Copy: When buffer is full, perform async GPU->CPU memory copy
// 3. Notify: Signal the consumer that owns the slot via its per-slot cv
//
// The loop uses adaptive polling with backoff based on estimated bandwidth to minimize
// CPU overhead while ensuring timely buffer flips before GPU overflow.
void
producer_loop(
    triple_buffer_producer_data_t parameters)  // NOLINT(performance-unnecessary-value-param)
{
    CHECK_NOTNULL(parameters.copy_data_fn);
    CHECK_NOTNULL(parameters.start_pkt_signal);

    auto& queue       = *CHECK_NOTNULL(parameters.shared->queue);
    auto& worker_flag = *CHECK_NOTNULL(parameters.producer_running);

    const size_t buffer_size = queue.buffer_size;
    auto&        buffers     = parameters.shared->buffers;
    const size_t num_buffers = parameters.shared->num_buffers;
    const auto   sqtt_bandwidth =
        std::max(1.0, common::get_env("ROCPROFILER_SQTT_BANDWIDTH", SQTT_BANDWIDTH_DEFAULT));
    const auto interval_microseconds = static_cast<size_t>(1E6 * buffer_size / sqtt_bandwidth);

    auto& buffer_packet = *CHECK_NOTNULL(parameters.buffer_packet);

    auto submit_signal = scoped_signal_t{};

    auto     start_t0 = std::chrono::system_clock::now();
    bool     do_sleep{false};
    uint64_t next_chunk_index = 0;
    int64_t  shader_engine_id = parameters.shader_engine_id;

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

        auto&       buffer      = buffers[slot_idx];
        const auto& near_cpu_v  = queue.near_cpu;
        const auto& hsa_agent_v = queue.hsa_agent;
        buffer.flags            = flags;
        buffer.size             = size;
        buffer.se_id            = shader_engine_id;
        buffer.chunk_index      = next_chunk_index++;
        buffer.read_offset      = read_offset;

        if(!isHeader)
            parameters.copy_data_fn(
                buffer.memory, src, near_cpu_v, hsa_agent_v, size, &submit_signal.sig);
        else
            std::memcpy(buffer.memory, src, size);

        auto copy_time = (std::chrono::system_clock::now() - t0).count() * 1E-9f;
        ROCP_TRACE << "Copy: " << copy_time << " s. BW: " << size / copy_time;

        // Publish: producer's writes above happen-before the consumer's
        // observation of `filled` via the slot mutex's release/acquire.
        {
            auto lk = std::unique_lock{buffer.mut};
        }
        buffer.filled.store(true);
        buffer.cv.notify_one();
    };

    auto submit_wait_timeout = [&]() {
        if(signal_wait(submit_signal.sig, 1 << 28)) return true;

        worker_flag.store(WORKER_FLAG_ERROR);
        ROCP_ERROR << "Submit timeout!";
        return false;
    };

    auto stop_trace = [&]() {
        ROCP_INFO << "Stopping the trace";
        if(!submit_wait_timeout()) return false;
        att_queue_submit(
            queue, &parameters.control_packet->after_krn_pkt.at(0), &submit_signal.sig);
        return submit_wait_timeout();
    };

    // Drain remaining ATT data after a stop; waits for a free slot to land it in.
    auto iterate_trace = [&]() {
        size_t idx  = wait_for_free_slot();
        auto   wptr = iterate_data(parameters.control_packet->GetHandle());
        buffer_packet.reset_current_buffer();
        ROCP_INFO << "Iterate data with size: " << wptr.size;
        send_to_consumer(wptr.data, wptr.size, ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_END, idx);
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

    // Wait until ATT start packets have been executed
    signal_wait(*parameters.start_pkt_signal);

    while(worker_flag.load() == WORKER_FLAG_RUNNING)
    {
        if(do_sleep) sleep_fn();
        do_sleep = true;  // Reset value

        // PHASE 1: Poll SQTT buffer status
        att_queue_submit(queue, &buffer_packet.query_status, &submit_signal.sig);
        if(!submit_wait_timeout()) break;

        if(auto status = buffer_packet.query_buffer_status())
        {
            ROCP_TRACE << "Sending buffer swap";
            // PHASE 2: trigger GPU buffer swap and stage the data into a CPU slot
            att_queue_submit(queue, &status->packet, &submit_signal.sig);
            ROCP_FATAL_IF(status->size != buffer_size)
                << "GPU buffer overflow: " << status->size << " vs " << buffer_size;

            // Try to claim a free CPU slot. If none free, the consumers haven't
            // kept up and we have to stop the trace.
            size_t     slot_idx = try_claim_slot();
            const bool cpu_full = (slot_idx == num_buffers);

            if(cpu_full || status->gpu_full) stop_trace();

            int flags = ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_NONE;
            if(cpu_full) flags |= ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL;
            if(status->gpu_full)
                flags |= ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_GPU_BUFFER_FULL;

            // If CPU was full we must wait for a slot before we can publish.
            if(cpu_full) slot_idx = wait_for_free_slot();
            send_to_consumer(
                status->data, buffer_size, flags, slot_idx, false, status->read_offset);

            if(cpu_full || status->gpu_full)
            {
                iterate_trace();
                send_header();

                att_queue_submit_and_wait_last(queue, parameters.control_packet->before_krn_pkt);
            }
            // The status_query test verifies we immediately poll again after consuming a
            // buffer, so skip the backoff when a flip just occurred.
            do_sleep = false;
            submit_wait_timeout();
        }
    }

    if(worker_flag.load() != WORKER_FLAG_ERROR && stop_trace()) iterate_trace();

    // Signal all consumers to exit. Setting `stopping` under each slot's
    // mutex ensures consumers about to enter cv.wait() observe it; the
    // subsequent notify_one wakes any consumer already parked.
    parameters.shared->stopping.store(true);
    for(size_t i = 0; i < num_buffers; i++)
    {
        auto lk = std::unique_lock{buffers[i].mut};
        buffers[i].cv.notify_one();
    }

    auto end_t0 = std::chrono::system_clock::now();
    ROCP_INFO << "Total trace time: " << (end_t0 - start_t0).count() * 1E-9f << " s.";
}
}  // namespace thread_trace
}  // namespace rocprofiler
