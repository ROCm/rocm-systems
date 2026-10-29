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
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <thread>

namespace rocprofiler
{
namespace thread_trace
{
constexpr double SQTT_BANDWIDTH_DEFAULT = 60E9;  // 60GB/s, for wiggle room
// Retired buffers copied between status polls, so engines that fill meanwhile are swapped promptly.
constexpr size_t COPIES_PER_POLL = 4;

rocprofiler_thread_trace_timestamp_t
convert_timestamp(rocprofiler_agent_id_t agent_id, uint64_t gpu_clock)
{
    auto result      = rocprofiler_thread_trace_timestamp_t{};
    result.gpu_clock = gpu_clock;
    if(gpu_clock == 0) return result;

    // KFD queues do not carry an HSA agent; resolve it from the SDK agent ID.
    auto hsa_agent = agent::get_hsa_agent(agent_id);
    if(!hsa_agent) return result;

    auto* ext_table = hsa::get_amd_ext_table();
    if(ext_table == nullptr ||
       ext_table->hsa_amd_profiling_convert_tick_to_system_domain_fn == nullptr)
        return result;

    auto status = ext_table->hsa_amd_profiling_convert_tick_to_system_domain_fn(
        *hsa_agent, gpu_clock, &result.system_clock);
    if(status != HSA_STATUS_SUCCESS)
    {
        result.system_clock = 0;
        ROCP_CI_LOG(ERROR) << "Failed to convert thread trace GPU clock: " << status;
    }
    return result;
}

namespace
{
struct trace_callback_data_t
{
    void*    data{};
    uint64_t size{};
    uint32_t se_id{};
};

std::vector<trace_callback_data_t>
iterate_data(hsa::SQTTBufferingPackets& packets, hsa_status_t& status)
{
    auto thread_trace_callback = [](uint32_t se_id, void* buffer, uint64_t size, void* userdata) {
        auto& data = *static_cast<std::vector<trace_callback_data_t>*>(userdata);
        data.emplace_back(trace_callback_data_t{buffer, size, se_id});
        return HSA_STATUS_SUCCESS;
    };
    std::vector<trace_callback_data_t> data{};
    status = packets.iterate_data(thread_trace_callback, &data);
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

        shader_data.start_timestamp = slot.start_timestamp;
        shader_data.end_timestamp   = slot.end_timestamp;

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

    auto buffer_packets = std::move(parameters.buffer_packets);
    for(auto& packet : buffer_packets)
        CHECK_NOTNULL(packet);

    auto& submit_signal = *parameters.submit_signal;

    auto   start_t0 = std::chrono::system_clock::now();
    bool   do_sleep{false};
    size_t num_shader_engines = 0;
    for(const auto& packet : buffer_packets)
        num_shader_engines =
            std::max(num_shader_engines, static_cast<size_t>(packet->shader_engine_id + 1));
    ROCP_FATAL_IF(num_shader_engines > 64)
        << "Triple buffering supports at most 64 shader engines, got " << num_shader_engines;

    std::vector<uint64_t>                             next_chunk_indices(num_shader_engines);
    std::vector<rocprofiler_thread_trace_timestamp_t> current_timestamps(num_shader_engines);

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

    auto send_to_consumer = [&](void*                                src,
                                size_t                               size,
                                int                                  flags,
                                size_t                               slot_idx,
                                int                                  shader_engine_id,
                                bool                                 isHeader    = false,
                                uint64_t                             read_offset = 0,
                                rocprofiler_thread_trace_timestamp_t start_ts    = {},
                                rocprofiler_thread_trace_timestamp_t end_ts      = {}) {
        auto t0 = std::chrono::system_clock::now();

        auto& buffer           = buffers[slot_idx];
        buffer.flags           = flags;
        buffer.size            = size;
        buffer.se_id           = shader_engine_id;
        buffer.chunk_index     = next_chunk_indices[shader_engine_id]++;
        buffer.read_offset     = read_offset;
        buffer.start_timestamp = start_ts;
        buffer.end_timestamp   = end_ts;

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
        hsa_status_t status = HSA_STATUS_SUCCESS;
        auto         data   = iterate_data(*buffer_packets.at(0), status);
        for(auto& packet : buffer_packets)
        {
            // A filled buffer awaiting its swap at stop arrives before the active buffer.
            auto samples = std::vector<trace_callback_data_t>{};
            for(const auto& sample : data)
                if(sample.se_id == static_cast<uint32_t>(packet->shader_engine_id))
                    samples.emplace_back(sample);
            if(samples.empty()) samples.emplace_back();

            auto& current_ts = current_timestamps.at(packet->shader_engine_id);
            auto  clock      = packet->get_gpu_clock();
            if(current_ts.gpu_clock == 0)
                current_ts = convert_timestamp(queue.agent_id, clock.start);
            auto end_ts = convert_timestamp(queue.agent_id, clock.latest);
            packet->reset_current_buffer();
            for(size_t i = 0; i < samples.size(); i++)
            {
                auto&      wptr  = samples[i];
                const bool last  = i + 1 == samples.size();
                int        flags = ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_NONE;
                if(last) flags |= ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_END;
                if(status == HSA_STATUS_ERROR_OUT_OF_RESOURCES)
                    flags |= ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_GPU_BUFFER_FULL;
                if(status != HSA_STATUS_SUCCESS || (wptr.size > 0 && !wptr.data) ||
                   wptr.size > buffer_size)
                {
                    if(status == HSA_STATUS_ERROR_OUT_OF_RESOURCES)
                        ROCP_WARNING << "Discarding ATT drain payload after GPU buffer overflow";
                    else
                        ROCP_CI_LOG(ERROR) << "Discarding ATT drain payload: status " << status
                                           << ", size " << wptr.size;
                    wptr.size = 0;
                }
                ROCP_INFO << "Iterate data for " << packet->shader_engine_id
                          << " with size: " << wptr.size;
                send_to_consumer(wptr.data,
                                 wptr.size,
                                 flags,
                                 wait_for_free_slot(),
                                 packet->shader_engine_id,
                                 false,
                                 0,
                                 current_ts,
                                 end_ts);
            }
        }
    };

    std::array<uint64_t, 4> header_plus_zeros{};  // Used for warmup the decoder path

    auto send_header = [&] {
        ROCP_INFO << "Restarting the trace!";
        for(auto& packet : buffer_packets)
        {
            if(packet->header == 0) continue;
            header_plus_zeros.at(0) = packet->header;
            size_t hidx             = wait_for_free_slot();
            send_to_consumer(header_plus_zeros.data(),
                             sizeof(header_plus_zeros),
                             ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_NONE,
                             hidx,
                             packet->shader_engine_id,
                             true);
        }
    };

    send_header();
    std::vector<hsa::sqtt_buffer_status_t> buffers_requiring_swap{};
    buffers_requiring_swap.reserve(buffer_packets.size());

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

    struct retired_buffer_t
    {
        hsa::sqtt_buffer_status_t            status{};
        rocprofiler_thread_trace_timestamp_t start_ts{};
        rocprofiler_thread_trace_timestamp_t end_ts{};
        bool                                 cpu_full{false};
    };
    std::deque<retired_buffer_t> retired_buffers{};
    uint64_t                     retired_mask = 0;  // Bit per SE with a buffer in retired_buffers
    bool                         any_full     = false;

    auto se_bit = [](int shader_engine_id) { return uint64_t{1} << shader_engine_id; };

    auto find_retired = [&](int shader_engine_id) {
        return std::find_if(
            retired_buffers.begin(), retired_buffers.end(), [&](const auto& retired) {
                return retired.status.shader_engine_id == shader_engine_id;
            });
    };

    // Copies a retired buffer to a CPU slot. With allow_stop, stops the trace once if consumers
    // cannot keep up; this must only happen after the swaps of the current poll were submitted.
    auto copy_retired = [&](const std::deque<retired_buffer_t>::iterator& it, bool allow_stop) {
        size_t slot_idx = try_claim_slot();
        if(slot_idx == num_buffers)
        {
            if(allow_stop)
            {
                if(!any_full && !stop_trace()) return false;
                any_full     = true;
                it->cpu_full = true;
            }
            slot_idx = wait_for_free_slot();
        }
        int flags = ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_NONE;
        if(it->status.gpu_full) flags |= ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_GPU_BUFFER_FULL;
        if(it->cpu_full) flags |= ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL;
        send_to_consumer(it->status.data,
                         it->status.size,
                         flags,
                         slot_idx,
                         it->status.shader_engine_id,
                         false,
                         it->status.read_offset,
                         it->start_ts,
                         it->end_ts);
        retired_mask &= ~se_bit(it->status.shader_engine_id);
        retired_buffers.erase(it);
        return true;
    };

    auto copy_all_retired = [&](bool allow_stop) {
        while(!retired_buffers.empty())
            if(!copy_retired(retired_buffers.begin(), allow_stop)) return false;
        return true;
    };

    while(worker_flag.load() == WORKER_FLAG_RUNNING)
    {
        if(do_sleep) sleep_fn();
        do_sleep = true;  // Reset value

        // The status packet queries every SE and orders the initial start packet.
        if(!att_queue_submit(queue, &buffer_packets.at(0)->query_status, &submit_signal)) return;
        signal_wait(submit_signal);

        // Only consumers free slots concurrently, so this many copies before the swaps never wait.
        size_t free_slots = 0;
        for(size_t i = 0; i < num_buffers; i++)
            if(!buffers[i].filled.load()) ++free_slots;

        // Check all engines before submitting any swaps: on gfx10+ overflow disables the queue.
        bool gpu_full = false;
        bool cpu_full = false;
        for(auto& packet : buffer_packets)
        {
            auto& current_ts = current_timestamps.at(packet->shader_engine_id);
            if(current_ts.gpu_clock == 0)
                current_ts = convert_timestamp(queue.agent_id, packet->get_gpu_clock().start);

            // Swapping an engine with an uncopied retired buffer needs a slot for that copy first.
            // Without one, waiting would let the GPU overflow: leave the engine unqueried, so its
            // filled buffer stays pending for the drain, and stop once the other swaps are out.
            const bool needs_copy = (retired_mask & se_bit(packet->shader_engine_id)) != 0;
            if(needs_copy && free_slots == 0)
            {
                find_retired(packet->shader_engine_id)->cpu_full = true;
                cpu_full                                         = true;
                continue;
            }

            if(auto status = packet->query_buffer_status())
            {
                if(needs_copy) --free_slots;
                if(status->gpu_full && parameters.restart_on_gpu_full)
                {
                    ROCP_WARNING << "GPU buffer overflow on agent " << queue.agent_id.handle
                                 << " SE " << packet->shader_engine_id
                                 << ": restarting thread trace";
                    gpu_full = true;
                }
                else if(status->gpu_full)
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
                    copy_all_retired(false);
                    return;
                }

                do_sleep = false;
                ROCP_FATAL_IF(status->size > buffer_size)
                    << "GPU buffer overflow: " << status->size << " vs " << buffer_size;
                buffers_requiring_swap.emplace_back(*status);
            }
        }

        // The buffer retired by an SE's previous swap becomes its next pending buffer, so it must
        // be copied before that SE is swapped again. Swap engines without such a copy first so
        // they are not delayed behind other engines' copies.
        std::stable_partition(
            buffers_requiring_swap.begin(), buffers_requiring_swap.end(), [&](const auto& status) {
                return (retired_mask & se_bit(status.shader_engine_id)) == 0;
            });
        // Swap packets carry the barrier bit and retire in order, so only the last one needs a
        // completion signal.
        for(size_t i = 0; i < buffers_requiring_swap.size(); i++)
        {
            auto& status = buffers_requiring_swap[i];
            if((retired_mask & se_bit(status.shader_engine_id)) != 0 &&
               !copy_retired(find_retired(status.shader_engine_id), false))
                return;

            const bool last = i + 1 == buffers_requiring_swap.size();
            ROCP_TRACE << "Sending buffer swap for SE " << status.shader_engine_id;
            if(!att_queue_submit(queue, &status.packet, last ? &submit_signal : nullptr)) return;
        }
        // The copy engine is not ordered by the AQL barrier bit; wait for the swaps.
        if(!buffers_requiring_swap.empty()) signal_wait(submit_signal);
        for(auto& status : buffers_requiring_swap)
        {
            auto packet =
                std::find_if(buffer_packets.begin(), buffer_packets.end(), [&](const auto& p) {
                    return p->shader_engine_id == status.shader_engine_id;
                });
            auto& current_ts = current_timestamps.at(status.shader_engine_id);
            auto  swap_ts    = convert_timestamp(queue.agent_id, (*packet)->get_gpu_clock().latest);
            retired_buffers.push_back({status, current_ts, swap_ts});
            retired_mask |= se_bit(status.shader_engine_id);
            current_ts = swap_ts;
        }
        buffers_requiring_swap.clear();

        if(gpu_full || cpu_full)
        {
            if(!stop_trace()) return;
            any_full = true;
        }

        // Poll again after a few copies so engines that fill meanwhile are swapped promptly.
        for(size_t i = 0; i < COPIES_PER_POLL && !retired_buffers.empty() && !any_full; i++)
            if(!copy_retired(retired_buffers.begin(), true)) return;
        if(!retired_buffers.empty()) do_sleep = false;

        if(any_full)
        {
            if(!copy_all_retired(true)) return;
            iterate_trace();
            send_header();

            if(!parameters.restart_trace(parameters.control_packet)) return;
            any_full = false;
            // Read each restarted engine's clock after the next query or final stop.
            std::fill(current_timestamps.begin(),
                      current_timestamps.end(),
                      rocprofiler_thread_trace_timestamp_t{});
        }
    }

    if(!copy_all_retired(false)) return;
    if(stop_trace()) iterate_trace();
}
}  // namespace thread_trace
}  // namespace rocprofiler
