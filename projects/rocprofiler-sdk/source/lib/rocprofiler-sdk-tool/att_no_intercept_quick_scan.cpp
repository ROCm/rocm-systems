// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "att_no_intercept_impl.hpp"

#include "lib/common/logging.hpp"

#include <rocprof_trace_decoder/rocprof_trace_decoder.h>
#include <rocprof_trace_decoder/trace_decoder_types.h>
#include <rocprof_trace_decoder/cxx/disassembly.hpp>

#include <fmt/core.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace tool
{
namespace att_no_intercept
{
namespace
{
void
wait_for_prior_chunks(agent_state_t& state, uint64_t chunk_index)
{
    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while(state.chunk_completed < chunk_index && !state.chunk_failed)
    {
        if(std::chrono::steady_clock::now() >= timeout)
        {
            state.chunk_failed = true;
            state.data_cv.notify_all();
            ROCP_CI_LOG(ERROR) << "Timed out after 5 seconds waiting for ATT chunk " << chunk_index;
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}

struct scan_context_t
{
    agent_state_t*                    state               = nullptr;
    const std::unordered_set<size_t>* kernel_filter_range = nullptr;
    trace_range_t                     trace               = {};
    std::vector<trace_range_t>*       completed           = {};

    uint64_t chunk_index          = 0;
    uint64_t first_valid_offset   = 0;
    int      remaining_dispatches = 0;
    int      first_flush_count    = 2;
};

std::shared_ptr<std::atomic<size_t>>
resolve_kernel_entry(agent_state_t& state, const rocprofiler_thread_trace_decoder_pc_t& entry_point)
{
    auto key         = entry_key_t{entry_point.code_object_id, entry_point.address};
    auto shared_lock = std::shared_lock{state.mutex};

    if(auto itr = state.kernel_iterations_by_entry.find(key);
       itr != state.kernel_iterations_by_entry.end())
        return itr->second;

    auto ranges_itr = state.kernel_ranges_by_code_object.find(key.code_object_id);
    if(ranges_itr == state.kernel_ranges_by_code_object.end()) return nullptr;

    for(const auto& range : ranges_itr->second)
    {
        if(key.address < range.begin || key.address >= range.end) continue;

        auto targeted = range.targeted;
        shared_lock.unlock();
        auto unique_lock = std::unique_lock{state.mutex};

        if(auto itr = state.kernel_iterations_by_entry.find(key);
           itr != state.kernel_iterations_by_entry.end())
            return itr->second;

        auto [itr, _] = state.kernel_iterations_by_entry.emplace(
            key, targeted ? std::make_shared<std::atomic<size_t>>(0) : nullptr);
        return itr->second;
    }

    return nullptr;
}

bool
is_targeted_iteration(std::atomic<size_t>&              iteration_count,
                      const std::unordered_set<size_t>& kernel_filter_range)
{
    const auto iteration = iteration_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if(kernel_filter_range.empty()) return iteration == 1;
    return kernel_filter_range.count(iteration) != 0;
}

void
handle_dispatch(scan_context_t&                                    context,
                const rocprofiler_thread_trace_decoder_dispatch_t& dispatch)
{
    thread_local uint64_t max_iteration_count = [&]() {
        uint64_t maxv = 0;
        for(auto value : *context.kernel_filter_range)
            maxv = std::max(maxv, value);
        return maxv;
    }();

    auto& state = *context.state;

    // Latch first valid dispatch, taking consecutive kernels into consideration
    if(--context.remaining_dispatches == 0) context.first_flush_count = 2;

    ROCP_TRACE << "Received dispatch: " << dispatch.entry_point.code_object_id << " / "
               << dispatch.entry_point.address << " at me/pipe " << int(dispatch.me_id) << "/"
               << int(dispatch.pipe_id);

    auto iteration_count = resolve_kernel_entry(state, dispatch.entry_point);
    if(iteration_count && *iteration_count <= max_iteration_count)
    {
        // Block so we dont prematurely increment the iteration count
        wait_for_prior_chunks(state, context.chunk_index);

        if(is_targeted_iteration(*iteration_count, *context.kernel_filter_range))
        {
            auto& trace = context.trace;
            if(!trace.active)
            {
                trace              = trace_range_t{};
                trace.active       = true;
                trace.offset_begin = dispatch.byte_offset;

                ROCP_INFO << "Dispatch cut at byte " << dispatch.byte_offset;
            }

            trace.remaining_dispatches = std::max<uint64_t>(state.consecutive_kernels, 1);
            trace.flush_count          = 0;
        }
    }

    auto& trace = context.trace;
    if(trace.active && trace.remaining_dispatches > 0)
    {
        trace.me_id   = dispatch.me_id;
        trace.pipe_id = dispatch.pipe_id;
        --trace.remaining_dispatches;
    }
}

void
handle_event(scan_context_t& context, const rocprofiler_thread_trace_decoder_event_t& event)
{
    // Latch first valid offset
    if(--context.first_flush_count == 0) context.first_valid_offset = event.byte_offset;

    // TODO: Handle terminal trace ranges that only have one trailing cut-end event
    // before chunk end. The current two-event rule avoids slicing off the end event but can
    // skip the last/only dispatch in a chunk.
    //
    auto& trace = context.trace;
    if(!trace.active) return;

    if(trace.remaining_dispatches > 0 || event.me_id != trace.me_id ||
       event.pipe_id != trace.pipe_id)
    {
        return;
    }

    bool is_dispatch_end = event.type == ROCPROF_TRACE_DECODER_EVENT_DISPATCH_END;
    bool is_flush        = event.type == ROCPROF_TRACE_DECODER_EVENT_CS_PARTIAL_FLUSH ||
                    event.type == ROCPROF_TRACE_DECODER_EVENT_BOTTOM_OF_PIPE_TS;

    if(!is_dispatch_end && !is_flush && trace.flush_count == 0) return;

    if(is_flush || trace.flush_count > 0) trace.flush_count++;

    if(trace.flush_count == 2)
    {
        ROCP_INFO << "End cut trace at byte: " << event.byte_offset;
        trace.offset_end = event.byte_offset;
        context.completed->emplace_back(trace);
        trace = {};
    }
}

rocprofiler_thread_trace_decoder_status_t
quick_scan_callback(rocprofiler_thread_trace_decoder_record_type_t record_type_id,
                    void*                                          records,
                    uint64_t                                       num_records,
                    void*                                          userdata)
{
    auto* context = static_cast<scan_context_t*>(userdata);
    if(context == nullptr || context->state == nullptr || records == nullptr)
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    if(record_type_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DISPATCH)
    {
        auto* dispatches = static_cast<rocprofiler_thread_trace_decoder_dispatch_t*>(records);
        for(uint64_t i = 0; i < num_records; ++i)
            handle_dispatch(*context, dispatches[i]);
    }
    else if(record_type_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_EVENT)
    {
        auto* events = static_cast<rocprofiler_thread_trace_decoder_event_t*>(records);
        for(uint64_t i = 0; i < num_records; ++i)
            handle_event(*context, events[i]);
    }

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

bool
build_standalone(rocprof_trace_decoder_handle_t handle,
                 uint64_t                       chunk_index,
                 const void*                    data,
                 uint64_t                       data_size,
                 uint64_t                       offset_begin,
                 uint64_t                       offset_end,
                 std::vector<uint8_t>&          output)
{
    if(offset_end <= offset_begin) return false;

    output.resize(offset_end - offset_begin + 4096);
    auto output_size = static_cast<uint64_t>(output.size());
    auto status      = rocprof_trace_decoder_build_standalone(handle,
                                                         chunk_index,
                                                         data,
                                                         data_size,
                                                         offset_begin,
                                                         offset_end,
                                                         output.data(),
                                                         &output_size);
    if(status == ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES)
    {
        output.resize(output_size);
        output_size = static_cast<uint64_t>(output.size());
        status      = rocprof_trace_decoder_build_standalone(handle,
                                                        chunk_index,
                                                        data,
                                                        data_size,
                                                        offset_begin,
                                                        offset_end,
                                                        output.data(),
                                                        &output_size);
    }

    if(status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS) return false;

    output.resize(output_size);
    return true;
}

void
forward_cut(agent_state_t&                         state,
            rocprofiler_thread_trace_shader_data_t original,
            std::vector<uint8_t>&                  standalone,
            shader_data_forwarder_t                shader_data_forwarder)
{
    static std::atomic<uint64_t> capture_id{1};

    auto shader_data             = rocprofiler_thread_trace_shader_data_t{};
    shader_data.size             = sizeof(rocprofiler_thread_trace_shader_data_t);
    shader_data.data             = standalone.data();
    shader_data.data_size        = standalone.size();
    shader_data.shader_engine_id = original.shader_engine_id;
    shader_data.chunk_index      = 0;
    shader_data.read_offset      = 0;
    shader_data.agent            = state.id;
    shader_data.flags            = ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_NONE;

    auto userdata  = rocprofiler_user_data_t{};
    userdata.value = capture_id.fetch_add(1);
    shader_data_forwarder(shader_data, userdata);
}

void
record_code_object_symbols(agent_state_t& state,
                           uint64_t       code_object_id,
                           const void*    code_object_data,
                           uint64_t       code_object_size)
{
    if(code_object_data == nullptr || code_object_size == 0) return;

    auto  lock        = std::unique_lock{state.mutex};
    auto& code_object = state.code_objects[code_object_id];

    rocprof_trace_decoder::codeobj::elf_inline::for_each_func_symbol(
        static_cast<const char*>(code_object_data),
        code_object_size,
        [&](auto&&, uint64_t vaddr, uint64_t size) {
            if(size == 0) return;
            if(vaddr != 0) code_object.symbol_sizes_by_vaddr[vaddr] = size;
        });
}
}  // namespace

bool
backend_supported()
{
    return true;
}

void
backend_create(agent_state_t& state)
{
    auto status = rocprof_trace_decoder_create_handle(&state.decoder);

    if(status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
    {
        ROCP_FATAL << fmt::format("failed to create ATT no-intercept decoder for agent {}: {}",
                                  state.id.handle,
                                  rocprof_trace_decoder_get_status_string(status));
    }

    status = rocprof_trace_decoder_quick_scan(state.decoder, 0, nullptr, 0, nullptr, nullptr);
    if(status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
    {
        auto msg = fmt::format("ATT no-intercept quick-scan support is unavailable for "
                               "agent {}: {}",
                               state.id.handle,
                               rocprof_trace_decoder_get_status_string(status));
        ROCP_FATAL << msg;
    }
}

void
backend_destroy(agent_state_t& state)
{
    (void) rocprof_trace_decoder_destroy_handle(state.decoder);
}

void
backend_code_object_load(agent_state_t&                                              state,
                         const rocprofiler_callback_tracing_code_object_load_data_t& data)
{
    if(data.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
    {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        record_code_object_symbols(state,
                                   data.code_object_id,
                                   reinterpret_cast<const void*>(data.memory_base),
                                   data.memory_size);
        return;
    }

    if(data.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
    {
        ROCP_WARNING << fmt::format("ATT no-intercept does not parse file-backed code object {} "
                                    "for symbol sizes; only memory-backed code objects are "
                                    "supported for quick-scan kernel range matching",
                                    data.code_object_id);
    }
}

uint64_t
send_overlapping_requests(scan_context_t&             context,
                          const uint8_t*              scan_data,
                          uint64_t                    scan_size,
                          std::vector<trace_range_t>& ranges,
                          bool                        is_end)
{
    auto  chunk_index = context.chunk_index;
    auto& state       = *context.state;

    // Signal next chunk to continue from an active trace
    if(context.trace.active && !is_end)
    {
        state.pending_requests++;
        {
            auto lock = std::unique_lock{state.request_mutex};
            state.chunk_requested.insert(chunk_index);
        }
        ROCP_INFO << "Requesting post-chunk: " << chunk_index;
    }

    wait_for_prior_chunks(state, chunk_index);

    state.chunk_completed.fetch_add(1);

    if(state.pending_requests == 0 || chunk_index == 0) return 0;

    uint64_t offset    = 0;
    bool     requested = false;
    {
        auto lock = std::unique_lock{state.request_mutex};
        if(state.chunk_requested.find(chunk_index - 1) != state.chunk_requested.end())
        {
            requested = true;
            state.chunk_requested.erase(chunk_index - 1);
        }
    }
    if(requested)
    {
        state.pending_requests--;
        offset =
            context.first_valid_offset != 0u ? context.first_valid_offset : scan_size;
        for(auto& range : ranges)
            if(range.offset_begin < offset) offset = std::max(offset, range.offset_end);

        std::vector<uint8_t> ret_data{};
        ret_data.resize(offset);
        std::memcpy(ret_data.data(), scan_data, offset);

        ROCP_INFO << "Request send: " << chunk_index << " with size " << ret_data.size();

        auto lock                         = std::unique_lock{state.data_mutex};
        state.chunk_data[chunk_index - 1] = std::move(ret_data);
    }
    state.data_cv.notify_all();
    return offset;
}

std::vector<uint8_t>
fetch_overlapping_requests(scan_context_t& context, uint64_t chunk_index)
{
    auto& state      = *context.state;
    auto& chunk_data = state.chunk_data;

    ROCP_INFO << "Waiting for chunk: " << chunk_index;

    std::vector<uint8_t> data_retrieved{};
    {
        auto lock = std::unique_lock{state.data_mutex};
        state.data_cv.wait(lock, [&]() {
            return state.chunk_failed || chunk_data.find(chunk_index) != chunk_data.end();
        });

        data_retrieved = std::move(chunk_data[chunk_index]);
        chunk_data.erase(chunk_index);
    }

    ROCP_INFO << "Chunk received with size: " << data_retrieved.size();
    return data_retrieved;
}

void
backend_shader_data(agent_state_t&                         state,
                    rocprofiler_thread_trace_shader_data_t shader_data,
                    shader_data_forwarder_t                shader_data_forwarder,
                    const std::unordered_set<size_t>&      kernel_filter_range)
{
    if((shader_data.flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_GPU_BUFFER_FULL) != 0 ||
       (shader_data.flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL) != 0)
        ROCP_WARNING << "SQTT Buffer full at chunk " << shader_data.chunk_index;

    if(shader_data.read_offset >= shader_data.data_size)
    {
        ROCP_CI_LOG(WARNING) << fmt::format(
            "Ignoring ATT no-intercept shader data for agent {} chunk {}: read offset {} is not "
            "less than data size {}",
            state.id.handle,
            shader_data.chunk_index,
            shader_data.read_offset,
            shader_data.data_size);
        return;
    }

    const auto* scan_data =
        static_cast<const uint8_t*>(shader_data.data) + shader_data.read_offset;
    auto scan_size = shader_data.data_size - shader_data.read_offset;

    // Thread local to prevent repeated allocations in critical section
    thread_local std::vector<trace_range_t> completed{};
    completed.clear();

    auto context                = scan_context_t{};
    context.state               = &state;
    context.kernel_filter_range = &kernel_filter_range;
    context.completed           = &completed;
    context.chunk_index         = shader_data.chunk_index;

    {
        auto status = rocprof_trace_decoder_quick_scan(state.decoder,
                                                       shader_data.chunk_index,
                                                       scan_data,
                                                       scan_size,
                                                       quick_scan_callback,
                                                       &context);

        if(status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
        {
            ROCP_ERROR << fmt::format(
                "ATT no-intercept quick scan failed for agent {} chunk {}: {}",
                state.id.handle,
                shader_data.chunk_index,
                rocprof_trace_decoder_get_status_string(status));
            state.chunk_failed = true;
            state.data_cv.notify_all();
            return;
        }
    }

    bool     is_end = (shader_data.flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_END) != 0;
    uint64_t overlap_offset =
        send_overlapping_requests(context, scan_data, scan_size, completed, is_end);

    uint64_t avg_trace_length = 0;
    for(auto& trace : completed)
        avg_trace_length += trace.offset_end - trace.offset_begin;

    avg_trace_length /= std::max<uint64_t>(completed.size(), 1);

    bool     is_begin_valid = false;
    uint64_t offset_begin   = 0;

    // Thread local to prevent repeated allocations in critical section
    thread_local auto standalone = std::vector<uint8_t>{};

    for(auto it = completed.begin(); it != completed.end(); it++)
    {
        // Remove duplicated portions
        if(it->offset_end <= overlap_offset) continue;

        // Merge traces that are small
        auto next = std::next(it);
        if(next != completed.end() && next->offset_begin - it->offset_end < avg_trace_length)
        {
            if(!is_begin_valid)
            {
                is_begin_valid = true;
                offset_begin   = it->offset_begin;
            }
            ROCP_INFO << "Merging: " << offset_begin << " to " << next->offset_begin;
            continue;
        }

        if(!is_begin_valid) offset_begin = it->offset_begin;
        is_begin_valid = false;

        bool status = build_standalone(state.decoder,
                                       shader_data.chunk_index,
                                       scan_data,
                                       scan_size,
                                       offset_begin,
                                       it->offset_end,
                                       standalone);

        if(!status)
        {
            ROCP_ERROR << fmt::format("ATT no-intercept standalone cut failed for agent {} "
                                      "chunk {} range {}..{}",
                                      state.id.handle,
                                      shader_data.chunk_index,
                                      it->offset_begin,
                                      it->offset_end);
            state.chunk_failed = true;
            state.data_cv.notify_all();
            return;
        }

        forward_cut(state, shader_data, standalone, shader_data_forwarder);
    }

    if(!context.trace.active) return;

    ROCP_INFO << "Cutting at: " << context.trace.offset_begin << " w/ size " << scan_size;

    // Handle traces where we reached the end of chunk
    bool status = build_standalone(state.decoder,
                                   shader_data.chunk_index,
                                   scan_data,
                                   scan_size,
                                   context.trace.offset_begin,
                                   scan_size,
                                   standalone);
    if(!status)
    {
        ROCP_ERROR << fmt::format("ATT no-intercept standalone cut failed for agent {} "
                                  "chunk {} range {}..{}",
                                  state.id.handle,
                                  shader_data.chunk_index,
                                  context.trace.offset_begin,
                                  scan_size);
        state.chunk_failed = true;
        state.data_cv.notify_all();
        return;
    }

    if(is_end)
    {
        forward_cut(state, shader_data, standalone, shader_data_forwarder);
        return;
    }

    auto fetched = fetch_overlapping_requests(context, shader_data.chunk_index);
    auto size    = standalone.size();

    ROCP_INFO << "Joining: " << size << " with fetched: " << fetched.size();

    standalone.resize(size + fetched.size());
    std::memcpy(standalone.data() + size, fetched.data(), fetched.size());

    forward_cut(state, shader_data, standalone, shader_data_forwarder);
}
}  // namespace att_no_intercept
}  // namespace tool
}  // namespace rocprofiler
