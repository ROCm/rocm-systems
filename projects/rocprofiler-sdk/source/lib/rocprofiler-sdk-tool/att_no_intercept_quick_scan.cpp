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

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iostream>

namespace rocprofiler
{
namespace tool
{
namespace att_no_intercept
{
namespace
{
struct trace_range_t
{
    bool                    active               = false;
    rocprofiler_kernel_id_t kernel_id            = 0;
    uint64_t                offset_begin         = 0;
    uint64_t                offset_end           = 0;
    uint64_t                remaining_dispatches = 0;
    uint8_t                 me_id                = 0;
    uint8_t                 pipe_id              = 0;
    uint64_t                flush_count          = 0;
};

struct scan_context
{
    agent_state_t*              state                = nullptr;
    kernel_target_filter_t      kernel_target_filter = nullptr;
    trace_range_t               trace                = {};
    std::vector<trace_range_t>* completed            = {};
};

std::optional<rocprofiler_kernel_id_t>
resolve_kernel_id(agent_state_t& state, const rocprofiler_thread_trace_decoder_pc_t& entry_point)
{
    auto shared_lock = std::shared_lock{state.mutex};

    auto find_exact = [&state](entry_key key) -> std::optional<rocprofiler_kernel_id_t> {
        if(auto itr = state.kernels_by_entry.find(key); itr != state.kernels_by_entry.end())
            return itr->second;
        return std::nullopt;
    };
    auto find_range = [&](entry_key key) -> std::optional<rocprofiler_kernel_id_t> {
        auto ranges_itr = state.kernel_ranges_by_code_object.find(key.code_object_id);
        if(ranges_itr == state.kernel_ranges_by_code_object.end()) return std::nullopt;

        for(const auto& range : ranges_itr->second)
        {
            if(key.address >= range.begin && key.address < range.end)
            {
                auto range_copy = range;
                shared_lock.unlock();
                auto unique_lock = std::unique_lock{state.mutex};

                state.kernels_by_entry[key] = range_copy.kernel_id;
                return range_copy.kernel_id;
            }
        }

        return std::nullopt;
    };

    auto key = entry_key{entry_point.code_object_id, entry_point.address};
    if(auto kernel_id = find_exact(key)) return kernel_id;

    if(auto kernel_id = find_range(key)) return kernel_id;

    return std::nullopt;
}

void
handle_dispatch(scan_context& context, const rocprofiler_thread_trace_decoder_dispatch_t& dispatch)
{
    auto& state = *context.state;

    ROCP_TRACE << "Received dispatch: " << dispatch.entry_point.code_object_id << " / "
              << dispatch.entry_point.address << " at me/pipe " << int(dispatch.me_id) << "/"
              << int(dispatch.pipe_id);

    auto kernel_id = resolve_kernel_id(state, dispatch.entry_point);
    if(!kernel_id)
    {
        ROCP_TRACE << "Unknown dispatch found.";
    }
    else if((context.kernel_target_filter == nullptr || context.kernel_target_filter(*kernel_id)))
    {
        auto& trace = context.trace;
        if(!trace.active)
        {
            trace              = trace_range_t{};
            trace.active       = true;
            trace.offset_begin = dispatch.byte_offset;

            ROCP_INFO << "Dispatch cut at : " << *kernel_id << " byte " << dispatch.byte_offset;
        }

        trace.kernel_id = *kernel_id;
        trace.remaining_dispatches = std::max<uint64_t>(state.consecutive_kernels, 1);
        trace.flush_count = 0;
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
handle_event(scan_context& context, const rocprofiler_thread_trace_decoder_event_t& event)
{
    // TODO: Handle terminal trace ranges that only have one trailing cut-end event
    // before chunk end. The current two-event rule avoids slicing off the end event but can
    // skip the last/only dispatch in a chunk.
    //
    auto& trace = context.trace;
    if(!trace.active) return;

    if(trace.remaining_dispatches > 0 ||
       event.me_id != trace.me_id || event.pipe_id != trace.pipe_id)
    {
        return;
    }

    bool is_dispatch_end = event.type == ROCPROF_TRACE_DECODER_EVENT_DISPATCH_END;
    bool is_flush = event.type == ROCPROF_TRACE_DECODER_EVENT_CS_PARTIAL_FLUSH || event.type == ROCPROF_TRACE_DECODER_EVENT_BOTTOM_OF_PIPE_TS;

    if(!is_dispatch_end && !is_flush && trace.flush_count == 0) return;

    if(is_flush || trace.flush_count > 0) trace.flush_count++;

    if(trace.flush_count == 2)
    {
        ROCP_INFO << "Cut trace at byte: " << event.byte_offset;
        trace.offset_end = event.byte_offset;
        context.completed->emplace_back(std::move(trace));
        trace = {};
    }
}

rocprofiler_thread_trace_decoder_status_t
quick_scan_callback(rocprofiler_thread_trace_decoder_record_type_t record_type_id,
                    void*                                          records,
                    uint64_t                                       num_records,
                    void*                                          userdata)
{
    auto* context = static_cast<scan_context*>(userdata);
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

rocprofiler_thread_trace_decoder_status_t
build_standalone(rocprof_trace_decoder_handle_t handle,
                 uint64_t                       chunk_index,
                 const void*                    data,
                 uint64_t                       data_size,
                 uint64_t                       offset_begin,
                 uint64_t                       offset_end,
                 std::vector<uint8_t>&          output)
{
    if(offset_end <= offset_begin) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;

    output.assign((offset_end - offset_begin) + 4096, 0);
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

    if(status == ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS) output.resize(output_size);

    return status;
}

void
forward_cut(agent_state_t&                           state,
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
                           uint64_t     code_object_id,
                           const void*  code_object_data,
                           uint64_t     code_object_size)
{
    if(code_object_data == nullptr || code_object_size == 0) return;

    auto sizes_by_name  = std::unordered_map<std::string, uint64_t>{};
    auto sizes_by_vaddr = std::unordered_map<uint64_t, uint64_t>{};

    rocprof_trace_decoder::codeobj::elf_inline::for_each_func_symbol(
        static_cast<const char*>(code_object_data),
        code_object_size,
        [&](std::string name, uint64_t vaddr, uint64_t size) {
            if(size == 0) return;
            if(!name.empty()) sizes_by_name.emplace(std::move(name), size);
            if(vaddr != 0) sizes_by_vaddr.emplace(vaddr, size);
        });

    if(sizes_by_name.empty() && sizes_by_vaddr.empty()) return;

    auto  lock                 = std::unique_lock{state.mutex};
    auto& code_object          = state.code_objects[code_object_id];
    code_object.code_object_id = code_object_id;
    for(auto& [name, size] : sizes_by_name)
        code_object.symbol_sizes_by_name[std::move(name)] = size;
    for(auto [vaddr, size] : sizes_by_vaddr)
        code_object.symbol_sizes_by_vaddr[vaddr] = size;
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
        ROCP_ERROR << fmt::format("failed to create ATT no-intercept decoder for agent {}: {}",
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
backend_code_object_load(agent_state_t&                                                state,
                         const rocprofiler_callback_tracing_code_object_load_data_t& data)
{
    if(data.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
    {
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

void
backend_shader_data(agent_state_t&                           state,
                    rocprofiler_thread_trace_shader_data_t shader_data,
                    shader_data_forwarder_t                shader_data_forwarder,
                    kernel_target_filter_t                 kernel_target_filter)
{
    if((shader_data.flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_GPU_BUFFER_FULL) != 0 ||
       (shader_data.flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL) != 0)
    {
        ROCP_WARNING << "SQTT Buffer full at chunk " << shader_data.chunk_index;
    }

    // TODO: Reconstruct cross-buffer read_offset data instead of only scanning the contiguous
    // unread suffix. This keeps the scanner from consuming already-read bytes for now.
    auto* scan_data = static_cast<const uint8_t*>(shader_data.data) + shader_data.read_offset;
    auto  scan_size = shader_data.data_size - shader_data.read_offset;

    thread_local std::vector<trace_range_t> completed{};
    completed.clear();

    auto context                 = scan_context{};
    context.state                = &state;
    context.kernel_target_filter = kernel_target_filter;
    context.completed            = &completed;

    auto status = rocprof_trace_decoder_quick_scan(state.decoder,
                                                   shader_data.chunk_index,
                                                   scan_data,
                                                   scan_size,
                                                   quick_scan_callback,
                                                   &context);

    if(status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
    {
        ROCP_WARNING << fmt::format("ATT no-intercept quick scan failed for agent {} chunk {}: {}",
                                 state.id.handle,
                                 shader_data.chunk_index,
                                 rocprof_trace_decoder_get_status_string(status));
        return;
    }

    ROCP_WARNING_IF(context.trace.active) << "Context not ended";

    for (auto& trace : completed)
    {
        auto standalone = std::vector<uint8_t>{};
        status          = build_standalone(state.decoder,
                                shader_data.chunk_index,
                                scan_data,
                                scan_size,
                                trace.offset_begin,
                                trace.offset_end,
                                standalone);

        if(status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
        {
            ROCP_WARNING << fmt::format("ATT no-intercept standalone cut failed for agent {} "
                                    "chunk {} range {}..{}: {}",
                                    state.id.handle,
                                    shader_data.chunk_index,
                                    trace.offset_begin,
                                    trace.offset_end,
                                    rocprof_trace_decoder_get_status_string(status));
            return;
        }

        forward_cut(state, shader_data, standalone, shader_data_forwarder);
    }
}
}  // namespace att_no_intercept
}  // namespace tool
}  // namespace rocprofiler
