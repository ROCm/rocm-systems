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

namespace rocprofiler
{
namespace tool
{
namespace att_no_intercept
{
namespace
{
struct backend_agent
{
    rocprof_trace_decoder_handle_t decoder = {};
};

struct cut_candidate
{
    bool                    ready        = false;
    rocprofiler_kernel_id_t kernel_id    = 0;
    uint64_t                offset_begin = 0;
    uint64_t                offset_end   = 0;
    uint8_t                 me_id        = 0;
    uint8_t                 pipe_id      = 0;
    uint64_t                flush_count  = 0;
};

struct scan_context
{
    agent_state*               state                = nullptr;
    kernel_target_filter_t     kernel_target_filter = nullptr;
    std::vector<cut_candidate> candidates           = {};
};

std::atomic<uint64_t>&
next_capture_id()
{
    static auto* _v = new std::atomic<uint64_t>{1};
    return *_v;
}

backend_agent*
get_backend(agent_state& state)
{
    return static_cast<backend_agent*>(state.backend);
}

void
warn_limited(std::atomic<uint64_t>& counter, std::string_view message)
{
    auto count = counter.fetch_add(1, std::memory_order_relaxed) + 1;
    if(count <= 16)
    {
        ROCP_WARNING << message;
    }
    else if(count == 17)
    {
        ROCP_WARNING << "suppressing additional ATT no-intercept warnings of this kind";
    }
}

std::optional<rocprofiler_kernel_id_t>
resolve_kernel_id(agent_state& state, const rocprofiler_thread_trace_decoder_pc_t& entry_point)
{
    auto lock       = std::lock_guard<std::mutex>{state.mutex};
    auto find_exact = [&state](entry_key key) -> std::optional<rocprofiler_kernel_id_t> {
        if(auto itr = state.kernels_by_entry.find(key); itr != state.kernels_by_entry.end())
            return itr->second;
        return std::nullopt;
    };
    auto find_range = [&state](entry_key key) -> std::optional<rocprofiler_kernel_id_t> {
        auto ranges_itr = state.kernel_ranges_by_code_object.find(key.code_object_id);
        if(ranges_itr == state.kernel_ranges_by_code_object.end()) return std::nullopt;

        for(const auto& range : ranges_itr->second)
        {
            if(key.address >= range.begin && key.address < range.end)
            {
                state.kernels_by_entry[key] = range.kernel_id;
                return range.kernel_id;
            }
        }

        return std::nullopt;
    };

    auto key = entry_key{entry_point.code_object_id, entry_point.address};
    if(auto kernel_id = find_exact(key)) return kernel_id;

    if(auto kernel_id = find_range(key)) return kernel_id;

    return std::nullopt;
}

bool
is_cut_end_event(const rocprofiler_thread_trace_decoder_event_t& event)
{
    return event.type == ROCPROF_TRACE_DECODER_EVENT_CS_PARTIAL_FLUSH ||
           event.type == ROCPROF_TRACE_DECODER_EVENT_BOTTOM_OF_PIPE_TS ||
           event.type == ROCPROF_TRACE_DECODER_EVENT_DISPATCH_END;
}

void
handle_dispatch(scan_context& context, const rocprofiler_thread_trace_decoder_dispatch_t& dispatch)
{
    auto& state = *context.state;
    state.stats.dispatches_seen.fetch_add(1, std::memory_order_relaxed);

    auto kernel_id = resolve_kernel_id(state, dispatch.entry_point);
    if(!kernel_id)
    {
        state.stats.unknown_dispatches.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    if(context.kernel_target_filter != nullptr && !context.kernel_target_filter(*kernel_id)) return;

    state.stats.target_dispatches.fetch_add(1, std::memory_order_relaxed);

    auto candidate         = cut_candidate{};
    candidate.kernel_id    = *kernel_id;
    candidate.offset_begin = dispatch.byte_offset;
    candidate.me_id        = dispatch.me_id;
    candidate.pipe_id      = dispatch.pipe_id;
    context.candidates.emplace_back(candidate);
}

void
handle_event(scan_context& context, const rocprofiler_thread_trace_decoder_event_t& event)
{
    // TODO: Handle terminal single-dispatch cuts that only have one trailing cut-end event
    // before chunk end. The current two-event rule avoids slicing off the end event but can
    // skip the last/only dispatch in a chunk.
    //
    // TODO: Track at most one active target candidate per ME/pipe, or explicitly bound
    // candidates by the next dispatch on that ME/pipe. Multiple target dispatches on the
    // same ME/pipe can currently consume the same trailing events.
    for(auto& candidate : context.candidates)
    {
        if(candidate.ready || event.byte_offset <= candidate.offset_begin ||
           event.me_id != candidate.me_id || event.pipe_id != candidate.pipe_id)
        {
            continue;
        }

        if(!is_cut_end_event(event) && candidate.flush_count == 0) continue;

        if(++candidate.flush_count == 2)
        {
            candidate.offset_end = event.byte_offset;
            candidate.ready      = true;
        }
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
forward_cut(agent_state&                           state,
            rocprofiler_thread_trace_shader_data_t original,
            uint64_t                               capture_id,
            std::vector<uint8_t>&                  standalone,
            shader_data_forwarder_t                shader_data_forwarder)
{
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
    userdata.value = capture_id;
    shader_data_forwarder(shader_data, userdata);
}

void
record_code_object_symbols(agent_state& state,
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

    auto  lock                 = std::lock_guard<std::mutex>{state.mutex};
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

void*
backend_create(agent_state& state)
{
    auto* backend = new backend_agent{};
    auto  status  = rocprof_trace_decoder_create_handle(&backend->decoder);
    if(status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
    {
        ROCP_ERROR << fmt::format("failed to create ATT no-intercept decoder for agent {}: {}",
                                  state.id.handle,
                                  rocprof_trace_decoder_get_status_string(status));
        delete backend;
        return nullptr;
    }

    status = rocprof_trace_decoder_quick_scan(backend->decoder, 0, nullptr, 0, nullptr, nullptr);
    if(status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
    {
        auto msg = fmt::format("ATT no-intercept quick-scan support is unavailable for "
                               "agent {}: {}",
                               state.id.handle,
                               rocprof_trace_decoder_get_status_string(status));
        (void) rocprof_trace_decoder_destroy_handle(backend->decoder);
        delete backend;
        ROCP_FATAL << msg;
        return nullptr;
    }

    return backend;
}

void
backend_destroy(agent_state& state)
{
    auto* backend = get_backend(state);
    if(backend == nullptr) return;

    auto status = rocprof_trace_decoder_destroy_handle(backend->decoder);
    if(status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
    {
        ROCP_WARNING << fmt::format("failed to destroy ATT no-intercept decoder for agent {}: {}",
                                    state.id.handle,
                                    rocprof_trace_decoder_get_status_string(status));
    }

    delete backend;
    state.backend = nullptr;
}

void
backend_code_object_load(agent_state&                                                state,
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
backend_shader_data(agent_state&                           state,
                    rocprofiler_thread_trace_shader_data_t shader_data,
                    shader_data_forwarder_t                shader_data_forwarder,
                    kernel_target_filter_t                 kernel_target_filter)
{
    auto* backend = get_backend(state);
    if(backend == nullptr) return;

    if((shader_data.flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_GPU_BUFFER_FULL) != 0 ||
       (shader_data.flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL) != 0)
    {
        state.stats.buffer_full_events.fetch_add(1, std::memory_order_relaxed);
    }

    if(shader_data.read_offset >= shader_data.data_size)
    {
        state.stats.quick_scan_failures.fetch_add(1, std::memory_order_relaxed);
        ROCP_CI_LOG(ERROR) << fmt::format("ATT no-intercept skipped agent {} chunk {}: "
                                          "read_offset {} is outside data_size {}",
                                          state.id.handle,
                                          shader_data.chunk_index,
                                          shader_data.read_offset,
                                          shader_data.data_size);
        return;
    }

    if(shader_data.data == nullptr)
    {
        state.stats.quick_scan_failures.fetch_add(1, std::memory_order_relaxed);
        ROCP_CI_LOG(ERROR) << fmt::format("ATT no-intercept skipped agent {} chunk {}: "
                                          "shader data pointer is null for data_size {}",
                                          state.id.handle,
                                          shader_data.chunk_index,
                                          shader_data.data_size);
        return;
    }

    // TODO: Reconstruct cross-buffer read_offset data instead of only scanning the contiguous
    // unread suffix. This keeps the scanner from consuming already-read bytes for now.
    auto* scan_data = static_cast<const uint8_t*>(shader_data.data) + shader_data.read_offset;
    auto  scan_size = shader_data.data_size - shader_data.read_offset;

    auto context                 = scan_context{};
    context.state                = &state;
    context.kernel_target_filter = kernel_target_filter;

    auto status = rocprof_trace_decoder_quick_scan(backend->decoder,
                                                   shader_data.chunk_index,
                                                   scan_data,
                                                   scan_size,
                                                   quick_scan_callback,
                                                   &context);

    state.stats.chunks_scanned.fetch_add(1, std::memory_order_relaxed);
    state.stats.bytes_scanned.fetch_add(scan_size, std::memory_order_relaxed);

    if(status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
    {
        warn_limited(state.stats.quick_scan_failures,
                     fmt::format("ATT no-intercept quick scan failed for agent {} chunk {}: {}",
                                 state.id.handle,
                                 shader_data.chunk_index,
                                 rocprof_trace_decoder_get_status_string(status)));
        return;
    }

    for(const auto& candidate : context.candidates)
    {
        if(!candidate.ready)
        {
            warn_limited(state.stats.cross_chunk_skips,
                         fmt::format("ATT no-intercept skipped dispatch for kernel {} on agent {}: "
                                     "single-dispatch cut crosses chunk boundary at chunk {}",
                                     candidate.kernel_id,
                                     state.id.handle,
                                     shader_data.chunk_index));
            continue;
        }

        auto standalone = std::vector<uint8_t>{};
        status          = build_standalone(backend->decoder,
                                  shader_data.chunk_index,
                                  scan_data,
                                  scan_size,
                                  candidate.offset_begin,
                                  candidate.offset_end,
                                  standalone);

        if(status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
        {
            warn_limited(state.stats.cut_build_failures,
                         fmt::format("ATT no-intercept standalone cut failed for agent {} "
                                     "chunk {} range {}..{}: {}",
                                     state.id.handle,
                                     shader_data.chunk_index,
                                     candidate.offset_begin,
                                     candidate.offset_end,
                                     rocprof_trace_decoder_get_status_string(status)));
            continue;
        }

        auto capture_id = next_capture_id().fetch_add(1, std::memory_order_relaxed);
        forward_cut(state, shader_data, capture_id, standalone, shader_data_forwarder);
        state.stats.cuts_built.fetch_add(1, std::memory_order_relaxed);
    }
}
}  // namespace att_no_intercept
}  // namespace tool
}  // namespace rocprofiler
