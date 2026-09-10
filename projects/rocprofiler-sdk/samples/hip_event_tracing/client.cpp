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
//
// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

/**
 * @file samples/hip_event_tracing/client.cpp
 *
 * @brief Example rocprofiler client (tool) for HIP event barrier tracing
 */

#include "client.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "common/call_stack.hpp"
#include "common/defines.hpp"
#include "common/filesystem.hpp"
#include "common/name_info.hpp"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace client
{
namespace
{
using common::buffer_name_info;
using common::call_stack_t;
using common::callback_name_info;
using common::source_location;

using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;

rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t      client_ctx       = {0};
rocprofiler_buffer_id_t       client_buffer    = {};
callback_name_info            client_cb_names  = {};
buffer_name_info              client_buf_names = {};
kernel_symbol_map_t           client_kernels   = {};

// Every line of output, whether it originates from a tracing callback, from the buffer
// callback thread, or from the application via client::narrate(), is serialized through
// this mutex, which guards client_call_stack as well. The HIP event enqueue callbacks fire
// from whichever thread submits to the HSA queue, the barrier-completion callback fires
// from the HSA async signal handler thread, the buffer callback fires from the dedicated
// rocprofiler callback thread, and narrate() fires from the application thread.
std::mutex    output_mutex;
call_stack_t* client_call_stack = nullptr;

// Counts HIP_EVENT_WAIT barrier completions. Incremented on the HSA async signal handler
// thread and read by the application thread, hence the atomic.
std::atomic<uint64_t> wait_completion_count{0};

// Output layout. Every entry is at most two lines: a timestamped explainer line saying in
// prose what happened, then one line carrying the record's raw fields, indented to the
// right edge of the timestamp column. The field line uses fixed-width columns so that
// records of the same type line up vertically.
constexpr size_t prefix_width = 18;  // "[+" + 11-wide value + " us] "
constexpr size_t tag_width    = 9;

// column widths for the field line
constexpr int w_op        = 16;  // longest operation name is HIP_EVENT_RECORD
constexpr int w_phase     = 5;
constexpr int w_cid       = 4;
constexpr int w_queue     = 1;
constexpr int w_kind      = 2;
constexpr int w_kernel_id = 3;
constexpr int w_time      = 16;
constexpr int w_elapsed   = 12;
constexpr int w_handle    = 12;  // 48-bit user-space pointer

// the field line is offset two columns past the timestamp so that it is visually
// subordinate to the explainer line above it
constexpr size_t field_indent = prefix_width + 2;

// right-align an already-formatted value in a fixed-width column
std::string
num_str(std::string s, int width)
{
    if(static_cast<int>(s.size()) < width) s.insert(0, width - s.size(), ' ');
    return s;
}

// right-align a numeric value in a fixed-width column
std::string
num(uint64_t v, int width)
{
    return num_str(std::to_string(v), width);
}

// left-align a text value in a fixed-width column
std::string
txt(std::string_view v, int width)
{
    auto s = std::string{v};
    if(static_cast<int>(s.size()) < width) s.append(width - s.size(), ' ');
    return s;
}

uint64_t
timestamp_ns()
{
    auto ts = rocprofiler_timestamp_t{};
    ROCPROFILER_CALL(rocprofiler_get_timestamp(&ts), "reading rocprofiler timestamp");
    return ts;
}

// Reference point for the "+N us" prefix. Deliberately the rocprofiler clock rather than
// std::chrono so that the prefix on a live line and the start_timestamp/end_timestamp
// printed inside a record are directly comparable.
uint64_t
program_start_ns()
{
    static uint64_t _v = timestamp_ns();
    return _v;
}

std::string
elapsed_prefix()
{
    auto ss = std::stringstream{};
    ss << "[+" << std::setw(prefix_width - 7) << std::fixed << std::setprecision(3)
       << (static_cast<double>(timestamp_ns() - program_start_ns()) / 1000.0) << " us] ";
    return ss.str();
}

// Emit one entry: the explainer line, then, if the entry has any, the field line. Both
// are assembled and written under a single lock so the two halves of an entry can never be
// separated by another thread's output, and the timestamp is read inside that lock so the
// printed order and the printed times agree. Written live to stdout for the interleaved
// view, and into the call stack so that tool_fini can write the conventional sample log.
// The enqueue callbacks run on the AQL submit path, where this I/O can add latency to GPU
// work submission; the sample writes directly anyway, for simplicity.
void
emit(const char*        func,
     const char*        file,
     uint32_t           lineno,
     std::string_view   tag,
     const std::string& explainer,
     const std::string& fields = {})
{
    auto lk    = std::unique_lock<std::mutex>{output_mutex};
    auto entry = std::stringstream{};

    entry << elapsed_prefix() << std::left << std::setw(tag_width) << tag << " :: " << explainer;

    if(!fields.empty()) entry << '\n' << std::string(field_indent, ' ') << fields;

    auto text = entry.str();

    std::cout << text << '\n' << std::flush;

    if(client_call_stack != nullptr)
        client_call_stack->emplace_back(source_location{func, file, lineno, std::move(text)});
}

// Abbreviates the source-location arguments to emit(), which the log file records for each
// entry.
#define EMIT_ARGS __FUNCTION__, __FILE__, __LINE__

template <typename Tp>
std::string
as_hex(Tp _v, size_t _width = 16)
{
    uintptr_t _vp = 0;
    if constexpr(std::is_pointer<Tp>::value)
        _vp = reinterpret_cast<uintptr_t>(_v);
    else
        _vp = _v;

    auto _ss = std::stringstream{};
    _ss.fill('0');
    _ss << "0x" << std::hex << std::setw(_width) << _vp;
    return _ss.str();
}

std::string
phase_name(rocprofiler_callback_phase_t phase)
{
    switch(phase)
    {
        // ROCPROFILER_CALLBACK_PHASE_LOAD/UNLOAD alias ENTER/EXIT
        case ROCPROFILER_CALLBACK_PHASE_ENTER: return "ENTER";
        case ROCPROFILER_CALLBACK_PHASE_EXIT: return "EXIT";
        case ROCPROFILER_CALLBACK_PHASE_NONE: return "NONE";
        default: break;
    }
    return "UNKNOWN";
}

// The three phases of a HIP event barrier. See the documentation on
// ::rocprofiler_hip_event_operation_t: ENTER/EXIT bracket the moment the barrier packet is
// enqueued (no GPU timestamps yet), and PHASE_NONE fires once the barrier has completed on
// the GPU (timestamps populated).
std::string
phase_meaning(rocprofiler_callback_phase_t phase)
{
    switch(phase)
    {
        case ROCPROFILER_CALLBACK_PHASE_ENTER: return "about to be enqueued";
        case ROCPROFILER_CALLBACK_PHASE_EXIT: return "enqueued";
        case ROCPROFILER_CALLBACK_PHASE_NONE: return "completed on the GPU";
        default: break;
    }
    return "in an unexpected phase";
}

// The trailing start/end/elapsed columns, shared by every record type so they line up
// across callback, buffer and dispatch entries alike.
std::string
timestamps_field(rocprofiler_timestamp_t start, rocprofiler_timestamp_t end)
{
    auto ss = std::stringstream{};

    // Timestamps are only populated in the barrier-completion record; on the enqueue
    // callbacks they are zero because the barrier has not run yet.
    if(start == 0 && end == 0)
    {
        // right-aligned like the populated case, so the columns line up either way
        ss << "start=" << num_str("<pending>", w_time) << " end=" << num_str("<pending>", w_time)
           << " elapsed=" << num_str("-", w_elapsed);
    }
    else
    {
        auto elapsed = std::stringstream{};
        elapsed << std::fixed << std::setprecision(3) << (static_cast<double>(end - start) / 1000.0)
                << "us";

        ss << "start=" << num(start, w_time) << " end=" << num(end, w_time)
           << " elapsed=" << num_str(elapsed.str(), w_elapsed);
    }

    return ss.str();
}

void
tool_code_object_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*              user_data,
                          void*                                 callback_data)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            auto lk = std::unique_lock<std::mutex>{output_mutex};
            client_kernels.emplace(data->kernel_id, *data);
        }
    }

    (void) user_data;
    (void) callback_data;
}

// The HIP event barrier callbacks. Fires three times per barrier that is actually
// scheduled: ENTER and EXIT around the enqueue, then NONE at completion. A
// hipStreamWaitEvent that does not need a barrier, because the event has already
// completed, was never recorded, or names the same stream, produces no callback at all.
void
tool_hip_event_callback(rocprofiler_callback_tracing_record_t record,
                        rocprofiler_user_data_t*              user_data,
                        void*                                 callback_data)
{
    assert(record.payload != nullptr);

    const auto* data = static_cast<rocprofiler_callback_tracing_hip_event_data_t*>(record.payload);

    auto op_name = client_cb_names.at(record.kind, record.operation);

    auto explainer = std::stringstream{};
    explainer << op_name << " barrier " << phase_meaning(record.phase) << " on queue "
              << data->queue_id.handle;

    // For a wait barrier the source queue is where the event was recorded; only worth
    // calling out when it differs from the queue the barrier is on.
    if(record.operation == ROCPROFILER_HIP_EVENT_WAIT &&
       data->source_queue_id.handle != data->queue_id.handle)
    {
        explainer << ", waiting on an event recorded in queue " << data->source_queue_id.handle;
    }

    auto fields = std::stringstream{};
    fields << "op=" << txt(op_name, w_op) << " phase=" << txt(phase_name(record.phase), w_phase)
           << " cid=" << num(record.correlation_id.internal, w_cid)
           << " queue=" << num(data->queue_id.handle, w_queue)
           << " src_queue=" << num(data->source_queue_id.handle, w_queue)
           << " event=" << as_hex(data->hip_event_handle, w_handle);

    // the enqueue phases carry no timestamps at all, so the columns are omitted rather than
    // printed empty
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_NONE)
        fields << " " << timestamps_field(data->start_timestamp, data->end_timestamp);

    emit(EMIT_ARGS, "EVENT CB", explainer.str(), fields.str());

    // published after the record is emitted so that an application thread waking on this count
    // cannot narrate past the line that reports the completion it is waiting for
    if(record.operation == ROCPROFILER_HIP_EVENT_WAIT &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_NONE)
        wait_completion_count.fetch_add(1, std::memory_order_release);

    (void) user_data;
    (void) callback_data;
}

// Kernel dispatch completions. Taken through the callback service rather than the buffered
// one so that they are delivered inline, at the moment the dispatch completes, and therefore
// interleave with the HIP event barrier callbacks in true chronological order. The buffered
// service would instead hold them until the buffer is flushed, which would report every
// dispatch after the barriers it actually preceded.
void
tool_kernel_dispatch_callback(rocprofiler_callback_tracing_record_t record,
                              rocprofiler_user_data_t*              user_data,
                              void*                                 callback_data)
{
    assert(record.payload != nullptr);

    const auto* data =
        static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(record.payload);

    assert(data->start_timestamp <= data->end_timestamp && "kernel dispatch: start > end");

    auto kernel_id   = data->dispatch_info.kernel_id;
    auto kernel_name = std::string{"??"};
    {
        auto lk  = std::unique_lock<std::mutex>{output_mutex};
        auto itr = client_kernels.find(kernel_id);
        if(itr != client_kernels.end()) kernel_name = std::string{itr->second.kernel_name};
    }

    auto explainer = std::stringstream{};
    explainer << kernel_name << " completed on queue " << data->dispatch_info.queue_id.handle;

    // the kernel name is variable length, so it lives on the explainer line and the field
    // line carries only fixed-width columns
    auto fields = std::stringstream{};
    fields << "op=" << txt("KERNEL_DISPATCH", w_op) << " kernel_id=" << num(kernel_id, w_kernel_id)
           << " cid=" << num(record.correlation_id.internal, w_cid)
           << " queue=" << num(data->dispatch_info.queue_id.handle, w_queue) << " "
           << timestamps_field(data->start_timestamp, data->end_timestamp);

    emit(EMIT_ARGS, "DISPATCH", explainer.str(), fields.str());

    (void) user_data;
    (void) callback_data;
}

void
tool_buffer_callback(rocprofiler_context_id_t      context,
                     rocprofiler_buffer_id_t       buffer_id,
                     rocprofiler_record_header_t** headers,
                     size_t                        num_headers,
                     void*                         user_data,
                     uint64_t                      drop_count)
{
    assert(drop_count == 0 && "drop count should be zero for lossless policy");

    if(num_headers == 0)
        throw std::runtime_error{
            "rocprofiler invoked a buffer callback with no headers. this should never happen"};
    else if(headers == nullptr)
        throw std::runtime_error{"rocprofiler invoked a buffer callback with a null pointer to the "
                                 "array of headers. this should never happen"};

    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
           header->kind == ROCPROFILER_BUFFER_TRACING_HIP_EVENT)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_hip_event_record_t*>(header->payload);

            if(record->start_timestamp > record->end_timestamp)
                throw std::runtime_error("hip event: start > end");

            auto op_name = client_buf_names.at(record->kind, record->operation);

            auto explainer = std::stringstream{};
            explainer << op_name << " barrier on queue " << record->queue_id.handle
                      << " delivered via the buffered service (context=" << context.handle
                      << ", buffer=" << buffer_id.handle << ")";

            auto fields = std::stringstream{};
            fields << "op=" << txt(op_name, w_op) << " kind=" << num(record->kind, w_kind)
                   << " op_id=" << num(record->operation, w_kind)
                   << " cid=" << num(record->correlation_id.internal, w_cid)
                   << " queue=" << num(record->queue_id.handle, w_queue)
                   << " src_queue=" << num(record->source_queue_id.handle, w_queue)
                   << " event=" << as_hex(record->hip_event_handle, w_handle) << " "
                   << timestamps_field(record->start_timestamp, record->end_timestamp);

            emit(EMIT_ARGS, "EVENT BUF", explainer.str(), fields.str());
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        {
            // The same dispatches already arrived inline through the callback service. They
            // are collected here as well so that the buffered service can be seen carrying
            // more than one record kind: a single flush of this buffer delivers the barrier
            // records and the dispatch records together, in the order they were appended
            // rather than in timestamp order.
            auto* record =
                static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);

            if(record->start_timestamp > record->end_timestamp)
                throw std::runtime_error("kernel dispatch: start > end");

            auto kernel_id   = record->dispatch_info.kernel_id;
            auto kernel_name = std::string{"??"};
            {
                auto lk  = std::unique_lock<std::mutex>{output_mutex};
                auto itr = client_kernels.find(kernel_id);
                if(itr != client_kernels.end()) kernel_name = std::string{itr->second.kernel_name};
            }

            auto explainer = std::stringstream{};
            explainer << kernel_name << " dispatch on queue "
                      << record->dispatch_info.queue_id.handle
                      << " delivered via the buffered service (context=" << context.handle
                      << ", buffer=" << buffer_id.handle << ")";

            // the operation name is KERNEL_DISPATCH_COMPLETE, too wide for the op column, so
            // this matches the label the callback path prints and lets op_id disambiguate
            auto fields = std::stringstream{};
            fields << "op=" << txt("KERNEL_DISPATCH", w_op) << " kind=" << num(record->kind, w_kind)
                   << " op_id=" << num(record->operation, w_kind)
                   << " cid=" << num(record->correlation_id.internal, w_cid)
                   << " queue=" << num(record->dispatch_info.queue_id.handle, w_queue)
                   << " kernel_id=" << num(kernel_id, w_kernel_id) << " "
                   << timestamps_field(record->start_timestamp, record->end_timestamp);

            emit(EMIT_ARGS, "DISP BUF", explainer.str(), fields.str());
        }
        else
        {
            auto _msg = std::stringstream{};
            _msg << "unexpected rocprofiler_record_header_t category + kind: (" << header->category
                 << " + " << header->kind << ")";
            throw std::runtime_error{_msg.str()};
        }
    }

    (void) user_data;
}

void
thread_precreate(rocprofiler_runtime_library_t lib, void* tool_data)
{
    emit(EMIT_ARGS,
         "NOTIFY",
         std::string{"internal thread about to be created by rocprofiler (lib="} +
             std::to_string(lib) + ")");
    (void) tool_data;
}

void
thread_postcreate(rocprofiler_runtime_library_t lib, void* tool_data)
{
    emit(EMIT_ARGS,
         "NOTIFY",
         std::string{"internal thread was created by rocprofiler (lib="} + std::to_string(lib) +
             ")");
    (void) tool_data;
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    assert(tool_data != nullptr);

    // establish the reference point for the "+N us" prefixes
    program_start_ns();

    {
        auto lk           = std::unique_lock<std::mutex>{output_mutex};
        client_call_stack = static_cast<call_stack_t*>(tool_data);
    }

    client_cb_names  = common::get_callback_tracing_names();
    client_buf_names = common::get_buffer_tracing_names();

    emit(EMIT_ARGS,
         "SETUP",
         std::string{"tracing kind '"} +
             std::string{client_cb_names.at(ROCPROFILER_CALLBACK_TRACING_HIP_EVENT)} +
             "' with operations " +
             std::string{client_cb_names.at(ROCPROFILER_CALLBACK_TRACING_HIP_EVENT,
                                            ROCPROFILER_HIP_EVENT_RECORD)} +
             " and " +
             std::string{client_cb_names.at(ROCPROFILER_CALLBACK_TRACING_HIP_EVENT,
                                            ROCPROFILER_HIP_EVENT_WAIT)});

    client_fini_func = fini_func;

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation");

    auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       code_object_ops.data(),
                                                       code_object_ops.size(),
                                                       tool_code_object_callback,
                                                       nullptr),
        "code object tracing service configure");

    // nullptr/0 subscribes to every operation in the domain, which here is record and wait
    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_HIP_EVENT,
                                                       nullptr,
                                                       0,
                                                       tool_hip_event_callback,
                                                       nullptr),
        "hip event callback tracing service configure");

    // Kernel dispatch completions, so that the barriers can be read against the kernels they
    // order. Restricted to COMPLETE: the ENQUEUE operation would add a callback pair around
    // every launch, which is noise here.
    auto kernel_dispatch_ops =
        std::vector<rocprofiler_tracing_operation_t>{ROCPROFILER_KERNEL_DISPATCH_COMPLETE};

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                       kernel_dispatch_ops.data(),
                                                       kernel_dispatch_ops.size(),
                                                       tool_kernel_dispatch_callback,
                                                       nullptr),
        "kernel dispatch callback tracing service configure");

    constexpr auto buffer_size_bytes      = 4096;
    constexpr auto buffer_watermark_bytes = buffer_size_bytes - (buffer_size_bytes / 8);

    ROCPROFILER_CALL(rocprofiler_create_buffer(client_ctx,
                                               buffer_size_bytes,
                                               buffer_watermark_bytes,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_buffer_callback,
                                               tool_data,
                                               &client_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_HIP_EVENT, nullptr, 0, client_buffer),
        "hip event buffer tracing service configure");

    // Kernel dispatch is traced through both services. The callback above reports each
    // dispatch inline; this reports the same dispatches through the buffer, into the same
    // buffer as the barrier records so that one flush delivers both kinds together.
    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, client_buffer),
        "kernel dispatch buffer tracing service configure");

    auto client_thread = rocprofiler_callback_thread_t{};
    ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                     "creating callback thread");

    ROCPROFILER_CALL(rocprofiler_assign_callback_thread(client_buffer, client_thread),
                     "assignment of thread for buffer");

    int valid_ctx = 0;
    ROCPROFILER_CALL(rocprofiler_context_is_valid(client_ctx, &valid_ctx),
                     "context validity check");
    if(valid_ctx == 0)
    {
        // notify rocprofiler that initialization failed
        // and all the contexts, buffers, etc. created
        // should be ignored
        return -1;
    }

    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start");

    // no errors
    return 0;
}

void
tool_fini(void* tool_data)
{
    assert(tool_data != nullptr);

    emit(EMIT_ARGS, "SHUTDOWN", "finalizing");

    auto _call_stack = call_stack_t{};
    {
        // detach the call stack from the emit path before printing it: the buffer callback
        // thread may still be alive
        auto lk           = std::unique_lock<std::mutex>{output_mutex};
        _call_stack       = std::move(*static_cast<call_stack_t*>(tool_data));
        client_call_stack = nullptr;
    }

    common::print_call_stack("hip_event_trace.log", _call_stack);

    delete static_cast<call_stack_t*>(tool_data);
}
}  // namespace

void
setup()
{
    if(int status = 0;
       rocprofiler_is_initialized(&status) == ROCPROFILER_STATUS_SUCCESS && status == 0)
    {
        ROCPROFILER_CALL(rocprofiler_force_configure(&rocprofiler_configure),
                         "force configuration");
    }
}

void
shutdown()
{
    if(client_id)
    {
        auto flush_status = rocprofiler_flush_buffer(client_buffer);
        // the buffer is already being flushed elsewhere; not an error here
        if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
            ROCPROFILER_CALL(flush_status, "buffer flush");
        client_fini_func(*client_id);
    }
}

void
start()
{
    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "context start");
}

void
stop()
{
    ROCPROFILER_CALL(rocprofiler_stop_context(client_ctx), "context stop");
}

void
flush()
{
    auto flush_status = rocprofiler_flush_buffer(client_buffer);
    // a concurrent flush from the callback thread is not an error
    if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
        ROCPROFILER_CALL(flush_status, "buffer flush");
}

uint64_t
wait_barriers_completed()
{
    return wait_completion_count.load(std::memory_order_acquire);
}

void
narrate(const char* msg)
{
    emit(EMIT_ARGS, "APP", msg);
}

void
banner(const char* msg)
{
    // deliberately taller than a record entry: this is the separator between the major
    // phases of the run
    const auto rule  = std::string(prefix_width, ' ') + std::string(72, '=');
    auto       lk    = std::unique_lock<std::mutex>{output_mutex};
    auto       entry = std::stringstream{};

    entry << rule << '\n'
          << elapsed_prefix() << std::left << std::setw(tag_width) << "APP"
          << " :: " << msg << '\n'
          << rule;

    auto text = entry.str();

    std::cout << '\n' << text << '\n' << std::flush;

    if(client_call_stack != nullptr)
        client_call_stack->emplace_back(
            source_location{__FUNCTION__, __FILE__, __LINE__, std::move(text)});
}
}  // namespace client

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // set the client name
    id->name = "HipEventTracingTool";

    // store client info
    client::client_id = id;

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " (priority=" << priority << ") is using rocprofiler-sdk v" << major << "."
         << minor << "." << patch << " (" << runtime_version << ")";

    std::clog << info.str() << std::endl;

    auto* client_tool_data = new std::vector<client::source_location>{};

    ROCPROFILER_CALL(rocprofiler_at_internal_thread_create(
                         client::thread_precreate,
                         client::thread_postcreate,
                         ROCPROFILER_LIBRARY | ROCPROFILER_HSA_LIBRARY | ROCPROFILER_HIP_LIBRARY |
                             ROCPROFILER_MARKER_LIBRARY,
                         static_cast<void*>(client_tool_data)),
                     "registration for thread creation notifications");

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &client::tool_init,
                                            &client::tool_fini,
                                            static_cast<void*>(client_tool_data)};

    // return pointer to configure data
    return &cfg;
}
