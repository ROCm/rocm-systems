// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

/**
 * @file samples/ompt/client.cpp
 *
 * @brief Example rocprofiler client (tool)
 */

#include "client.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "common/call_stack.hpp"
#include "common/defines.hpp"
#include "common/filesystem.hpp"
#include "common/name_info.hpp"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <ratio>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>
namespace client
{
namespace
{
using common::call_stack_t;
using common::callback_name_info;
using common::source_location;
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;

rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
auto                          client_ctx       = rocprofiler_context_id_t{};
auto                          cb_name_info     = common::get_callback_tracing_names();
auto                          bf_name_info     = common::get_buffer_tracing_names();
auto                          client_buffer    = rocprofiler_buffer_id_t{};
auto                          client_kernels   = kernel_symbol_map_t{};
auto                          call_stack_mtx   = std::mutex{};

auto
get_call_stack_lock()
{
    return std::unique_lock<std::mutex>{call_stack_mtx};
}

void
print_call_stack(const call_stack_t& _call_stack)
{
    common::print_call_stack("openmp_target_trace.log", _call_stack);
}

void
tool_tracing_ctrl_callback(rocprofiler_callback_tracing_record_t record,
                           rocprofiler_user_data_t*,
                           void* client_data)
{
    auto* ctx = static_cast<rocprofiler_context_id_t*>(client_data);

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER &&
       record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API &&
       record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause)
    {
        ROCPROFILER_CALL(rocprofiler_stop_context(*ctx), "pausing client context");
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT &&
            record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API &&
            record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume)
    {
        ROCPROFILER_CALL(rocprofiler_start_context(*ctx), "resuming client context");
    }
}

void
tool_callback_tracing_callback(rocprofiler_callback_tracing_record_t record,
                               rocprofiler_user_data_t*              user_data,
                               void*                                 callback_data)
{
    assert(callback_data != nullptr);

    if(record.kind == ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API)
    {
        if(record.operation == ROCPROFILER_HSA_CORE_API_ID_hsa_queue_destroy)
        {
            // skip hsa_queue_destroy for now, it tries to print the queue after it is destroyed
            return;
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
            record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            // flush the buffer to ensure that any lookups for the client kernel names for the code
            // object are completed
            auto flush_status = rocprofiler_flush_buffer(client_buffer);
            if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
                ROCPROFILER_CALL(flush_status, "buffer flush");
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
            record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            client_kernels.emplace(data->kernel_id, *data);
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            client_kernels.erase(data->kernel_id);
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_OMPT)
    {
        // demonstrate the use of the ompt_data_t* fields from OMPT
        // The client has its own version of those fields as well as an interface to the
        // ompt API entry points.
        auto* data = static_cast<rocprofiler_callback_tracing_ompt_data_t*>(record.payload);

        if(record.operation == ROCPROFILER_OMPT_ID_parallel_begin)
        {
            // set the parallel_data value
            auto& args                = data->args.parallel_begin;
            args.parallel_data->value = record.correlation_id.internal;
        }
        else if(record.operation == ROCPROFILER_OMPT_ID_parallel_end)
        {
            // set the parallel_data value
            auto& args                = data->args.parallel_end;
            args.parallel_data->value = 0;
        }
        else if(record.operation == ROCPROFILER_OMPT_ID_thread_begin)
        {
            // set the thread_data value
            auto& args              = data->args.thread_begin;
            args.thread_data->value = record.thread_id;
        }
        else if(record.operation == ROCPROFILER_OMPT_ID_thread_end)
        {
            // set the thread_data value
            auto& args              = data->args.thread_end;
            args.thread_data->value = 0;
        }
        else if(record.operation == ROCPROFILER_OMPT_ID_implicit_task)
        {
            auto& args = data->args.implicit_task;
            // set the task_data value
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
                args.task_data->value = record.correlation_id.internal;
            else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
                args.task_data->value = 0;
            else
                assert(0);
        }
        else if(record.operation == ROCPROFILER_OMPT_ID_target_emi)
        {
            auto& args = data->args.target_emi;
            // set the target_data value
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
                args.target_data->value = record.correlation_id.internal;
            else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
                args.target_data->value = 0;
            else
                assert(0);
        }
        else if(record.operation == ROCPROFILER_OMPT_ID_target_data_op_emi)
        {
            auto& args = data->args.target_data_op_emi;
            // set the host_op_id value
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
                args.host_op_id->value = record.correlation_id.internal;
            else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
                args.host_op_id->value = 0;
            else
                assert(0);
        }
        else if(record.operation == ROCPROFILER_OMPT_ID_target_submit_emi)
        {
            // set the host_op_id value
            auto& args = data->args.target_submit_emi;
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
                args.host_op_id->value = record.correlation_id.internal;
            else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
                args.host_op_id->value = 0;
            else
                assert(0);
        }
    }

    auto     now = std::chrono::steady_clock::now().time_since_epoch().count();
    uint64_t dt  = 0;
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        user_data->value = now;
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
        dt = (now - user_data->value);

    const char* name = nullptr;
    rocprofiler_query_callback_tracing_kind_operation_name(
        record.kind, record.operation, &name, nullptr);

    auto info = std::stringstream{};
    info << std::left << "tid=" << record.thread_id << ", cid=" << std::setw(3)
         << record.correlation_id.internal << ", kind=" << std::setw(2) << record.kind
         << ", operation=" << std::setw(3) << record.operation << ", phase=" << record.phase
         << ", dt_nsec=" << std::setw(8) << dt << ", name=" << name;

    auto info_data_cb = [](rocprofiler_callback_tracing_kind_t,
                           int,
                           uint32_t          arg_num,
                           const void* const arg_value_addr,
                           int32_t           indirection_count,
                           const char*       arg_type,
                           const char*       arg_name,
                           const char*       arg_value_str,
                           int32_t           dereference_count,
                           void*             cb_data) -> int {
        auto& dss = *static_cast<std::stringstream*>(cb_data);
        dss << ((arg_num == 0) ? "(" : ", ");
        dss << arg_num << ": " << arg_name << "=" << arg_value_str;
        (void) arg_value_addr;
        (void) arg_type;
        (void) indirection_count;
        (void) dereference_count;
        return 0;
    };

    int32_t max_deref = 1;
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
        // not for PHASE_NONE
        max_deref = 2;
    auto info_data = std::stringstream{};
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT)
    {
        ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kind_operation_args(
                             record, info_data_cb, max_deref, static_cast<void*>(&info_data)),
                         "Failure iterating trace operation args");
    }

    auto info_data_str = info_data.str();
    if(!info_data_str.empty()) info << " " << info_data_str << ")";

    auto* call_stack_v = static_cast<call_stack_t*>(callback_data);
    auto  _lk          = get_call_stack_lock();
    call_stack_v->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});
}

void
tool_buffered_tracing_callback(rocprofiler_context_id_t      context,
                               rocprofiler_buffer_id_t       buffer_id,
                               rocprofiler_record_header_t** headers,
                               size_t                        num_headers,
                               void*                         user_data,
                               uint64_t                      drop_count)
{
    assert(drop_count == 0 && "drop count should be zero for lossless policy");

    auto* call_stack_v = static_cast<call_stack_t*>(user_data);

    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
           header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);

            auto info = std::stringstream{};

            auto dt = (record->end_timestamp - record->start_timestamp);
            info << std::left << "tid=" << record->thread_id << ", cid=" << std::setw(3)
                 << record->correlation_id.internal << ", kind=" << std::setw(2) << record->kind
                 << ", operation=" << std::setw(3) << record->operation << ", phase= "
                 << ", dt_nsec=" << std::setw(8) << dt
                 << ", agent_id=" << record->dispatch_info.agent_id.handle
                 << ", queue_id=" << record->dispatch_info.queue_id.handle
                 << ", kernel_id=" << record->dispatch_info.kernel_id
                 << ", kernel=" << client_kernels.at(record->dispatch_info.kernel_id).kernel_name
                 << ", start=" << record->start_timestamp << ", stop=" << record->end_timestamp
                 << ", private_segment_size=" << record->dispatch_info.private_segment_size
                 << ", group_segment_size=" << record->dispatch_info.group_segment_size
                 << ", workgroup_size=(" << record->dispatch_info.workgroup_size.x << ","
                 << record->dispatch_info.workgroup_size.y << ","
                 << record->dispatch_info.workgroup_size.z << "), grid_size=("
                 << record->dispatch_info.grid_size.x << "," << record->dispatch_info.grid_size.y
                 << "," << record->dispatch_info.grid_size.z << ")";

            if(record->start_timestamp > record->end_timestamp)
                throw std::runtime_error("kernel dispatch: start > end");

            auto _lk = get_call_stack_lock();
            call_stack_v->emplace_back(
                source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload);

            auto info = std::stringstream{};

            auto dt = (record->end_timestamp - record->start_timestamp);
            info << std::left << "tid=" << record->thread_id << ", cid=" << std::setw(3)
                 << record->correlation_id.internal << ", kind=" << std::setw(2) << record->kind
                 << ", operation=" << std::setw(3) << record->operation << ", phase= "
                 << ", dt_nsec=" << std::setw(8) << dt
                 << ", src_agent_id=" << record->src_agent_id.handle
                 << ", dst_agent_id=" << record->dst_agent_id.handle
                 << ", direction=" << record->operation << ", start=" << record->start_timestamp
                 << ", stop=" << record->end_timestamp
                 << ", name=" << bf_name_info.at(record->kind, record->operation);

            if(record->start_timestamp > record->end_timestamp)
                throw std::runtime_error("memory copy: start > end");

            auto _lk = get_call_stack_lock();
            call_stack_v->emplace_back(
                source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_scratch_memory_record_t*>(header->payload);

            auto info = std::stringstream{};

            auto _elapsed =
                std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
                    std::chrono::nanoseconds{record->end_timestamp - record->start_timestamp})
                    .count();

            auto dt = (record->end_timestamp - record->start_timestamp);
            info << std::left << "tid=" << record->thread_id << ", cid=" << std::setw(3)
                 << record->correlation_id.internal << ", kind=" << std::setw(2) << record->kind
                 << ", operation=" << std::setw(3) << record->operation << ", phase= "
                 << ", dt_nsec=" << std::setw(8) << dt << ", agent_id=" << record->agent_id.handle
                 << ", queue_id=" << record->queue_id.handle << ", thread_id=" << record->thread_id
                 << ", elapsed=" << std::setprecision(3) << std::fixed << _elapsed
                 << " usec, flags=" << record->flags
                 << ", name=" << bf_name_info.at(record->kind, record->operation);

            auto _lk = get_call_stack_lock();
            call_stack_v->emplace_back(
                source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});
        }
        else
        {
            auto _msg = std::stringstream{};
            _msg << "unexpected rocprofiler_record_header_t category + kind: (" << header->category
                 << " + " << header->kind << ")";
            throw std::runtime_error{_msg.str()};
        }
    }

    (void) context;
    (void) buffer_id;
}

void
tool_control_init(rocprofiler_context_id_t& primary_ctx)
{
    // Create a specialized (throw-away) context for handling ROCTx profiler pause and resume.
    // A separate context is used because if the context that is associated with roctxProfilerPause
    // disabled that same context, a call to roctxProfilerResume would be ignored because the
    // context that enables the callback for that API call is disabled.
    auto cntrl_ctx = rocprofiler_context_id_t{};
    ROCPROFILER_CALL(rocprofiler_create_context(&cntrl_ctx), "control context creation failed");

    // enable callback marker tracing with only the pause/resume operations
    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         cntrl_ctx,
                         ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                         nullptr,
                         0,
                         tool_tracing_ctrl_callback,
                         &primary_ctx),
                     "callback tracing service failed to configure");

    // start the context so that it is always active
    ROCPROFILER_CALL(rocprofiler_start_context(cntrl_ctx), "start of control context");
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    assert(tool_data != nullptr);

    auto* call_stack_v = static_cast<call_stack_t*>(tool_data);

    call_stack_v->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    for(const auto& itr : cb_name_info)
    {
        auto name_idx = std::stringstream{};
        name_idx << " [" << std::setw(3) << itr.value << "]";
        call_stack_v->emplace_back(
            source_location{"rocprofiler_callback_tracing_kind_names          " + name_idx.str(),
                            __FILE__,
                            __LINE__,
                            std::string{itr.name}});

        for(auto [didx, ditr] : itr.items())
        {
            auto operation_idx = std::stringstream{};
            operation_idx << " [" << std::setw(3) << didx << "]";
            call_stack_v->emplace_back(source_location{
                "rocprofiler_callback_tracing_kind_operation_names" + operation_idx.str(),
                __FILE__,
                __LINE__,
                std::string{"- "} + std::string{*ditr}});
        }
    }

    for(const auto& itr : bf_name_info)
    {
        auto name_idx = std::stringstream{};
        name_idx << " [" << std::setw(3) << itr.value << "]";
        call_stack_v->emplace_back(
            source_location{"rocprofiler_buffer_tracing_kind_names          " + name_idx.str(),
                            __FILE__,
                            __LINE__,
                            std::string{itr.name}});

        for(auto [didx, ditr] : itr.items())
        {
            auto operation_idx = std::stringstream{};
            operation_idx << " [" << std::setw(3) << didx << "]";
            call_stack_v->emplace_back(source_location{
                "rocprofiler_buffer_tracing_kind_operation_names" + operation_idx.str(),
                __FILE__,
                __LINE__,
                std::string{"- "} + std::string{*ditr}});
        }
    }

    client_fini_func = fini_func;

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation failed");

    // enable the control
    tool_control_init(client_ctx);

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       tool_callback_tracing_callback,
                                                       tool_data),
        "callback tracing service failed to configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_OMPT,
                                                       nullptr,
                                                       0,
                                                       tool_callback_tracing_callback,
                                                       tool_data),
        "callback tracing service failed to configure")

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                                                       nullptr,
                                                       0,
                                                       tool_callback_tracing_callback,
                                                       tool_data),
        "callback tracing service failed to configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API,
                                                       nullptr,
                                                       0,
                                                       tool_callback_tracing_callback,
                                                       tool_data),
        "callback tracing service failed to configure");

    constexpr auto buffer_size_bytes      = 4096;
    constexpr auto buffer_watermark_bytes = buffer_size_bytes - (buffer_size_bytes / 8);

    ROCPROFILER_CALL(rocprofiler_create_buffer(client_ctx,
                                               buffer_size_bytes,
                                               buffer_watermark_bytes,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_buffered_tracing_callback,
                                               tool_data,
                                               &client_buffer),
                     "buffer creation");

    for(auto itr : {ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                    ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                    ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY})
    {
        ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
                             client_ctx, itr, nullptr, 0, client_buffer),
                         "buffer tracing service configure");
    }

    int valid_ctx = 0;
    ROCPROFILER_CALL(rocprofiler_context_is_valid(client_ctx, &valid_ctx),
                     "failure checking context validity");
    if(valid_ctx == 0)
    {
        // notify rocprofiler that initialization failed
        // and all the contexts, buffers, etc. created
        // should be ignored
        return -1;
    }
    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start failed");

    // no errors
    return 0;
}

void
tool_fini(void* tool_data)
{
    assert(tool_data != nullptr);

    auto* _call_stack = static_cast<call_stack_t*>(tool_data);
    auto  _lk         = get_call_stack_lock();
    _call_stack->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    print_call_stack(*_call_stack);

    delete _call_stack;
}
}  // namespace

void
setup()
{}

void
shutdown()
{
    if(client_id) client_fini_func(*client_id);
}

void
start()
{
    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start failed");
}

void
stop()
{
    int status = 0;
    ROCPROFILER_CALL(rocprofiler_is_initialized(&status), "failed to retrieve init status");
    if(status != 0)
    {
        ROCPROFILER_CALL(rocprofiler_stop_context(client_ctx), "rocprofiler context stop failed");
    }
}
}  // namespace client

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // set the client name
    id->name = "ExampleTool";

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

    // demonstration of alternative way to get the version info
    {
        auto version_info = std::array<uint32_t, 3>{};
        ROCPROFILER_CALL(
            rocprofiler_get_version(&version_info.at(0), &version_info.at(1), &version_info.at(2)),
            "failed to get version info");

        if(std::array<uint32_t, 3>{major, minor, patch} != version_info)
        {
            throw std::runtime_error{"version info mismatch"};
        }
    }

    // data passed around all the callbacks
    auto* client_tool_data = new std::vector<client::source_location>{};

    // add first entry
    client_tool_data->emplace_back(
        client::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &client::tool_init,
                                            &client::tool_fini,
                                            static_cast<void*>(client_tool_data)};

    // return pointer to configure data
    return &cfg;
}
