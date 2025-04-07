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
//
// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

/**
 * @file samples/api_buffered_tracing/client.cpp
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

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
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
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

namespace client
{
namespace
{
struct external_corr_id_data;

using common::buffer_name_info;
using common::call_stack_t;
using common::source_location;

using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;
using external_corr_id_set_t = std::unordered_set<external_corr_id_data*>;
using retired_corr_id_set_t  = std::unordered_set<uint64_t>;

rocprofiler_client_id_t*      client_id                = nullptr;
rocprofiler_client_finalize_t client_fini_func         = nullptr;
rocprofiler_context_id_t      client_ctx               = {0};
rocprofiler_buffer_id_t       client_buffer            = {};
buffer_name_info*             client_name_info         = new buffer_name_info{};
kernel_symbol_map_t*          client_kernels           = new kernel_symbol_map_t{};
auto                          client_mutex             = std::shared_mutex{};
auto                          client_external_corr_ids = external_corr_id_set_t{};
auto                          client_retired_corr_ids  = retired_corr_id_set_t{};

void
print_call_stack(const call_stack_t& _call_stack)
{
    common::print_call_stack("external_correlation_id_request.log", _call_stack);
}

void
tool_code_object_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*              user_data,
                          void*                                 callback_data)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            // flush the buffer to ensure that any lookups for the client kernel names for the code
            // object are completed
            auto flush_status = rocprofiler_flush_buffer(client_buffer);
            if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
                ROCPROFILER_CHECK(flush_status);
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
            record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            client_kernels->emplace(data->kernel_id, *data);
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            auto flush_status = rocprofiler_flush_buffer(client_buffer);
            if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
                ROCPROFILER_CHECK(flush_status);

            client_kernels->erase(data->kernel_id);
        }
    }

    (void) user_data;
    (void) callback_data;
}

struct external_corr_id_data
{
    using request_kind_t               = rocprofiler_external_correlation_id_request_kind_t;
    static constexpr auto request_none = ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_NONE;

    rocprofiler_thread_id_t         thread_id        = 0;
    rocprofiler_context_id_t        context_id       = {.handle = 0};
    request_kind_t                  kind             = request_none;
    rocprofiler_tracing_operation_t operation        = 0;
    uint64_t                        internal_corr_id = 0;
    void*                           user_data        = nullptr;
    uint64_t                        seen_count       = 0;

    bool valid() const;

    friend std::ostream& operator<<(std::ostream& os, external_corr_id_data data)
    {
        if(!data.valid()) return os;
        auto ss = std::stringstream{};
        ss << "seen=" << data.seen_count << ", thr_id=" << data.thread_id
           << ", context_id=" << data.context_id.handle << ", kind=" << data.kind
           << ", operation=" << data.operation << ", corr_id=" << data.internal_corr_id
           << ", user_data=" << data.user_data;
        return (os << ss.str());
    }
};

bool
operator==(external_corr_id_data lhs, external_corr_id_data rhs)
{
    return std::tie(lhs.thread_id,
                    lhs.context_id.handle,
                    lhs.kind,
                    lhs.operation,
                    lhs.internal_corr_id,
                    lhs.user_data) == std::tie(rhs.thread_id,
                                               rhs.context_id.handle,
                                               rhs.kind,
                                               rhs.operation,
                                               rhs.internal_corr_id,
                                               rhs.user_data);
}

bool
operator!=(external_corr_id_data lhs, external_corr_id_data rhs)
{
    return !(lhs == rhs);
}

bool
external_corr_id_data::valid() const
{
    static constexpr auto invalid_v = external_corr_id_data{};
    return (*this != invalid_v);
}

int
set_external_correlation_id(rocprofiler_thread_id_t                            thr_id,
                            rocprofiler_context_id_t                           ctx_id,
                            rocprofiler_external_correlation_id_request_kind_t kind,
                            rocprofiler_tracing_operation_t                    op,
                            uint64_t                                           internal_corr_id,
                            rocprofiler_user_data_t*                           external_corr_id,
                            void*                                              user_data)
{
    auto* _data =
        new external_corr_id_data{thr_id, ctx_id, kind, op, internal_corr_id, user_data, 0};

    {
        static auto _mtx = std::mutex{};
        auto        _lk  = std::unique_lock{_mtx};
        client_external_corr_ids.emplace(_data);
    }

    external_corr_id->ptr = _data;

    return 0;
}

void
tool_tracing_callback(rocprofiler_context_id_t      context,
                      rocprofiler_buffer_id_t       buffer_id,
                      rocprofiler_record_header_t** headers,
                      size_t                        num_headers,
                      void*                         user_data,
                      uint64_t /*drop_count*/)
{
    static const auto ensure_internal_correlation_id_retirement_ordering = [](uint64_t _corr_id) {
        auto _lk = std::shared_lock<std::shared_mutex>{client_mutex};
        // this correlation ID should not have reported as retired yet so
        // we are demoing the expectation here
        if(client_retired_corr_ids.count(_corr_id) > 0)
        {
            auto msg = std::stringstream{};
            msg << "internal correlation id " << _corr_id << " was retired prematurely";
            throw std::runtime_error{msg.str()};
        }
    };

    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        auto kind_name = std::string{};
        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING)
        {
            const char* _name = nullptr;
            auto        _kind = static_cast<rocprofiler_buffer_tracing_kind_t>(header->kind);
            ROCPROFILER_CHECK(rocprofiler_query_buffer_tracing_kind_name(_kind, &_name, nullptr));
            if(_name)
            {
                static size_t len = 15;

                kind_name = std::string{_name};
                len       = std::max(len, kind_name.length());
                kind_name.resize(len, ' ');
                kind_name += " :: ";
            }
        }

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
           header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload);

            // this should always be empty
            auto _extern_corr_id = external_corr_id_data{};

            // demonstrate reliability of correlation ID retirement ordering
            ensure_internal_correlation_id_retirement_ordering(record->correlation_id.internal);

            auto info = std::stringstream{};
            info << "tid=" << record->thread_id << ", context=" << context.handle
                 << ", buffer_id=" << buffer_id.handle
                 << ", corr_id=" << record->correlation_id.internal << ", kind=" << record->kind
                 << ", operation=" << record->operation
                 << ", name=" << (*client_name_info)[record->kind][record->operation]
                 << ", extern_corr_id={" << _extern_corr_id << "}";

            static_cast<call_stack_t*>(user_data)->emplace_back(
                source_location{__FUNCTION__, __FILE__, __LINE__, kind_name + info.str()});
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);

            // demonstrate reliability of correlation ID retirement ordering
            ensure_internal_correlation_id_retirement_ordering(record->correlation_id.internal);

            auto _extern_corr_id = external_corr_id_data{};
            if(record->correlation_id.external.ptr)
            {
                auto* _extcid =
                    static_cast<external_corr_id_data*>(record->correlation_id.external.ptr);
                _extcid->seen_count++;
                _extern_corr_id = *_extcid;
                // demonstrate reliability of correlation ID retirement ordering
                ensure_internal_correlation_id_retirement_ordering(_extcid->internal_corr_id);
            }

            auto info = std::stringstream{};

            info << "tid=" << record->thread_id << ", context=" << context.handle
                 << ", buffer_id=" << buffer_id.handle
                 << ", corr_id=" << record->correlation_id.internal << ", kind=" << record->kind
                 << ", operation=" << record->operation
                 << ", agent_id=" << record->dispatch_info.agent_id.handle
                 << ", queue_id=" << record->dispatch_info.queue_id.handle
                 << ", dispatch_id=" << record->dispatch_info.dispatch_id
                 << ", kernel_id=" << record->dispatch_info.kernel_id
                 << ", kernel=" << client_kernels->at(record->dispatch_info.kernel_id).kernel_name
                 << ", extern_corr_id={" << _extern_corr_id << "}";

            static_cast<call_stack_t*>(user_data)->emplace_back(
                source_location{__FUNCTION__, __FILE__, __LINE__, kind_name + info.str()});
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload);

            // demonstrate reliability of correlation ID retirement ordering
            ensure_internal_correlation_id_retirement_ordering(record->correlation_id.internal);

            auto _extern_corr_id = external_corr_id_data{};
            if(record->correlation_id.external.ptr)
            {
                auto* _extcid =
                    static_cast<external_corr_id_data*>(record->correlation_id.external.ptr);
                _extcid->seen_count++;
                _extern_corr_id = *_extcid;
                // demonstrate reliability of correlation ID retirement ordering
                ensure_internal_correlation_id_retirement_ordering(_extcid->internal_corr_id);
            }

            auto info = std::stringstream{};

            info << "tid=" << record->thread_id << ", context=" << context.handle
                 << ", buffer_id=" << buffer_id.handle
                 << ", corr_id=" << record->correlation_id.internal << ", kind=" << record->kind
                 << ", operation=" << record->operation
                 << ", src_agent_id=" << record->src_agent_id.handle
                 << ", dst_agent_id=" << record->dst_agent_id.handle
                 << ", direction=" << record->operation << ", start=" << record->start_timestamp
                 << ", stop=" << record->end_timestamp
                 << ", name=" << client_name_info->at(record->kind, record->operation)
                 << ", extern_corr_id={" << _extern_corr_id << "}";

            static_cast<call_stack_t*>(user_data)->emplace_back(
                source_location{__FUNCTION__, __FILE__, __LINE__, kind_name + info.str()});
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_correlation_id_retirement_record_t*>(
                    header->payload);

            {
                auto _lk = std::unique_lock<std::shared_mutex>{client_mutex};
                client_retired_corr_ids.emplace(record->internal_correlation_id);
            }

            auto _extern_corr_id = external_corr_id_data{};
            auto info            = std::stringstream{};

            info << "context=" << context.handle << ", buffer_id=" << buffer_id.handle
                 << ", corr_id=" << record->internal_correlation_id << ", kind=" << record->kind
                 << ", timestamp=" << record->timestamp
                 << ", name=" << client_name_info->at(record->kind) << ", extern_corr_id={"
                 << _extern_corr_id << "}";

            static_cast<call_stack_t*>(user_data)->emplace_back(
                source_location{__FUNCTION__, __FILE__, __LINE__, kind_name + info.str()});
        }
        else
        {
            auto _msg = std::stringstream{};
            _msg << "unexpected rocprofiler_record_header_t category + kind: (" << header->category
                 << " + " << header->kind << ")";
            throw std::runtime_error{_msg.str()};
        }
    }
}

template <typename Arg, typename... Args>
auto
make_array(Arg arg, Args&&... args)
{
    constexpr auto N = 1 + sizeof...(Args);
    return std::array<Arg, N>{std::forward<Arg>(arg), std::forward<Args>(args)...};
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    assert(tool_data != nullptr);

    auto* call_stack_v = static_cast<call_stack_t*>(tool_data);

    call_stack_v->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    *client_name_info = common::get_buffer_tracing_names();
    client_fini_func  = fini_func;

    ROCPROFILER_CHECK(rocprofiler_create_context(&client_ctx));

    auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

    ROCPROFILER_CHECK(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       code_object_ops.data(),
                                                       code_object_ops.size(),
                                                       tool_code_object_callback,
                                                       nullptr));

    constexpr auto buffer_size_bytes      = 4096;
    constexpr auto buffer_watermark_bytes = buffer_size_bytes - (buffer_size_bytes / 8);

    ROCPROFILER_CHECK(rocprofiler_create_buffer(client_ctx,
                                                buffer_size_bytes,
                                                buffer_watermark_bytes,
                                                ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                tool_tracing_callback,
                                                tool_data,
                                                &client_buffer));

    auto external_corr_id_request_kinds =
        make_array(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
                   ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY);

    ROCPROFILER_CHECK(rocprofiler_configure_external_correlation_id_request_service(
        client_ctx,
        external_corr_id_request_kinds.data(),
        external_corr_id_request_kinds.size(),
        set_external_correlation_id,
        nullptr));

    auto       hip_runtime_ops         = std::vector<rocprofiler_tracing_operation_t>{};
    const auto desired_hip_runtime_ops = std::unordered_set<std::string_view>{
        "hipLaunchKernel", "hipMemcpyAsync", "hipMemsetAsync", "hipMalloc"};
    for(auto [idx, itr] : (*client_name_info)[ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API].items())
    {
        if(desired_hip_runtime_ops.count(*itr) > 0) hip_runtime_ops.emplace_back(idx);
    }

    if(desired_hip_runtime_ops.size() != hip_runtime_ops.size())
        throw std::runtime_error{"missing hip operations"};

    ROCPROFILER_CHECK(
        rocprofiler_configure_buffer_tracing_service(client_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
                                                     hip_runtime_ops.data(),
                                                     hip_runtime_ops.size(),
                                                     client_buffer));

    ROCPROFILER_CHECK(rocprofiler_configure_buffer_tracing_service(
        client_ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, client_buffer));

    ROCPROFILER_CHECK(rocprofiler_configure_buffer_tracing_service(
        client_ctx, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0, client_buffer));

    ROCPROFILER_CHECK(rocprofiler_configure_buffer_tracing_service(
        client_ctx,
        ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT,
        nullptr,
        0,
        client_buffer));

    int valid_ctx = 0;
    ROCPROFILER_CHECK(rocprofiler_context_is_valid(client_ctx, &valid_ctx));
    if(valid_ctx == 0)
    {
        // notify rocprofiler that initialization failed
        // and all the contexts, buffers, etc. created
        // should be ignored
        return -1;
    }

    ROCPROFILER_CHECK(rocprofiler_start_context(client_ctx));

    // no errors
    return 0;
}

void
tool_fini(void* tool_data)
{
    assert(tool_data != nullptr);
    client_fini_func = nullptr;
    client_id        = nullptr;

    std::cout << "finalizing...\n" << std::flush;
    rocprofiler_stop_context(client_ctx);
    ROCPROFILER_CHECK(rocprofiler_flush_buffer(client_buffer));

    auto* _call_stack = static_cast<call_stack_t*>(tool_data);
    _call_stack->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    print_call_stack(*_call_stack);

    size_t unretired = 0;
    size_t unseen    = 0;
    for(auto* itr : client_external_corr_ids)
    {
        if(itr->seen_count != 1)
        {
            std::cerr << "external correlation ID seen " << itr->seen_count << " times: {" << *itr
                      << "}\n"
                      << std::flush;
            ++unseen;
        }
        if(client_retired_corr_ids.count(itr->internal_corr_id) != 1)
        {
            std::cerr << "internal correlation ID passed to external correlation ID request was "
                         "not retired: {"
                      << itr->internal_corr_id << "}\n"
                      << std::flush;
            ++unretired;
        }

        delete itr;
    }

    std::cerr << "external correlation IDs not seen   : " << unseen << "\n" << std::flush;
    std::cerr << "internal correlation IDs not retired: " << unretired << "\n" << std::flush;

    if(unseen > 0) throw std::runtime_error{"unseen external correlation id data"};
    if(unretired > 0) throw std::runtime_error{"unretired internal correlation id values"};

    delete _call_stack;
}
}  // namespace

void
setup()
{
    if(int status = 0;
       rocprofiler_is_initialized(&status) == ROCPROFILER_STATUS_SUCCESS && status == 0)
    {
        ROCPROFILER_CHECK(rocprofiler_force_configure(&rocprofiler_configure));
    }
}

void
shutdown()
{
    if(client_id)
    {
        ROCPROFILER_CHECK(rocprofiler_flush_buffer(client_buffer));
        client_fini_func(*client_id);
    }
}

void
start()
{
    ROCPROFILER_CHECK(rocprofiler_start_context(client_ctx));
}

void
identify(uint64_t val)
{
    auto _tid = rocprofiler_thread_id_t{};
    rocprofiler_get_thread_id(&_tid);
    rocprofiler_user_data_t user_data = {};
    user_data.value                   = val;
    rocprofiler_push_external_correlation_id(client_ctx, _tid, user_data);
}

void
stop()
{
    ROCPROFILER_CHECK(rocprofiler_stop_context(client_ctx));
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

    auto* client_tool_data = new std::vector<client::source_location>{};

    client_tool_data->emplace_back(
        client::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});

    std::atexit([]() {
        std::cout << "atexit handler...\n" << std::flush;
        if(client::client_fini_func && client::client_id)
            client::client_fini_func(*client::client_id);
    });

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &client::tool_init,
                                            &client::tool_fini,
                                            static_cast<void*>(client_tool_data)};

    // return pointer to configure data
    return &cfg;
}
