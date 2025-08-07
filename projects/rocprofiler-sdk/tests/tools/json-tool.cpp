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
 * @file tests/tools/json-tool.cpp
 *
 * @brief Test rocprofiler tool
 */

#include "common/defines.hpp"
#include "common/filesystem.hpp"
#include "common/hash.hpp"
#include "common/name_info.hpp"
#include "common/perfetto.hpp"
#include "common/serialization.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/utility.hpp>

#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <variant>
#include <vector>

namespace client
{
namespace
{
template <typename Tp>
size_t
get_hash_id(Tp&& _val)
{
    if constexpr(!std::is_pointer<Tp>::value)
        return std::hash<Tp>{}(std::forward<Tp>(_val));
    else if constexpr(std::is_same<Tp, const char*>::value)
        return get_hash_id(std::string_view{_val});
    else
        return get_hash_id(*_val);
}

std::string
demangle(std::string_view _mangled_name, int& _status)
{
    constexpr size_t buffer_len = 4096;
    // return the mangled since there is no buffer
    if(_mangled_name.empty())
    {
        _status = -2;
        return std::string{};
    }

    auto _demangled_name = std::string{_mangled_name};

    // PARAMETERS to __cxa_demangle
    //  mangled_name:
    //      A NULL-terminated character string containing the name to be demangled.
    //  buffer:
    //      A region of memory, allocated with malloc, of *length bytes, into which the
    //      demangled name is stored. If output_buffer is not long enough, it is expanded
    //      using realloc. output_buffer may instead be NULL; in that case, the demangled
    //      name is placed in a region of memory allocated with malloc.
    //  _buflen:
    //      If length is non-NULL, the length of the buffer containing the demangled name
    //      is placed in *length.
    //  status:
    //      *status is set to one of the following values
    size_t _demang_len = 0;
    char*  _demang = abi::__cxa_demangle(_demangled_name.c_str(), nullptr, &_demang_len, &_status);
    switch(_status)
    {
        //  0 : The demangling operation succeeded.
        // -1 : A memory allocation failure occurred.
        // -2 : mangled_name is not a valid name under the C++ ABI mangling rules.
        // -3 : One of the arguments is invalid.
        case 0:
        {
            if(_demang) _demangled_name = std::string{_demang};
            break;
        }
        case -1:
        {
            char _msg[buffer_len];
            ::memset(_msg, '\0', buffer_len * sizeof(char));
            ::snprintf(_msg,
                       buffer_len,
                       "memory allocation failure occurred demangling %s",
                       _demangled_name.c_str());
            ::perror(_msg);
            break;
        }
        case -2: break;
        case -3:
        {
            char _msg[buffer_len];
            ::memset(_msg, '\0', buffer_len * sizeof(char));
            ::snprintf(_msg,
                       buffer_len,
                       "Invalid argument in: (\"%s\", nullptr, nullptr, %p)",
                       _demangled_name.c_str(),
                       (void*) &_status);
            ::perror(_msg);
            break;
        }
        default: break;
    };

    // if it "demangled" but the length is zero, set the status to -2
    if(_demang_len == 0 && _status == 0) _status = -2;

    // free allocated buffer
    ::free(_demang);
    return _demangled_name;
}

std::string
demangle(std::string_view symbol)
{
    int  _status       = 0;
    auto demangled_str = demangle(symbol, _status);
    if(_status == 0) return demangled_str;
    return std::string{symbol};
}

struct source_location
{
    std::string function = {};
    std::string file     = {};
    uint32_t    line     = 0;
    std::string context  = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("function", function));
        ar(cereal::make_nvp("file", file));
        ar(cereal::make_nvp("line", line));
        ar(cereal::make_nvp("context", context));
    }
};

template <typename Tp, typename... Args>
auto
make_array(Tp&& arg, Args&&... args)
{
    constexpr auto N = sizeof...(Args) + 1;
    return std::array<Tp, N>{std::forward<Tp>(arg), std::forward<Args>(args)...};
}

using call_stack_t = std::vector<source_location>;

using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using host_function_data_t =
    rocprofiler_callback_tracing_code_object_host_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;
using host_functions_map_t = std::unordered_map<uint64_t, host_function_data_t>;

rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;

using callback_payload_t =
    std::variant<rocprofiler_callback_tracing_code_object_load_data_t,
                 rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t,
                 rocprofiler_callback_tracing_hsa_api_data_t,
                 rocprofiler_callback_tracing_marker_api_data_t>;

using callback_arg_array_t = std::vector<std::pair<std::string, std::string>>;

template <typename ArchiveT>
void
serialize_args(ArchiveT& ar, const callback_arg_array_t& data)
{
    if constexpr(std::is_same<ArchiveT, cereal::BinaryOutputArchive>::value ||
                 std::is_same<ArchiveT, cereal::PortableBinaryOutputArchive>::value)
    {
        ar(cereal::make_nvp("args", data));
    }
    else
    {
        ar.setNextName("args");
        ar.startNode();
        for(const auto& itr : data)
            ar(cereal::make_nvp(itr.first, itr.second));
        ar.finishNode();
    }
}

template <typename... Args>
void
consume_args(Args&&...)
{}

int
save_args(rocprofiler_callback_tracing_kind_t domain_idx,
          rocprofiler_tracing_operation_t     op_idx,
          uint32_t                            arg_num,
          const void* const                   arg_value_addr,
          int32_t                             arg_indirection_count,
          const char*                         arg_type,
          const char*                         arg_name,
          const char*                         arg_value_str,
          int32_t                             arg_dereference_count,
          void*                               data)
{
    auto* argvec = static_cast<callback_arg_array_t*>(data);
    argvec->emplace_back(arg_name, arg_value_str);
    return 0;

    consume_args(domain_idx,
                 op_idx,
                 arg_num,
                 arg_value_addr,
                 arg_indirection_count,
                 arg_type,
                 arg_dereference_count);
}

struct code_object_callback_record_t
{
    uint64_t                                             timestamp = 0;
    rocprofiler_callback_tracing_record_t                record    = {};
    rocprofiler_callback_tracing_code_object_load_data_t payload   = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        cereal::save(ar, record);
        ar(cereal::make_nvp("payload", payload));
    }
};

struct kernel_symbol_callback_record_t
{
    uint64_t                                                               timestamp = 0;
    rocprofiler_callback_tracing_record_t                                  record    = {};
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t payload   = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        cereal::save(ar, record);
        ar(cereal::make_nvp("payload", payload));
    }
};

struct host_function_callback_record_t
{
    uint64_t                                                                    timestamp = 0;
    rocprofiler_callback_tracing_record_t                                       record    = {};
    rocprofiler_callback_tracing_code_object_host_kernel_symbol_register_data_t payload   = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        cereal::save(ar, record);
        ar(cereal::make_nvp("payload", payload));
    }
};

struct runtime_init_callback_record_t
{
    uint64_t                                                   timestamp = 0;
    rocprofiler_callback_tracing_record_t                      record    = {};
    rocprofiler_callback_tracing_runtime_initialization_data_t payload   = {};
    callback_arg_array_t                                       args      = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        cereal::save(ar, record);
        ar(cereal::make_nvp("payload", payload));
        serialize_args(ar, args);
    }
};

struct hsa_api_callback_record_t
{
    uint64_t                                    timestamp = 0;
    rocprofiler_callback_tracing_record_t       record    = {};
    rocprofiler_callback_tracing_hsa_api_data_t payload   = {};
    callback_arg_array_t                        args      = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        cereal::save(ar, record);
        ar(cereal::make_nvp("payload", payload));
        serialize_args(ar, args);
    }
};

struct hip_api_callback_record_t
{
    uint64_t                                    timestamp = 0;
    rocprofiler_callback_tracing_record_t       record    = {};
    rocprofiler_callback_tracing_hip_api_data_t payload   = {};
    callback_arg_array_t                        args      = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        cereal::save(ar, record);
        ar(cereal::make_nvp("payload", payload));
        serialize_args(ar, args);
    }
};

struct marker_api_callback_record_t
{
    uint64_t                                       timestamp = 0;
    rocprofiler_callback_tracing_record_t          record    = {};
    rocprofiler_callback_tracing_marker_api_data_t payload   = {};
    callback_arg_array_t                           args      = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        cereal::save(ar, record);
        ar(cereal::make_nvp("payload", payload));
        serialize_args(ar, args);
    }
};

struct rccl_api_callback_record_t
{
    uint64_t                                     timestamp = 0;
    rocprofiler_callback_tracing_record_t        record    = {};
    rocprofiler_callback_tracing_rccl_api_data_t payload   = {};
    callback_arg_array_t                         args      = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        cereal::save(ar, record);
        ar(cereal::make_nvp("payload", payload));
        serialize_args(ar, args);
    }
};

struct rocdecode_api_callback_record_t
{
    uint64_t                                          timestamp = 0;
    rocprofiler_callback_tracing_record_t             record    = {};
    rocprofiler_callback_tracing_rocdecode_api_data_t payload   = {};
    callback_arg_array_t                              args      = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        cereal::save(ar, record);
        ar(cereal::make_nvp("payload", payload));
        serialize_args(ar, args);
    }
};

struct rocjpeg_api_callback_record_t
{
    uint64_t                                        timestamp = 0;
    rocprofiler_callback_tracing_record_t           record    = {};
    rocprofiler_callback_tracing_rocjpeg_api_data_t payload   = {};
    callback_arg_array_t                            args      = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        cereal::save(ar, record);
        ar(cereal::make_nvp("payload", payload));
        serialize_args(ar, args);
    }
};

struct ompt_callback_record_t
{
    uint64_t                                 timestamp = 0;
    rocprofiler_callback_tracing_record_t    record    = {};
    rocprofiler_callback_tracing_ompt_data_t payload   = {};
    callback_arg_array_t                     args      = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        cereal::save(ar, record);
        ar(cereal::make_nvp("payload", payload));
        serialize_args(ar, args);
    }
};

struct kernel_dispatch_callback_record_t
{
    uint64_t                                            timestamp = 0;
    rocprofiler_callback_tracing_record_t               record    = {};
    rocprofiler_callback_tracing_kernel_dispatch_data_t payload   = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        cereal::save(ar, record);
        ar(cereal::make_nvp("payload", payload));
    }
};

struct memory_copy_callback_record_t
{
    uint64_t                                        timestamp = 0;
    rocprofiler_callback_tracing_record_t           record    = {};
    rocprofiler_callback_tracing_memory_copy_data_t payload   = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        cereal::save(ar, record);
        ar(cereal::make_nvp("payload", payload));
    }
};

struct memory_allocation_callback_record_t
{
    uint64_t                                              timestamp = 0;
    rocprofiler_callback_tracing_record_t                 record    = {};
    rocprofiler_callback_tracing_memory_allocation_data_t payload   = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        cereal::save(ar, record);
        ar(cereal::make_nvp("payload", payload));
    }
};

struct scratch_memory_callback_record_t
{
    uint64_t                                           timestamp = 0;
    rocprofiler_callback_tracing_record_t              record    = {};
    rocprofiler_callback_tracing_scratch_memory_data_t payload   = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        cereal::save(ar, record);
        ar(cereal::make_nvp("payload", payload));

        if constexpr(std::is_same<ArchiveT, cereal::BinaryOutputArchive>::value ||
                     std::is_same<ArchiveT, cereal::PortableBinaryOutputArchive>::value)
        {}
        else
        {
            ar.setNextName("args");
            ar.startNode();
            if(payload.args_kind == HSA_AMD_TOOL_EVENT_SCRATCH_ALLOC_START)
            {
                ar(cereal::make_nvp("dispatch_id", payload.args.alloc_start.dispatch_id));
            }
            else if(payload.args_kind == HSA_AMD_TOOL_EVENT_SCRATCH_ALLOC_END)
            {
                ar(cereal::make_nvp("dispatch_id", payload.args.alloc_end.dispatch_id));
                ar(cereal::make_nvp("size", payload.args.alloc_end.size));
                ar(cereal::make_nvp("num_slots", payload.args.alloc_end.num_slots));
            }
            ar.finishNode();
        }
    }
};

struct profile_counting_record
{
    profile_counting_record(rocprofiler_dispatch_counting_service_record_t hdr)
    : header{hdr}
    {}

    rocprofiler_dispatch_counting_service_record_t header = {};
    std::vector<rocprofiler_record_counter_t>      data   = {};

    profile_counting_record()                                   = default;
    ~profile_counting_record()                                  = default;
    profile_counting_record(const profile_counting_record&)     = default;
    profile_counting_record(profile_counting_record&&) noexcept = default;
    profile_counting_record& operator=(const profile_counting_record&) = default;
    profile_counting_record& operator=(profile_counting_record&&) noexcept = default;

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        cereal::save(ar, header);
        auto _data = data;
        for(auto& itr : _data)
        {
            auto _counter_id = rocprofiler_counter_id_t{};
            ROCPROFILER_CALL(rocprofiler_query_record_counter_id(itr.id, &_counter_id),
                             "failed to query counter id");
            itr.id = _counter_id.handle;
        }

        ar(cereal::make_nvp("records", _data));
    }

    void emplace_back(rocprofiler_record_counter_t val)
    {
        if(*this != val)
            throw std::runtime_error{"invalid profile_counting_record::emplace_back(...)"};
        data.emplace_back(val);
    }

    bool operator==(rocprofiler_record_counter_t rhs) const
    {
        return (header.dispatch_info.dispatch_id == rhs.dispatch_id);
    }

    bool operator!=(rocprofiler_record_counter_t rhs) const { return !(*this == rhs); }
};

auto counter_info                  = std::deque<rocprofiler_counter_info_v0_t>{};
auto runtime_init_cb_records       = std::deque<runtime_init_callback_record_t>{};
auto code_object_records           = std::deque<code_object_callback_record_t>{};
auto kernel_symbol_records         = std::deque<kernel_symbol_callback_record_t>{};
auto host_function_records         = std::deque<host_function_callback_record_t>{};
auto hsa_api_cb_records            = std::deque<hsa_api_callback_record_t>{};
auto marker_api_cb_records         = std::deque<marker_api_callback_record_t>{};
auto counter_collection_bf_records = std::deque<profile_counting_record>{};
auto hip_api_cb_records            = std::deque<hip_api_callback_record_t>{};
auto scratch_memory_cb_records     = std::deque<scratch_memory_callback_record_t>{};
auto kernel_dispatch_cb_records    = std::deque<kernel_dispatch_callback_record_t>{};
auto memory_copy_cb_records        = std::deque<memory_copy_callback_record_t>{};
auto memory_allocation_cb_records  = std::deque<memory_allocation_callback_record_t>{};
auto rccl_api_cb_records           = std::deque<rccl_api_callback_record_t>{};
auto rocdecode_api_cb_records      = std::deque<rocdecode_api_callback_record_t>{};
auto rocjpeg_api_cb_records        = std::deque<rocjpeg_api_callback_record_t>{};
auto ompt_cb_records               = std::deque<ompt_callback_record_t>{};

int
set_external_correlation_id(rocprofiler_thread_id_t                            thr_id,
                            rocprofiler_context_id_t                           ctx_id,
                            rocprofiler_external_correlation_id_request_kind_t kind,
                            rocprofiler_tracing_operation_t                    op,
                            uint64_t                                           internal_corr_id,
                            rocprofiler_user_data_t*                           external_corr_id,
                            void*                                              user_data)
{
    consume_args(ctx_id, kind, op, internal_corr_id, user_data);

    external_corr_id->value = thr_id;
    return 0;
}

void
dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                  rocprofiler_counter_config_id_t*             config,
                  rocprofiler_user_data_t* /*user_data*/,
                  void* /*callback_data_args*/)
{
    static std::shared_mutex                                             m_mutex       = {};
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> profile_cache = {};

    auto search_cache = [&]() {
        if(auto pos = profile_cache.find(dispatch_data.dispatch_info.agent_id.handle);
           pos != profile_cache.end())
        {
            *config = pos->second;
            return true;
        }
        return false;
    };

    {
        auto rlock = std::shared_lock{m_mutex};
        if(search_cache()) return;
    }

    auto wlock = std::unique_lock{m_mutex};
    if(search_cache()) return;

    // Counters we want to collect (here its SQ_WAVES_sum)
    auto* counters_env = getenv("ROCPROF_COUNTERS");
    if(std::string(counters_env) != "SQ_WAVES_sum")
        throw std::runtime_error{"Counter not supported in the test tool"};

    std::set<std::string> counters_to_collect = {"SQ_WAVES_sum"};
    // GPU Counter IDs
    std::vector<rocprofiler_counter_id_t> gpu_counters;

    // Iterate through the agents and get the counters available on that agent
    ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
                         dispatch_data.dispatch_info.agent_id,
                         []([[maybe_unused]] rocprofiler_agent_id_t id,
                            rocprofiler_counter_id_t*               counters,
                            size_t                                  num_counters,
                            void*                                   user_data) {
                             std::vector<rocprofiler_counter_id_t>* vec =
                                 static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
                             for(size_t i = 0; i < num_counters; i++)
                             {
                                 vec->push_back(counters[i]);
                             }
                             return ROCPROFILER_STATUS_SUCCESS;
                         },
                         static_cast<void*>(&gpu_counters)),
                     "Could not fetch supported counters");

    for(auto& counter : gpu_counters)
    {
        auto info = rocprofiler_counter_info_v0_t{};

        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&info)),
            "Could not query counter_id");

        counter_info.emplace_back(info);
    }

    std::vector<rocprofiler_counter_id_t> collect_counters;
    // Look for the counters contained in counters_to_collect in gpu_counters
    for(auto& counter : gpu_counters)
    {
        rocprofiler_counter_info_v0_t info;

        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&info)),
            "Could not query counter_id");

        if(counters_to_collect.count(std::string(info.name)) > 0)
        {
            collect_counters.push_back(counter);
        }
    }

    // Create a colleciton profile for the counters
    rocprofiler_counter_config_id_t profile = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_create_counter_config(dispatch_data.dispatch_info.agent_id,
                                                       collect_counters.data(),
                                                       collect_counters.size(),
                                                       &profile),
                     "Could not construct profile cfg");

    profile_cache.emplace(dispatch_data.dispatch_info.agent_id.handle, profile);
    // Return the profile to collect those counters for this dispatch
    *config = profile;
}

void
tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                      rocprofiler_user_data_t* /*user_data*/,
                      void* /*callback_data*/)
{
    auto ts = rocprofiler_timestamp_t{};
    ROCPROFILER_CALL(rocprofiler_get_timestamp(&ts), "get timestamp");

    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT)
    {
        if(record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
        {
            auto data_v =
                *static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);

            static auto _mutex = std::mutex{};
            auto        _lk    = std::unique_lock<std::mutex>{_mutex};
            code_object_records.emplace_back(code_object_callback_record_t{ts, record, data_v});
        }
        else if(record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
        {
            auto data_v = *static_cast<kernel_symbol_data_t*>(record.payload);

            static auto _mutex = std::mutex{};
            auto        _lk    = std::unique_lock<std::mutex>{_mutex};
            kernel_symbol_records.emplace_back(kernel_symbol_callback_record_t{ts, record, data_v});
        }
        else if(record.operation == ROCPROFILER_CODE_OBJECT_HOST_KERNEL_SYMBOL_REGISTER)
        {
            auto        data_v = *static_cast<host_function_data_t*>(record.payload);
            static auto _mutex = std::mutex{};
            auto        _lk    = std::unique_lock<std::mutex>{_mutex};
            host_function_records.emplace_back(host_function_callback_record_t{ts, record, data_v});
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API ||
            record.kind == ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API ||
            record.kind == ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API ||
            record.kind == ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API)
    {
        auto* data = static_cast<rocprofiler_callback_tracing_hsa_api_data_t*>(record.payload);
        auto  args = callback_arg_array_t{};
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
            rocprofiler_iterate_callback_tracing_kind_operation_args(
                record, save_args, record.phase, &args);

        static auto _mutex = std::mutex{};
        auto        _lk    = std::unique_lock<std::mutex>{_mutex};
        hsa_api_cb_records.emplace_back(
            hsa_api_callback_record_t{ts, record, *data, std::move(args)});
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API ||
            record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API)
    {
        auto* data = static_cast<rocprofiler_callback_tracing_hip_api_data_t*>(record.payload);
        auto  args = callback_arg_array_t{};
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
            rocprofiler_iterate_callback_tracing_kind_operation_args(
                record, save_args, record.phase, &args);

        static auto _mutex = std::mutex{};
        auto        _lk    = std::unique_lock<std::mutex>{_mutex};
        hip_api_cb_records.emplace_back(
            hip_api_callback_record_t{ts, record, *data, std::move(args)});
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API ||
            record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API ||
            record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API ||
            record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_RANGE_API)
    {
        auto* data = static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);
        auto  args = callback_arg_array_t{};
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
            rocprofiler_iterate_callback_tracing_kind_operation_args(
                record, save_args, record.phase, &args);

        static auto _mutex = std::mutex{};
        auto        _lk    = std::unique_lock<std::mutex>{_mutex};
        marker_api_cb_records.emplace_back(
            marker_api_callback_record_t{ts, record, *data, std::move(args)});
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_SCRATCH_MEMORY)
    {
        auto* data =
            static_cast<rocprofiler_callback_tracing_scratch_memory_data_t*>(record.payload);

        static auto _mutex = std::mutex{};
        auto        _lk    = std::unique_lock<std::mutex>{_mutex};
        scratch_memory_cb_records.emplace_back(scratch_memory_callback_record_t{ts, record, *data});
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH)
    {
        auto* data =
            static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(record.payload);

        static auto _mutex = std::mutex{};
        auto        _lk    = std::unique_lock<std::mutex>{_mutex};
        kernel_dispatch_cb_records.emplace_back(
            kernel_dispatch_callback_record_t{ts, record, *data});
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY)
    {
        auto* data = static_cast<rocprofiler_callback_tracing_memory_copy_data_t*>(record.payload);

        static auto _mutex = std::mutex{};
        auto        _lk    = std::unique_lock<std::mutex>{_mutex};
        memory_copy_cb_records.emplace_back(memory_copy_callback_record_t{ts, record, *data});
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION)
    {
        auto* data =
            static_cast<rocprofiler_callback_tracing_memory_allocation_data_t*>(record.payload);

        static auto _mutex = std::mutex{};
        auto        _lk    = std::unique_lock<std::mutex>{_mutex};
        memory_allocation_cb_records.emplace_back(
            memory_allocation_callback_record_t{ts, record, *data});
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_RCCL_API)
    {
        auto* data = static_cast<rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);
        auto  args = callback_arg_array_t{};
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
            rocprofiler_iterate_callback_tracing_kind_operation_args(
                record, save_args, record.phase, &args);

        static auto _mutex = std::mutex{};
        auto        _lk    = std::unique_lock<std::mutex>{_mutex};
        rccl_api_cb_records.emplace_back(
            rccl_api_callback_record_t{ts, record, *data, std::move(args)});
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_OMPT)
    {
        auto* data = static_cast<rocprofiler_callback_tracing_ompt_data_t*>(record.payload);
        auto  args = callback_arg_array_t{};
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
            rocprofiler_iterate_callback_tracing_kind_operation_args(
                record, save_args, record.phase, &args);

        static auto _mutex = std::mutex{};
        auto        _lk    = std::unique_lock<std::mutex>{_mutex};
        ompt_cb_records.emplace_back(ompt_callback_record_t{ts, record, *data, std::move(args)});
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION)
    {
        auto* data = static_cast<rocprofiler_callback_tracing_runtime_initialization_data_t*>(
            record.payload);
        auto args = callback_arg_array_t{};
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
            rocprofiler_iterate_callback_tracing_kind_operation_args(
                record, save_args, record.phase, &args);

        static auto _mutex = std::mutex{};
        auto        _lk    = std::unique_lock<std::mutex>{_mutex};
        runtime_init_cb_records.emplace_back(
            runtime_init_callback_record_t{ts, record, *data, std::move(args)});
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API)
    {
        auto* data =
            static_cast<rocprofiler_callback_tracing_rocdecode_api_data_t*>(record.payload);
        auto args = callback_arg_array_t{};
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
            rocprofiler_iterate_callback_tracing_kind_operation_args(
                record, save_args, record.phase, &args);

        static auto _mutex = std::mutex{};
        auto        _lk    = std::unique_lock<std::mutex>{_mutex};
        rocdecode_api_cb_records.emplace_back(
            rocdecode_api_callback_record_t{ts, record, *data, std::move(args)});
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API)
    {
        auto* data = static_cast<rocprofiler_callback_tracing_rocjpeg_api_data_t*>(record.payload);
        auto  args = callback_arg_array_t{};
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
            rocprofiler_iterate_callback_tracing_kind_operation_args(
                record, save_args, record.phase, &args);

        static auto _mutex = std::mutex{};
        auto        _lk    = std::unique_lock<std::mutex>{_mutex};
        rocjpeg_api_cb_records.emplace_back(
            rocjpeg_api_callback_record_t{ts, record, *data, std::move(args)});
    }
    else
    {
        throw std::runtime_error{"unsupported callback kind"};
    }
}

auto runtime_init_bf_records =
    std::deque<rocprofiler_buffer_tracing_runtime_initialization_record_t>{};
auto hsa_api_bf_records         = std::deque<rocprofiler_buffer_tracing_hsa_api_record_t>{};
auto marker_api_bf_records      = std::deque<rocprofiler_buffer_tracing_marker_api_record_t>{};
auto hip_api_bf_records         = std::deque<rocprofiler_buffer_tracing_hip_api_record_t>{};
auto kernel_dispatch_bf_records = std::deque<rocprofiler_buffer_tracing_kernel_dispatch_record_t>{};
auto memory_copy_bf_records     = std::deque<rocprofiler_buffer_tracing_memory_copy_record_t>{};
auto memory_allocation_bf_records =
    std::deque<rocprofiler_buffer_tracing_memory_allocation_record_t>{};
auto scratch_memory_records = std::deque<rocprofiler_buffer_tracing_scratch_memory_record_t>{};
auto corr_id_retire_records =
    std::deque<rocprofiler_buffer_tracing_correlation_id_retirement_record_t>{};
auto rccl_api_ext_bf_records  = std::deque<rocprofiler_buffer_tracing_rccl_api_ext_record_t>{};
auto rocdecode_api_bf_records = std::deque<rocprofiler_buffer_tracing_rocdecode_api_record_t>{};
auto rocdecode_api_ext_bf_records =
    std::deque<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t>{};
auto rocjpeg_api_bf_records = std::deque<rocprofiler_buffer_tracing_rocjpeg_api_record_t>{};
auto ompt_bf_records        = std::deque<rocprofiler_buffer_tracing_ompt_record_t>{};
auto kfd_page_migrate_event_records =
    std::deque<rocprofiler_buffer_tracing_kfd_event_page_migrate_record_t>{};
auto kfd_page_fault_event_records =
    std::deque<rocprofiler_buffer_tracing_kfd_event_page_fault_record_t>{};
auto kfd_queue_event_records = std::deque<rocprofiler_buffer_tracing_kfd_event_queue_record_t>{};
auto kfd_unmap_from_gpu_event_records =
    std::deque<rocprofiler_buffer_tracing_kfd_event_unmap_from_gpu_record_t>{};
auto kfd_dropped_events_event_records =
    std::deque<rocprofiler_buffer_tracing_kfd_event_dropped_events_record_t>{};
auto kfd_page_migrate_records = std::deque<rocprofiler_buffer_tracing_kfd_page_migrate_record_t>{};
auto kfd_page_fault_records   = std::deque<rocprofiler_buffer_tracing_kfd_page_fault_record_t>{};
auto kfd_queue_records        = std::deque<rocprofiler_buffer_tracing_kfd_queue_record_t>{};

void
tool_tracing_buffered(rocprofiler_context_id_t /*context*/,
                      rocprofiler_buffer_id_t /*buffer_id*/,
                      rocprofiler_record_header_t** headers,
                      size_t                        num_headers,
                      void*                         user_data,
                      uint64_t                      drop_count)
{
    assert(user_data != nullptr);
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

        if(header == nullptr)
        {
            throw std::runtime_error{
                "rocprofiler provided a null pointer to header. this should never happen"};
        }
        else if(header->hash !=
                rocprofiler_record_header_compute_hash(header->category, header->kind))
        {
            throw std::runtime_error{"rocprofiler_record_header_t (category | kind) != hash"};
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING)
        {
            if(header->kind == ROCPROFILER_BUFFER_TRACING_HSA_CORE_API ||
               header->kind == ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API ||
               header->kind == ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API ||
               header->kind == ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_hsa_api_record_t*>(header->payload);

                hsa_api_bf_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_marker_api_record_t*>(header->payload);

                marker_api_bf_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload);

                hip_api_bf_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
                    header->payload);

                kernel_dispatch_bf_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload);

                memory_copy_bf_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_memory_allocation_record_t*>(
                    header->payload);

                memory_allocation_bf_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_scratch_memory_record_t*>(
                    header->payload);

                scratch_memory_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_correlation_id_retirement_record_t*>(
                        header->payload);

                corr_id_retire_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_RCCL_API_EXT)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_rccl_api_ext_record_t*>(header->payload);

                rccl_api_ext_bf_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_OMPT)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_ompt_record_t*>(header->payload);

                ompt_bf_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_runtime_initialization_record_t*>(
                        header->payload);

                runtime_init_bf_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_ROCDECODE_API)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_rocdecode_api_record_t*>(
                    header->payload);

                rocdecode_api_bf_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_ROCDECODE_API_EXT)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t*>(
                    header->payload);

                rocdecode_api_ext_bf_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_ROCJPEG_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_rocjpeg_api_record_t*>(header->payload);

                rocjpeg_api_bf_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_kfd_event_page_migrate_record_t*>(
                        header->payload);

                kfd_page_migrate_event_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_kfd_event_page_fault_record_t*>(
                        header->payload);

                kfd_page_fault_event_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_kfd_event_queue_record_t*>(
                    header->payload);

                kfd_queue_event_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_kfd_event_unmap_from_gpu_record_t*>(
                        header->payload);

                kfd_unmap_from_gpu_event_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_kfd_event_dropped_events_record_t*>(
                        header->payload);

                kfd_dropped_events_event_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_kfd_page_migrate_record_t*>(
                    header->payload);

                kfd_page_migrate_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_kfd_page_fault_record_t*>(
                    header->payload);

                kfd_page_fault_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KFD_QUEUE)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_kfd_queue_record_t*>(header->payload);

                kfd_queue_records.emplace_back(*record);
            }
            else
            {
                throw std::runtime_error{
                    "unexpected rocprofiler_record_header_t tracing category kind"};
            }
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
                header->kind == ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER)
        {
            auto* profiler_record =
                static_cast<rocprofiler_dispatch_counting_service_record_t*>(header->payload);
            counter_collection_bf_records.emplace_back(*profiler_record);
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
                header->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
        {
            auto* profiler_record = static_cast<rocprofiler_record_counter_t*>(header->payload);
            if(counter_collection_bf_records.empty())
                throw std::runtime_error{
                    "missing rocprofiler_dispatch_counting_service_record_t (header)"};
            counter_collection_bf_records.back().emplace_back(*profiler_record);
        }
        else
        {
            throw std::runtime_error{"unexpected rocprofiler_record_header_t category + kind"};
        }
    }
}

void
thread_precreate(rocprofiler_runtime_library_t lib, void* tool_data)
{
    static_cast<call_stack_t*>(tool_data)->emplace_back(
        source_location{__FUNCTION__,
                        __FILE__,
                        __LINE__,
                        std::string{"internal thread about to be created by rocprofiler (lib="} +
                            std::to_string(lib) + ")"});
}

void
thread_postcreate(rocprofiler_runtime_library_t lib, void* tool_data)
{
    static_cast<call_stack_t*>(tool_data)->emplace_back(
        source_location{__FUNCTION__,
                        __FILE__,
                        __LINE__,
                        std::string{"internal thread was created by rocprofiler (lib="} +
                            std::to_string(lib) + ")"});
}

bool
is_active(rocprofiler_context_id_t ctx)
{
    int  status = 0;
    auto errc   = rocprofiler_context_is_active(ctx, &status);
    return (errc == ROCPROFILER_STATUS_SUCCESS && status > 0);
}

void
start();

void
stop();

void
flush();

void
push_external_correlation(uint64_t _value);

void
pop_external_correlation();

// contexts
rocprofiler_context_id_t hsa_api_callback_ctx           = {0};
rocprofiler_context_id_t hip_api_callback_ctx           = {0};
rocprofiler_context_id_t marker_api_callback_ctx        = {0};
rocprofiler_context_id_t code_object_ctx                = {0};
rocprofiler_context_id_t rccl_api_callback_ctx          = {0};
rocprofiler_context_id_t ompt_callback_ctx              = {0};
rocprofiler_context_id_t hsa_api_buffered_ctx           = {0};
rocprofiler_context_id_t hip_api_buffered_ctx           = {0};
rocprofiler_context_id_t marker_api_buffered_ctx        = {0};
rocprofiler_context_id_t memory_copy_callback_ctx       = {0};
rocprofiler_context_id_t memory_copy_buffered_ctx       = {0};
rocprofiler_context_id_t memory_allocation_callback_ctx = {0};
rocprofiler_context_id_t memory_allocation_buffered_ctx = {0};
rocprofiler_context_id_t rccl_api_ext_buffered_ctx      = {0};
rocprofiler_context_id_t ompt_buffered_ctx              = {0};
rocprofiler_context_id_t counter_collection_ctx         = {0};
rocprofiler_context_id_t scratch_memory_ctx             = {0};
rocprofiler_context_id_t corr_id_retire_ctx             = {0};
rocprofiler_context_id_t kernel_dispatch_callback_ctx   = {0};
rocprofiler_context_id_t kernel_dispatch_buffered_ctx   = {0};
rocprofiler_context_id_t runtime_init_callback_ctx      = {};
rocprofiler_context_id_t runtime_init_buffered_ctx      = {};
rocprofiler_context_id_t rocdecode_api_callback_ctx     = {0};
rocprofiler_context_id_t rocdecode_api_buffered_ctx     = {0};
rocprofiler_context_id_t rocdecode_api_ext_buffered_ctx = {0};
rocprofiler_context_id_t rocjpeg_api_callback_ctx       = {0};
rocprofiler_context_id_t rocjpeg_api_buffered_ctx       = {0};
rocprofiler_context_id_t page_migrate_event_ctx         = {0};
rocprofiler_context_id_t kfd_page_fault_event_ctx       = {0};
rocprofiler_context_id_t kfd_queue_event_ctx            = {0};
rocprofiler_context_id_t kfd_unmap_from_gpu_event_ctx   = {0};
rocprofiler_context_id_t kfd_droped_events_event_ctx    = {0};
rocprofiler_context_id_t kfd_page_migrate_records_ctx   = {0};
rocprofiler_context_id_t kfd_page_fault_records_ctx     = {0};
rocprofiler_context_id_t kfd_queue_records_ctx          = {0};

// buffers
rocprofiler_buffer_id_t runtime_init_buffered_buffer    = {};
rocprofiler_buffer_id_t hsa_api_buffered_buffer         = {};
rocprofiler_buffer_id_t hip_api_buffered_buffer         = {};
rocprofiler_buffer_id_t marker_api_buffered_buffer      = {};
rocprofiler_buffer_id_t kernel_dispatch_buffer          = {};
rocprofiler_buffer_id_t memory_copy_buffer              = {};
rocprofiler_buffer_id_t memory_allocation_buffer        = {};
rocprofiler_buffer_id_t counter_collection_buffer       = {};
rocprofiler_buffer_id_t scratch_memory_buffer           = {};
rocprofiler_buffer_id_t corr_id_retire_buffer           = {};
rocprofiler_buffer_id_t rccl_api_ext_buffered_buffer    = {};
rocprofiler_buffer_id_t rocdecode_api_buffer            = {};
rocprofiler_buffer_id_t rocdecode_api_ext_buffer        = {};
rocprofiler_buffer_id_t rocjpeg_api_buffer              = {};
rocprofiler_buffer_id_t ompt_buffered_buffer            = {};
rocprofiler_buffer_id_t page_migrate_event_buffer       = {};
rocprofiler_buffer_id_t kfd_page_fault_event_buffer     = {};
rocprofiler_buffer_id_t kfd_queue_event_buffer          = {};
rocprofiler_buffer_id_t kfd_unmap_from_gpu_event_buffer = {};
rocprofiler_buffer_id_t kfd_droped_events_event_buffer  = {};
rocprofiler_buffer_id_t kfd_page_migrate_records_buffer = {};
rocprofiler_buffer_id_t kfd_page_fault_records_buffer   = {};
rocprofiler_buffer_id_t kfd_queue_records_buffer        = {};

auto contexts = std::unordered_map<std::string_view, rocprofiler_context_id_t*>{
    {"RUNTIME_INIT_CALLBACK", &runtime_init_callback_ctx},
    {"HSA_API_CALLBACK", &hsa_api_callback_ctx},
    {"HIP_API_CALLBACK", &hip_api_callback_ctx},
    {"MARKER_API_CALLBACK", &marker_api_callback_ctx},
    {"CODE_OBJECT", &code_object_ctx},
    {"KERNEL_DISPATCH_CALLBACK", &kernel_dispatch_callback_ctx},
    {"MEMORY_COPY_CALLBACK", &memory_copy_callback_ctx},
    {"MEMORY_ALLOCATION_CALLBACK", &memory_allocation_callback_ctx},
    {"RCCL_API_CALLBACK", &rccl_api_callback_ctx},
    {"OMPT_CALLBACK", &ompt_callback_ctx},
    {"RUNTIME_INIT_BUFFERED", &runtime_init_buffered_ctx},
    {"HSA_API_BUFFERED", &hsa_api_buffered_ctx},
    {"HIP_API_BUFFERED", &hip_api_buffered_ctx},
    {"MARKER_API_BUFFERED", &marker_api_buffered_ctx},
    {"KERNEL_DISPATCH_BUFFERED", &kernel_dispatch_buffered_ctx},
    {"MEMORY_COPY_BUFFERED", &memory_copy_buffered_ctx},
    {"MEMORY_ALLOCATION_BUFFERED", &memory_allocation_buffered_ctx},
    {"COUNTER_COLLECTION", &counter_collection_ctx},
    {"SCRATCH_MEMORY", &scratch_memory_ctx},
    {"CORRELATION_ID_RETIREMENT", &corr_id_retire_ctx},
    {"RCCL_API_EXT_BUFFERED", &rccl_api_ext_buffered_ctx},
    {"ROCDECODE_API_CALLBACK", &rocdecode_api_callback_ctx},
    {"ROCDECODE_API_BUFFERED", &rocdecode_api_buffered_ctx},
    {"ROCDECODE_API_EXT_BUFFERED", &rocdecode_api_ext_buffered_ctx},
    {"ROCJPEG_API_CALLBACK", &rocjpeg_api_callback_ctx},
    {"ROCJPEG_API_BUFFERED", &rocjpeg_api_buffered_ctx},
    {"OMPT_BUFFERED", &ompt_buffered_ctx},
    {"KFD_EVENT_PAGE_MIGRATE", &page_migrate_event_ctx},
    {"KFD_EVENT_PAGE_FAULT", &kfd_page_fault_event_ctx},
    {"KFD_EVENT_QUEUE", &kfd_queue_event_ctx},
    {"KFD_EVENT_UNMAP_FROM_GPU", &kfd_unmap_from_gpu_event_ctx},
    {"KFD_DROPPED_EVENTS", &kfd_droped_events_event_ctx},
    {"KFD_PAGE_MIGRATE", &kfd_page_migrate_records_ctx},
    {"KFD_PAGE_FAULT", &kfd_page_fault_records_ctx},
    {"KFD_QUEUE", &kfd_queue_records_ctx},
};

auto buffers = std::array<rocprofiler_buffer_id_t*, 22>{&runtime_init_buffered_buffer,
                                                        &hsa_api_buffered_buffer,
                                                        &hip_api_buffered_buffer,
                                                        &marker_api_buffered_buffer,
                                                        &kernel_dispatch_buffer,
                                                        &memory_copy_buffer,
                                                        &memory_allocation_buffer,
                                                        &scratch_memory_buffer,
                                                        &counter_collection_buffer,
                                                        &corr_id_retire_buffer,
                                                        &rccl_api_ext_buffered_buffer,
                                                        &ompt_buffered_buffer,
                                                        &rocdecode_api_buffer,
                                                        &rocdecode_api_ext_buffer,
                                                        &rocjpeg_api_buffer,
                                                        &kfd_page_fault_event_buffer,
                                                        &kfd_queue_event_buffer,
                                                        &kfd_unmap_from_gpu_event_buffer,
                                                        &kfd_droped_events_event_buffer,
                                                        &kfd_page_migrate_records_buffer,
                                                        &kfd_page_fault_records_buffer,
                                                        &kfd_queue_records_buffer};

auto agents     = std::vector<rocprofiler_agent_t>{};
auto agents_map = std::unordered_map<rocprofiler_agent_id_t, rocprofiler_agent_t>{};

rocprofiler_timestamp_t init_time            = 0;
rocprofiler_timestamp_t fini_time            = 0;
rocprofiler_thread_id_t main_tid             = 0;
auto                    kfd_configure_status = ROCPROFILER_STATUS_SUCCESS;

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    rocprofiler_get_timestamp(&init_time);
    rocprofiler_get_thread_id(&main_tid);

    assert(tool_data != nullptr);

    rocprofiler_query_available_agents_cb_t iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                                                            const void**                agents_arr,
                                                            size_t                      num_agents,
                                                            void*                       user_data) {
        if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
            throw std::runtime_error{"unexpected rocprofiler agent version"};
        auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(user_data);
        for(size_t i = 0; i < num_agents; ++i)
            agents_v->emplace_back(*static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]));
        return ROCPROFILER_STATUS_SUCCESS;
    };

    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           iterate_cb,
                                           sizeof(rocprofiler_agent_t),
                                           const_cast<void*>(static_cast<const void*>(&agents))),
        "query available agents");

    for(auto itr : agents)
        agents_map.emplace(itr.id, itr);

    auto* call_stack_v = static_cast<call_stack_t*>(tool_data);

    call_stack_v->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    client_fini_func = fini_func;

    for(auto itr : contexts)
    {
        ROCPROFILER_CALL(rocprofiler_create_context(itr.second), "context creation");
        ROCPROFILER_CALL(rocprofiler_configure_external_correlation_id_request_service(
                             *itr.second, nullptr, 0, set_external_correlation_id, nullptr),
                         "external correlation id request service configure");
    }

    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         runtime_init_callback_ctx,
                         ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION,
                         nullptr,
                         0,
                         tool_tracing_callback,
                         nullptr),
                     "runtime init callback tracing service configure");

    for(auto itr : {ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API,
                    ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
                    ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API,
                    ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API})
    {
        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                             hsa_api_callback_ctx, itr, nullptr, 0, tool_tracing_callback, nullptr),
                         "hsa api callback tracing service configure");
    }

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(hip_api_callback_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       nullptr),
        "hip runtime api callback tracing service configure");

    // ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
    //                      hip_api_callback_ctx,
    //                      ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API,
    //                      nullptr,
    //                      0,
    //                      tool_tracing_callback,
    //                      nullptr),
    //                  "hip compiler api callback tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(code_object_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       nullptr),
        "code object tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(marker_api_callback_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       nullptr),
        "marker core api tracing service configure");

    // ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
    //                      marker_api_callback_ctx,
    //                      ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_RANGE_API,
    //                      nullptr,
    //                      0,
    //                      tool_tracing_callback,
    //                      nullptr),
    //                  "marker core api tracing service configure");

    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         marker_api_callback_ctx,
                         ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                         nullptr,
                         0,
                         tool_tracing_callback,
                         nullptr),
                     "marker control api tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(marker_api_callback_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       nullptr),
        "marker name api tracing service configure");

    auto kernel_dispatch_cb_ops = make_array<rocprofiler_tracing_operation_t>(
        ROCPROFILER_KERNEL_DISPATCH_ENQUEUE, ROCPROFILER_KERNEL_DISPATCH_COMPLETE);

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(kernel_dispatch_callback_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                       kernel_dispatch_cb_ops.data(),
                                                       kernel_dispatch_cb_ops.size(),
                                                       tool_tracing_callback,
                                                       nullptr),
        "kernel dispatch callback tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(memory_copy_callback_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       nullptr),
        "memory copy callback tracing service configure");

    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         memory_allocation_callback_ctx,
                         ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION,
                         nullptr,
                         0,
                         tool_tracing_callback,
                         nullptr),
                     "memory allocation callback tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(scratch_memory_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_SCRATCH_MEMORY,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       nullptr),
        "scratch memory tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(rccl_api_callback_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_RCCL_API,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       nullptr),
        "rccl api callback tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(rocdecode_api_callback_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       nullptr),
        "rocdecode api callback tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(rocjpeg_api_callback_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       nullptr),
        "rocjpeg api callback tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(ompt_callback_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_OMPT,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       nullptr),
        "ompt callback tracing service configure");

    constexpr auto buffer_size = 8192;
    constexpr auto watermark   = 7936;

    ROCPROFILER_CALL(rocprofiler_create_buffer(runtime_init_buffered_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &runtime_init_buffered_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(hsa_api_buffered_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &hsa_api_buffered_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(hip_api_buffered_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &hip_api_buffered_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(marker_api_buffered_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &marker_api_buffered_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(kernel_dispatch_buffered_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &kernel_dispatch_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(memory_copy_buffered_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &memory_copy_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(memory_allocation_buffered_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &memory_allocation_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(scratch_memory_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &scratch_memory_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(corr_id_retire_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &corr_id_retire_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(counter_collection_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &counter_collection_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(rccl_api_ext_buffered_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &rccl_api_ext_buffered_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(rocdecode_api_buffered_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &rocdecode_api_buffer),
                     "buffer creation");
    ROCPROFILER_CALL(rocprofiler_create_buffer(rocdecode_api_ext_buffered_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &rocdecode_api_ext_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(rocjpeg_api_buffered_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &rocjpeg_api_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(ompt_buffered_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &ompt_buffered_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(page_migrate_event_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &page_migrate_event_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(kfd_page_fault_event_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &kfd_page_fault_event_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(kfd_queue_event_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &kfd_queue_event_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(kfd_unmap_from_gpu_event_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &kfd_unmap_from_gpu_event_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(kfd_droped_events_event_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &kfd_droped_events_event_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(kfd_page_migrate_records_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &kfd_page_migrate_records_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(kfd_page_fault_records_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &kfd_page_fault_records_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(kfd_queue_records_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &kfd_queue_records_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
                         runtime_init_buffered_ctx,
                         ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION,
                         nullptr,
                         0,
                         runtime_init_buffered_buffer),
                     "buffer tracing service configure");

    for(auto itr : {ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,
                    ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API,
                    ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API,
                    ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API})
    {
        ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
                             hsa_api_buffered_ctx, itr, nullptr, 0, hsa_api_buffered_buffer),
                         "buffer tracing service configure");
    }

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(hip_api_buffered_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
                                                     nullptr,
                                                     0,
                                                     hip_api_buffered_buffer),
        "buffer tracing service configure");

    // ROCPROFILER_CALL(
    //     rocprofiler_configure_buffer_tracing_service(hip_api_buffered_ctx,
    //                                                  ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API,
    //                                                  nullptr,
    //                                                  0,
    //                                                  hip_api_buffered_buffer),
    //     "buffer tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(marker_api_buffered_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API,
                                                     nullptr,
                                                     0,
                                                     marker_api_buffered_buffer),
        "buffer tracing service configure");

    // ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
    //                      marker_api_buffered_ctx,
    //                      ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API,
    //                      nullptr,
    //                      0,
    //                      marker_api_buffered_buffer),
    //                  "buffer tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(marker_api_buffered_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API,
                                                     nullptr,
                                                     0,
                                                     marker_api_buffered_buffer),
        "buffer tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(marker_api_buffered_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API,
                                                     nullptr,
                                                     0,
                                                     marker_api_buffered_buffer),
        "buffer tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(kernel_dispatch_buffered_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                     nullptr,
                                                     0,
                                                     kernel_dispatch_buffer),
        "buffer tracing service for kernel dispatch configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(memory_copy_buffered_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                                                     nullptr,
                                                     0,
                                                     memory_copy_buffer),
        "buffer tracing service for memory copy configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(memory_allocation_buffered_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION,
                                                     nullptr,
                                                     0,
                                                     memory_allocation_buffer),
        "buffer tracing service for memory allocation configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(scratch_memory_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY,
                                                     nullptr,
                                                     0,
                                                     scratch_memory_buffer),
        "buffer tracing service for scratch memory configure");

    {
        kfd_configure_status = rocprofiler_configure_buffer_tracing_service(
            page_migrate_event_ctx,
            ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE,
            nullptr,
            0,
            page_migrate_event_buffer);

        constexpr auto message = "buffer tracing service for page migration configure";
        if(kfd_configure_status == ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL)
            std::cerr << message
                      << " failed: " << rocprofiler_get_status_string(kfd_configure_status)
                      << std::endl;
        else
            ROCPROFILER_CALL(kfd_configure_status, message);
    }

    {
        kfd_configure_status = rocprofiler_configure_buffer_tracing_service(
            kfd_page_fault_event_ctx,
            ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT,
            nullptr,
            0,
            kfd_page_fault_event_buffer);

        constexpr auto message = "buffer tracing service for page migration configure";
        if(kfd_configure_status == ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL)
            std::cerr << message
                      << " failed: " << rocprofiler_get_status_string(kfd_configure_status)
                      << std::endl;
        else
            ROCPROFILER_CALL(kfd_configure_status, message);
    }

    {
        kfd_configure_status =
            rocprofiler_configure_buffer_tracing_service(kfd_queue_event_ctx,
                                                         ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE,
                                                         nullptr,
                                                         0,
                                                         kfd_queue_event_buffer);

        constexpr auto message = "buffer tracing service for page migration configure";
        if(kfd_configure_status == ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL)
            std::cerr << message
                      << " failed: " << rocprofiler_get_status_string(kfd_configure_status)
                      << std::endl;
        else
            ROCPROFILER_CALL(kfd_configure_status, message);
    }

    {
        kfd_configure_status = rocprofiler_configure_buffer_tracing_service(
            kfd_unmap_from_gpu_event_ctx,
            ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
            nullptr,
            0,
            kfd_unmap_from_gpu_event_buffer);

        constexpr auto message = "buffer tracing service for page migration configure";
        if(kfd_configure_status == ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL)
            std::cerr << message
                      << " failed: " << rocprofiler_get_status_string(kfd_configure_status)
                      << std::endl;
        else
            ROCPROFILER_CALL(kfd_configure_status, message);
    }

    {
        kfd_configure_status = rocprofiler_configure_buffer_tracing_service(
            kfd_droped_events_event_ctx,
            ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS,
            nullptr,
            0,
            kfd_droped_events_event_buffer);

        constexpr auto message = "buffer tracing service for page migration configure";
        if(kfd_configure_status == ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL)
            std::cerr << message
                      << " failed: " << rocprofiler_get_status_string(kfd_configure_status)
                      << std::endl;
        else
            ROCPROFILER_CALL(kfd_configure_status, message);
    }

    {
        kfd_configure_status = rocprofiler_configure_buffer_tracing_service(
            kfd_page_migrate_records_ctx,
            ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE,
            nullptr,
            0,
            kfd_page_migrate_records_buffer);

        constexpr auto message = "buffer tracing service for page migration configure";
        if(kfd_configure_status == ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL)
            std::cerr << message
                      << " failed: " << rocprofiler_get_status_string(kfd_configure_status)
                      << std::endl;
        else
            ROCPROFILER_CALL(kfd_configure_status, message);
    }

    {
        kfd_configure_status =
            rocprofiler_configure_buffer_tracing_service(kfd_page_fault_records_ctx,
                                                         ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT,
                                                         nullptr,
                                                         0,
                                                         kfd_page_fault_records_buffer);

        constexpr auto message = "buffer tracing service for page migration configure";
        if(kfd_configure_status == ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL)
            std::cerr << message
                      << " failed: " << rocprofiler_get_status_string(kfd_configure_status)
                      << std::endl;
        else
            ROCPROFILER_CALL(kfd_configure_status, message);
    }

    {
        kfd_configure_status =
            rocprofiler_configure_buffer_tracing_service(kfd_queue_records_ctx,
                                                         ROCPROFILER_BUFFER_TRACING_KFD_QUEUE,
                                                         nullptr,
                                                         0,
                                                         kfd_queue_records_buffer);

        constexpr auto message = "buffer tracing service for page migration configure";
        if(kfd_configure_status == ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL)
            std::cerr << message
                      << " failed: " << rocprofiler_get_status_string(kfd_configure_status)
                      << std::endl;
        else
            ROCPROFILER_CALL(kfd_configure_status, message);
    }

    ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
                         corr_id_retire_ctx,
                         ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT,
                         nullptr,
                         0,
                         corr_id_retire_buffer),
                     "buffer tracing service for memory copy configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(rccl_api_ext_buffered_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_RCCL_API_EXT,
                                                     nullptr,
                                                     0,
                                                     rccl_api_ext_buffered_buffer),
        "buffer tracing service for rccl api configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(rocdecode_api_buffered_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_ROCDECODE_API,
                                                     nullptr,
                                                     0,
                                                     rocdecode_api_buffer),
        "buffer tracing service for rocdecode api configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(rocdecode_api_ext_buffered_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_ROCDECODE_API_EXT,
                                                     nullptr,
                                                     0,
                                                     rocdecode_api_ext_buffer),
        "buffer tracing service for rocdecode ext api configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(rocjpeg_api_buffered_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_ROCJPEG_API,
                                                     nullptr,
                                                     0,
                                                     rocjpeg_api_buffer),
        "buffer tracing service for rocjpeg api configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            ompt_buffered_ctx, ROCPROFILER_BUFFER_TRACING_OMPT, nullptr, 0, ompt_buffered_buffer),
        "buffer tracing service for ompt configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_dispatch_counting_service(
            counter_collection_ctx, counter_collection_buffer, dispatch_callback, nullptr),
        "setup buffered service");

    for(auto* itr : buffers)
    {
        if(itr->handle == 0) continue;

        auto client_thread = rocprofiler_callback_thread_t{};
        ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                         "creating callback thread");

        ROCPROFILER_CALL(rocprofiler_assign_callback_thread(*itr, client_thread),
                         "assignment of thread for buffer");
    }

    for(auto itr : contexts)
    {
        int valid_ctx = 0;
        ROCPROFILER_CALL(rocprofiler_context_is_valid(*itr.second, &valid_ctx),
                         "context validity check");
        if(valid_ctx == 0)
        {
            // notify rocprofiler that initialization failed
            // and all the contexts, buffers, etc. created
            // should be ignored
            return -1;
        }
    }

    // environment variable to select which contexts to collect
    auto* context_settings_env = getenv("ROCPROFILER_TOOL_CONTEXTS");
    if(context_settings_env != nullptr && !std::string_view{context_settings_env}.empty())
    {
        auto context_settings = std::string{context_settings_env};

        // ignore case
        for(auto& itr : context_settings)
            itr = toupper(itr);

        // if context is not in string, set the pointer to null in the contexts array
        auto options = std::stringstream{};
        for(auto& itr : contexts)
        {
            options << "\n\t- " << itr.first;
            auto pos = context_settings.find(itr.first);
            if(pos == std::string::npos)
            {
                std::cerr << "Excluding context: " << itr.first << std::endl;
                itr.second = nullptr;
            }
            else
            {
                std::cerr << "Enabling context: " << itr.first << std::endl;
                context_settings.erase(pos, itr.first.length());
            }
        }

        // detect if there are any invalid entries
        if(context_settings.find_first_not_of(" ,;:\t\n\r") != std::string::npos)
        {
            auto filename = std::string_view{__FILE__};
            auto msg      = std::stringstream{};
            msg << "[rocprofiler-sdk-json-tool][" << filename.substr(filename.find_last_of('/') + 1)
                << ":" << __LINE__ << "] invalid specification of ROCPROFILER_TOOL_CONTEXTS ('"
                << context_settings_env << "'). Valid choices are: " << options.str()
                << "\nRemainder: " << context_settings << "\n";
            throw std::runtime_error{msg.str()};
        }
    }

    auto* context_settings_excl_env = getenv("ROCPROFILER_TOOL_CONTEXTS_EXCLUDE");
    if(context_settings_excl_env != nullptr && !std::string_view{context_settings_excl_env}.empty())
    {
        auto context_settings = std::string{context_settings_excl_env};

        // ignore case
        for(auto& itr : context_settings)
            itr = toupper(itr);

        // if context is not in string, set the pointer to null in the contexts array
        auto options = std::stringstream{};
        for(auto& itr : contexts)
        {
            options << "\n\t- " << itr.first;
            auto pos = context_settings.find(itr.first);
            if(pos != std::string::npos)
            {
                std::cerr << "Excluding context: " << itr.first << std::endl;
                itr.second = nullptr;
                context_settings.erase(pos, itr.first.length());
            }
        }

        // detect if there are any invalid entries
        if(context_settings.find_first_not_of(" ,;:\t\n\r") != std::string::npos)
        {
            auto filename = std::string_view{__FILE__};
            auto msg      = std::stringstream{};
            msg << "[rocprofiler-sdk-json-tool][" << filename.substr(filename.find_last_of('/') + 1)
                << ":" << __LINE__
                << "] invalid specification of ROCPROFILER_TOOL_CONTEXTS_EXCLUDE ('"
                << context_settings_excl_env << "'). Valid choices are: " << options.str()
                << "\nRemainder: " << context_settings << "\n";
            throw std::runtime_error{msg.str()};
        }
    }

    push_external_correlation(getppid());

    start();

    // no errors
    return 0;
}

void
write_json(call_stack_t* _call_stack);

void
write_perfetto();

void
tool_fini(void* tool_data)
{
    static std::atomic_flag _once = ATOMIC_FLAG_INIT;
    if(_once.test_and_set()) return;

    stop();
    flush();

    pop_external_correlation();

    rocprofiler_get_timestamp(&fini_time);

    std::cerr << "[" << getpid() << "][" << __FUNCTION__
              << "] Finalizing... agents=" << agents.size()
              << ", runtime_init_callback_records=" << runtime_init_cb_records.size()
              << ", code_object_callback_records=" << code_object_records.size()
              << ", kernel_symbol_callback_records=" << kernel_symbol_records.size()
              << ", host_function_callback_records=" << host_function_records.size()
              << ", hsa_api_callback_records=" << hsa_api_cb_records.size()
              << ", hip_api_callback_records=" << hip_api_cb_records.size()
              << ", marker_api_callback_records=" << marker_api_cb_records.size()
              << ", scratch_memory_callback_records=" << scratch_memory_cb_records.size()
              << ", kernel_dispatch_callback_records=" << kernel_dispatch_cb_records.size()
              << ", memory_copy_callback_records=" << memory_copy_cb_records.size()
              << ", memory_allocation_callback_records=" << memory_allocation_cb_records.size()
              << ", rccl_api_callback_records=" << rccl_api_cb_records.size()
              << ", ompt_callback_records=" << ompt_cb_records.size()
              << ", kernel_dispatch_bf_records=" << kernel_dispatch_bf_records.size()
              << ", memory_copy_bf_records=" << memory_copy_bf_records.size()
              << ", memory_allocation_bf_records=" << memory_allocation_bf_records.size()
              << ", scratch_memory_records=" << scratch_memory_records.size()
              << ", kfd_page_migrate_event=" << kfd_page_migrate_event_records.size()
              << ", kfd_page_fault_event=" << kfd_page_fault_event_records.size()
              << ", kfd_queue_event=" << kfd_queue_event_records.size()
              << ", kfd_unmap_from_gpu_event=" << kfd_unmap_from_gpu_event_records.size()
              << ", kfd_droped_events_event=" << kfd_dropped_events_event_records.size()
              << ", kfd_page_migrate_record=" << kfd_page_migrate_records.size()
              << ", kfd_page_fault_record=" << kfd_page_fault_records.size()
              << ", kfd_queue_record=" << kfd_queue_records.size()
              << ", runtime_init_bf_records=" << runtime_init_bf_records.size()
              << ", hsa_api_bf_records=" << hsa_api_bf_records.size()
              << ", hip_api_bf_records=" << hip_api_bf_records.size()
              << ", marker_api_bf_records=" << marker_api_bf_records.size()
              << ", corr_id_retire_records=" << corr_id_retire_records.size()
              << ", rccl_api_ext_bf_records=" << rccl_api_ext_bf_records.size()
              << ", ompt_bf_records=" << ompt_bf_records.size()
              << ", counter_collection_value_records=" << counter_collection_bf_records.size()
              << ", rocdecode_api_callback_records=" << rocdecode_api_cb_records.size()
              << ", rocdecode_api_bf_records=" << rocdecode_api_bf_records.size()
              << ", rocdecode_api_ext_bf_records=" << rocdecode_api_ext_bf_records.size()
              << ", rocjpeg_api_callback_records=" << rocjpeg_api_cb_records.size()
              << ", rocjpeg_api_bf_records=" << rocjpeg_api_bf_records.size() << "...\n"
              << std::flush;

    auto* _call_stack = static_cast<call_stack_t*>(tool_data);
    if(_call_stack)
    {
        _call_stack->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});
    }

    write_json(_call_stack);
    write_perfetto();

    std::cerr << "[" << getpid() << "][" << __FUNCTION__ << "] Finalization complete.\n"
              << std::flush;

    delete _call_stack;
}

std::string
get_filename(std::string fname, std::string extension)
{
    auto ofname = fname + "." + extension;
    if(auto* eofname = getenv("ROCPROFILER_TOOL_OUTPUT_FILE")) ofname = eofname;

    if(getenv("ROCPROFILER_TOOL_OUTPUT_FILE_APPEND_PID"))
    {
        const auto  fpath = common::fs::path(ofname);
        std::string filename_buf(512, '\0');

        if(fpath.has_extension())
        {
            std::snprintf(filename_buf.data(),
                          filename_buf.size() - 1,
                          "%s-%d.%s",
                          fpath.stem().c_str(),
                          getpid(),
                          fpath.extension().c_str());
        }
        else
        {
            std::snprintf(filename_buf.data(),
                          filename_buf.size() - 1,
                          "%s-%d.json",
                          fpath.stem().c_str(),
                          getpid());
        }
        return filename_buf;
    }
    else
    {
        return ofname;
    }
}

void
write_json(call_stack_t* _call_stack)
{
    auto ofname = get_filename("rocprofiler-tool-results", "json");

    std::ostream* ofs     = nullptr;
    auto          cleanup = std::function<void(std::ostream*&)>{};
    if(ofname == "stdout")
        ofs = &std::cout;
    else if(ofname == "stderr")
        ofs = &std::cerr;
    else
    {
        ofs = new std::ofstream{ofname};
        if(ofs && *ofs)
        {
            std::cerr << "[" << getpid() << "][" << __FUNCTION__
                      << "] Outputting collected data to " << ofname << "...\n"
                      << std::flush;
            cleanup = [](std::ostream*& _os) { delete _os; };
        }
        else
        {
            std::cerr << "Error outputting to " << ofname << ". Redirecting to stderr...\n"
                      << std::flush;
            ofname = "stderr";
            ofs    = &std::cerr;
        }
    }

    {
        namespace sdk           = ::rocprofiler::sdk;
        using JSONOutputArchive = cereal::MinimalJSONOutputArchive;

        constexpr auto json_prec    = 32;
        constexpr auto json_indent  = JSONOutputArchive::Options::IndentChar::space;
        auto           json_opts    = JSONOutputArchive::Options{json_prec, json_indent, 1};
        auto           json_ar      = JSONOutputArchive{*ofs, json_opts};
        auto           buffer_names = sdk::get_buffer_tracing_names();
        auto           callbk_names = sdk::get_callback_tracing_names();
        auto           validate_kfd_events =
            (kfd_configure_status != ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL);

        json_ar.setNextName("rocprofiler-sdk-json-tool");
        json_ar.startNode();

        json_ar.setNextName("metadata");
        json_ar.startNode();
        json_ar(cereal::make_nvp("pid", getpid()));
        json_ar(cereal::make_nvp("main_tid", main_tid));
        json_ar(cereal::make_nvp("init_time", init_time));
        json_ar(cereal::make_nvp("fini_time", fini_time));
        json_ar(cereal::make_nvp("validate_kfd_events", validate_kfd_events));
        json_ar.finishNode();

        json_ar(cereal::make_nvp("agents", agents));
        json_ar(cereal::make_nvp("counter_info", counter_info));
        if(_call_stack) json_ar(cereal::make_nvp("call_stack", *_call_stack));

        json_ar.setNextName("callback_records");
        json_ar.startNode();
        try
        {
            json_ar(cereal::make_nvp("names", callbk_names));
            json_ar(cereal::make_nvp("runtime_init", runtime_init_cb_records));
            json_ar(cereal::make_nvp("code_objects", code_object_records));
            json_ar(cereal::make_nvp("kernel_symbols", kernel_symbol_records));
            json_ar(cereal::make_nvp("host_functions", host_function_records));
            json_ar(cereal::make_nvp("hsa_api_traces", hsa_api_cb_records));
            json_ar(cereal::make_nvp("hip_api_traces", hip_api_cb_records));
            json_ar(cereal::make_nvp("marker_api_traces", marker_api_cb_records));
            json_ar(cereal::make_nvp("rccl_api_traces", rccl_api_cb_records));
            json_ar(cereal::make_nvp("ompt_traces", ompt_cb_records));
            json_ar(cereal::make_nvp("scratch_memory_traces", scratch_memory_cb_records));
            json_ar(cereal::make_nvp("kernel_dispatch", kernel_dispatch_cb_records));
            json_ar(cereal::make_nvp("memory_copies", memory_copy_cb_records));
            json_ar(cereal::make_nvp("memory_allocations", memory_allocation_cb_records));
            json_ar(cereal::make_nvp("rocdecode_api_traces", rocdecode_api_cb_records));
            json_ar(cereal::make_nvp("rocjpeg_api_traces", rocjpeg_api_cb_records));
        } catch(std::exception& e)
        {
            std::cerr << "[" << getpid() << "][" << __FUNCTION__
                      << "] threw an exception: " << e.what() << "\n"
                      << std::flush;
        }
        json_ar.finishNode();

        json_ar.setNextName("buffer_records");
        json_ar.startNode();
        try
        {
            json_ar(cereal::make_nvp("names", buffer_names));
            json_ar(cereal::make_nvp("runtime_init", runtime_init_bf_records));
            json_ar(cereal::make_nvp("kernel_dispatch", kernel_dispatch_bf_records));
            json_ar(cereal::make_nvp("memory_copies", memory_copy_bf_records));
            json_ar(cereal::make_nvp("memory_allocations", memory_allocation_bf_records));
            json_ar(cereal::make_nvp("scratch_memory_traces", scratch_memory_records));
            json_ar(cereal::make_nvp("kfd_page_migrate_events", kfd_page_migrate_event_records));
            json_ar(cereal::make_nvp("kfd_page_fault_events", kfd_page_fault_event_records));
            json_ar(cereal::make_nvp("kfd_queue_events", kfd_queue_event_records));
            json_ar(
                cereal::make_nvp("kfd_unmap_from_gpu_events", kfd_unmap_from_gpu_event_records));
            json_ar(cereal::make_nvp("kfd_droped_events", kfd_dropped_events_event_records));
            json_ar(cereal::make_nvp("kfd_page_migrate_records", kfd_page_migrate_records));
            json_ar(cereal::make_nvp("kfd_page_fault_records", kfd_page_fault_records));
            json_ar(cereal::make_nvp("kfd_queue_records", kfd_queue_records));
            json_ar(cereal::make_nvp("hsa_api_traces", hsa_api_bf_records));
            json_ar(cereal::make_nvp("hip_api_traces", hip_api_bf_records));
            json_ar(cereal::make_nvp("marker_api_traces", marker_api_bf_records));
            json_ar(cereal::make_nvp("rccl_api_traces", rccl_api_ext_bf_records));
            json_ar(cereal::make_nvp("ompt_traces", ompt_bf_records));
            json_ar(cereal::make_nvp("retired_correlation_ids", corr_id_retire_records));
            json_ar(cereal::make_nvp("counter_collection", counter_collection_bf_records));
            json_ar(cereal::make_nvp("rocdecode_api_traces", rocdecode_api_bf_records));
            json_ar(cereal::make_nvp("rocdecode_api_ext_traces", rocdecode_api_ext_bf_records));
            json_ar(cereal::make_nvp("rocjpeg_api_traces", rocjpeg_api_bf_records));
        } catch(std::exception& e)
        {
            std::cerr << "[" << getpid() << "][" << __FUNCTION__
                      << "] threw an exception: " << e.what() << "\n"
                      << std::flush;
        }
        json_ar.finishNode();

        json_ar.finishNode();
    }

    *ofs << std::flush;

    if(cleanup) cleanup(ofs);
}

void
write_perfetto()
{
    auto args            = ::perfetto::TracingInitArgs{};
    auto track_event_cfg = ::perfetto::protos::gen::TrackEventConfig{};
    auto cfg             = ::perfetto::TraceConfig{};

    // Enabled by default, set this env to any value to disable
    bool enable_debug_annotations = []() {
        return getenv("ROCPROFILER_DISABLE_PERFETTO_ANNOTATIONS") == nullptr;
    }();

    // environment settings
    auto shmem_size_hint = size_t{64};
    auto buffer_size_kb  = size_t{1024000};

    auto* buffer_config = cfg.add_buffers();
    buffer_config->set_size_kb(buffer_size_kb);
    buffer_config->set_fill_policy(
        ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_DISCARD);

    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");  // this MUST be track_event
    ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

    args.shmem_size_hint_kb = shmem_size_hint;
    args.backends |= ::perfetto::kInProcessBackend;

    ::perfetto::Tracing::Initialize(args);
    ::perfetto::TrackEvent::Register();

    auto tracing_session = ::perfetto::Tracing::NewTrace();

    tracing_session->Setup(cfg);
    tracing_session->StartBlocking();

    auto tids            = std::set<rocprofiler_thread_id_t>{};
    auto agent_ids       = std::set<uint64_t>{};
    auto agent_ids_alloc = std::set<uint64_t>{};
    auto agent_queue_ids = std::map<uint64_t, std::set<uint64_t>>{};

    auto _get_agent = [](uint64_t id_handle) -> const rocprofiler_agent_t* {
        for(const auto& itr : agents)
        {
            if(id_handle == itr.id.handle) return &itr;
        }
        return nullptr;
    };

    {
        for(const auto& itr : hsa_api_bf_records)
            tids.emplace(itr.thread_id);
        for(const auto& itr : hip_api_bf_records)
            tids.emplace(itr.thread_id);
        for(const auto& itr : marker_api_bf_records)
            tids.emplace(itr.thread_id);
        for(const auto& itr : rccl_api_ext_bf_records)
            tids.emplace(itr.thread_id);
        for(const auto& itr : ompt_bf_records)
            tids.emplace(itr.thread_id);
        for(const auto& itr : rocdecode_api_bf_records)
            tids.emplace(itr.thread_id);
        for(const auto& itr : rocjpeg_api_bf_records)
            tids.emplace(itr.thread_id);

        for(const auto& itr : memory_copy_bf_records)
        {
            tids.emplace(itr.thread_id);
            agent_ids.emplace(itr.dst_agent_id.handle);
            agent_ids.emplace(itr.src_agent_id.handle);
        }

        for(const auto& itr : memory_allocation_bf_records)
        {
            tids.emplace(itr.thread_id);
            agent_ids_alloc.emplace(itr.agent_id.handle);
        }

        for(const auto& itr : kernel_dispatch_bf_records)
        {
            tids.emplace(itr.thread_id);
            agent_queue_ids[itr.dispatch_info.agent_id.handle].emplace(
                itr.dispatch_info.queue_id.handle);
        }
    }

    auto thread_tracks = std::unordered_map<rocprofiler_thread_id_t, ::perfetto::Track>{};

    uint64_t nthrn = 0;
    for(const auto& itr : tids)
    {
        if(itr == main_tid)
            thread_tracks.emplace(main_tid, ::perfetto::ThreadTrack::Current());
        else
        {
            auto _track  = ::perfetto::Track{itr};
            auto _desc   = _track.Serialize();
            auto _namess = std::stringstream{};
            _namess << "Thread " << ++nthrn << " (" << itr << ")";
            _desc.set_name(_namess.str());
            perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

            thread_tracks.emplace(itr, _track);
        }
    }

    auto agent_tracks = std::unordered_map<uint64_t, ::perfetto::Track>{};

    for(const auto& itr : agent_ids)
    {
        const auto* _agent = _get_agent(itr);
        if(!_agent) throw std::runtime_error{"agent lookup error"};

        auto _namess = std::stringstream{};

        if(_agent->type == ROCPROFILER_AGENT_TYPE_CPU)
            _namess << "CPU COPY [" << itr << "] ";
        else if(_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
            _namess << "GPU COPY [" << itr << "] ";

        if(!std::string_view{_agent->model_name}.empty())
            _namess << _agent->model_name;
        else
            _namess << _agent->product_name;

        auto _track = ::perfetto::Track{get_hash_id(_namess.str())};
        auto _desc  = _track.Serialize();
        _desc.set_name(_namess.str());

        perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

        agent_tracks.emplace(itr, _track);
    }

    for(const auto& itr : agent_ids_alloc)
    {
        const auto* _agent  = _get_agent(itr);
        auto        _namess = std::stringstream{};

        if(_agent != nullptr)
        {
            if(_agent->type == ROCPROFILER_AGENT_TYPE_CPU)
                _namess << "CPU MEMORY OPERATION [" << itr << "] ";
            else if(_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                _namess << "GPU MEMORY OPERATION [" << itr << "] ";

            if(!std::string_view{_agent->model_name}.empty())
                _namess << _agent->model_name;
            else
                _namess << _agent->product_name;
        }
        else
        {
            _namess << "UNKNOWN MEMORY OPERATION [" << itr << "] ";
        }
        auto _track = ::perfetto::Track{get_hash_id(_namess.str())};
        auto _desc  = _track.Serialize();
        _desc.set_name(_namess.str());

        perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

        agent_tracks.emplace(itr, _track);
    }

    auto agent_queue_tracks =
        std::unordered_map<uint64_t, std::unordered_map<uint64_t, ::perfetto::Track>>{};

    for(const auto& aitr : agent_queue_ids)
    {
        uint32_t nqueue = 0;
        for(const auto& qitr : aitr.second)
        {
            const auto* _agent = _get_agent(aitr.first);
            if(!_agent) throw std::runtime_error{"agent lookup error"};

            auto _namess = std::stringstream{};

            if(_agent->type == ROCPROFILER_AGENT_TYPE_CPU)
                _namess << "CPU COMPUTE [" << aitr.first << "] ";
            else if(_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                _namess << "GPU COMPUTE [" << aitr.first << "] ";

            _namess << " Queue [" << nqueue++ << "]";

            auto _track = ::perfetto::Track{get_hash_id(_namess.str())};
            auto _desc  = _track.Serialize();
            _desc.set_name(_namess.str());

            perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

            agent_queue_tracks[aitr.first].emplace(qitr, _track);
        }
    }

    {
        namespace sdk = ::rocprofiler::sdk;

        auto buffer_names     = sdk::get_buffer_tracing_names();
        auto callbk_name_info = sdk::get_callback_tracing_names();

        for(const auto& itr : hsa_api_bf_records)
        {
            auto  name  = buffer_names.at(itr.kind, itr.operation);
            auto& track = thread_tracks.at(itr.thread_id);

            auto _args = callback_arg_array_t{};
            if(enable_debug_annotations)
            {
                auto ritr = std::find_if(
                    hsa_api_cb_records.begin(), hsa_api_cb_records.end(), [&itr](const auto& citr) {
                        return (citr.record.correlation_id.internal ==
                                    itr.correlation_id.internal &&
                                !citr.args.empty());
                    });
                if(ritr != hsa_api_cb_records.end()) _args = ritr->args;
            }

            TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::hsa_api>::name,
                              ::perfetto::StaticString(name.data()),
                              track,
                              itr.start_timestamp,
                              ::perfetto::Flow::ProcessScoped(itr.correlation_id.internal),
                              "begin_ns",
                              itr.start_timestamp,
                              "tid",
                              itr.thread_id,
                              "kind",
                              itr.kind,
                              "operation",
                              itr.operation,
                              "corr_id",
                              itr.correlation_id.internal,
                              "ancestor_id",
                              itr.correlation_id.ancestor,
                              [&](::perfetto::EventContext ctx) {
                                  for(const auto& aitr : _args)
                                      sdk::add_perfetto_annotation(ctx, aitr.first, aitr.second);
                              });
            TRACE_EVENT_END(sdk::perfetto_category<sdk::category::hsa_api>::name,
                            track,
                            itr.end_timestamp,
                            "end_ns",
                            itr.end_timestamp);
        }

        for(const auto& itr : hip_api_bf_records)
        {
            auto  name  = buffer_names.at(itr.kind, itr.operation);
            auto& track = thread_tracks.at(itr.thread_id);

            auto _args = callback_arg_array_t{};
            if(enable_debug_annotations)
            {
                auto ritr = std::find_if(
                    hip_api_cb_records.begin(), hip_api_cb_records.end(), [&itr](const auto& citr) {
                        return (citr.record.correlation_id.internal ==
                                    itr.correlation_id.internal &&
                                !citr.args.empty());
                    });
                if(ritr != hip_api_cb_records.end()) _args = ritr->args;
            }

            TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::hip_api>::name,
                              ::perfetto::StaticString(name.data()),
                              track,
                              itr.start_timestamp,
                              ::perfetto::Flow::ProcessScoped(itr.correlation_id.internal),
                              "begin_ns",
                              itr.start_timestamp,
                              "tid",
                              itr.thread_id,
                              "kind",
                              itr.kind,
                              "operation",
                              itr.operation,
                              "corr_id",
                              itr.correlation_id.internal,
                              "ancestor_id",
                              itr.correlation_id.ancestor,
                              [&](::perfetto::EventContext ctx) {
                                  for(const auto& aitr : _args)
                                      sdk::add_perfetto_annotation(ctx, aitr.first, aitr.second);
                              });
            TRACE_EVENT_END(sdk::perfetto_category<sdk::category::hip_api>::name,
                            track,
                            itr.end_timestamp,
                            "end_ns",
                            itr.end_timestamp);
        }

        for(const auto& itr : rccl_api_ext_bf_records)
        {
            auto  name  = buffer_names.at(itr.kind, itr.operation);
            auto& track = thread_tracks.at(itr.thread_id);

            auto _args = callback_arg_array_t{};
            if(enable_debug_annotations)
            {
                auto ritr = std::find_if(rccl_api_cb_records.begin(),
                                         rccl_api_cb_records.end(),
                                         [&itr](const auto& citr) {
                                             return (citr.record.correlation_id.internal ==
                                                         itr.correlation_id.internal &&
                                                     !citr.args.empty());
                                         });
                if(ritr != rccl_api_cb_records.end()) _args = ritr->args;
            }

            TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::rccl_api>::name,
                              ::perfetto::StaticString(name.data()),
                              track,
                              itr.start_timestamp,
                              ::perfetto::Flow::ProcessScoped(itr.correlation_id.internal),
                              "begin_ns",
                              itr.start_timestamp,
                              "tid",
                              itr.thread_id,
                              "kind",
                              itr.kind,
                              "operation",
                              itr.operation,
                              "corr_id",
                              itr.correlation_id.internal,
                              "ancestor_id",
                              itr.correlation_id.ancestor,
                              [&](::perfetto::EventContext ctx) {
                                  for(const auto& aitr : _args)
                                      sdk::add_perfetto_annotation(ctx, aitr.first, aitr.second);
                              });
            TRACE_EVENT_END(sdk::perfetto_category<sdk::category::rccl_api>::name,
                            track,
                            itr.end_timestamp,
                            "end_ns",
                            itr.end_timestamp);
        }

        for(const auto& itr : rocdecode_api_bf_records)
        {
            auto  name  = buffer_names.at(itr.kind, itr.operation);
            auto& track = thread_tracks.at(itr.thread_id);

            auto _args = callback_arg_array_t{};
            if(enable_debug_annotations)
            {
                auto ritr = std::find_if(rocdecode_api_cb_records.begin(),
                                         rocdecode_api_cb_records.end(),
                                         [&itr](const auto& citr) {
                                             return (citr.record.correlation_id.internal ==
                                                         itr.correlation_id.internal &&
                                                     !citr.args.empty());
                                         });
                if(ritr != rocdecode_api_cb_records.end()) _args = ritr->args;
            }

            TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::rocdecode_api>::name,
                              ::perfetto::StaticString(name.data()),
                              track,
                              itr.start_timestamp,
                              ::perfetto::Flow::ProcessScoped(itr.correlation_id.internal),
                              "begin_ns",
                              itr.start_timestamp,
                              "tid",
                              itr.thread_id,
                              "kind",
                              itr.kind,
                              "operation",
                              itr.operation,
                              "corr_id",
                              itr.correlation_id.internal,
                              "ancestor_id",
                              itr.correlation_id.ancestor,
                              [&](::perfetto::EventContext ctx) {
                                  for(const auto& aitr : _args)
                                      sdk::add_perfetto_annotation(ctx, aitr.first, aitr.second);
                              });
            TRACE_EVENT_END(sdk::perfetto_category<sdk::category::rocdecode_api>::name,
                            track,
                            itr.end_timestamp,
                            "end_ns",
                            itr.end_timestamp);
        }

        for(const auto& itr : rocjpeg_api_bf_records)
        {
            auto  name  = buffer_names.at(itr.kind, itr.operation);
            auto& track = thread_tracks.at(itr.thread_id);

            auto _args = callback_arg_array_t{};
            if(enable_debug_annotations)
            {
                auto ritr = std::find_if(rocjpeg_api_cb_records.begin(),
                                         rocjpeg_api_cb_records.end(),
                                         [&itr](const auto& citr) {
                                             return (citr.record.correlation_id.internal ==
                                                         itr.correlation_id.internal &&
                                                     !citr.args.empty());
                                         });
                if(ritr != rocjpeg_api_cb_records.end()) _args = ritr->args;
            }

            TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::rocjpeg_api>::name,
                              ::perfetto::StaticString(name.data()),
                              track,
                              itr.start_timestamp,
                              ::perfetto::Flow::ProcessScoped(itr.correlation_id.internal),
                              "begin_ns",
                              itr.start_timestamp,
                              "tid",
                              itr.thread_id,
                              "kind",
                              itr.kind,
                              "operation",
                              itr.operation,
                              "corr_id",
                              itr.correlation_id.internal,
                              "ancestor_id",
                              itr.correlation_id.ancestor,
                              [&](::perfetto::EventContext ctx) {
                                  for(const auto& aitr : _args)
                                      sdk::add_perfetto_annotation(ctx, aitr.first, aitr.second);
                              });
            TRACE_EVENT_END(sdk::perfetto_category<sdk::category::rocjpeg_api>::name,
                            track,
                            itr.end_timestamp,
                            "end_ns",
                            itr.end_timestamp);
        }

        for(const auto& itr : ompt_bf_records)
        {
            auto  name  = buffer_names.at(itr.kind, itr.operation);
            auto& track = thread_tracks.at(itr.thread_id);

            auto _args = callback_arg_array_t{};
            if(enable_debug_annotations)
            {
                auto ritr = std::find_if(
                    ompt_cb_records.begin(), ompt_cb_records.end(), [&itr](const auto& citr) {
                        return (citr.record.correlation_id.internal ==
                                    itr.correlation_id.internal &&
                                !citr.args.empty());
                    });
                if(ritr != ompt_cb_records.end()) _args = ritr->args;
            }

            TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::openmp>::name,
                              ::perfetto::StaticString(name.data()),
                              track,
                              itr.start_timestamp,
                              ::perfetto::Flow::ProcessScoped(itr.correlation_id.internal),
                              "begin_ns",
                              itr.start_timestamp,
                              "tid",
                              itr.thread_id,
                              "kind",
                              itr.kind,
                              "operation",
                              itr.operation,
                              "corr_id",
                              itr.correlation_id.internal,
                              "ancestor_id",
                              itr.correlation_id.ancestor,
                              [&](::perfetto::EventContext ctx) {
                                  for(const auto& aitr : _args)
                                      sdk::add_perfetto_annotation(ctx, aitr.first, aitr.second);
                              });
            TRACE_EVENT_END(sdk::perfetto_category<sdk::category::openmp>::name,
                            track,
                            itr.end_timestamp,
                            "end_ns",
                            itr.end_timestamp);
        }

        for(const auto& itr : memory_copy_bf_records)
        {
            auto  name  = buffer_names.at(itr.kind, itr.operation);
            auto& track = agent_tracks.at(itr.dst_agent_id.handle);

            TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::memory_copy>::name,
                              ::perfetto::StaticString(name.data()),
                              track,
                              itr.start_timestamp,
                              ::perfetto::Flow::ProcessScoped(itr.correlation_id.internal),
                              "begin_ns",
                              itr.start_timestamp,
                              "kind",
                              itr.kind,
                              "operation",
                              itr.operation,
                              "src_agent",
                              agents_map.at(itr.src_agent_id).logical_node_id,
                              "dst_agent",
                              agents_map.at(itr.dst_agent_id).logical_node_id,
                              "copy_bytes",
                              itr.bytes);
            TRACE_EVENT_END(sdk::perfetto_category<sdk::category::memory_copy>::name,
                            track,
                            itr.end_timestamp,
                            "end_ns",
                            itr.end_timestamp);
        }

        auto demangled = std::unordered_map<std::string_view, std::string>{};
        for(const auto& itr : kernel_dispatch_bf_records)
        {
            const auto&                            info = itr.dispatch_info;
            const kernel_symbol_callback_record_t* sym  = nullptr;
            for(const auto& kitr : kernel_symbol_records)
            {
                if(kitr.payload.kernel_id == info.kernel_id)
                {
                    sym = &kitr;
                    break;
                }
            }

            auto  name  = std::string_view{sym->payload.kernel_name};
            auto& track = agent_queue_tracks.at(info.agent_id.handle).at(info.queue_id.handle);

            if(demangled.find(name) == demangled.end())
            {
                demangled.emplace(name, demangle(name));
            }

            TRACE_EVENT_BEGIN(sdk::perfetto_category<sdk::category::kernel_dispatch>::name,
                              ::perfetto::StaticString(demangled.at(name).c_str()),
                              track,
                              itr.start_timestamp,
                              ::perfetto::Flow::ProcessScoped(itr.correlation_id.internal),
                              "begin_ns",
                              itr.start_timestamp,
                              "kind",
                              itr.kind,
                              "agent",
                              agents_map.at(info.agent_id).logical_node_id,
                              "corr_id",
                              itr.correlation_id.internal,
                              "queue",
                              info.queue_id.handle,
                              "kernel_id",
                              info.kernel_id,
                              "private_segment_size",
                              info.private_segment_size,
                              "group_segment_size",
                              info.group_segment_size,
                              "workgroup_size",
                              info.workgroup_size.x * info.workgroup_size.y * info.workgroup_size.z,
                              "grid_size",
                              info.grid_size.x * info.grid_size.y * info.grid_size.z);

            TRACE_EVENT_END(sdk::perfetto_category<sdk::category::kernel_dispatch>::name,
                            track,
                            itr.end_timestamp,
                            "end_ns",
                            itr.end_timestamp);
        }
    }

    ::perfetto::TrackEvent::Flush();
    tracing_session->FlushBlocking();
    tracing_session->StopBlocking();

    using char_vec_t = std::vector<char>;

    auto trace_data = char_vec_t{tracing_session->ReadTraceBlocking()};

    if(!trace_data.empty())
    {
        auto ofname = get_filename("rocprofiler-tool-results", "pftrace");

        auto jpos = ofname.find(".json");
        if(jpos != std::string::npos) ofname = ofname.substr(0, jpos) + std::string{".pftrace"};

        std::clog << "Writing perfetto trace file: " << ofname << std::endl;
        auto ofs = std::ofstream{ofname};
        // Write the trace into a file.
        ofs.write(trace_data.data(), trace_data.size());
    }
    else
    {
        throw std::runtime_error{"no trace data"};
    }
}

void
start()
{
    for(auto itr : contexts)
    {
        if(itr.second && !is_active(*itr.second))
        {
            if(itr.first == "COUNTER_COLLECTION")
            {
                auto* counters = getenv("ROCPROF_COUNTERS");
                if(!counters) continue;
            }
            ROCPROFILER_CALL(rocprofiler_start_context(*itr.second), "context start");
        }
    }
}

void
stop()
{
    for(auto itr : contexts)
    {
        if(itr.second && is_active(*itr.second))
        {
            ROCPROFILER_CALL(rocprofiler_stop_context(*itr.second), "context stop");
        }
    }
}

void
flush()
{
    for(auto* itr : buffers)
    {
        if(itr && itr->handle > 0)
        {
            ROCPROFILER_CALL(rocprofiler_flush_buffer(*itr), "buffer flush");
        }
    }
}

void
push_external_correlation(uint64_t _value)
{
    auto tid = rocprofiler_thread_id_t{};
    ROCPROFILER_CALL(rocprofiler_get_thread_id(&tid), "get thread id");

    for(auto itr : contexts)
    {
        if(itr.second)
        {
            ROCPROFILER_CALL(rocprofiler_push_external_correlation_id(
                                 *itr.second, tid, rocprofiler_user_data_t{.value = _value}),
                             "push external correlation");
        }
    }
}

void
pop_external_correlation()
{
    auto tid = rocprofiler_thread_id_t{};
    ROCPROFILER_CALL(rocprofiler_get_thread_id(&tid), "get thread id");

    for(auto itr : contexts)
    {
        if(itr.second)
        {
            auto _data = rocprofiler_user_data_t{.value = 0};
            ROCPROFILER_CALL(rocprofiler_pop_external_correlation_id(*itr.second, tid, &_data),
                             "push external correlation");
            if(_data.value != static_cast<uint64_t>(getppid()))
            {
                auto _msg = std::stringstream{};
                _msg << "rocprofiler_pop_external_correlation_id(context.handle="
                     << itr.second->handle << ", tid=" << tid
                     << ", ...) returned external correlation id value of " << _data.value
                     << ". expected: " << getppid();
                throw std::runtime_error{_msg.str()};
            }
        }
    }
}
}  // namespace
}  // namespace client

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // set the client name
    id->name = "rocprofiler-sdk-json-tool";

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

    ROCPROFILER_CALL(rocprofiler_at_internal_thread_create(
                         client::thread_precreate,
                         client::thread_postcreate,
                         ROCPROFILER_LIBRARY | ROCPROFILER_HSA_LIBRARY | ROCPROFILER_HIP_LIBRARY |
                             ROCPROFILER_MARKER_LIBRARY,
                         static_cast<void*>(client_tool_data)),
                     "registration for thread creation notifications");

    std::atexit([]() {
        if(client::client_fini_func) client::client_fini_func(*client::client_id);
    });
    std::at_quick_exit([]() {
        if(client::client_fini_func) client::client_fini_func(*client::client_id);
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

PERFETTO_TRACK_EVENT_STATIC_STORAGE();
