// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#include "common/synchronized.hpp"
#include "core/agent_manager.hpp"
#include "core/timemory.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include <memory>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
using hardware_counter_info = ::tim::hardware_counters::info;

using kernel_symbol_data_t =
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t =
    std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;
using callback_arg_array_t = std::vector<std::pair<std::string, std::string>>;

using rocprofsys_agent_t = agent;

struct code_object_callback_record_t
{
    uint64_t                                             timestamp = 0;
    rocprofiler_callback_tracing_record_t                record    = {};
    rocprofiler_callback_tracing_code_object_load_data_t payload   = {};
};

struct kernel_symbol_callback_record_t
{
    uint64_t                              timestamp = 0;
    rocprofiler_callback_tracing_record_t record    = {};
    kernel_symbol_data_t                  payload   = {};
};

struct rocprofiler_tool_counter_info_t : rocprofiler_counter_info_v0_t
{
    using this_type            = rocprofiler_tool_counter_info_t;
    using parent_type          = rocprofiler_counter_info_v0_t;
    using dimension_info_vec_t = std::vector<rocprofiler_record_dimension_info_t>;

    rocprofiler_tool_counter_info_t(rocprofiler_agent_id_t _agent_id, parent_type _info,
                                    dimension_info_vec_t&& _dim_info);

    rocprofiler_tool_counter_info_t()                                           = default;
    ~rocprofiler_tool_counter_info_t()                                          = default;
    rocprofiler_tool_counter_info_t(const rocprofiler_tool_counter_info_t&)     = default;
    rocprofiler_tool_counter_info_t(rocprofiler_tool_counter_info_t&&) noexcept = default;
    rocprofiler_tool_counter_info_t& operator=(const rocprofiler_tool_counter_info_t&) =
        default;
    rocprofiler_tool_counter_info_t& operator=(
        rocprofiler_tool_counter_info_t&&) noexcept = default;

    rocprofiler_agent_id_t                           agent_id       = {};
    std::vector<rocprofiler_record_dimension_info_t> dimension_info = {};
};

struct tool_agent
{
    uint64_t                  device_id = 0;
    const rocprofsys_agent_t* agent     = nullptr;
};

struct timing_interval
{
    rocprofiler_timestamp_t start = 0;
    rocprofiler_timestamp_t end   = 0;
};

struct argument_info
{
    uint32_t    arg_number = 0;
    std::string arg_type   = {};
    std::string arg_name   = {};
    std::string arg_value  = {};
};

using function_args_t = std::vector<argument_info>;

using agent_counter_info_map_t =
    std::unordered_map<rocprofiler_agent_id_t,
                       std::vector<rocprofiler_tool_counter_info_t>>;

using agent_counter_profile_map_t =
    std::unordered_map<rocprofiler_agent_id_t,
                       std::optional<rocprofiler_profile_config_id_t>>;

using counter_id_vec_t = std::vector<rocprofiler_counter_id_t>;

using agent_counter_id_map_t =
    std::unordered_map<rocprofiler_agent_id_t, counter_id_vec_t>;

using backtrace_operation_map_t =
    std::unordered_map<rocprofiler_callback_tracing_kind_t,
                       std::unordered_set<rocprofiler_tracing_operation_t>>;

struct client_data
{
    static constexpr size_t num_buffers  = 4;
    static constexpr size_t num_contexts = 2;

    using buffer_name_info_t   = rocprofiler::sdk::buffer_name_info_t<std::string_view>;
    using callback_name_info_t = rocprofiler::sdk::callback_name_info_t<std::string_view>;
    using kernel_symbol_vec_t  = std::vector<kernel_symbol_callback_record_t*>;
    using code_object_vec_t    = std::vector<code_object_callback_record_t>;
    using buffer_id_vec_t      = std::array<rocprofiler_buffer_id_t, num_buffers>;
    using context_id_vec_t     = std::array<rocprofiler_context_id_t, num_contexts>;
    using agent_vec_t          = std::vector<rocprofiler_agent_v0_t>;

    rocprofiler_client_id_t*                  client_id                 = nullptr;
    rocprofiler_client_finalize_t             client_fini               = nullptr;
    rocprofiler_context_id_t                  primary_ctx               = { 0 };
    rocprofiler_context_id_t                  counter_ctx               = { 0 };
    rocprofiler_buffer_id_t                   kernel_dispatch_buffer    = { 0 };
    rocprofiler_buffer_id_t                   memory_copy_buffer        = { 0 };
    rocprofiler_buffer_id_t                   memory_alloc_buffer       = { 0 };
    rocprofiler_buffer_id_t                   counter_collection_buffer = { 0 };
    std::vector<tool_agent>                   cpu_agents                = {};
    std::vector<tool_agent>                   gpu_agents                = {};
    std::vector<hardware_counter_info>        events_info               = {};
    agent_counter_id_map_t                    agent_events              = {};
    agent_counter_info_map_t                  agent_counter_info        = {};
    agent_counter_profile_map_t               agent_counter_profiles    = {};
    common::synchronized<code_object_vec_t>   code_object_records       = {};
    common::synchronized<kernel_symbol_vec_t> kernel_symbol_records     = {};
    buffer_name_info_t                        buffered_tracing_info     = {};
    callback_name_info_t                      callback_tracing_info     = {};
    backtrace_operation_map_t                 backtrace_operations      = {};

    void                        initialize();
    void                        initialize_event_info();
    void                        set_agents();
    context_id_vec_t            get_contexts() const;
    buffer_id_vec_t             get_buffers() const;
    const rocprofsys_agent_t*   get_agent(rocprofiler_agent_id_t _id) const;
    const tool_agent*           get_gpu_tool_agent(rocprofiler_agent_id_t id) const;
    const kernel_symbol_data_t* get_kernel_symbol_info(uint64_t _kernel_id) const;
    const rocprofiler_tool_counter_info_t* get_tool_counter_info(
        rocprofiler_agent_id_t _agent_id, rocprofiler_counter_id_t _counter_id) const;
    const rocprofiler_callback_tracing_code_object_load_data_t* get_code_object_info(
        uint64_t code_object_id) const;
};

inline client_data::context_id_vec_t
client_data::get_contexts() const
{
    return context_id_vec_t{
        primary_ctx,
        counter_ctx,
    };
}

inline client_data::buffer_id_vec_t
client_data::get_buffers() const
{
    return buffer_id_vec_t{
        kernel_dispatch_buffer,
        memory_copy_buffer,
        memory_alloc_buffer,
        counter_collection_buffer,
    };
}

inline const rocprofsys_agent_t*
client_data::get_agent(rocprofiler_agent_id_t _id) const
{
    const auto& agent = agent_manager::get_instance().get_agent_by_handle(_id.handle);

    return &agent;
}

inline const tool_agent*
client_data::get_gpu_tool_agent(rocprofiler_agent_id_t id) const
{
    for(const auto& itr : gpu_agents)
        if(id.handle == itr.agent->handle) return &itr;
    return nullptr;
}

inline const kernel_symbol_data_t*
client_data::get_kernel_symbol_info(uint64_t _kernel_id) const
{
    return kernel_symbol_records.rlock(
        [_kernel_id](const auto& _data) -> const kernel_symbol_data_t* {
            for(const auto& itr : _data)
            {
                if(_kernel_id == itr->payload.kernel_id)
                {
                    return &itr->payload;
                    break;
                }
            }
            return nullptr;
        });
}

inline const rocprofiler_tool_counter_info_t*
client_data::get_tool_counter_info(rocprofiler_agent_id_t   _agent_id,
                                   rocprofiler_counter_id_t _counter_id) const
{
    for(const auto& itr : agent_counter_info.at(_agent_id))
    {
        if(itr.id == _counter_id) return &itr;
    }
    return nullptr;
}

inline const rocprofiler_callback_tracing_code_object_load_data_t*
client_data::get_code_object_info(uint64_t code_object_id) const
{
    return code_object_records.rlock(
        [code_object_id](const auto& _data)
            -> const rocprofiler_callback_tracing_code_object_load_data_t* {
            for(const auto& itr : _data)
            {
                if(code_object_id == itr.payload.code_object_id)
                {
                    return &itr.payload;
                    break;
                }
            }
            return nullptr;
        });
}

inline constexpr client_data*
as_client_data(void* _ptr)
{
    return static_cast<client_data*>(_ptr);
}
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys

#if !defined(ROCPROFILER_CALL)
#    define ROCPROFILER_CALL(result)                                                     \
        {                                                                                \
            rocprofiler_status_t ROCPROFSYS_VARIABLE(_rocp_status_, __LINE__) =          \
                (result);                                                                \
            if(ROCPROFSYS_VARIABLE(_rocp_status_, __LINE__) !=                           \
               ROCPROFILER_STATUS_SUCCESS)                                               \
            {                                                                            \
                auto        msg        = std::stringstream{};                            \
                std::string status_msg = rocprofiler_get_status_string(                  \
                    ROCPROFSYS_VARIABLE(_rocp_status_, __LINE__));                       \
                msg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] "           \
                    << "rocprofiler-sdk call [" << #result                               \
                    << "] failed with error code "                                       \
                    << ROCPROFSYS_VARIABLE(_rocp_status_, __LINE__)                      \
                    << " :: " << status_msg;                                             \
                ROCPROFSYS_WARNING(0, "%s\n", msg.str().c_str());                        \
            }                                                                            \
        }
#endif
