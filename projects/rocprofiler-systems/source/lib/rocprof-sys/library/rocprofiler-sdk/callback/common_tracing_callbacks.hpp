// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"
#include <cstdint>

#include "core/common_types.hpp"
#include "core/demangler.hpp"

#include <string>

namespace rocprofsys::domains::callback
{
namespace detail
{
inline auto
// NOLINTNEXTLINE(readability-function-size)
iterate_args_callback(auto /*kind*/, std::int32_t /*operation*/, std::uint32_t arg_number,
                      const void* const /*arg_value_addr*/,
                      std::int32_t /*arg_indirection_count*/, const char* arg_type,
                      const char* arg_name, const char*             arg_value_str,
                      std::int32_t /*arg_dereference_count*/, void* data)
{
    auto* func_args = static_cast<function_args_t*>(data);
    if(arg_type && arg_name && arg_value_str)
    {
        func_args->emplace_back(
            argument_info{ .arg_number = arg_number,
                           .arg_type   = rocprofsys::utility::demangle(arg_type),
                           .arg_name   = arg_name,
                           .arg_value  = arg_value_str });
    }
    return 0;
}
}  // namespace detail

template <policies::domain_service::externals Externals>
inline void
on_tracing_api_configure()
{}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
inline void
on_tracing_api_enter(typename SdkBackend::callback_tracing_record_t record,
                     typename SdkBackend::user_data_t* user_data, void* callback_data)
{
    (void) callback_data;

    if(!Externals::is_active())
    {
        return;
    }

    typename SdkBackend::timestamp_t timestamp = SdkBackend::get_timestamp();

    if(user_data)
    {
        user_data->value = timestamp;
    }

    auto name =
        SdkBackend::get_callback_tracing_names().at(record.kind, record.operation);

    if(Externals::get_use_timemory())
    {
        Externals::tracing_push_timemory(typename Category<Externals>::type{}, name);
    }
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
inline void
on_tracing_api_exit(typename SdkBackend::callback_tracing_record_t record,
                    typename SdkBackend::user_data_t* user_data, void* callback_data)
{
    (void) callback_data;

    typename SdkBackend::timestamp_t timestamp = SdkBackend::get_timestamp();

    if(!Externals::is_active() || !user_data)
    {
        return;
    }

    auto backtrace_data = Externals::get_backtrace_data(
        Externals::check_backtrace_operations(record.kind, record.operation));

    auto name =
        SdkBackend::get_callback_tracing_names().at(record.kind, record.operation);

    const auto begin_timestamp = user_data->value;
    const auto end_timestamp   = timestamp;

    if(Externals::get_use_timemory())
    {
        Externals::tracing_pop_timemory(typename Category<Externals>::type{}, name);
    }

    auto args = function_args_t{};

    SdkBackend::iterate_callback_tracing_kind_operation_args(
        record, detail::iterate_args_callback, 2, &args);

    auto call_stack = Externals::get_backtrace_json(backtrace_data);

    Externals::metadata_add_string(Category<Externals>::k_name);

    Externals::metadata_add_thread_info(
        { Externals::get_ppid(), Externals::get_pid(), record.thread_id, 0, 0, "{}" });

    const std::string args_str = get_args_string(args);

    Externals::buffer_storage_store(typename Externals::region_sample{
        record.thread_id, name, record.correlation_id.internal,
        SdkBackend::get_parent_stack_id(record.correlation_id), begin_timestamp,
        end_timestamp, call_stack.dump(), args_str, Category<Externals>::k_name });
}

}  // namespace rocprofsys::domains::callback
