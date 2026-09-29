// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common_types.hpp"

#include "library/rocprofiler-sdk/callback/common_tracing_callbacks.hpp"
#include "library/rocprofiler-sdk/callback/ompt/decoder.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofsys::domains::callback
{

namespace detail
{

template <policies::domain_service::backend SdkBackend>
struct rocprofsys_ompt_data_storage_t
{
    SdkBackend::callback_tracing_record_t record;
    SdkBackend::timestamp_t               begin_timestamp;
    function_args_t                       args;  // Required for orphan ENTER events
};

template <policies::domain_service::backend SdkBackend>
auto
ompt_get_unified_name(const typename SdkBackend::callback_tracing_record_t& record)
{
    std::string_view name =
        SdkBackend::get_callback_tracing_names().at(record.kind, record.operation);

    // Forces omp_parallel begin and end to have same name, allowing track to connect
    if(record.operation == SdkBackend::OMPT_ID_parallel_begin ||
       record.operation == SdkBackend::OMPT_ID_parallel_end)
    {
        name = "omp_parallel";
    }

    return name;
}

template <policies::domain_service::backend SdkBackend>
void
ompt_iterate_operation_args(const typename SdkBackend::callback_tracing_record_t& record,
                            function_args_t&                                      args)
{
    const auto operation = static_cast<SdkBackend::ompt_operation_t>(record.operation);

    // ROCProfiler-SDK recommends one dereference on ENTER to avoid faults.
    const auto max_deref = (record.phase == SdkBackend::CALLBACK_PHASE_ENTER ||
                            operation == SdkBackend::OMPT_ID_parallel_begin)
                               ? 1
                               : 2;

    SdkBackend::iterate_callback_tracing_kind_operation_args(
        record, detail::iterate_args_callback, max_deref, &args);

    const auto* payload =
        static_cast<const SdkBackend::callback_tracing_ompt_data_t*>(record.payload);

    if(!payload)
    {
        return;
    }

    const auto representation =
        rocprofsys::domains::callback::ompt::make_ompt_flag_representation<SdkBackend>(
            operation, *payload);

    if(!representation)
    {
        return;
    }

    for(auto& decoded : representation->decode())
    {
        args.emplace_back(argument_info{
            .arg_number = static_cast<std::uint32_t>(args.size()),
            .arg_type   = std::string{ decoded.type },
            .arg_name   = std::string{ decoded.key },
            .arg_value  = std::move(decoded.value),
        });
    }
}

// Records a completed OMPT region (instant, standard, or parallel) into the trace
// cache: registers the category/thread-info metadata once and stores the region
// sample, mirroring domains::callback::on_tracing_api_exit's tail.
template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename BacktraceDataT>
void
ompt_emit_region(const typename SdkBackend::callback_tracing_record_t& record,
                 typename SdkBackend::timestamp_t                      begin_timestamp,
                 typename SdkBackend::timestamp_t                      end_timestamp,
                 BacktraceDataT& backtrace_data, const function_args_t& args)
{
    const std::string_view name = ompt_get_unified_name<SdkBackend>(record);

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

// An instant event is one that has its begin_timestamp = end_timestamp
template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename BacktraceDataT>
void
ompt_cache_instant_event(typename SdkBackend::callback_tracing_record_t record,
                         typename SdkBackend::timestamp_t               instant_timestamp,
                         BacktraceDataT&                                backtrace_data)
{
    auto args = function_args_t{};
    ompt_iterate_operation_args<SdkBackend>(record, args);

    ompt_emit_region<SdkBackend, Externals, Category>(
        record, instant_timestamp, instant_timestamp, backtrace_data, args);
}

// OMPT callbacks with no corresponding begin/end are treated as "instant"
template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename BacktraceDataT>
void
ompt_cache_orphan_event(const rocprofsys_ompt_data_storage_t<SdkBackend>& stored_data,
                        BacktraceDataT&                                   backtrace_data)
{
    ompt_emit_region<SdkBackend, Externals, Category>(
        stored_data.record, stored_data.begin_timestamp, stored_data.begin_timestamp,
        backtrace_data, stored_data.args);
}

template <policies::domain_service::backend SdkBackend>
struct ompt_storage
{
    static auto& get_standard() { return s_standard_cb; }

    static auto& get_parallel() { return s_parallel_cb; }

    static void clear()
    {
        s_standard_cb.clear();
        s_parallel_cb.clear();
    }

private:
    // Any OMPT callback that can be of phase ENTER or EXIT is a standard callback.
    //  I.e. it has an ompt_scope_endpoint_t in its definition (excluding
    //  ROCPROFILER_OMPT_ID_nest_lock as it is a mutex)

    // std::uint64_t -> internal id from rocprofiler_correlation_id_t
    static inline thread_local auto s_standard_cb =
        std::unordered_map<std::uint64_t, rocprofsys_ompt_data_storage_t<SdkBackend>>{};

    // An OMPT parallel callback consists of ROCPROFILER_OMPT_ID_parallel_begin and
    // ROCPROFILER_OMPT_ID_parallel_end
    //  As the beginning and end can only occur on the same thread, they are connected
    //  into a single track called "omp_parallel" for clarity. In this track, the
    //  information contained within parallel_begin should be displayed as it contains all
    //  the information that parallel_end has as well as the flags and number of
    //  threads/teams that were requested.
    // uintptr_t -> parallel_data (see callback definition)
    static inline thread_local auto s_parallel_cb =
        std::unordered_map<uintptr_t, rocprofsys_ompt_data_storage_t<SdkBackend>>{};
};

template <policies::domain_service::backend SdkBackend>
void
ompt_push_standard_callback(const typename SdkBackend::callback_tracing_record_t& record,
                            const typename SdkBackend::timestamp_t& begin_timestamp)
{
    auto args = function_args_t{};
    ompt_iterate_operation_args<SdkBackend>(record, args);
    ompt_storage<SdkBackend>::get_standard().emplace(
        record.correlation_id.internal,
        rocprofsys_ompt_data_storage_t<SdkBackend>{ record, begin_timestamp, args });
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename BacktraceDataT>
void
ompt_pop_standard_callback(const typename SdkBackend::callback_tracing_record_t& record,
                           const typename SdkBackend::timestamp_t& end_timestamp,
                           BacktraceDataT&                         backtrace_data)
{
    auto& storage = ompt_storage<SdkBackend>::get_standard();
    auto  itr     = storage.find(record.correlation_id.internal);

    if(itr == storage.end())
    {
        auto args = function_args_t{};
        ompt_iterate_operation_args<SdkBackend>(record, args);
        ompt_cache_orphan_event<SdkBackend, Externals, Category>(
            rocprofsys_ompt_data_storage_t<SdkBackend>{ record, end_timestamp, args },
            backtrace_data);
        return;
    }

    auto stored_data = itr->second;
    storage.erase(itr);

    ompt_emit_region<SdkBackend, Externals, Category>(record, stored_data.begin_timestamp,
                                                      end_timestamp, backtrace_data,
                                                      stored_data.args);
}

template <policies::domain_service::backend SdkBackend>
void
ompt_push_parallel_callback(const typename SdkBackend::callback_tracing_record_t& record,
                            const typename SdkBackend::timestamp_t& begin_timestamp)
{
    auto* payload_data =
        static_cast<SdkBackend::callback_tracing_ompt_data_t*>(record.payload);
    const void* parallel_data_address = payload_data->args.parallel_begin.parallel_data;

    auto args = function_args_t{};
    ompt_iterate_operation_args<SdkBackend>(record, args);
    ompt_storage<SdkBackend>::get_parallel().emplace(
        reinterpret_cast<uintptr_t>(parallel_data_address),
        rocprofsys_ompt_data_storage_t<SdkBackend>{ record, begin_timestamp, args });
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename BacktraceDataT>
void
ompt_pop_parallel_callback(const typename SdkBackend::callback_tracing_record_t& record,
                           const typename SdkBackend::timestamp_t& end_timestamp,
                           BacktraceDataT&                         backtrace_data)
{
    auto* payload_data =
        static_cast<SdkBackend::callback_tracing_ompt_data_t*>(record.payload);
    const void* parallel_data_address = payload_data->args.parallel_end.parallel_data;

    auto& storage = ompt_storage<SdkBackend>::get_parallel();
    auto  itr     = storage.find(reinterpret_cast<uintptr_t>(parallel_data_address));

    if(itr == storage.end())
    {
        auto args = function_args_t{};
        ompt_iterate_operation_args<SdkBackend>(record, args);
        ompt_cache_orphan_event<SdkBackend, Externals, Category>(
            rocprofsys_ompt_data_storage_t<SdkBackend>{ record, end_timestamp, args },
            backtrace_data);
        return;
    }

    auto stored_data = itr->second;
    storage.erase(itr);

    ompt_emit_region<SdkBackend, Externals, Category>(record, stored_data.begin_timestamp,
                                                      end_timestamp, backtrace_data,
                                                      stored_data.args);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
void
ompt_finalize_orphan_events()
{
    auto empty_backtrace_data = Externals::get_backtrace_data(false);
    for(const auto& [parallel_data, stored_data] :
        ompt_storage<SdkBackend>::get_parallel())
    {
        ompt_cache_orphan_event<SdkBackend, Externals, Category>(stored_data,
                                                                 empty_backtrace_data);
    }

    for(const auto& [correlation_id, stored_data] :
        ompt_storage<SdkBackend>::get_standard())
    {
        ompt_cache_orphan_event<SdkBackend, Externals, Category>(stored_data,
                                                                 empty_backtrace_data);
    }

    ompt_storage<SdkBackend>::clear();
}

// To handle events without finalization, perfetto push must occur in start
// Allows capture of worker thread implicit tasks and sync regions
template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
void
timemory_store_start(typename SdkBackend::callback_tracing_record_t record)
{
    const std::string_view name = ompt_get_unified_name<SdkBackend>(record);

    if(Externals::get_use_timemory())
    {
        Externals::tracing_push_timemory(typename Category<Externals>::type{}, name);
    }
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
void
timemory_store_stop(typename SdkBackend::callback_tracing_record_t record)
{
    const std::string_view name = ompt_get_unified_name<SdkBackend>(record);

    if(Externals::get_use_timemory())
    {
        Externals::tracing_pop_timemory(typename Category<Externals>::type{}, name);
    }
}

template <policies::domain_service::backend SdkBackend>
bool
should_skip(const typename SdkBackend::callback_tracing_record_t& record)
{
    // Skip implicit_task associated with an "initial-task-begin" occurrence as
    // well as the thread_begin associated with an "initial-thread-begin" occurrence
    // as they are generated by our tool.
    // The two callbacks occur after our tool initializes OMPT but before the
    // first OpenMP region (user code) begins.
    // Note: Can occur multiple times (Ex: MPI+OpenMP hybrid)

    auto* payload_data =
        static_cast<SdkBackend::callback_tracing_ompt_data_t*>(record.payload);
    if(!payload_data)
    {
        return true;
    }

    const auto operation = record.operation;

    if(operation == SdkBackend::OMPT_ID_implicit_task)
    {
        const int flag = payload_data->args.implicit_task.flags;
        if(ompt::has_flag(flag, ompt::ompt_task_flag_t::ompt_task_initial))
        {
            return true;  // Skips both the start and end
        }
    }
    else if(operation == SdkBackend::OMPT_ID_thread_begin)
    {
        const auto thread_type =
            static_cast<ompt::ompt_thread_t>(payload_data->args.thread_begin.thread_type);
        if(thread_type == ompt::ompt_thread_t::ompt_thread_initial)
        {
            return true;
        }
    }

    // TODO: Once finalization issue is fixed, skip the corresponding end
    // of the thread_begin callback. Can be identified with:
    // - thread_end: The thread_data ptr from the thread_begin callback generated
    //    by the "initial-thread-begin" needs to match the thread_end's
    //    thread_data ptr
    return false;
}

}  // namespace detail

template <policies::domain_service::externals Externals>
inline void
on_ompt_configure()
{}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
inline void
on_ompt_enter(typename SdkBackend::callback_tracing_record_t record,
              typename SdkBackend::user_data_t* /*user_data*/, void* /*callback_data*/,
              typename SdkBackend::timestamp_t timestamp = SdkBackend::get_timestamp())
{
    if(!Externals::is_active())
    {
        return;
    }

    if(detail::should_skip<SdkBackend>(record))
    {
        return;
    }

    detail::timemory_store_start<SdkBackend, Externals, Category>(record);
    detail::ompt_push_standard_callback<SdkBackend>(record, timestamp);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
inline void
on_ompt_exit(typename SdkBackend::callback_tracing_record_t record,
             typename SdkBackend::user_data_t* /*user_data*/, void* /*callback_data*/,
             typename SdkBackend::timestamp_t timestamp = SdkBackend::get_timestamp())
{
    if(!Externals::is_active())
    {
        return;
    }

    if(detail::should_skip<SdkBackend>(record))
    {
        return;
    }

    auto backtrace_data = Externals::get_backtrace_data(
        Externals::check_backtrace_operations(record.kind, record.operation));

    detail::timemory_store_stop<SdkBackend, Externals, Category>(record);
    detail::ompt_pop_standard_callback<SdkBackend, Externals, Category>(record, timestamp,
                                                                        backtrace_data);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
inline void
on_ompt_none(typename SdkBackend::callback_tracing_record_t record,
             typename SdkBackend::user_data_t* /*user_data*/, void* /*callback_data*/,
             typename SdkBackend::timestamp_t timestamp = SdkBackend::get_timestamp())
{
    if(!Externals::is_active())
    {
        return;
    }

    if(detail::should_skip<SdkBackend>(record))
    {
        return;
    }

    // Callbacks that are received but that we do not process
    static const std::set<typename SdkBackend::ompt_operation_t> k_ompt_no_process = {
        SdkBackend::OMPT_ID_callback_functions,  // "Fake" callback
        // Not processed as these are received after our tool
        // finalizes
        SdkBackend::OMPT_ID_thread_end,
    };

    auto ompt_operation_type =
        static_cast<SdkBackend::ompt_operation_t>(record.operation);
    if(k_ompt_no_process.contains(ompt_operation_type))
    {
        return;
    }

    auto backtrace_data = Externals::get_backtrace_data(
        Externals::check_backtrace_operations(record.kind, record.operation));

    switch(ompt_operation_type)
    {
        case SdkBackend::OMPT_ID_parallel_begin:
            detail::timemory_store_start<SdkBackend, Externals, Category>(record);
            detail::ompt_push_parallel_callback<SdkBackend>(record, timestamp);
            break;
        case SdkBackend::OMPT_ID_parallel_end:
            detail::timemory_store_stop<SdkBackend, Externals, Category>(record);
            detail::ompt_pop_parallel_callback<SdkBackend, Externals, Category>(
                record, timestamp, backtrace_data);
            break;
        // Unlike parallel callbacks, we cannot receive the corresponding
        // end to thread_begin. Set thread_begin as "instant" so the user
        // can see callback without it spanning the entire track
        case SdkBackend::OMPT_ID_thread_begin:
        case SdkBackend::OMPT_ID_lock_init:
        case SdkBackend::OMPT_ID_lock_destroy:
        // Although this has endpoint arg, treat it as instant event
        case SdkBackend::OMPT_ID_nest_lock:
        case SdkBackend::OMPT_ID_dispatch:
        case SdkBackend::OMPT_ID_flush:
        case SdkBackend::OMPT_ID_cancel:
        case SdkBackend::OMPT_ID_device_initialize:
        case SdkBackend::OMPT_ID_device_finalize:
        case SdkBackend::OMPT_ID_device_load:
        // case ROCPROFILER_OMPT_ID_device_unload: // Unsupported by
        // runtime
        case SdkBackend::OMPT_ID_task_create:
        case SdkBackend::OMPT_ID_task_schedule:
        case SdkBackend::OMPT_ID_mutex_released:
        case SdkBackend::OMPT_ID_mutex_acquire:
        case SdkBackend::OMPT_ID_mutex_acquired:
        case SdkBackend::OMPT_ID_dependences:
        case SdkBackend::OMPT_ID_task_dependence:
        case SdkBackend::OMPT_ID_error:
        {
            // These callbacks are considered instant events and should
            // start and immediately call stop as no corresponding "end"
            // will be received
            const auto instant_timestamp = timestamp;
            detail::timemory_store_start<SdkBackend, Externals, Category>(record);
            detail::timemory_store_stop<SdkBackend, Externals, Category>(record);
            detail::ompt_cache_instant_event<SdkBackend, Externals, Category>(
                record, instant_timestamp, backtrace_data);
            break;
        }
        default:
            LOG_WARNING("tool_tracing_callback: unhandled PHASE_NONE "
                        "for OMPT callback record.");
    }
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
void
on_ompt_finalize()
{
    detail::ompt_finalize_orphan_events<SdkBackend, Externals, Category>();
}

template <typename Externals>
struct ompt_api_category
{
    using type = Externals::rocm_ompt_api_category;

    static constexpr std::string_view k_name = Externals::rocm_ompt_api_category_name;
};

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_ompt_api = callback_domain_definition<SdkBackend>{
    .meta      = domain_descriptor{ .name  = "ompt",
                                    .id    = SdkBackend::CALLBACK_TRACING_OMPT,
                                    .mode  = collection_mode::callback,
                                    .group = std::nullopt },
    .on_record = tracing_callback_dispatcher<
        SdkBackend, on_ompt_enter<SdkBackend, Externals, ompt_api_category>,
        on_ompt_exit<SdkBackend, Externals, ompt_api_category>,
        on_ompt_none<SdkBackend, Externals, ompt_api_category>>::callback,
    .on_configure = on_ompt_configure<Externals>,
    .on_finalize  = on_ompt_finalize<SdkBackend, Externals, ompt_api_category>
};

}  // namespace rocprofsys::domains::callback
