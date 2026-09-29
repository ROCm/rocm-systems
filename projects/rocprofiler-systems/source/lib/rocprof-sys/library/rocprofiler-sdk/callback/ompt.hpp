// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common_types.hpp"

#include "library/rocprofiler-sdk/callback/common_tracing_callbacks.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::domains::callback
{

namespace detail
{

using callback_arg_array_t = std::vector<std::pair<std::string, std::string>>;

template <policies::domain_service::backend SdkBackend>
int
save_args(typename SdkBackend::callback_tracing_kind_t /*kind*/,
          std::int32_t /*operation*/, std::uint32_t /*arg_number*/,
          const void* const /*arg_value_addr*/, std::int32_t /*arg_indirection_count*/,
          const char* /*arg_type*/, const char* arg_name, const char* arg_value_str,
          std::int32_t /*arg_dereference_count*/, void* data)
{
    auto* argvec = static_cast<callback_arg_array_t*>(data);
    argvec->emplace_back(arg_name, arg_value_str);
    return 0;
}

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

namespace external_types
{

enum class ompt_parallel_flag_t : std::size_t
{
    ompt_parallel_invoker_program = 0x00000001,
    ompt_parallel_invoker_runtime = 0x00000002,
    ompt_parallel_league          = 0x40000000,
    ompt_parallel_team            = 0x80000000
};

enum class ompt_task_flag_t : std::size_t
{
    ompt_task_initial    = 0x00000001,
    ompt_task_implicit   = 0x00000002,
    ompt_task_explicit   = 0x00000004,
    ompt_task_target     = 0x00000008,
    ompt_task_taskwait   = 0x00000010,
    ompt_task_undeferred = 0x08000000,
    ompt_task_untied     = 0x10000000,
    ompt_task_final      = 0x20000000,
    ompt_task_mergeable  = 0x40000000,
    ompt_task_merged     = 0x80000000
};

enum class ompt_cancel_flag_t : std::size_t
{
    ompt_cancel_parallel       = 0x01,
    ompt_cancel_sections       = 0x02,
    ompt_cancel_loop           = 0x04,
    ompt_cancel_taskgroup      = 0x08,
    ompt_cancel_activated      = 0x10,
    ompt_cancel_detected       = 0x20,
    ompt_cancel_discarded_task = 0x40
};

enum class ompt_thread_t
{
    ompt_thread_initial = 1,
    ompt_thread_worker  = 2,
    ompt_thread_other   = 3,
    ompt_thread_unknown = 4
};

template <typename EnumT>
constexpr auto
to_underlying(EnumT value)
{
    return static_cast<std::underlying_type_t<EnumT>>(value);
}

template <typename EnumT>
constexpr bool
has_flag(int flags_val, EnumT flag)
{
    return (flags_val & to_underlying(flag)) != 0;
}

}  // namespace external_types

/**
 * @brief rocprofiler_iterate_callback_tracing_kind_operation_args wrapper for OMPT
 * callbacks.
 *
 * Certain OMPT callbacks have a "flags" argument that contains a bitmask of flags.
 * This function decodes the "flags" into a human readable form.
 * Works with both callback_arg_array_t (perfetto) and function_args_t (rocpd).
 */

template <policies::domain_service::backend SdkBackend, typename ArgsT>
void
ompt_iterate_operation_args(const typename SdkBackend::callback_tracing_record_t& record,
                            ArgsT&                                                args)
{
    static_assert(std::is_same_v<ArgsT, callback_arg_array_t> ||
                      std::is_same_v<ArgsT, function_args_t>,
                  "ompt_iterate_operation_args: ArgsT must be callback_arg_array_t or "
                  "function_args_t");

    auto ompt_operation_type =
        static_cast<SdkBackend::ompt_operation_t>(record.operation);
    // ROCProfiler-SDK documentation recommends using 1 for the ENTER phase to avoid seg.
    // faults.
    auto max_deref = (record.phase == SdkBackend::CALLBACK_PHASE_ENTER ||
                      ompt_operation_type == SdkBackend::OMPT_ID_parallel_begin)
                         ? 1
                         : 2;

    // Perform standard iteration of arguments
    if constexpr(std::is_same_v<ArgsT, callback_arg_array_t>)
    {
        SdkBackend::iterate_callback_tracing_kind_operation_args(
            record, save_args<SdkBackend>, max_deref, &args);
    }
    else
    {
        SdkBackend::iterate_callback_tracing_kind_operation_args(
            record, iterate_args_callback, max_deref, &args);
    }

    static const auto k_ompt_has_flags = std::set<typename SdkBackend::ompt_operation_t>{
        SdkBackend::OMPT_ID_parallel_begin, SdkBackend::OMPT_ID_parallel_end,
        SdkBackend::OMPT_ID_task_create,    SdkBackend::OMPT_ID_implicit_task,
        SdkBackend::OMPT_ID_cancel,
    };
    if(k_ompt_has_flags.find(ompt_operation_type) == k_ompt_has_flags.end())
    {
        return;
    }

    auto append = [&args](const std::string& flag_type, const std::string& key,
                          const std::string& val) {
        if constexpr(std::is_same_v<ArgsT, callback_arg_array_t>)
        {
            args.emplace_back(key, val);
        }
        else
        {
            args.emplace_back(
                argument_info{ .arg_number = static_cast<std::uint32_t>(args.size()),
                               .arg_type   = flag_type,
                               .arg_name   = key,
                               .arg_value  = val });
        }
    };

    int   flags_val = 0;
    auto* payload_data =
        static_cast<SdkBackend::callback_tracing_ompt_data_t*>(record.payload);
    if(!payload_data)
    {
        return;
    }

    // Extract flags value
    switch(ompt_operation_type)
    {
        case SdkBackend::OMPT_ID_parallel_begin:
            flags_val = payload_data->args.parallel_begin.flags;
            break;
        case SdkBackend::OMPT_ID_parallel_end:
            flags_val = payload_data->args.parallel_end.flags;
            break;
        case SdkBackend::OMPT_ID_task_create:
            flags_val = payload_data->args.task_create.flags;
            break;
        case SdkBackend::OMPT_ID_implicit_task:
            flags_val = payload_data->args.implicit_task.flags;
            break;
        case SdkBackend::OMPT_ID_cancel:
            flags_val = payload_data->args.cancel.flags;
            break;
        default: break;
    }

    // Textual representation of flags adapted from OMPT 5.0 specification
    switch(ompt_operation_type)
    {
        case SdkBackend::OMPT_ID_parallel_begin:  // ompt_parallel_flag_t
        case SdkBackend::OMPT_ID_parallel_end:    // ompt_parallel_flag_t
        {
            const auto flag_str = std::string{ "ompt_parallel_flag_t" };
            using external_types::ompt_parallel_flag_t;
            if(external_types::has_flag(
                   flags_val, ompt_parallel_flag_t::ompt_parallel_invoker_program))
            {
                append(flag_str, "invoker", "program");
            }
            else if(external_types::has_flag(
                        flags_val, ompt_parallel_flag_t::ompt_parallel_invoker_runtime))
            {
                append(flag_str, "invoker", "runtime");
            }

            if(external_types::has_flag(flags_val,
                                        ompt_parallel_flag_t::ompt_parallel_league))
            {
                append(flag_str, "invoker_cause", "teams_construct");
            }
            else if(external_types::has_flag(flags_val,
                                             ompt_parallel_flag_t::ompt_parallel_team))
            {
                append(flag_str, "invoker_cause", "parallel_construct");
            }
            break;
        }
        case SdkBackend::OMPT_ID_task_create:  // ompt_task_flag_t
        {
            const auto flag_str = std::string{ "ompt_task_flag_t" };
            using external_types::ompt_task_flag_t;
            if(external_types::has_flag(flags_val, ompt_task_flag_t::ompt_task_initial))
            {
                append(flag_str, "classification", "initial");
            }
            else if(external_types::has_flag(flags_val,
                                             ompt_task_flag_t::ompt_task_implicit))
            {
                append(flag_str, "classification", "implicit");
            }
            else if(external_types::has_flag(flags_val,
                                             ompt_task_flag_t::ompt_task_explicit))
            {
                append(flag_str, "classification", "explicit");
            }
            else if(external_types::has_flag(flags_val,
                                             ompt_task_flag_t::ompt_task_target))
            {
                append(flag_str, "classification", "target");
            }

            // Multiple/none can be set
            constexpr std::size_t k_task_properties_reserve = 60;
            std::string           task_properties;
            task_properties.reserve(k_task_properties_reserve);
            if(external_types::has_flag(flags_val,
                                        ompt_task_flag_t::ompt_task_undeferred))
            {
                task_properties += "undeferred, ";
            }
            if(external_types::has_flag(flags_val, ompt_task_flag_t::ompt_task_untied))
            {
                task_properties += "untied, ";
            }
            if(external_types::has_flag(flags_val, ompt_task_flag_t::ompt_task_final))
            {
                task_properties += "final, ";
            }
            if(external_types::has_flag(flags_val, ompt_task_flag_t::ompt_task_mergeable))
            {
                task_properties += "mergeable, ";
            }
            if(external_types::has_flag(flags_val, ompt_task_flag_t::ompt_task_merged))
            {
                task_properties += "merged, ";
            }

            if(!task_properties.empty())
            {
                task_properties.erase(task_properties.size() - 2);
            }
            else
            {
                task_properties = "none";
            }
            append(flag_str, "properties", task_properties);
            break;
        }
        case SdkBackend::OMPT_ID_implicit_task:  // initial (1) or implicit (2)
        {
            const auto flag_str = std::string{ "flags" };
            // As of now, implicit_tasks with ompt_task_initial are filtered out
            if(external_types::has_flag(
                   flags_val, external_types::ompt_task_flag_t::ompt_task_initial))
            {
                append(flag_str, "kind", "initial");
            }
            else if(external_types::has_flag(
                        flags_val, external_types::ompt_task_flag_t::ompt_task_implicit))
            {
                append(flag_str, "kind", "implicit");
            }
            break;
        }
        case SdkBackend::OMPT_ID_cancel:  // ompt_cancel_flag_t
        {
            const auto flag_str = std::string{ "ompt_cancel_flag_t" };
            using external_types::ompt_cancel_flag_t;
            if(external_types::has_flag(flags_val,
                                        ompt_cancel_flag_t::ompt_cancel_parallel))
            {
                append(flag_str, "construct", "parallel");
            }
            else if(external_types::has_flag(flags_val,
                                             ompt_cancel_flag_t::ompt_cancel_sections))
            {
                append(flag_str, "construct", "sections");
            }
            else if(external_types::has_flag(flags_val,
                                             ompt_cancel_flag_t::ompt_cancel_loop))
            {
                append(flag_str, "construct", "loop");
            }
            else if(external_types::has_flag(flags_val,
                                             ompt_cancel_flag_t::ompt_cancel_taskgroup))
            {
                append(flag_str, "construct", "taskgroup");
            }

            if(external_types::has_flag(flags_val,
                                        ompt_cancel_flag_t::ompt_cancel_activated))
            {
                append(flag_str, "state", "activated");
            }
            else if(external_types::has_flag(flags_val,
                                             ompt_cancel_flag_t::ompt_cancel_detected))
            {
                append(flag_str, "state", "detected");
            }
            else if(external_types::has_flag(
                        flags_val, ompt_cancel_flag_t::ompt_cancel_discarded_task))
            {
                append(flag_str, "state", "discarded_task");
            }
            break;
        }
        default: break;
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

// Any OMPT callback that can be of phase ENTER or EXIT is a standard callback.
//  I.e. it has an ompt_scope_endpoint_t in its definition (excluding
//  ROCPROFILER_OMPT_ID_nest_lock as it is a mutex)
template <policies::domain_service::backend SdkBackend>
auto&
get_ompt_standard_cb_storage()
{
    // std::uint64_t -> internal id from rocprofiler_correlation_id_t
    static thread_local auto s_storage =
        std::unordered_map<std::uint64_t, rocprofsys_ompt_data_storage_t<SdkBackend>>{};
    return s_storage;
}

// An OMPT parallel callback consists of ROCPROFILER_OMPT_ID_parallel_begin and
// ROCPROFILER_OMPT_ID_parallel_end
//  As the beginning and end can only occur on the same thread, they are connected
//  into a single track called "omp_parallel" for clarity. In this track, the
//  information contained within parallel_begin should be displayed as it contains all
//  the information that parallel_end has as well as the flags and number of
//  threads/teams that were requested.
template <policies::domain_service::backend SdkBackend>
auto&
get_ompt_parallel_cb_storage()
{
    // uintptr_t -> parallel_data (see callback definition)
    static thread_local auto s_storage =
        std::unordered_map<uintptr_t, rocprofsys_ompt_data_storage_t<SdkBackend>>{};
    return s_storage;
}

template <policies::domain_service::backend SdkBackend>
void
ompt_push_standard_callback(const typename SdkBackend::callback_tracing_record_t& record,
                            const typename SdkBackend::timestamp_t& begin_timestamp)
{
    auto args = function_args_t{};
    ompt_iterate_operation_args<SdkBackend>(record, args);
    get_ompt_standard_cb_storage<SdkBackend>().emplace(
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
    auto& storage = get_ompt_standard_cb_storage<SdkBackend>();
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
    get_ompt_parallel_cb_storage<SdkBackend>().emplace(
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

    auto& storage = get_ompt_parallel_cb_storage<SdkBackend>();
    auto  it      = storage.find(reinterpret_cast<uintptr_t>(parallel_data_address));

    if(it == storage.end())
    {
        auto args = function_args_t{};
        ompt_iterate_operation_args<SdkBackend>(record, args);
        ompt_cache_orphan_event<SdkBackend, Externals, Category>(
            rocprofsys_ompt_data_storage_t<SdkBackend>{ record, end_timestamp, args },
            backtrace_data);
        return;
    }

    auto stored_data = it->second;
    storage.erase(it);

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
        get_ompt_parallel_cb_storage<SdkBackend>())
    {
        ompt_cache_orphan_event<SdkBackend, Externals, Category>(stored_data,
                                                                 empty_backtrace_data);
    }

    for(const auto& [correlation_id, stored_data] :
        get_ompt_standard_cb_storage<SdkBackend>())
    {
        ompt_cache_orphan_event<SdkBackend, Externals, Category>(stored_data,
                                                                 empty_backtrace_data);
    }

    get_ompt_parallel_cb_storage<SdkBackend>().clear();
    get_ompt_standard_cb_storage<SdkBackend>().clear();
}

// To handle events without finalization, perfetto push must occur in start
// Allows capture of worker thread implicit tasks and sync regions
template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
void
ompt_tracing_callback_start(typename SdkBackend::callback_tracing_record_t record,
                            typename SdkBackend::user_data_t* /*user_data*/,
                            typename SdkBackend::timestamp_t /*timestamp*/)
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
ompt_tracing_callback_stop(typename SdkBackend::callback_tracing_record_t record,
                           typename SdkBackend::user_data_t* /*user_data*/,
                           typename SdkBackend::timestamp_t /*timestamp*/)
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
        if(external_types::has_flag(flag,
                                    external_types::ompt_task_flag_t::ompt_task_initial))
        {
            return true;  // Skips both the start and end
        }
    }
    else if(operation == SdkBackend::OMPT_ID_thread_begin)
    {
        const auto thread_type = static_cast<external_types::ompt_thread_t>(
            payload_data->args.thread_begin.thread_type);
        if(thread_type == external_types::ompt_thread_t::ompt_thread_initial)
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
              typename SdkBackend::user_data_t* user_data, void* /*callback_data*/,
              typename SdkBackend::timestamp_t  timestamp = SdkBackend::get_timestamp())
{
    if(!Externals::is_active())
    {
        return;
    }

    if(detail::should_skip<SdkBackend>(record))
    {
        return;
    }

    detail::ompt_tracing_callback_start<SdkBackend, Externals, Category>(
        record, user_data, timestamp);
    detail::ompt_push_standard_callback<SdkBackend>(record, timestamp);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
inline void
on_ompt_exit(typename SdkBackend::callback_tracing_record_t record,
             typename SdkBackend::user_data_t* user_data, void* /*callback_data*/,
             typename SdkBackend::timestamp_t  timestamp = SdkBackend::get_timestamp())
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

    detail::ompt_tracing_callback_stop<SdkBackend, Externals, Category>(record, user_data,
                                                                        timestamp);
    detail::ompt_pop_standard_callback<SdkBackend, Externals, Category>(record, timestamp,
                                                                        backtrace_data);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
inline void
on_ompt_none(typename SdkBackend::callback_tracing_record_t record,
             typename SdkBackend::user_data_t* user_data, void* /*callback_data*/,
             typename SdkBackend::timestamp_t  timestamp = SdkBackend::get_timestamp())
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
    if(k_ompt_no_process.find(ompt_operation_type) != k_ompt_no_process.end())
    {
        return;
    }

    auto backtrace_data = Externals::get_backtrace_data(
        Externals::check_backtrace_operations(record.kind, record.operation));

    switch(ompt_operation_type)
    {
        case SdkBackend::OMPT_ID_parallel_begin:
            detail::ompt_tracing_callback_start<SdkBackend, Externals, Category>(
                record, user_data, timestamp);
            detail::ompt_push_parallel_callback<SdkBackend>(record, timestamp);
            break;
        case SdkBackend::OMPT_ID_parallel_end:
            detail::ompt_tracing_callback_stop<SdkBackend, Externals, Category>(
                record, user_data, timestamp);
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
            detail::ompt_tracing_callback_start<SdkBackend, Externals, Category>(
                record, user_data, instant_timestamp);
            detail::ompt_tracing_callback_stop<SdkBackend, Externals, Category>(
                record, user_data, instant_timestamp);
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
