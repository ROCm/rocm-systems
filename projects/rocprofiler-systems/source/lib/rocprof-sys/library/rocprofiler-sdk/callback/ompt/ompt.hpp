// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/static_object.hpp"
#include "core/common_types.hpp"

#include "library/rocprofiler-sdk/callback/common_tracing_callbacks.hpp"
#include "library/rocprofiler-sdk/callback/ompt/decoder.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofsys::domains::callback::ompt
{

namespace detail
{

// Begin-side data of a region whose end has not been received yet.
template <policies::domain_service::backend SdkBackend>
struct pending_region
{
    SdkBackend::callback_tracing_record_t record;
    SdkBackend::timestamp_t               begin_timestamp;
    function_args_t                       args;  // Required for orphan ENTER events
};

template <policies::domain_service::backend SdkBackend>
using pending_regions_t = std::unordered_map<std::uint64_t, pending_region<SdkBackend>>;

// The regions of one thread that have begun but not yet ended.
template <policies::domain_service::backend SdkBackend>
struct thread_open_regions
{
    // Any OMPT callback that can be of phase ENTER or EXIT is a standard callback.
    //  I.e. it has an ompt_scope_endpoint_t in its definition (excluding
    //  ROCPROFILER_OMPT_ID_nest_lock as it is a mutex)
    // Keyed by the internal id from rocprofiler_correlation_id_t.
    pending_regions_t<SdkBackend> standard = {};

    // An OMPT parallel callback consists of ROCPROFILER_OMPT_ID_parallel_begin and
    // ROCPROFILER_OMPT_ID_parallel_end
    //  As the beginning and end can only occur on the same thread, they are connected
    //  into a single track called "omp_parallel" for clarity. In this track, the
    //  information contained within parallel_begin should be displayed as it contains all
    //  the information that parallel_end has as well as the flags and number of
    //  threads/teams that were requested.
    // Keyed by the parallel_data address (see callback definition).
    pending_regions_t<SdkBackend> parallel = {};
};

// Directory of every thread's open regions. Some OMPT ends arrive after our tool has
// finalized, leaving worker-thread regions open, so finalization must drain the maps of
// all threads, not just the caller's. A thread's entry is dropped when the thread exits.
template <policies::domain_service::backend SdkBackend>
struct open_regions
{
    struct registry_t
    {
        std::vector<std::unique_ptr<thread_open_regions<SdkBackend>>> threads = {};
        std::mutex                                                    mutex;
    };

    // Returns the calling thread's regions, registering them on first use.
    static thread_open_regions<SdkBackend>& current()
    {
        static thread_local handle_t handle{};
        return *handle.regions;
    }

    // Invokes @p visit on every registered thread's regions while holding the registry
    // lock.
    template <typename VisitorT>
    static void for_each_thread(VisitorT&& visit)
    {
        auto&                       registry = get_registry();
        std::lock_guard<std::mutex> lock{ registry.mutex };
        for(auto& thread_regions : registry.threads)
        {
            visit(*thread_regions);
        }
    }

private:
    // Owns the calling thread's slot: registers on construction, removes on thread exit.
    struct handle_t
    {
        handle_t()
        {
            auto&                       registry = get_registry();
            std::lock_guard<std::mutex> lock{ registry.mutex };
            regions =
                registry.threads
                    .emplace_back(std::make_unique<thread_open_regions<SdkBackend>>())
                    .get();
        }

        ~handle_t()
        {
            auto&                       registry = get_registry();
            std::lock_guard<std::mutex> lock{ registry.mutex };
            std::erase_if(registry.threads,
                          [this](const auto& entry) { return entry.get() == regions; });
        }

        handle_t(const handle_t&)            = delete;
        handle_t& operator=(const handle_t&) = delete;

        thread_open_regions<SdkBackend>* regions = nullptr;
    };

    static registry_t& get_registry()
    {
        // Never destroyed: handle_t destructors run at thread exit, which can follow
        // static destruction. registry_t is the context type so every translation unit
        // including this header shares one registry.
        static auto*& s_registry =
            common::static_object<registry_t, registry_t>::construct(
                common::do_not_destroy{});
        return *s_registry;
    }
};

// Map and key under which the begin and end records of one region meet.
template <policies::domain_service::backend SdkBackend>
struct region_slot
{
    pending_regions_t<SdkBackend>& regions;
    std::uint64_t                  key;
};

template <policies::domain_service::backend SdkBackend>
auto
get_unified_name(const typename SdkBackend::callback_tracing_record_t& record)
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
function_args_t
collect_args(const typename SdkBackend::callback_tracing_record_t& record)
{
    auto       args      = function_args_t{};
    const auto operation = static_cast<SdkBackend::ompt_operation_t>(record.operation);

    // ROCProfiler-SDK recommends one dereference on ENTER to avoid faults.
    const auto max_deref = (record.phase == SdkBackend::CALLBACK_PHASE_ENTER ||
                            operation == SdkBackend::OMPT_ID_parallel_begin)
                               ? 1
                               : 2;

    SdkBackend::iterate_callback_tracing_kind_operation_args(
        record, callback::detail::iterate_args_callback, max_deref, &args);

    const auto* payload =
        static_cast<const SdkBackend::callback_tracing_ompt_data_t*>(record.payload);

    if(payload)
    {
        append_flag_args<SdkBackend>(args, operation, *payload);
    }

    return args;
}

// Requires a non-null payload for parallel operations (guaranteed by should_skip).
template <policies::domain_service::backend SdkBackend>
region_slot<SdkBackend>
find_slot(const typename SdkBackend::callback_tracing_record_t& record)
{
    auto& thread_regions = open_regions<SdkBackend>::current();

    if(record.operation != SdkBackend::OMPT_ID_parallel_begin &&
       record.operation != SdkBackend::OMPT_ID_parallel_end)
    {
        return { .regions = thread_regions.standard,
                 .key     = record.correlation_id.internal };
    }

    const auto* payload =
        static_cast<const SdkBackend::callback_tracing_ompt_data_t*>(record.payload);
    const void* parallel_data_address =
        (record.operation == SdkBackend::OMPT_ID_parallel_begin)
            ? payload->args.parallel_begin.parallel_data
            : payload->args.parallel_end.parallel_data;

    static_assert(sizeof(std::uintptr_t) <= sizeof(std::uint64_t));
    // The address is only compared for identity, never dereferenced.
    return { .regions = thread_regions.parallel,
             .key     = reinterpret_cast<std::uintptr_t>(parallel_data_address) };
}

// Records a completed OMPT region (instant, standard, or parallel) into the trace
// cache: registers the category/thread-info metadata once and stores the region
// sample, mirroring domains::callback::on_tracing_api_exit's tail. A non-empty
// @p name_suffix is appended to the region name as " [suffix]".
template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename BacktraceDataT>
void
emit_region(const typename SdkBackend::callback_tracing_record_t& record,
            typename SdkBackend::timestamp_t                      begin_timestamp,
            typename SdkBackend::timestamp_t                      end_timestamp,
            BacktraceDataT& backtrace_data, const function_args_t& args,
            std::string_view name_suffix = {})
{
    std::string_view name{ get_unified_name<SdkBackend>(record) };

    auto call_stack = Externals::get_backtrace_json(backtrace_data);

    Externals::get_metadata_registry().add_string(Category<Externals>::k_name);
    Externals::get_metadata_registry().ensure_thread(record.thread_id, [&] {
        return
            typename Externals::thread_info_t{ .parent_process_id = Externals::get_ppid(),
                                               .process_id        = Externals::get_pid(),
                                               .thread_id         = record.thread_id,
                                               .start             = 0,
                                               .end               = 0,
                                               .extdata           = "{}" };
    });

    const std::string args_str = get_args_string(args);

    Externals::get_buffer_storage().store(typename Externals::region_sample{
        record.thread_id,
        (name_suffix.empty() ? name : fmt::format("{} [{}]", name, name_suffix)),
        record.correlation_id.internal,
        SdkBackend::get_parent_stack_id(record.correlation_id), begin_timestamp,
        end_timestamp, call_stack.dump(), args_str, Category<Externals>::k_name });
}

// To handle events without finalization, perfetto push must occur in start
// Allows capture of worker thread implicit tasks and sync regions
template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
void
begin_region(const typename SdkBackend::callback_tracing_record_t& record,
             typename SdkBackend::timestamp_t                      begin_timestamp)
{
    if(Externals::get_use_timemory())
    {
        Externals::tracing_push_timemory(typename Category<Externals>::type{},
                                         get_unified_name<SdkBackend>(record));
    }

    auto slot = find_slot<SdkBackend>(record);
    slot.regions.emplace(slot.key,
                         pending_region<SdkBackend>{ record, begin_timestamp,
                                                     collect_args<SdkBackend>(record) });
}

// Closes the region opened by the matching begin_region. An end without a matching
// begin is emitted as an instant event.
template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename BacktraceDataT>
void
end_region(const typename SdkBackend::callback_tracing_record_t& record,
           typename SdkBackend::timestamp_t end_timestamp, BacktraceDataT& backtrace_data)
{
    if(Externals::get_use_timemory())
    {
        Externals::tracing_pop_timemory(typename Category<Externals>::type{},
                                        get_unified_name<SdkBackend>(record));
    }

    auto slot = find_slot<SdkBackend>(record);
    auto node = slot.regions.extract(slot.key);

    if(node.empty())
    {
        emit_region<SdkBackend, Externals, Category>(record, end_timestamp, end_timestamp,
                                                     backtrace_data,
                                                     collect_args<SdkBackend>(record));
        return;
    }

    const auto& begin = node.mapped();
    emit_region<SdkBackend, Externals, Category>(
        record, begin.begin_timestamp, end_timestamp, backtrace_data, begin.args);
}

// An instant event is one that has its begin_timestamp = end_timestamp
template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename BacktraceDataT>
void
instant_region(const typename SdkBackend::callback_tracing_record_t& record,
               typename SdkBackend::timestamp_t timestamp, BacktraceDataT& backtrace_data)
{
    if(Externals::get_use_timemory())
    {
        const std::string_view name = get_unified_name<SdkBackend>(record);
        Externals::tracing_push_timemory(typename Category<Externals>::type{}, name);
        Externals::tracing_pop_timemory(typename Category<Externals>::type{}, name);
    }

    emit_region<SdkBackend, Externals, Category>(
        record, timestamp, timestamp, backtrace_data, collect_args<SdkBackend>(record));
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
        if(has_flag(flag, k_task_initial))
        {
            return true;  // Skips both the start and end
        }
    }
    else if(operation == SdkBackend::OMPT_ID_thread_begin)
    {
        if(payload_data->args.thread_begin.thread_type == SdkBackend::OMPT_THREAD_INITIAL)
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

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
inline void
on_ompt_enter(typename SdkBackend::callback_tracing_record_t record,
              typename SdkBackend::user_data_t* /*user_data*/, void* /*callback_data*/,
              typename SdkBackend::timestamp_t timestamp = SdkBackend::get_timestamp())
{
    if(!Externals::is_active() || detail::should_skip<SdkBackend>(record))
    {
        return;
    }

    detail::begin_region<SdkBackend, Externals, Category>(record, timestamp);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
inline void
on_ompt_exit(typename SdkBackend::callback_tracing_record_t record,
             typename SdkBackend::user_data_t* /*user_data*/, void* /*callback_data*/,
             typename SdkBackend::timestamp_t timestamp = SdkBackend::get_timestamp())
{
    if(!Externals::is_active() || detail::should_skip<SdkBackend>(record))
    {
        return;
    }

    auto backtrace_data = Externals::get_backtrace_data(
        Externals::check_backtrace_operations(record.kind, record.operation));

    detail::end_region<SdkBackend, Externals, Category>(record, timestamp,
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
    if(!Externals::is_active() || detail::should_skip<SdkBackend>(record))
    {
        return;
    }

    const auto operation = static_cast<SdkBackend::ompt_operation_t>(record.operation);

    // Received but not processed: callback_functions is a "fake" callback and
    // thread_end arrives after our tool finalizes.
    if(operation == SdkBackend::OMPT_ID_callback_functions ||
       operation == SdkBackend::OMPT_ID_thread_end)
    {
        return;
    }

    auto backtrace_data = Externals::get_backtrace_data(
        Externals::check_backtrace_operations(record.kind, record.operation));

    switch(operation)
    {
        case SdkBackend::OMPT_ID_parallel_begin:
            detail::begin_region<SdkBackend, Externals, Category>(record, timestamp);
            break;
        case SdkBackend::OMPT_ID_parallel_end:
            detail::end_region<SdkBackend, Externals, Category>(record, timestamp,
                                                                backtrace_data);
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
            // No corresponding "end" will be received for these callbacks
            detail::instant_region<SdkBackend, Externals, Category>(record, timestamp,
                                                                    backtrace_data);
            break;
        default:
            LOG_WARNING("unhandled PHASE_NONE operation {} for OMPT callback record",
                        record.operation);
    }
}

// Regions still open on any thread at finalization never received their end. They are
// emitted with the finalization timestamp as their end and an "[incomplete]" name suffix.
template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
void
on_ompt_finalize()
{
    // Draining before the tool has stopped would race with callbacks still in flight.
    if(!Externals::is_finalized())
    {
        LOG_WARNING("on_ompt_finalize() called while state is not Finalized; "
                    "skipping orphan drain");
        return;
    }

    static constexpr std::string_view k_incomplete_suffix{ "incomplete" };

    auto empty_backtrace_data = Externals::get_backtrace_data(false);

    // One shared end-of-program timestamp for every region drained in this pass.
    const auto finalize_timestamp = SdkBackend::get_timestamp();

    detail::open_regions<SdkBackend>::for_each_thread(
        [&](detail::thread_open_regions<SdkBackend>& thread_regions) {
            for(auto* regions : { &thread_regions.parallel, &thread_regions.standard })
            {
                for(const auto& [key, pending] : *regions)
                {
                    detail::emit_region<SdkBackend, Externals, Category>(
                        pending.record, pending.begin_timestamp, finalize_timestamp,
                        empty_backtrace_data, pending.args, k_incomplete_suffix);
                }
                regions->clear();
            }
        });
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
    .on_configure = on_tracing_api_configure<Externals>,
    .on_finalize  = on_ompt_finalize<SdkBackend, Externals, ompt_api_category>
};

}  // namespace rocprofsys::domains::callback::ompt
