// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common_types.hpp"

#include "library/rocprofiler-sdk/callback/common_tracing_callbacks.hpp"
#include "library/rocprofiler-sdk/types.hpp"
#include "logger/debug.hpp"

#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <atomic>
#include <cstdint>
#include <optional>
#include <stack>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::domains::callback::roctx
{

namespace detail
{

constexpr std::int32_t k_args_max_deref_depth = 2;

template <policies::domain_service::externals Externals>
struct open_range
{
    Externals::string_id_t name_id;
    std::uint64_t          begin_timestamp;
    bool                   write_enabled;
    std::uint64_t          range_id{ 0 };
};

template <policies::domain_service::externals Externals>
using range_stack_t =
    std::stack<open_range<Externals>, std::vector<open_range<Externals>>>;

template <policies::domain_service::backend SdkBackend>
struct region_span
{
    SdkBackend::timestamp_t begin;
    SdkBackend::timestamp_t end;
};

struct range_origin
{
    std::uint64_t begin_timestamp;
    std::uint64_t range_id;
};

template <policies::domain_service::externals Externals>
struct open_ranges
{
    static inline thread_local auto s_pushed = range_stack_t<Externals>{};

    static inline thread_local auto s_started = range_stack_t<Externals>{};
};

inline std::atomic<std::uint64_t>&
push_range_id_counter()
{
    static std::atomic<std::uint64_t> s_counter{ UINT64_MAX };
    return s_counter;
}

[[nodiscard]] constexpr const char*
message_or_empty(const char* message) noexcept
{
    return message != nullptr ? message : "";
}

template <policies::domain_service::externals Externals, typename Trigger>
[[nodiscard]] bool
should_write(const Trigger& trigger)
{
    auto* session = Externals::get_session();
    return trigger.should_write_markers() && session != nullptr &&
           session->is_active_without(Externals::roctx_trigger_name);
}

template <policies::domain_service::backend SdkBackend>
[[nodiscard]] std::string
collect_args(const typename SdkBackend::callback_tracing_record_t& record)
{
    auto args = function_args_t{};
    SdkBackend::iterate_callback_tracing_kind_operation_args(
        record, callback::detail::iterate_args_callback, k_args_max_deref_depth, &args);
    return get_args_string(args);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
void
emit_region(std::string_view name, const region_span<SdkBackend>& timespan,
            const typename SdkBackend::callback_tracing_record_t& record)
{
    constexpr const char* k_empty_json = "{}";

    constexpr std::uint32_t k_unknown_time = 0;
    Externals::get_metadata_registry().add_thread_info(
        { Externals::get_ppid(), Externals::get_pid(), record.thread_id, k_unknown_time,
          k_unknown_time, k_empty_json });

    Externals::get_buffer_storage().store(typename Externals::region_sample{
        record.thread_id, name, record.correlation_id.internal,
        SdkBackend::get_parent_stack_id(record.correlation_id), timespan.begin,
        timespan.end, k_empty_json, collect_args<SdkBackend>(record),
        Category<Externals>::k_name });
}

template <policies::domain_service::externals Externals,
          template <typename> class Category>
void
begin_region(std::string_view name)
{
    if(Externals::get_use_timemory())
    {
        Externals::tracing_push_timemory(typename Category<Externals>::type{}, name);
    }
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
void
end_region(std::string_view name, const region_span<SdkBackend>& span,
           const typename SdkBackend::callback_tracing_record_t& record)
{
    if(Externals::get_use_timemory())
    {
        Externals::tracing_pop_timemory(typename Category<Externals>::type{}, name);
    }

    emit_region<SdkBackend, Externals, Category>(name, span, record);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
void
close_range(range_stack_t<Externals>&                             ranges,
            typename SdkBackend::timestamp_t                      end_timestamp,
            const typename SdkBackend::callback_tracing_record_t& record)
{
    const auto range = ranges.top();
    ranges.pop();

    const char* name = Externals::lookup_string(range.name_id);
    if(range.write_enabled && name != nullptr)
    {
        end_region<SdkBackend, Externals, Category>(
            name, region_span<SdkBackend>{ range.begin_timestamp, end_timestamp },
            record);
    }
}

template <policies::domain_service::externals Externals,
          template <typename> class Category, typename Trigger>
void
open_range_on(range_stack_t<Externals>& ranges, const Trigger& trigger, const char* name,
              const range_origin& origin)
{
    const bool write_enabled = should_write<Externals>(trigger);
    ranges.push({ Externals::intern_string(name), origin.begin_timestamp, write_enabled,
                  origin.range_id });
    if(write_enabled)
    {
        begin_region<Externals, Category>(name);
    }
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename Trigger>
void
enter_range_push(Trigger& trigger, const typename SdkBackend::marker_payload_t& data,
                 typename SdkBackend::timestamp_t timestamp)
{
    const char*         name = message_or_empty(data.args.roctxRangePushA.message);
    const std::uint64_t range_id =
        push_range_id_counter().fetch_sub(1, std::memory_order_relaxed);

    trigger.on_range_start(range_id, name);
    open_range_on<Externals, Category>(open_ranges<Externals>::s_pushed, trigger, name,
                                       { timestamp, range_id });
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename Trigger>
void
enter_mark(const Trigger& trigger, const typename SdkBackend::marker_payload_t& data)
{
    const char* name = message_or_empty(data.args.roctxMarkA.message);
    static_cast<void>(Externals::intern_string(name));
    if(should_write<Externals>(trigger))
    {
        begin_region<Externals, Category>(name);
    }
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename Trigger>
void
enter_other(const Trigger&                                        trigger,
            const typename SdkBackend::callback_tracing_record_t& record)
{
    if(should_write<Externals>(trigger))
    {
        begin_region<Externals, Category>(
            SdkBackend::get_callback_tracing_names().at(record.kind, record.operation));
    }
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename Trigger>
void
exit_range_pop(Trigger&                                              trigger,
               const typename SdkBackend::callback_tracing_record_t& record,
               typename SdkBackend::timestamp_t                      timestamp)
{
    auto& ranges = open_ranges<Externals>::s_pushed;
    if(ranges.empty())
    {
        LOG_WARNING("roctxRangePop does not have corresponding roctxRangePush "
                    "(skipping)");
        return;
    }

    const auto range_id = ranges.top().range_id;
    close_range<SdkBackend, Externals, Category>(ranges, timestamp, record);
    trigger.on_range_stop(range_id);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename Trigger>
void
exit_range_stop(Trigger& trigger, const typename SdkBackend::marker_payload_t& data,
                const typename SdkBackend::callback_tracing_record_t& record,
                typename SdkBackend::timestamp_t                      timestamp)
{
    auto& ranges = open_ranges<Externals>::s_started;
    if(ranges.empty())
    {
        LOG_WARNING("roctxRangeStop does not have corresponding roctxRangeStart "
                    "(skipping)");
        return;
    }

    close_range<SdkBackend, Externals, Category>(ranges, timestamp, record);
    trigger.on_range_stop(data.args.roctxRangeStop.id);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename Trigger>
void
exit_mark(const Trigger& trigger, const typename SdkBackend::marker_payload_t& data,
          const typename SdkBackend::callback_tracing_record_t& record,
          const region_span<SdkBackend>&                        span)
{
    if(should_write<Externals>(trigger))
    {
        end_region<SdkBackend, Externals, Category>(
            message_or_empty(data.args.roctxMarkA.message), span, record);
    }
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename Trigger>
void
exit_range_start(Trigger& trigger, const typename SdkBackend::marker_payload_t& data,
                 typename SdkBackend::timestamp_t begin_timestamp)
{
    const char* name     = message_or_empty(data.args.roctxRangeStartA.message);
    const auto  range_id = data.retval.roctx_range_id_t_retval;

    trigger.on_range_start(range_id, name);
    open_range_on<Externals, Category>(open_ranges<Externals>::s_started, trigger, name,
                                       { begin_timestamp, 0 });
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category, typename Trigger>
void
exit_other(const Trigger&                                        trigger,
           const typename SdkBackend::callback_tracing_record_t& record,
           const region_span<SdkBackend>&                        span)
{
    if(should_write<Externals>(trigger))
    {
        end_region<SdkBackend, Externals, Category>(
            SdkBackend::get_callback_tracing_names().at(record.kind, record.operation),
            span, record);
    }
}

}  // namespace detail

template <policies::domain_service::externals Externals,
          template <typename> class Category>
inline void
on_roctx_core_configure()
{
    Externals::get_metadata_registry().add_string(Category<Externals>::k_name);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
inline void
on_roctx_core_enter(
    typename SdkBackend::callback_tracing_record_t record,
    typename SdkBackend::user_data_t* user_data, [[maybe_unused]] void* callback_data,
    typename SdkBackend::timestamp_t timestamp = SdkBackend::get_timestamp())
{
    auto* trigger = Externals::get_roctx_trigger();
    if(trigger == nullptr)
    {
        return;
    }

    const auto& data = *static_cast<const SdkBackend::marker_payload_t*>(record.payload);

    switch(record.operation)
    {
        case SdkBackend::MARKER_CORE_API_ID_roctxRangePushA:
            detail::enter_range_push<SdkBackend, Externals, Category>(*trigger, data,
                                                                      timestamp);
            break;
        case SdkBackend::MARKER_CORE_API_ID_roctxMarkA:
            detail::enter_mark<SdkBackend, Externals, Category>(*trigger, data);
            break;
        case SdkBackend::MARKER_CORE_API_ID_roctxRangeStartA:
        case SdkBackend::MARKER_CORE_API_ID_roctxRangePop:
        case SdkBackend::MARKER_CORE_API_ID_roctxRangeStop: break;
        default:
            detail::enter_other<SdkBackend, Externals, Category>(*trigger, record);
            break;
    }

    user_data->value = timestamp;
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals,
          template <typename> class Category>
inline void
on_roctx_core_exit(
    typename SdkBackend::callback_tracing_record_t record,
    typename SdkBackend::user_data_t* user_data, [[maybe_unused]] void* callback_data,
    typename SdkBackend::timestamp_t timestamp = SdkBackend::get_timestamp())
{
    auto* trigger = Externals::get_roctx_trigger();
    if(trigger == nullptr)
    {
        return;
    }

    const auto& data = *static_cast<const SdkBackend::marker_payload_t*>(record.payload);
    const auto  begin_timestamp = static_cast<SdkBackend::timestamp_t>(user_data->value);
    const detail::region_span<SdkBackend> span{ begin_timestamp, timestamp };

    switch(record.operation)
    {
        case SdkBackend::MARKER_CORE_API_ID_roctxRangePop:
            detail::exit_range_pop<SdkBackend, Externals, Category>(*trigger, record,
                                                                    timestamp);
            break;
        case SdkBackend::MARKER_CORE_API_ID_roctxRangeStop:
            detail::exit_range_stop<SdkBackend, Externals, Category>(*trigger, data,
                                                                     record, timestamp);
            break;
        case SdkBackend::MARKER_CORE_API_ID_roctxMarkA:
            detail::exit_mark<SdkBackend, Externals, Category>(*trigger, data, record,
                                                               span);
            break;
        case SdkBackend::MARKER_CORE_API_ID_roctxRangePushA: break;
        case SdkBackend::MARKER_CORE_API_ID_roctxRangeStartA:
            detail::exit_range_start<SdkBackend, Externals, Category>(*trigger, data,
                                                                      begin_timestamp);
            break;
        default:
            detail::exit_other<SdkBackend, Externals, Category>(*trigger, record, span);
            break;
    }
}

template <typename Externals>
struct roctx_api_category
{
    using type = Externals::rocm_marker_api_category;

    static constexpr std::string_view k_name = Externals::rocm_marker_api_category_name;
};

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_core_api = callback_domain_definition<SdkBackend>{
    .meta      = domain_descriptor{ .name  = "marker_core_api",
                                    .id    = SdkBackend::CALLBACK_TRACING_MARKER_CORE_API,
                                    .mode  = collection_mode::callback,
                                    .group = std::nullopt },
    .on_record = tracing_callback_dispatcher<
        SdkBackend, on_roctx_core_enter<SdkBackend, Externals, roctx_api_category>,
        on_roctx_core_exit<SdkBackend, Externals, roctx_api_category>>::callback,
    .on_configure = on_roctx_core_configure<Externals, roctx_api_category>
};

}  // namespace rocprofsys::domains::callback::roctx
