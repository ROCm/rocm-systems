// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// All rocprofiler-sdk headers come transitively through the backend shim.
#include "library/rocprofiler-sdk/fwd.hpp"
#include "library/rocprofiler-sdk/marker_writer.hpp"
#include "library/rocprofiler-sdk/trace_control.hpp"

#include "core/common_types.hpp"
#include "core/demangler.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "logger/debug.hpp"

#include <timemory/hash/types.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{

struct roctx_client_config
{
    bool        pause_resume_enabled{ false };
    bool        use_perfetto{ false };
    bool        use_timemory{ false };
    bool        perfetto_annotations{ false };
    std::string selected_trace_regions{};
};

// ─── roctx_client<Wrapper, MarkerWriterPolicy> ───────────────────────────────
//
// Wrapper first matches marker_writer<Wrapper, Policy> convention:
//   roctx_client<>                               → uses backend + default_marker_policy
//   roctx_client<MockBackend>                    → MockBackend + default_marker_policy
//   roctx_client<MockBackend, MockPolicy>        → fully injectable for tests

template <typename Wrapper, typename MarkerWriterPolicy = default_marker_policy>
class roctx_client
{
public:
    explicit roctx_client(const roctx_client_config& roctx_cfg);

    ~roctx_client()                              = default;
    roctx_client(const roctx_client&)            = delete;
    roctx_client& operator=(const roctx_client&) = delete;
    roctx_client(roctx_client&&)                 = default;
    roctx_client& operator=(roctx_client&&)      = default;

    void configure_services(typename Wrapper::context_id ctx);

    std::shared_ptr<control::trace_control> get_controller() const
    {
        return m_controller;
    }

private:
    struct marker_range_entry
    {
        tim::hash_value_t             hash;
        typename Wrapper::timestamp_t begin_ts;
        bool                          write_enabled;
        std::uint64_t                 range_id{ 0 };
    };

    using marker_range_stack_t = std::vector<marker_range_entry>;

    typename Wrapper::context_id               m_ctx{};
    roctx_client_config                        m_config;
    marker_writer<Wrapper, MarkerWriterPolicy> m_writer;
    std::shared_ptr<control::trace_control>    m_controller{};

    static thread_local marker_range_stack_t m_pushed_ranges;
    static thread_local marker_range_stack_t m_started_ranges;

    void handle_marker_core_enter(typename Wrapper::callback_tracing_record record,
                                  typename Wrapper::user_data_t*            user_data,
                                  typename Wrapper::timestamp_t             ts);
    void handle_marker_core_exit(typename Wrapper::callback_tracing_record record,
                                 typename Wrapper::user_data_t*            user_data,
                                 typename Wrapper::timestamp_t             ts);
    void handle_marker_control(typename Wrapper::callback_tracing_record record);

    static void marker_core_callback(typename Wrapper::callback_tracing_record record,
                                     typename Wrapper::user_data_t*            user_data,
                                     void* callback_data);
    static void marker_control_callback(typename Wrapper::callback_tracing_record record,
                                        typename Wrapper::user_data_t* user_data,
                                        void*                          callback_data);
};

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys

// ─── Template implementations ────────────────────────────────────────────────

namespace rocprofsys::rocprofiler_sdk
{

// ─── Thread-local storage ────────────────────────────────────────────────────

template <typename Wrapper, typename MarkerWriterPolicy>
thread_local typename roctx_client<Wrapper, MarkerWriterPolicy>::marker_range_stack_t
    roctx_client<Wrapper, MarkerWriterPolicy>::m_pushed_ranges{};

template <typename Wrapper, typename MarkerWriterPolicy>
thread_local typename roctx_client<Wrapper, MarkerWriterPolicy>::marker_range_stack_t
    roctx_client<Wrapper, MarkerWriterPolicy>::m_started_ranges{};

// ─── Constructor ─────────────────────────────────────────────────────────────

template <typename Wrapper, typename MarkerWriterPolicy>
roctx_client<Wrapper, MarkerWriterPolicy>::roctx_client(
    const roctx_client_config& roctx_cfg)
: m_config{ roctx_cfg }
, m_writer{ roctx_cfg.use_perfetto, roctx_cfg.use_timemory,
            roctx_cfg.perfetto_annotations }
, m_controller{ std::make_shared<control::trace_control>(
      roctx_cfg.selected_trace_regions) }
{}

// ─── Synthetic push-range ID ──────────────────────────────────────────────────
// Starts at UINT64_MAX and decrements to stay away from SDK-allocated IDs.

namespace roctx_client_detail
{
inline std::atomic<std::uint64_t>&
s_push_range_id()
{
    static std::atomic<std::uint64_t> id{ UINT64_MAX };
    return id;
}

template <typename Wrapper>
int
iterate_args_callback(typename Wrapper::callback_tracing_kind, std::int32_t,
                      std::uint32_t arg_number, const void* const, std::int32_t,
                      const char* arg_type, const char* arg_name,
                      const char* arg_value_str, std::int32_t, void* data)
{
    auto* args = static_cast<function_args_t*>(data);
    if(arg_type && arg_name && arg_value_str)
    {
        args->emplace_back(argument_info{ arg_number,
                                          rocprofsys::utility::demangle(arg_type),
                                          arg_name, arg_value_str });
    }
    return 0;
}

template <typename Wrapper>
void
configure_callback_tracing(typename Wrapper::context_id            context_id,
                           typename Wrapper::callback_tracing_kind kind,
                           typename Wrapper::tracing_operation*    operations,
                           size_t                                  operations_count,
                           typename Wrapper::callback_tracing_cb_t callback,
                           void*                                   callback_args)
{
    auto status = Wrapper::configure_callback_tracing_service(
        context_id, kind, operations, operations_count, callback, callback_args);
    if(status != Wrapper::STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to configure marker core callback : {}",
                    Wrapper::get_status_string(status));
    }
}

template <typename Wrapper>
std::string
collect_args(typename Wrapper::callback_tracing_record record)
{
    auto args = function_args_t{};
    Wrapper::iterate_callback_tracing_kind_operation_args(
        record, iterate_args_callback<Wrapper>, 2, &args);
    return get_args_string(args);
}

}  // namespace roctx_client_detail

// ─── configure_services ──────────────────────────────────────────────────────

template <typename Wrapper, typename MarkerWriterPolicy>
void
roctx_client<Wrapper, MarkerWriterPolicy>::configure_services(
    typename Wrapper::context_id ctx)
{
    m_ctx = ctx;

    roctx_client_detail::configure_callback_tracing<Wrapper>(
        m_ctx, Wrapper::CALLBACK_TRACING_MARKER_CORE_API, nullptr, 0,
        marker_core_callback, this);

    if(m_config.pause_resume_enabled)
    {
        auto control_ops = std::array<typename Wrapper::tracing_operation, 2>{
            Wrapper::MARKER_CONTROL_API_ID_roctxProfilerPause,
            Wrapper::MARKER_CONTROL_API_ID_roctxProfilerResume
        };
        roctx_client_detail::configure_callback_tracing<Wrapper>(
            m_ctx, Wrapper::CALLBACK_TRACING_MARKER_CONTROL_API, control_ops.data(),
            control_ops.size(), marker_control_callback, this);
    }
}

// ─── handle_marker_core_enter ─────────────────────────────────────────────────

template <typename Wrapper, typename MarkerWriterPolicy>
void
roctx_client<Wrapper, MarkerWriterPolicy>::handle_marker_core_enter(
    typename Wrapper::callback_tracing_record record,
    typename Wrapper::user_data_t* user_data, typename Wrapper::timestamp_t ts)
{
    auto*      data = static_cast<typename Wrapper::marker_payload_t*>(record.payload);
    const bool write_enabled = m_controller->should_write_markers();

    switch(record.operation)
    {
        case Wrapper::MARKER_CORE_API_ID_roctxRangePushA:
        {
            const char*         name = data->args.roctxRangePushA.message;
            const std::uint64_t range_id =
                roctx_client_detail::s_push_range_id().fetch_sub(
                    1, std::memory_order_relaxed);
            m_controller->handle_range_start(range_id, name);
            const bool pushed_write_enabled = m_controller->should_write_markers();
            m_pushed_ranges.push_back(
                { tim::add_hash_id(name), ts, pushed_write_enabled, range_id });
            if(pushed_write_enabled) m_writer.write_begin(name);
            break;
        }
        case Wrapper::MARKER_CORE_API_ID_roctxRangeStartA:
        {
            tim::add_hash_id(data->args.roctxRangeStartA.message);
            break;
        }
        case Wrapper::MARKER_CORE_API_ID_roctxMarkA:
        {
            const char* name = data->args.roctxMarkA.message;
            tim::add_hash_id(name);
            if(write_enabled) m_writer.write_begin(name);
            break;
        }
        default:
        {
            if(write_enabled)
            {
                const auto& name =
                    trace_cache::get_metadata_registry().get_callback_tracing_info().at(
                        record.kind, record.operation);
                m_writer.write_begin(name);
            }
            break;
        }
    }

    user_data->value = ts;
}

// ─── handle_marker_core_exit ──────────────────────────────────────────────────

template <typename Wrapper, typename MarkerWriterPolicy>
void
roctx_client<Wrapper, MarkerWriterPolicy>::handle_marker_core_exit(
    typename Wrapper::callback_tracing_record record,
    typename Wrapper::user_data_t* user_data, typename Wrapper::timestamp_t ts)
{
    auto* data = static_cast<typename Wrapper::marker_payload_t*>(record.payload);
    const std::uint64_t begin_ts = user_data->value;
    const auto          args_str = roctx_client_detail::collect_args<Wrapper>(record);

    auto pop_and_write = [&](marker_range_stack_t& stack) {
        auto        range = stack.back();
        const char* name  = nullptr;
        stack.pop_back();
        tim::get_hash_identifier_fast(range.hash, name);
        if(range.write_enabled && name)
            m_writer.write_end(name, range.begin_ts, ts, args_str, record);
    };

    switch(record.operation)
    {
        case Wrapper::MARKER_CORE_API_ID_roctxRangePop:
        {
            if(m_pushed_ranges.empty())
            {
                LOG_CRITICAL("roctxRangePop does not have corresponding roctxRangePush "
                             "(skipping)");
                return;
            }
            const auto range_id = m_pushed_ranges.back().range_id;
            pop_and_write(m_pushed_ranges);
            m_controller->handle_range_stop(range_id);
            break;
        }
        case Wrapper::MARKER_CORE_API_ID_roctxRangeStop:
        {
            if(m_started_ranges.empty())
            {
                LOG_CRITICAL("roctxRangeStop does not have corresponding roctxRangeStart "
                             "(skipping)");
                return;
            }
            pop_and_write(m_started_ranges);
            m_controller->handle_range_stop(data->args.roctxRangeStop.id);
            break;
        }
        case Wrapper::MARKER_CORE_API_ID_roctxMarkA:
        {
            if(m_controller->should_write_markers())
                m_writer.write_end(data->args.roctxMarkA.message, begin_ts, ts, args_str,
                                   record);
            break;
        }
        case Wrapper::MARKER_CORE_API_ID_roctxRangePushA:
        {
            return;
        }
        case Wrapper::MARKER_CORE_API_ID_roctxRangeStartA:
        {
            const char* name     = data->args.roctxRangeStartA.message;
            auto        range_id = data->retval.roctx_range_id_t_retval;
            m_controller->handle_range_start(range_id, name);
            const bool write_enabled = m_controller->should_write_markers();
            m_started_ranges.push_back(
                { tim::get_hash_id(name), begin_ts, write_enabled });
            if(write_enabled) m_writer.write_begin(name);
            return;
        }
        default:
        {
            if(m_controller->should_write_markers())
            {
                const auto& name =
                    trace_cache::get_metadata_registry().get_callback_tracing_info().at(
                        record.kind, record.operation);
                m_writer.write_end(name, begin_ts, ts, args_str, record);
            }
            break;
        }
    }
}

// ─── handle_marker_control ────────────────────────────────────────────────────

template <typename Wrapper, typename MarkerWriterPolicy>
void
roctx_client<Wrapper, MarkerWriterPolicy>::handle_marker_control(
    typename Wrapper::callback_tracing_record record)
{
    if(record.operation == Wrapper::MARKER_CONTROL_API_ID_roctxProfilerPause &&
       record.phase == Wrapper::CALLBACK_PHASE_ENTER)
    {
        m_controller->handle_pause(record.thread_id);
    }
    else if(record.operation == Wrapper::MARKER_CONTROL_API_ID_roctxProfilerResume &&
            record.phase == Wrapper::CALLBACK_PHASE_EXIT)
    {
        m_controller->handle_resume(record.thread_id);
    }
}

// ─── Static callbacks ─────────────────────────────────────────────────────────

template <typename Wrapper, typename MarkerWriterPolicy>
void
roctx_client<Wrapper, MarkerWriterPolicy>::marker_core_callback(
    typename Wrapper::callback_tracing_record record,
    typename Wrapper::user_data_t* user_data, void* callback_data)
{
    if(!callback_data) return;
    auto* client = static_cast<roctx_client*>(callback_data);

    typename Wrapper::timestamp_t ts{};
    Wrapper::get_timestamp(&ts);

    if(record.phase == Wrapper::CALLBACK_PHASE_ENTER)
        client->handle_marker_core_enter(record, user_data, ts);
    else if(record.phase == Wrapper::CALLBACK_PHASE_EXIT)
        client->handle_marker_core_exit(record, user_data, ts);
}

template <typename Wrapper, typename MarkerWriterPolicy>
void
roctx_client<Wrapper, MarkerWriterPolicy>::marker_control_callback(
    typename Wrapper::callback_tracing_record record, typename Wrapper::user_data_t*,
    void*                                     callback_data)
{
    if(!callback_data) return;
    static_cast<roctx_client*>(callback_data)->handle_marker_control(record);
}

}  // namespace rocprofsys::rocprofiler_sdk
