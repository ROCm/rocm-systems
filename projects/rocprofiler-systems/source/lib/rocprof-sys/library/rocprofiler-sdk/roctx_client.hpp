// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/control/session.hpp"
#include "core/control/triggers/roctx.hpp"
#include "library/rocprofiler-sdk/marker_writer.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <timemory/hash/types.hpp>

#include <cstdint>
#include <memory>
#include <string>
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

template <typename MarkerWriterPolicy = default_marker_policy>
class roctx_client
{
public:
    explicit roctx_client(const roctx_client_config& roctx_cfg);

    ~roctx_client()                              = default;
    roctx_client(const roctx_client&)            = delete;
    roctx_client& operator=(const roctx_client&) = delete;
    roctx_client(roctx_client&&)                 = default;
    roctx_client& operator=(roctx_client&&)      = default;

    void configure_services(rocprofiler_context_id_t ctx);

    std::shared_ptr<control::session> get_session() const { return m_session; }
    control::triggers::roctx&         get_trigger() { return *m_trigger; }

private:
    struct marker_range_entry
    {
        tim::hash_value_t       hash;
        rocprofiler_timestamp_t begin_ts;
        bool                    write_enabled;
        std::uint64_t           range_id{ 0 };
    };

    using marker_range_stack_t = std::vector<marker_range_entry>;

    rocprofiler_context_id_t                  m_ctx{ 0 };
    roctx_client_config                       m_config;
    marker_writer<MarkerWriterPolicy>         m_writer;
    std::shared_ptr<control::session>         m_session;
    std::unique_ptr<control::triggers::roctx> m_trigger;

    static thread_local marker_range_stack_t m_pushed_ranges;
    static thread_local marker_range_stack_t m_started_ranges;

    void handle_marker_core_enter(rocprofiler_callback_tracing_record_t record,
                                  rocprofiler_user_data_t*              user_data,
                                  rocprofiler_timestamp_t               ts);
    void handle_marker_core_exit(rocprofiler_callback_tracing_record_t record,
                                 rocprofiler_user_data_t*              user_data,
                                 rocprofiler_timestamp_t               ts);
    void handle_marker_control(rocprofiler_callback_tracing_record_t record);

    static void marker_core_callback(rocprofiler_callback_tracing_record_t record,
                                     rocprofiler_user_data_t*              user_data,
                                     void*                                 callback_data);
    static void marker_control_callback(rocprofiler_callback_tracing_record_t record,
                                        rocprofiler_user_data_t*              user_data,
                                        void* callback_data);
};

template <typename MarkerWriterPolicy>
thread_local typename roctx_client<MarkerWriterPolicy>::marker_range_stack_t
    roctx_client<MarkerWriterPolicy>::m_pushed_ranges{};

template <typename MarkerWriterPolicy>
thread_local typename roctx_client<MarkerWriterPolicy>::marker_range_stack_t
    roctx_client<MarkerWriterPolicy>::m_started_ranges{};

template <typename MarkerWriterPolicy>
roctx_client<MarkerWriterPolicy>::roctx_client(const roctx_client_config& roctx_cfg)
: m_config{ roctx_cfg }
, m_writer{ roctx_cfg.use_perfetto, roctx_cfg.use_timemory,
            roctx_cfg.perfetto_annotations }
, m_session{ std::make_shared<control::session>() }
, m_trigger{ std::make_unique<control::triggers::roctx>(
      *m_session, roctx_cfg.selected_trace_regions) }
{
    m_session->attach(*m_trigger);
}

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
