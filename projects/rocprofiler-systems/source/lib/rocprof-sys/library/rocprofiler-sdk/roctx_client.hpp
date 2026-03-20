// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/marker_writer.hpp"
#include "library/rocprofiler-sdk/trace_control.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <timemory/hash/types.hpp>

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

    std::shared_ptr<control::trace_control> get_controller() const
    {
        return m_controller;
    }

private:
    struct marker_range_entry
    {
        tim::hash_value_t       hash;
        rocprofiler_timestamp_t begin_ts;
        bool                    write_enabled;
    };

    using marker_range_stack_t = std::vector<marker_range_entry>;

    rocprofiler_context_id_t                m_ctx{ 0 };
    roctx_client_config                     m_config;
    marker_writer<MarkerWriterPolicy>       m_writer;
    std::shared_ptr<control::trace_control> m_controller{};

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

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
