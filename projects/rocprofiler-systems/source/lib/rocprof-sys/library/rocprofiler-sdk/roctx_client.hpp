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
#include <tuple>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{

struct roctx_client_config
{
    bool        is_write_enabled{ false };
    bool        use_perfetto{ false };
    bool        use_timemory{ false };
    std::string selected_trace_regions{};
};

class roctx_client
{
public:
    roctx_client(roctx_client_config roctx_cfg);
    ~roctx_client() = default;

    roctx_client(const roctx_client&)            = delete;
    roctx_client& operator=(const roctx_client&) = delete;
    roctx_client(roctx_client&&)                 = default;
    roctx_client& operator=(roctx_client&&)      = default;

    void configure_services(rocprofiler_context_id_t ctx);

    bool should_write_markers() const;
    void shutdown();

    std::shared_ptr<control::trace_control> get_controller() { return m_controller; }

private:
    using marker_range_stack_t =
        std::vector<std::tuple<tim::hash_value_t, rocprofiler_timestamp_t, bool>>;

    rocprofiler_context_id_t m_ctx{ 0 };

    roctx_client_config                     m_config;
    marker_writer                           m_writer;
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
