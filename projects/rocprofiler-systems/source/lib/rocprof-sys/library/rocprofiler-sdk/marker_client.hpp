// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/marker_writer.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

namespace rocprofsys
{
namespace control
{
class trace_control;
}

namespace rocprofiler_sdk
{

// Manages the control context and handles all marker callbacks.
// Routes to marker_writer for output and trace_control for pause/resume/region filtering.
class marker_client
{
public:
    explicit marker_client(control::trace_control& controller);
    ~marker_client() = default;

    marker_client(const marker_client&)            = delete;
    marker_client& operator=(const marker_client&) = delete;
    marker_client(marker_client&&)                 = default;
    marker_client& operator=(marker_client&&)      = default;

    // Configure MARKER_CORE_API and MARKER_CONTROL_API services on the context
    void configure_services(rocprofiler_context_id_t ctx);

    rocprofiler_context_id_t get_context() const noexcept { return m_ctx; }

    // Check if marker writing is enabled (ROCPROFSYS_ROCM_MARKERS domain)
    bool is_write_enabled() const;

    // Delegate to trace_control
    bool region_filter_active() const;
    bool should_write_markers() const;
    void shutdown();

private:
    rocprofiler_context_id_t m_ctx{ 0 };
    marker_writer            m_writer;
    control::trace_control&  m_controller;

    // Handle MARKER_CORE_API callback - contains all logic
    void handle_marker_core_enter(rocprofiler_callback_tracing_record_t record,
                                  rocprofiler_user_data_t*              user_data,
                                  rocprofiler_timestamp_t               ts);

    void handle_marker_core_exit(rocprofiler_callback_tracing_record_t record,
                                 rocprofiler_user_data_t*              user_data,
                                 rocprofiler_timestamp_t               ts);

    // Handle MARKER_CONTROL_API callback - delegates to trace_control
    void handle_marker_control(rocprofiler_callback_tracing_record_t record);

    // Static callbacks registered with rocprofiler-sdk
    static void marker_core_callback(rocprofiler_callback_tracing_record_t record,
                                     rocprofiler_user_data_t*              user_data,
                                     void*                                 callback_data);

    static void marker_control_callback(rocprofiler_callback_tracing_record_t record,
                                        rocprofiler_user_data_t*              user_data,
                                        void* callback_data);
};

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
