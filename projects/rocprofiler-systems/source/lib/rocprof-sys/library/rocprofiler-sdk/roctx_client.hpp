// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/marker_writer.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstdint>
#include <functional>

namespace rocprofsys
{
namespace rocprofiler_sdk
{

struct marker_handlers
{
    std::function<bool()>                      should_write;
    std::function<void(uint64_t, const char*)> on_range_start;
    std::function<void(uint64_t)>              on_range_stop;
    std::function<void()>                      on_pause;
    std::function<void()>                      on_resume;
    std::function<void()>                      on_shutdown;
};

class roctx_client
{
public:
    explicit roctx_client(marker_handlers handlers);
    ~roctx_client() = default;

    roctx_client(const roctx_client&)            = delete;
    roctx_client& operator=(const roctx_client&) = delete;
    roctx_client(roctx_client&&)                 = default;
    roctx_client& operator=(roctx_client&&)      = default;

    void configure_services(rocprofiler_context_id_t ctx);

    rocprofiler_context_id_t get_context() const noexcept { return m_ctx; }
    bool                     is_write_enabled() const noexcept { return m_write_enabled; }

    bool should_write_markers() const;
    void shutdown();

private:
    rocprofiler_context_id_t m_ctx{ 0 };
    marker_writer            m_writer;
    marker_handlers          m_handlers;
    bool                     m_write_enabled{ false };

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
