// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <utility>

namespace perfetto
{
class TracingSession;
}

namespace rocprofsys::core
{
struct session_backend
{
    [[nodiscard]] std::unique_ptr<::perfetto::TracingSession> new_trace() const;
    void                                                      flush_track_events() const;
};

template <typename Backend, typename TraceConfig, typename ErrorCallback>
[[nodiscard]] auto
start_tracing_session(Backend& backend, const TraceConfig& trace_cfg, int fd,
                      ErrorCallback&& on_error)
{
    auto session = backend.new_trace();
    session->SetOnErrorCallback(std::forward<ErrorCallback>(on_error));
    session->Setup(trace_cfg, fd);
    session->StartBlocking();
    return session;
}

template <typename Backend, typename Session>
void
flush_and_stop_session(Backend& backend, Session& session)
{
    backend.flush_track_events();
    session.FlushBlocking();
    session.StopBlocking();
}
}  // namespace rocprofsys::core
