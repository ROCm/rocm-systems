// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <type_traits>
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

template <typename Backend>
concept trace_session_backend = requires(Backend& backend) {
    backend.new_trace();
    backend.flush_track_events();
};

template <typename Backend>
concept flushable_trace_session_backend =
    requires(Backend& backend) { backend.flush_track_events(); };

template <typename Session>
concept stoppable_trace_session = requires(Session& session) {
    session.FlushBlocking();
    session.StopBlocking();
};

template <typename Session, typename TraceConfig, typename ErrorCallback>
concept trace_session = requires(Session& session, const TraceConfig& trace_cfg, int fd,
                                 ErrorCallback callback) {
    session.SetOnErrorCallback(std::move(callback));
    session.Setup(trace_cfg, fd);
    session.StartBlocking();
} && stoppable_trace_session<Session>;

template <typename Backend, typename TraceConfig, typename ErrorCallback>
concept trace_session_backend_for =
    trace_session_backend<Backend> && requires(Backend& backend) {
        requires trace_session<std::remove_reference_t<decltype(*backend.new_trace())>,
                               TraceConfig, ErrorCallback>;
    };

template <typename Backend, typename TraceConfig, typename ErrorCallback>
    requires trace_session_backend_for<Backend, TraceConfig, std::decay_t<ErrorCallback>>
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
    requires flushable_trace_session_backend<Backend> && stoppable_trace_session<Session>
void
flush_and_stop_session(Backend& backend, Session& session)
{
    backend.flush_track_events();
    session.FlushBlocking();
    session.StopBlocking();
}
}  // namespace rocprofsys::core
