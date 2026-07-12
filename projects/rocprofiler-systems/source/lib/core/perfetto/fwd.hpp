// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <memory>

namespace perfetto
{
class TracingSession;
}

namespace rocprofsys::core
{
struct engine_config;

struct tracing_session_deleter
{
    void operator()(::perfetto::TracingSession* session) const noexcept;
};

using tracing_session_ptr =
    std::unique_ptr<::perfetto::TracingSession, tracing_session_deleter>;

struct perfetto_sdk_backend
{
    using session_ptr = tracing_session_ptr;

    void                      init_sdk(const engine_config& cfg) const;
    [[nodiscard]] session_ptr start_cached_session(const engine_config& cfg) const;
    void                      flush_and_stop(session_ptr& session) const;
};

template <typename Backend>
concept perfetto_backend = requires(Backend backend, const engine_config& cfg,
                                    typename Backend::session_ptr& session) {
    typename Backend::session_ptr;
    { backend.init_sdk(cfg) } -> std::same_as<void>;
    { backend.start_cached_session(cfg) } -> std::same_as<typename Backend::session_ptr>;
    { backend.flush_and_stop(session) } -> std::same_as<void>;
};

template <perfetto_backend Backend = perfetto_sdk_backend>
class basic_cached_perfetto_engine;

using cached_perfetto_engine = basic_cached_perfetto_engine<perfetto_sdk_backend>;
}  // namespace rocprofsys::core
