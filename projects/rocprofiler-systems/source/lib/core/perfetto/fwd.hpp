// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstddef>
#include <memory>
#include <vector>

namespace perfetto
{
class TracingSession;
}

namespace rocprofsys
{

class output_file_registry;

namespace core
{
struct engine_config;

enum class perfetto_engine_mode;

struct tracing_session_deleter
{
    void operator()(::perfetto::TracingSession* session) const noexcept;
};

using tracing_session_ptr =
    std::unique_ptr<::perfetto::TracingSession, tracing_session_deleter>;

struct perfetto_sdk_backend
{
    using session_ptr = tracing_session_ptr;

    void                            init_sdk(const engine_config& cfg) const;
    [[nodiscard]] session_ptr       start_session(const engine_config& cfg,
                                                  perfetto_engine_mode mode, int fd) const;
    void                            flush_and_stop(session_ptr& session) const;
    [[nodiscard]] std::vector<char> read_trace(session_ptr& session) const;
    void                            destroy_session(session_ptr& session) const noexcept;
    void                            release_session(session_ptr& session) const noexcept;
};

template <typename Backend>
concept perfetto_backend =
    requires(Backend backend, const engine_config& cfg, perfetto_engine_mode mode, int fd,
             typename Backend::session_ptr& session) {
        typename Backend::session_ptr;
        { backend.init_sdk(cfg) } -> std::same_as<void>;
        {
            backend.start_session(cfg, mode, fd)
        } -> std::same_as<typename Backend::session_ptr>;
        { backend.flush_and_stop(session) } -> std::same_as<void>;
        { backend.read_trace(session) } -> std::same_as<std::vector<char>>;
        { backend.destroy_session(session) } noexcept -> std::same_as<void>;
        { backend.release_session(session) } noexcept -> std::same_as<void>;
    };

template <perfetto_backend Backend = perfetto_sdk_backend>
class basic_perfetto_engine;

using perfetto_engine = basic_perfetto_engine<perfetto_sdk_backend>;
}  // namespace core

namespace perfetto
{
void
setup();

void
start();

void
stop();

void
post_process(bool&, output_file_registry&);
}  // namespace perfetto
}  // namespace rocprofsys
