// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/perfetto/fwd.hpp"
#include "core/perfetto/output_layout.hpp"
#include "core/perfetto/sinks.hpp"

#include <memory>
#include <vector>

#include <sys/types.h>

namespace rocprofsys
{
class output_file_registry;
class track_registry;

namespace trace_cache
{
class post_processor;

// RAII orchestrator for the cached-perfetto post-processing pipeline. It owns
// the perfetto_engine, trace_sink, and track_registry trio, wires them into the
// active post_processor on construction, and drains/finalizes on destruction.
class cached_perfetto_session
{
public:
    cached_perfetto_session(output_file_registry& registry, pid_t root_pid,
                            core::perfetto_output_layout layout,
                            const std::vector<int>&      source_pids,
                            post_processor&              processor);
    ~cached_perfetto_session() noexcept;

    cached_perfetto_session(const cached_perfetto_session&)            = delete;
    cached_perfetto_session& operator=(const cached_perfetto_session&) = delete;
    cached_perfetto_session(cached_perfetto_session&&)                 = delete;
    cached_perfetto_session& operator=(cached_perfetto_session&&)      = delete;

private:
    std::unique_ptr<core::perfetto_engine> m_engine;
    std::unique_ptr<core::trace_sink>      m_sink;
    std::unique_ptr<track_registry>        m_tracks;
    bool                                   m_started{ false };
};
}  // namespace trace_cache
}  // namespace rocprofsys
