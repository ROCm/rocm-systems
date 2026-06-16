// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "engine.hpp"
#include "logger/debug.hpp"
#include "packet_framing.hpp"

#include <cstdint>
#include <exception>
#include <utility>
#include <variant>

#include <unistd.h>

namespace rocprofsys::core
{
template <perfetto_backend Backend>
basic_perfetto_engine<Backend>::basic_perfetto_engine(engine_config cfg)
: m_cfg{ std::move(cfg) }
{}

template <perfetto_backend Backend>
basic_perfetto_engine<Backend>::~basic_perfetto_engine() noexcept
{
    if(m_running)
    {
        // Last-ditch teardown so the engine never outlives its tracing session.
        // Destructors must not throw.
        try
        {
            stop();
        } catch(const std::exception& e)
        {
            LOG_ERROR("perfetto_engine destructor ignored stop() exception: {}",
                      e.what());
        } catch(...)
        {
            LOG_ERROR("perfetto_engine destructor ignored unknown stop() exception");
        }
    }
}

template <perfetto_backend Backend>
void
basic_perfetto_engine<Backend>::init_sdk()
{
    m_backend.init_sdk(m_cfg);
}

template <perfetto_backend Backend>
void
basic_perfetto_engine<Backend>::start(mode m, int fd)
{
    if(m != mode::live_fd)
    {
        LOG_WARNING("perfetto_engine::start(mode, fd) requires mode::live_fd; "
                    "use start(mode, trace_sink&) for cached_interceptor");
        return;
    }

    if(is_system_backend())
    {
        LOG_WARNING(
            "Perfetto's system backend owns the session - Engine has nothing to drive");
        return;
    }

    if(m_running)
    {
        LOG_WARNING("perfetto_engine::start() called while a session is "
                    "already active; replacing it");
    }

    const auto pid = static_cast<pid_t>(::getpid());
    start_session(pid, fd, mode::live_fd);

    m_active_pid   = pid;
    m_running      = true;
    m_current_mode = mode::live_fd;
    m_active_sink  = nullptr;
}

template <perfetto_backend Backend>
void
basic_perfetto_engine<Backend>::start(mode m, trace_sink& sink)
{
    if(m != mode::cached_interceptor)
    {
        LOG_WARNING("perfetto_engine::start(mode, sink) requires "
                    "mode::cached_interceptor; use start(mode, fd) for live_fd");
        return;
    }

    if(is_system_backend())
    {
        LOG_WARNING("Perfetto cached output is unsupported with the system/all "
                    "backend; no cached Perfetto trace will be produced. Use the "
                    "inprocess backend for ROCPROFSYS_PERFETTO_OUTPUT_LAYOUT "
                    "single-file or full cached output.");
        return;
    }

    if(m_running)
    {
        LOG_WARNING("perfetto_engine::start() called while a session is "
                    "already active; replacing it");
    }

    const auto pid = static_cast<pid_t>(::getpid());
    start_session(pid, /*fd=*/-1, mode::cached_interceptor);

    {
        std::lock_guard<std::mutex> lk{ m_collector_mutex };
        m_collected_bytes.clear();
    }
    m_collected_bytes_frozen.store(false, std::memory_order_release);

    m_active_pid   = pid;
    m_running      = true;
    m_current_mode = mode::cached_interceptor;
    m_active_sink  = &sink;

    void* prev = activate_cached_engine(this, &basic_perfetto_engine::collect_thunk);
    if(prev != nullptr && prev != this)
    {
        LOG_WARNING("perfetto_engine: cached engine already active when start() was "
                    "called; replacing it. Worker threads attached to the prior engine "
                    "may emit into stale state until they exit.");
    }
}

template <perfetto_backend Backend>
void
basic_perfetto_engine<Backend>::stop()
{
    if(!m_running) return;

    const auto current_mode = m_current_mode;
    const auto pid          = m_active_pid;
    auto*      sink         = m_active_sink;

    session_ptr* session = nullptr;
    {
        std::lock_guard<std::mutex> lk{ m_sessions_mutex };
        auto                        it = m_sessions.find(pid);
        if(it != m_sessions.end()) session = &it->second;
    }

    if(current_mode == mode::cached_interceptor)
    {
        void* observed = nullptr;
        if(!clear_active_cached_engine(this, &observed) && observed != nullptr)
        {
            LOG_WARNING(
                "perfetto_engine::stop() saw active cached engine pointing at a "
                "different instance ({} vs this={}) -- overlapping cached engines "
                "are not supported and worker threads attached to the prior engine "
                "may emit into stale state until they exit",
                observed, static_cast<void*>(this));
        }
    }

    if(session == nullptr || !static_cast<bool>(*session))
    {
        m_running     = false;
        m_active_pid  = 0;
        m_active_sink = nullptr;
        if(current_mode == mode::cached_interceptor && sink != nullptr)
            std::visit([](auto& s) { s.finalize(); }, *sink);
        return;
    }

    LOG_DEBUG("Flushing the perfetto trace data...");
    std::exception_ptr first_exc{};
    try
    {
        LOG_DEBUG("Stopping the perfetto trace session (blocking)...");
        m_backend.flush_and_stop(*session);
    } catch(...)
    {
        first_exc = std::current_exception();
    }

    m_running     = false;
    m_active_pid  = 0;
    m_active_sink = nullptr;

    if(current_mode != mode::cached_interceptor)
    {
        if(first_exc) std::rethrow_exception(first_exc);
        return;
    }

    std::unordered_map<int, std::vector<char>> drained;
    {
        std::lock_guard<std::mutex> lk{ m_collector_mutex };
        drained.swap(m_collected_bytes);
    }

    if(sink == nullptr)
    {
        if(first_exc) std::rethrow_exception(first_exc);
        return;
    }

    const auto dropped = m_dropped_packet_count.exchange(0, std::memory_order_relaxed);
    if(dropped > 0)
    {
        LOG_WARNING("perfetto cached collector dropped {} packet(s) during the "
                    "session (most likely allocation pressure)",
                    dropped);
    }

    for(auto& entry : drained)
    {
        auto&      bytes      = entry.second;
        const auto source_pid = entry.first;
        if(bytes.empty()) continue;
        try
        {
            std::visit(
                [source_pid, &bytes](auto& s) {
                    s.on_source_drained(source_pid, std::move(bytes));
                },
                *sink);
        } catch(...)
        {
            if(!first_exc) first_exc = std::current_exception();
        }
    }

    try
    {
        std::visit([](auto& s) { s.finalize(); }, *sink);
    } catch(...)
    {
        if(!first_exc) first_exc = std::current_exception();
    }

    if(first_exc) std::rethrow_exception(first_exc);
}

template <perfetto_backend Backend>
std::vector<char>
basic_perfetto_engine<Backend>::read_trace(pid_t pid)
{
    std::lock_guard<std::mutex> lk{ m_sessions_mutex };
    auto                        it = m_sessions.find(pid);
    if(it == m_sessions.end() || !static_cast<bool>(it->second)) return {};
    return m_backend.read_trace(it->second);
}

template <perfetto_backend Backend>
void
basic_perfetto_engine<Backend>::destroy_session(pid_t pid)
{
    std::lock_guard<std::mutex> lk{ m_sessions_mutex };
    auto                        it = m_sessions.find(pid);
    if(it != m_sessions.end()) m_backend.destroy_session(it->second);
}

template <perfetto_backend Backend>
void
basic_perfetto_engine<Backend>::forget_session(pid_t pid)
{
    std::lock_guard<std::mutex> lk{ m_sessions_mutex };
    auto                        it = m_sessions.find(pid);
    if(it != m_sessions.end()) m_backend.release_session(it->second);
}

template <perfetto_backend Backend>
bool
basic_perfetto_engine<Backend>::is_running() const noexcept
{
    return m_running;
}

template <perfetto_backend Backend>
void
basic_perfetto_engine<Backend>::preregister_pids(const std::vector<int>& source_pids)
{
    if(m_collected_bytes_frozen.load(std::memory_order_acquire))
    {
        LOG_ERROR("preregister_pids called after the collector map was frozen; "
                  "new pids cannot be added without restarting the engine");
        return;
    }

    {
        std::lock_guard<std::mutex> lk{ m_collector_mutex };
        for(int pid : source_pids)
        {
            auto& bytes = m_collected_bytes[pid];
            bytes.reserve(COLLECTED_BYTES_SLAB_SIZE);
        }
    }

    m_collected_bytes_frozen.store(true, std::memory_order_release);
}

template <perfetto_backend Backend>
void
basic_perfetto_engine<Backend>::collect_packet_bytes(int pid, const void* data,
                                                     std::size_t size) noexcept
{
    if(data == nullptr || size == 0) return;

    if(!m_collected_bytes_frozen.load(std::memory_order_acquire))
    {
        LOG_ERROR("perfetto cached collector dropped packet for pid {} -- "
                  "preregister_pids has not run yet",
                  pid);
        return;
    }

    try
    {
        auto it = m_collected_bytes.find(pid);
        if(it == m_collected_bytes.end())
        {
            LOG_ERROR("perfetto cached collector dropped packet for unregistered pid {}",
                      pid);
            return;
        }

        auto& bytes = it->second;
        bytes.push_back(static_cast<char>(TRACE_PACKETS_TAG));
        append_varint(bytes, static_cast<std::uint64_t>(size));
        bytes.insert(bytes.end(), static_cast<const char*>(data),
                     static_cast<const char*>(data) + size);
    } catch(...)
    {
        m_dropped_packet_count.fetch_add(1, std::memory_order_relaxed);
        LOG_ERROR("perfetto cached collector dropped packet for pid {} on internal "
                  "exception",
                  pid);
    }
}

template <perfetto_backend Backend>
bool
basic_perfetto_engine<Backend>::is_system_backend() const noexcept
{
    return m_cfg.backend != engine_config::backend_t::inprocess;
}

template <perfetto_backend Backend>
void
basic_perfetto_engine<Backend>::start_session(pid_t pid, int fd, mode m)
{
    std::lock_guard<std::mutex> lk{ m_sessions_mutex };
    m_sessions[pid] = m_backend.start_session(m_cfg, m, fd);
}

template <perfetto_backend Backend>
void
basic_perfetto_engine<Backend>::collect_thunk(void* engine, int pid, const void* data,
                                              std::size_t size) noexcept
{
    auto* typed = static_cast<basic_perfetto_engine*>(engine);
    if(typed == nullptr) return;
    typed->collect_packet_bytes(pid, data, size);
}
}  // namespace rocprofsys::core
