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

namespace rocprofsys::core
{
template <perfetto_backend Backend>
basic_cached_perfetto_engine<Backend>::basic_cached_perfetto_engine(engine_config cfg)
: m_cfg{ std::move(cfg) }
{}

template <perfetto_backend Backend>
basic_cached_perfetto_engine<Backend>::~basic_cached_perfetto_engine() noexcept
{
    if(m_running)
    {
        try
        {
            stop();
        } catch(const std::exception& e)
        {
            LOG_ERROR("cached_perfetto_engine destructor ignored stop() exception: {}",
                      e.what());
        } catch(...)
        {
            LOG_ERROR(
                "cached_perfetto_engine destructor ignored unknown stop() exception");
        }
    }
}

template <perfetto_backend Backend>
void
basic_cached_perfetto_engine<Backend>::init_sdk()
{
    m_backend.init_sdk(m_cfg);
}

template <perfetto_backend Backend>
void
basic_cached_perfetto_engine<Backend>::start(trace_sink& sink)
{
    if(is_system_backend())
    {
        LOG_WARNING("Perfetto cached output is unsupported with the system/all "
                    "backend; no cached Perfetto trace will be produced. Use the "
                    "inprocess backend for cached Perfetto output.");
        return;
    }

    if(m_running)
    {
        LOG_WARNING("cached_perfetto_engine::start() called while a session is "
                    "already active; replacing it");
    }

    m_session = m_backend.start_cached_session(m_cfg);

    {
        std::lock_guard<std::mutex> lk{ m_collector_mutex };
        m_collected_bytes.clear();
    }
    m_collected_bytes_frozen.store(false, std::memory_order_release);

    m_running     = true;
    m_active_sink = sink;

    void* prev =
        activate_cached_engine(this, &basic_cached_perfetto_engine::collect_thunk);
    if(prev != nullptr && prev != this)
    {
        LOG_WARNING("cached_perfetto_engine: another cached engine was already "
                    "active; overlapping cached engines are not supported");
    }
}

template <perfetto_backend Backend>
void
basic_cached_perfetto_engine<Backend>::stop()
{
    if(!m_running) return;

    void* observed = nullptr;
    if(!clear_active_cached_engine(this, &observed) && observed != nullptr)
    {
        LOG_WARNING("cached_perfetto_engine::stop() saw a different active engine "
                    "({} vs this={}); overlapping cached engines are not supported",
                    observed, static_cast<void*>(this));
    }

    std::exception_ptr first_exc{};
    try
    {
        m_backend.flush_and_stop(m_session);
    } catch(...)
    {
        first_exc = std::current_exception();
    }

    m_running = false;

    std::unordered_map<int, std::vector<char>> drained;
    {
        std::lock_guard<std::mutex> lk{ m_collector_mutex };
        drained.swap(m_collected_bytes);
    }

    if(!m_active_sink.has_value())
    {
        if(first_exc) std::rethrow_exception(first_exc);
        return;
    }

    auto& sink = m_active_sink->get();
    m_active_sink.reset();

    const auto dropped = m_dropped_packet_count.exchange(0, std::memory_order_relaxed);
    if(dropped > 0)
    {
        LOG_WARNING("perfetto cached collector dropped {} packet(s) during the session",
                    dropped);
    }

    for(auto& [source_pid, bytes] : drained)
    {
        if(bytes.empty()) continue;
        try
        {
            std::visit(
                [source_pid, &bytes](auto& s) {
                    s.on_source_drained(source_pid, std::move(bytes));
                },
                sink);
        } catch(...)
        {
            if(!first_exc) first_exc = std::current_exception();
        }
    }

    try
    {
        std::visit([](auto& s) { s.finalize(); }, sink);
    } catch(...)
    {
        if(!first_exc) first_exc = std::current_exception();
    }

    m_session = {};
    if(first_exc) std::rethrow_exception(first_exc);
}

template <perfetto_backend Backend>
bool
basic_cached_perfetto_engine<Backend>::is_running() const noexcept
{
    return m_running;
}

template <perfetto_backend Backend>
void
basic_cached_perfetto_engine<Backend>::preregister_pids(
    const std::vector<int>& source_pids)
{
    if(m_collected_bytes_frozen.load(std::memory_order_acquire))
    {
        LOG_ERROR("preregister_pids called after the collector map was frozen");
        return;
    }

    {
        std::lock_guard<std::mutex> lk{ m_collector_mutex };
        for(int pid : source_pids)
        {
            auto& bytes = m_collected_bytes[pid];
            bytes.reserve(COLLECTED_BYTES_INITIAL_CAPACITY);
        }
    }

    m_collected_bytes_frozen.store(true, std::memory_order_release);
}

template <perfetto_backend Backend>
void
basic_cached_perfetto_engine<Backend>::collect_packet_bytes(int pid, const void* data,
                                                            std::size_t size) noexcept
{
    if(data == nullptr || size == 0) return;

    if(!m_collected_bytes_frozen.load(std::memory_order_acquire))
    {
        LOG_ERROR("perfetto cached collector dropped packet for pid {} before "
                  "preregister_pids",
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
basic_cached_perfetto_engine<Backend>::is_system_backend() const noexcept
{
    return m_cfg.backend != engine_config::backend_t::inprocess;
}

template <perfetto_backend Backend>
void
basic_cached_perfetto_engine<Backend>::collect_thunk(void* engine, int pid,
                                                     const void* data,
                                                     std::size_t size) noexcept
{
    auto* typed = static_cast<basic_cached_perfetto_engine*>(engine);
    if(typed == nullptr) return;
    typed->collect_packet_bytes(pid, data, size);
}
}  // namespace rocprofsys::core
