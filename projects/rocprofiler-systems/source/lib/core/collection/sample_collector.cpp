// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/collection/sample_collector.hpp"
#include "common/clocks.hpp"
#include "core/collection/detail/source_runner.hpp"
#include "core/collection/fork_gate.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/source_interface.hpp"
#include "logger/debug.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::collection
{
template <typename Fn>
bool
sample_collector::guarded(entry& item, std::string_view step, Fn&& func)
{
    try
    {
        std::forward<Fn>(func)();
        return true;
    } catch(const std::exception& error)
    {
        LOG_ERROR("collection: source '{}' {} failed: {}", item.runner->source().name(),
                  step, error.what());
    } catch(...)
    {
        LOG_ERROR("collection: source '{}' {} failed: unknown exception",
                  item.runner->source().name(), step);
    }
    return false;
}

sample_collector::sample_collector(sink_interface& out, fork_gate& gate,
                                   detail::run_options options)
: m_sink(out)
, m_context{ .gate = gate, .options = std::move(options) }
{}

sample_collector::~sample_collector()
{
    try
    {
        stop();
    } catch(...)
    {
        LOG_ERROR("collection: stopping the sources failed");
    }
}

void
sample_collector::add_source(std::unique_ptr<source_interface> source,
                             std::chrono::nanoseconds          interval)
{
    if(!source)
    {
        throw std::invalid_argument("collection: add_source() needs a source");
    }
    if(interval <= std::chrono::nanoseconds::zero())
    {
        throw std::invalid_argument("collection: a polling interval must be positive");
    }
    const std::lock_guard lock(m_mutex);
    if(m_stopped)
    {
        throw std::logic_error("collection: add_source() after stop()");
    }
    m_entries.push_back(std::make_unique<entry>(
        entry{ .runner = std::make_unique<detail::source_runner>(
                   std::move(source), interval, m_context, m_sink) }));
}

void
sample_collector::start()
{
    std::vector<entry*> pending;
    std::unique_lock    lock(m_mutex);
    for(const auto& item : m_entries)
    {
        if(item->status == state::created)
        {
            pending.push_back(item.get());
        }
    }
    lock.unlock();

    for(auto* item : pending)
    {
        const state status = launch(*item);
        lock.lock();
        item->status = status;
        if(status == state::running && m_context.paused.load(std::memory_order_relaxed))
        {
            item->runner->pause();
        }
        lock.unlock();
    }
}

std::vector<sample_collector::entry*>
sample_collector::take_active()
{
    m_stopped = true;
    std::vector<entry*> active;
    for(const auto& item : m_entries)
    {
        if(item->status == state::running || item->status == state::failed)
        {
            active.push_back(item.get());
        }
        item->status = state::stopped;
    }
    return active;
}

sample_collector::state
sample_collector::launch(entry& item)
{
    bool described = false;
    if(!guarded(item, "describe", [&] {
           // describe() may hold the registry write lock; a child forked then would
           // inherit it held
           const fork_gate::pass pass{ m_context.gate, fork_gate::pass::k_wait };
           described = item.runner->source().describe(m_sink);
       }))
    {
        return state::failed;
    }
    if(!described)
    {
        return state::stopped;
    }
    if(!guarded(item, "start", [&] { item.runner->start(); }))
    {
        return state::failed;
    }
    return state::running;
}

void
sample_collector::pause()
{
    const std::lock_guard lock(m_mutex);
    if(m_context.paused.load(std::memory_order_relaxed))
    {
        return;
    }
    m_context.paused.store(true, std::memory_order_release);
    m_paused_at = clocks::timeline::now_ns();
    for(const auto& item : m_entries)
    {
        if(item->status == state::running)
        {
            item->runner->pause();
        }
    }
}

void
sample_collector::resume()
{
    const std::lock_guard lock(m_mutex);
    if(!m_context.paused.load(std::memory_order_relaxed))
    {
        return;
    }
    m_context.paused.store(false, std::memory_order_release);
    m_sink.write_gap(m_paused_at, clocks::timeline::now_ns());
    for(const auto& item : m_entries)
    {
        if(item->status == state::running)
        {
            item->runner->resume();
        }
    }
}

void
sample_collector::stop()
{
    std::unique_lock    lock(m_mutex);
    const auto          active     = take_active();
    const bool          was_paused = m_context.paused.exchange(false);
    const std::uint64_t paused_at  = m_paused_at;
    lock.unlock();

    for(auto* item : active)
    {
        static_cast<void>(guarded(*item, "stop", [item] { item->runner->stop(); }));
    }
    if(was_paused)
    {
        m_sink.write_gap(paused_at, clocks::timeline::now_ns());
    }
}
}  // namespace rocprofsys::collection
