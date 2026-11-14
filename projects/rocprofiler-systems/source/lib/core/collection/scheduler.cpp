// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/collection/scheduler.hpp"
#include "common/clocks.hpp"
#include "core/collection/tick_grid.hpp"
#include "logger/debug.hpp"

#include <chrono>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace rocprofsys::collection
{
template <typename Fn>
bool
scheduler::guarded(Fn&& func) const noexcept
{
    try
    {
        return std::forward<Fn>(func)();
    } catch(const std::exception& error)
    {
        log_stopped(error.what());
    } catch(...)
    {
        log_stopped("unknown exception");
    }
    return false;
}

scheduler::scheduler(std::string name)
: m_name(std::move(name))
{}

scheduler::~scheduler() { stop(); }

void
scheduler::start(std::chrono::nanoseconds interval, callback func, thread_start on_start)
{
    const std::lock_guard lock(m_mutex);
    if(m_grid)
    {
        throw std::logic_error("scheduler '" + m_name + "' started twice");
    }
    tick_grid& grid = m_grid.emplace(tick_grid::clock::now(), interval);
    m_callback      = std::move(func);
    m_on_start      = std::move(on_start);
    try
    {
        m_thread = std::jthread(
            [this, &grid](const std::stop_token& token) { run(token, grid); });
    } catch(...)
    {
        m_grid.reset();
        throw;
    }
}

void
scheduler::pause()
{
    {
        const std::lock_guard lock(m_mutex);
        if(!m_grid)
        {
            return;
        }
        m_grid->pause();
        m_changed = true;
        ++m_pauses;
    }
    m_wake.notify_all();
}

void
scheduler::resume()
{
    {
        const std::lock_guard lock(m_mutex);
        if(!m_grid)
        {
            return;
        }
        m_grid->resume(tick_grid::clock::now());
        m_changed = true;
    }
    m_wake.notify_all();
}

void
scheduler::stop()
{
    if(!m_thread.joinable())
    {
        return;
    }
    m_thread.request_stop();
    if(m_thread.get_id() == std::this_thread::get_id())
    {
        m_thread.detach();
        return;
    }
    m_thread.join();
}

void
scheduler::run(const std::stop_token& token, tick_grid& grid)
{
    if(m_on_start && !guarded([this] {
           m_on_start();
           return true;
       }))
    {
        return;
    }
    const auto changed = [this] { return m_changed; };

    std::unique_lock lock(m_mutex);
    while(!token.stop_requested())
    {
        const auto decision = grid.next(tick_grid::clock::now());
        switch(decision.what)
        {
            case tick_grid::step::exit: return;
            case tick_grid::step::idle:
                m_wake.wait(lock, token, changed);
                m_changed = false;
                break;
            case tick_grid::step::wait:
                m_wake.wait_until(lock, token, decision.until, changed);
                m_changed = false;
                break;
            case tick_grid::step::fire: fire(lock, grid); break;
            default: throw std::logic_error("tick_grid::step not handled");
        }
    }
}

void
scheduler::fire(std::unique_lock<std::mutex>& lock, tick_grid& grid)
{
    const auto pauses = m_pauses;
    lock.unlock();
    const bool keep_going =
        guarded([this] { return m_callback(clocks::timeline::now_ns()); });
    lock.lock();
    if(!keep_going)
    {
        grid.finish();
        return;
    }
    // A pause during the callback left the grid paused or resumed at its resume time
    if(m_pauses == pauses)
    {
        grid.advance(tick_grid::clock::now());
    }
}

void
scheduler::log_stopped(std::string_view reason) const noexcept
{
    // A throw here would escape the thread body and terminate the process
    try
    {
        LOG_ERROR("collection: '{}' stopped polling: {}", m_name, reason);
    } catch(...)
    {
        return;
    }
}
}  // namespace rocprofsys::collection
