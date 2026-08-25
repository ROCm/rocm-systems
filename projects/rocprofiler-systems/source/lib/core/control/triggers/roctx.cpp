// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "roctx.hpp"

#include "common/delimit.hpp"
#include "core/control/session.hpp"
#include "core/state.hpp"

#include "logger/debug.hpp"
#include <spdlog/fmt/ranges.h>  // NOLINT

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

namespace rocprofsys::control::triggers
{
namespace
{
/// Runs @p fn when the guard goes out of scope, on every exit path
/// (early return or fall-through). Used to keep refresh_state() calls out
/// of every early-return branch in the roctx entry points below.
template <typename Func>
class scope_exit
{
public:
    explicit scope_exit(Func on_exit_action)
    : m_func{ std::move(on_exit_action) }
    {}
    ~scope_exit() { m_func(); }

    scope_exit(const scope_exit&)            = delete;
    scope_exit& operator=(const scope_exit&) = delete;
    scope_exit(scope_exit&&)                 = delete;
    scope_exit& operator=(scope_exit&&)      = delete;

private:
    Func m_func;
};

template <typename Func>
scope_exit(Func) -> scope_exit<Func>;
}  // namespace

roctx::roctx(std::shared_ptr<session> sess, std::string_view trace_regions)
: m_session{ std::move(sess) }
{
    if(!trace_regions.empty())
    {
        const auto delimited =
            rocprofsys::common::delimit(std::string{ trace_regions }, ",");
        m_trace_regions.insert(delimited.begin(), delimited.end());
    }

    if(filter_active())
    {
        LOG_INFO("roctx trigger: filter active for regions: [{}]",
                 fmt::join(m_trace_regions, ", "));  // NOLINT(misc-include-cleaner)
    }

    m_should_write.store(compute_should_write(), std::memory_order_relaxed);
    m_session->register_trigger(trigger_name, compute_action());
}

roctx::~roctx() { m_session->unregister_trigger(trigger_name); }

void
roctx::on_range_start(std::uint64_t range_id, const char* message)
{
    if(message == nullptr || !m_trace_regions.contains(message))
    {
        return;
    }

    const auto       _thread_state_guard = state::thread::scoped(state::thread::Internal);
    const scope_exit refresh{ [this] { refresh_state(); } };
    const std::scoped_lock notify_lk{ m_mutex };
    const bool             was_empty = m_active_range_ids.empty();
    m_active_range_ids.insert(range_id);
    if(was_empty)
    {
        m_in_region.store(true, std::memory_order_relaxed);
    }
}

void
roctx::on_range_stop(std::uint64_t range_id)
{
    const scope_exit refresh{ [this] { refresh_state(); } };
    if(!filter_active())
    {
        return;
    }

    if(!remove_active_range(range_id))
    {
        return;
    }

    warn_if_paused_region_ended();
}

bool
roctx::remove_active_range(std::uint64_t range_id)
{
    const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
    const std::scoped_lock notify_lk{ m_mutex };
    if(m_active_range_ids.erase(range_id) == 0)
    {
        return false;
    }

    const bool now_empty = m_active_range_ids.empty();
    if(now_empty)
    {
        m_in_region.store(false, std::memory_order_relaxed);
    }
    return now_empty;
}

void
roctx::warn_if_paused_region_ended()
{
    // Region ended while paused: silently clear user_paused so a later region
    // push behaves as a fresh start. Subsequent on_resume becomes a no-op.
    if(m_user_paused.exchange(false, std::memory_order_relaxed))
    {
        LOG_WARNING(
            "Target region ended while paused. Subsequent resume will be ignored.");
    }
}

void
roctx::on_pause()
{
    const auto       _thread_state_guard = state::thread::scoped(state::thread::Internal);
    const scope_exit refresh{ [this] { refresh_state(); } };
    if(filter_active())
    {
        const std::scoped_lock notify_lk{ m_mutex };
        if(m_active_range_ids.empty())
        {
            LOG_WARNING("Pause requested outside of target region - ignoring");
            return;
        }
    }

    bool expected = false;
    if(!m_user_paused.compare_exchange_strong(expected, true, std::memory_order_relaxed))
    {
        LOG_WARNING("Pause requested but tracing is already paused - ignoring");
        return;
    }

    LOG_INFO("Pausing tracing session...");
}

void
roctx::on_resume()
{
    if(!m_user_paused.load(std::memory_order_relaxed))
    {
        LOG_WARNING("Resume requested but tracing was not paused by user - ignoring");
        return;
    }

    const auto       _thread_state_guard = state::thread::scoped(state::thread::Internal);
    const scope_exit refresh{ [this] { refresh_state(); } };
    if(filter_active())
    {
        const std::scoped_lock notify_lk{ m_mutex };
        if(m_active_range_ids.empty())
        {
            LOG_WARNING("Resume requested outside of target region - ignoring");
            return;
        }
    }

    m_user_paused.store(false, std::memory_order_relaxed);
    LOG_INFO("Resuming tracing session...");
}

action
roctx::compute_action() const noexcept
{
    return compute_should_write() ? action::trace : action::pause;
}

bool
roctx::compute_should_write() const noexcept
{
    const bool filter    = filter_active();
    const bool in_region = m_in_region.load(std::memory_order_relaxed);
    const bool paused    = m_user_paused.load(std::memory_order_relaxed);
    return !paused && (!filter || in_region);
}

void
roctx::refresh_state()
{
    m_should_write.store(compute_should_write(), std::memory_order_relaxed);
    m_session->set_action(trigger_name, compute_action());
}
}  // namespace rocprofsys::control::triggers
