// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/control/clock.hpp"
#include "core/control/session.hpp"
#include "core/control/trigger.hpp"

#include <chrono>
#include <cstdint>
#include <limits>
#include <stop_token>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace rocprofsys::control::triggers
{
/// Clock-independent schedule entry. Hoisted out of the time_window template so
/// all clock instantiations share one concrete type — allowing callers to build
/// a schedule without knowing which ClockPolicy will be used.
struct time_window_config
{
    clock_duration delay{};
    clock_duration duration{};
    std::uint64_t  repeat{ 1 };
};

using time_window_schedule = std::vector<time_window_config>;

/// Time-windowed pause/resume trigger driven by an injected clock.
///
/// Lifecycle votes:
///   first delay > 0  -> initial vote paused; publishes active when delay elapses
///   duration > 0     -> publishes paused when the active window elapses
///   duration == 0    -> active window remains open until shutdown
///
/// A trigger owns a sequential schedule. Each entry waits `delay`, opens
/// collection, then closes it after `duration`; `repeat` repeats that entry
/// before advancing to the next entry. This preserves TRACE_PERIODS semantics
/// with one trigger/vote, avoiding conflicting votes from multiple time-window
/// triggers in the same scope.
///
/// Templated on Clock so production wires `clocks::steady` and tests wire
/// `clocks::manual`. Methods on the Clock parameter are duck-typed against
/// the concept in core/control/clock.hpp.
template <ClockPolicy Clock>
class time_window : public trigger
{
public:
    using config        = time_window_config;
    using schedule_type = time_window_schedule;

    time_window(session& sess, Clock& clk, config cfg, scope event_scope = scope::global)
    : time_window{ sess, clk, schedule_type{ cfg }, event_scope }
    {}

    time_window(session& sess, Clock& clk, schedule_type cfgs,
                scope event_scope = scope::global)
    : m_session{ sess }
    , m_clock{ clk }
    , m_schedule{ std::move(cfgs) }
    , m_scope{ event_scope }
    {}

    ~time_window() override = default;

    time_window(const time_window&)            = delete;
    time_window& operator=(const time_window&) = delete;
    time_window(time_window&&)                 = delete;
    time_window& operator=(time_window&&)      = delete;

    [[nodiscard]] std::string_view name() const noexcept override
    {
        return "time_window";
    }

    [[nodiscard]] vote initial_vote() const noexcept override
    {
        if(const auto* cfg = first_window_config())
        {
            if(cfg->delay > clock_duration::zero()) return vote::paused;
            return vote::active;
        }
        return vote::abstain;
    }

    [[nodiscard]] scope event_scope() const noexcept override { return m_scope; }

    void start()
    {
        if(!has_window() || m_thread.joinable()) return;
        m_clock.reset();
        m_thread = std::jthread{ [this](std::stop_token st) { worker(st); } };
    }

    void stop() noexcept
    {
        m_thread.request_stop();
        if(m_thread.joinable()) m_thread.join();
    }

private:
    session&      m_session;
    Clock&        m_clock;
    schedule_type m_schedule;
    scope         m_scope;
    std::jthread  m_thread;

    [[nodiscard]] static bool has_window(const config& cfg) noexcept
    {
        return cfg.delay > clock_duration::zero() ||
               cfg.duration > clock_duration::zero();
    }

    [[nodiscard]] const config* first_window_config() const noexcept
    {
        for(const auto& cfg : m_schedule)
        {
            if(has_window(cfg)) return &cfg;
        }
        return nullptr;
    }

    [[nodiscard]] bool has_window() const noexcept
    {
        return first_window_config() != nullptr;
    }

    void worker(std::stop_token st)
    {
        // When stop is requested (e.g. from the jthread destructor or stop()),
        // this callback interrupts any in-progress sleep_until so the worker exits.
        std::stop_callback const stop_cb{ st, [this] { m_clock.interrupt(); } };

        auto current = m_clock.now();

        for(const auto& cfg : m_schedule)
        {
            if(!has_window(cfg)) continue;

            const auto repeats = (cfg.repeat == 0)
                                     ? std::numeric_limits<std::uint64_t>::max()
                                     : cfg.repeat;

            for(std::uint64_t rep = 0; rep < repeats; ++rep)
            {
                if(cfg.delay > clock_duration::zero())
                {
                    current += cfg.delay;
                    if(!m_clock.sleep_until(current)) return;
                }

                m_session.publish(*this, vote::active);

                if(cfg.duration == clock_duration::zero()) return;

                current += cfg.duration;
                if(!m_clock.sleep_until(current)) return;
                m_session.publish(*this, vote::paused);
            }
        }
    }
};
}  // namespace rocprofsys::control::triggers
