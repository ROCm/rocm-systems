// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/c/profiler_hub_types.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <mutex>
#include <string>
#include <utility>

namespace profiler_hub
{

/**
 * @brief Reports every phase of one trace read as a single rising percentage.
 *
 * A client that shows the progress message only when the percentage changes
 * would drop a message that arrives at the same percentage as the one before
 * it. So no message carries a made-up value such as 0.0. A phase message takes
 * the next whole percent through step(). A loop that has a real fraction
 * reports it through within() on a window set by set_window(). The value never
 * falls, and only finish() reaches 100%.
 *
 * Callbacks can arrive from worker threads, so every call is serialized.
 */
class read_progress_t
{
public:
    using sink_t = std::function<void(double, ph_progress_description_t)>;

    explicit read_progress_t(sink_t sink)
    : m_sink{ std::move(sink) }
    {}

    /** @brief Reports @p text at the next whole percent above the last report. */
    void step(const std::string& text)
    {
        const std::scoped_lock lock{ m_mutex };
        // The value sits in the middle of its percent. A client that truncates
        // the percentage never rounds it down into the previous one.
        const int next = std::min(whole_percent() + 1, k_last_percent);
        emit((static_cast<double>(next) + 0.5) / 100.0, text);
    }

    /** @brief Reports @p text at 100%. The read's last message. */
    void finish(const std::string& text)
    {
        const std::scoped_lock lock{ m_mutex };
        m_last = 1.0;
        if(m_sink) m_sink(m_last, text.c_str());
    }

    /** @brief The next within() calls spread over @p span of the full range. The
     *         window starts one percent above the last reported value, so its
     *         first message does not share a percentage with the one before. */
    void set_window(double span)
    {
        const std::scoped_lock lock{ m_mutex };
        open_window(span);
    }

    /** @brief Like set_window(), with the window ending at @p end, a value in
     *         (0, 1). */
    void set_window_until(double end)
    {
        const std::scoped_lock lock{ m_mutex };
        open_window(end - m_last);
    }

    /** @brief Reports @p text at @p fraction (0 to 1) of the current window. */
    void within(double fraction, const std::string& text)
    {
        const std::scoped_lock lock{ m_mutex };
        const double           clamped = std::clamp(fraction, 0.0, 1.0);
        emit(m_window_begin + (m_window_end - m_window_begin) * clamped, text);
    }

private:
    static constexpr int    k_last_percent = 99;
    static constexpr double k_last_value   = 0.995;

    void open_window(double span)
    {
        m_window_begin = std::min(m_last + 0.01, k_last_value);
        m_window_end   = std::clamp(m_window_begin + span, m_window_begin, k_last_value);
    }

    int whole_percent() const { return static_cast<int>(std::floor(m_last * 100.0)); }

    void emit(double value, const std::string& text)
    {
        m_last = std::clamp(std::max(value, m_last), 0.0, k_last_value);
        if(m_sink) m_sink(m_last, text.c_str());
    }

    sink_t     m_sink;
    std::mutex m_mutex;
    double     m_last{ 0.0 };
    double     m_window_begin{ 0.0 };
    double     m_window_end{ 0.0 };
};

}  // namespace profiler_hub
