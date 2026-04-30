// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/timer_sample.hpp"

#include <cstdint>
#include <vector>

namespace rocprofsys::sampling::test
{

struct recording_trace_sink
{
    recording_trace_sink() = default;

    template <class T>
    explicit recording_trace_sink(T& /*resolver*/)
    {}

    // Satisfies TraceSinkPolicy: store_timer + store_overflow.
    // Records tid and sample count; no core/ dependency introduced.

    void store_timer(int64_t tid, std::vector<timer_sample> const& samples)
    {
        m_timer_records.push_back({ tid, static_cast<int>(samples.size()) });
    }

    void store_overflow(int64_t tid, std::vector<overflow_sample> const& samples)
    {
        m_overflow_records.push_back({ tid, static_cast<int>(samples.size()) });
    }

    // Per-thread counter tracks emission shape (TF-3) — production code calls
    // this from sampling_service::do_emit_resolved alongside store_timer.
    // The recording double keeps an independent record so tests can assert on
    // the call separately from the timer/overflow paths.
    void store_thread_counters(int64_t tid, std::vector<timer_sample> const& samples)
    {
        m_thread_counter_records.push_back({ tid, static_cast<int>(samples.size()) });
    }

    struct call_record
    {
        int64_t tid;
        int     sample_count;
    };

    [[nodiscard]] auto const& timer_records() const noexcept { return m_timer_records; }
    [[nodiscard]] auto const& overflow_records() const noexcept
    {
        return m_overflow_records;
    }
    [[nodiscard]] auto const& thread_counter_records() const noexcept
    {
        return m_thread_counter_records;
    }
    void clear() noexcept
    {
        m_timer_records.clear();
        m_overflow_records.clear();
        m_thread_counter_records.clear();
    }

    // Total samples forwarded across all calls (convenience for assertions).
    [[nodiscard]] size_t total_timer_samples() const noexcept
    {
        size_t n = 0;
        for(auto const& r : m_timer_records)
            n += static_cast<size_t>(r.sample_count);
        return n;
    }
    [[nodiscard]] size_t total_overflow_samples() const noexcept
    {
        size_t n = 0;
        for(auto const& r : m_overflow_records)
            n += static_cast<size_t>(r.sample_count);
        return n;
    }

    std::vector<call_record> m_timer_records;
    std::vector<call_record> m_overflow_records;
    std::vector<call_record> m_thread_counter_records;
};

}  // namespace rocprofsys::sampling::test
