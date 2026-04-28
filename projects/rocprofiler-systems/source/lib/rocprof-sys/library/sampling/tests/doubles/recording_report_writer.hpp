// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/timer_sample.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofsys::sampling::test
{

// Captures every write_timer_samples / write_overflow_samples / flush call.
// Used for post_process() assertions.
struct recording_report_writer
{
    struct timer_write_call
    {
        int64_t                   tid;
        std::vector<timer_sample> samples;
    };

    struct overflow_write_call
    {
        int64_t                      tid;
        std::vector<overflow_sample> samples;
    };

    void write_timer_samples(int64_t tid, std::vector<timer_sample> const& samples)
    {
        timer_calls.push_back({ tid, samples });
    }

    void write_overflow_samples(int64_t tid, std::vector<overflow_sample> const& samples)
    {
        overflow_calls.push_back({ tid, samples });
    }

    void flush() noexcept { flush_count++; }

    void clear() noexcept
    {
        timer_calls.clear();
        overflow_calls.clear();
        flush_count = 0;
    }

    std::vector<timer_write_call>    timer_calls;
    std::vector<overflow_write_call> overflow_calls;
    int                              flush_count{ 0 };
};

// Throws on write_timer_samples — used for exception-safety test.
struct throwing_report_writer
{
    void write_timer_samples(int64_t /*tid*/,
                             std::vector<timer_sample> const& /*samples*/)
    {
        throw std::runtime_error("write_timer_samples: injected failure");
    }

    void write_overflow_samples(int64_t /*tid*/,
                                std::vector<overflow_sample> const& /*samples*/)
    {}

    void flush() noexcept {}
};

// Throws from flush() — used to verify offload.reset() runs even when flush throws
// (C-15).
struct throwing_flush_report_writer
{
    void write_timer_samples(int64_t /*tid*/,
                             std::vector<timer_sample> const& /*samples*/)
    {}

    void write_overflow_samples(int64_t /*tid*/,
                                std::vector<overflow_sample> const& /*samples*/)
    {}

    void flush() { throw std::runtime_error("flush: injected failure"); }
};

}  // namespace rocprofsys::sampling::test
