// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/timer_sample.hpp"

#include <cstdint>
#include <vector>

namespace rocprofsys::sampling::test
{

// Captures every emit_timer / emit_overflow call for post_process() assertions.
struct recording_perfetto_sink
{
    struct timer_emit_call
    {
        int64_t                   tid;
        std::vector<timer_sample> samples;
    };

    struct overflow_emit_call
    {
        int64_t                      tid;
        std::vector<overflow_sample> samples;
    };

    void emit_timer(int64_t                          tid, void const* /*info*/,
                    std::vector<timer_sample> const& samples)
    {
        timer_calls.push_back({ tid, samples });
    }

    void emit_overflow(int64_t                             tid, void const* /*info*/,
                       std::vector<overflow_sample> const& samples)
    {
        overflow_calls.push_back({ tid, samples });
    }

    void clear() noexcept
    {
        timer_calls.clear();
        overflow_calls.clear();
    }

    std::vector<timer_emit_call>    timer_calls;
    std::vector<overflow_emit_call> overflow_calls;
};

}  // namespace rocprofsys::sampling::test
