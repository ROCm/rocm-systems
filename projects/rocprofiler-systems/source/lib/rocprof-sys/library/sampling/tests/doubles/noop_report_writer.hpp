// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/timer_sample.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace rocprofsys::sampling::test
{

struct noop_report_writer
{
    void write_timer_samples(int64_t tid, std::vector<timer_sample> const& samples)
    {
        m_timer_counts[tid] += static_cast<int>(samples.size());
    }

    void write_overflow_samples(int64_t tid, std::vector<overflow_sample> const& samples)
    {
        m_overflow_counts[tid] += static_cast<int>(samples.size());
    }

    void flush() noexcept {}

    std::unordered_map<int64_t, int> m_timer_counts;
    std::unordered_map<int64_t, int> m_overflow_counts;
};

}  // namespace rocprofsys::sampling::test
