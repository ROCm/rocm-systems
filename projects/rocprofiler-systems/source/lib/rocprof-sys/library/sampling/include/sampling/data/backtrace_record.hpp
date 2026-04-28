// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/backtrace_metrics_data.hpp"

#include <array>
#include <cstdint>

namespace rocprofsys::sampling
{

enum class trigger_type : uint8_t
{
    TIMER    = 0,
    OVERFLOW = 1
};

// Single record pushed by the signal handler into sample_ring_buffer.
// Raw instruction pointers only — DWARF resolution happens post-signal in parse().
// backtrace_record is trivially copyable so the ring buffer can memcpy it safely.
struct backtrace_record
{
    int64_t                   tid          = 0;
    uint64_t                  timestamp_ns = 0;
    trigger_type              trigger      = trigger_type::TIMER;
    backtrace_metrics_data    metrics      = {};
    std::array<uintptr_t, 64> raw_pcs      = {};
    uint8_t                   pc_count     = 0;
};

static_assert(std::is_trivially_copyable_v<backtrace_record>,
              "backtrace_record must be trivially copyable for signal-safe ring buffer");

}  // namespace rocprofsys::sampling
