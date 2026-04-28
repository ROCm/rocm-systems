// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "pause_interval_registry.hpp"
#include "sampling/data/backtrace_record.hpp"
#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/timer_sample.hpp"

#include <cstdint>
#include <vector>

namespace rocprofsys::sampling
{

// sample_parser — stateless helper; converts raw backtrace_record sequences
// into parsed timer_sample / overflow_sample, applying pause-interval filtering.
// AC-11: single-sample buffers (raw.size()==1 with pc_count<=1) are discarded.
// AC-13: samples overlapping any pause interval are dropped.
// NFR-T-6: output is sorted by beg_ns.
class sample_parser
{
public:
    sample_parser() = default;

    // Parse timer samples.
    // init_rec: the backtrace_record sampled at sampler start (its timestamp
    //           is used as beg_ns for the first real sample).
    // raw:      the sequence of backtrace_records captured by the ring buffer.
    // pause_reg: used to filter out overlapping intervals.
    template <class ClockPolicy>
    [[nodiscard]] std::vector<timer_sample> parse_timer(
        int64_t tid, backtrace_record const& init_rec,
        std::vector<backtrace_record> const&  raw,
        pause_interval_registry<ClockPolicy>& pause_reg) const;

    // Parse overflow samples.
    template <class ClockPolicy>
    [[nodiscard]] std::vector<overflow_sample> parse_overflow(
        int64_t tid, std::vector<backtrace_record> const& raw,
        pause_interval_registry<ClockPolicy>& pause_reg) const;

private:
    // Builds a minimal stack_frame sequence from raw PCs (no DWARF resolution).
    // DWARF resolution is deferred to post_process (R-A1 mitigation).
    static std::vector<stack_frame> frames_from_pcs(backtrace_record const& rec);
};

}  // namespace rocprofsys::sampling

#include "sample_parser_impl.hpp"
