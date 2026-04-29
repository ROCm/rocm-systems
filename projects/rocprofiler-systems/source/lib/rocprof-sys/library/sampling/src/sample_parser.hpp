// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "pause_interval_registry.hpp"
#include "sampling/data/backtrace_record.hpp"
#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/timer_sample.hpp"

#include <algorithm>
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

template <class ClockPolicy>
std::vector<timer_sample>
sample_parser::parse_timer(int64_t tid, backtrace_record const& init_rec,
                           std::vector<backtrace_record> const&  raw,
                           pause_interval_registry<ClockPolicy>& pause_reg) const
{
    // AC-11: discard if raw has only 1 entry with <=1 frame.
    if(raw.size() == 1 && raw[0].pc_count <= 1)
    {
        return {};
    }

    std::vector<timer_sample> result;
    result.reserve(raw.size());

    uint64_t last_ts     = init_rec.timestamp_ns;
    int64_t  last_cpu_ns = init_rec.metrics.cpu_ns;

    for(auto const& rec : raw)
    {
        uint64_t beg        = last_ts;
        uint64_t end        = rec.timestamp_ns;
        int64_t  cur_cpu_ns = static_cast<int64_t>(rec.metrics.cpu_ns);

        last_ts = end;
        // Advance last_cpu_ns unconditionally so skipped records do not inflate
        // the delta of the next surviving sample (mirrors how last_ts advances).
        int64_t cpu_delta = cur_cpu_ns - last_cpu_ns;
        last_cpu_ns       = cur_cpu_ns;

        if(end <= beg) continue;  // clock skew / tie — skip

        // AC-13: filter samples that overlap a pause interval.
        if(pause_reg.spans_pause_interval(beg, end)) continue;

        timer_sample s;
        s.tid    = tid;
        s.beg_ns = beg;
        s.end_ns = end;
        s.stack  = frames_from_pcs(rec);

        // cpu_ns is the delta: CPU time consumed during this sample interval.
        s.metrics        = rec.metrics;
        s.metrics.cpu_ns = cpu_delta;

        result.push_back(std::move(s));
    }

    // NFR-T-6: output sorted by beg_ns.
    std::sort(result.begin(), result.end());
    return result;
}

template <class ClockPolicy>
std::vector<overflow_sample>
sample_parser::parse_overflow(int64_t tid, std::vector<backtrace_record> const& raw,
                              pause_interval_registry<ClockPolicy>& pause_reg) const
{
    std::vector<overflow_sample> result;
    result.reserve(raw.size());

    uint64_t last_ts = 0;
    for(auto const& rec : raw)
    {
        uint64_t beg = last_ts;
        uint64_t end = rec.timestamp_ns;
        last_ts      = end;

        // AC-13: filter samples that overlap a pause interval.
        if(beg < end && pause_reg.spans_pause_interval(beg, end)) continue;

        overflow_sample s;
        s.tid    = tid;
        s.beg_ns = beg;
        s.end_ns = end;
        s.stack  = frames_from_pcs(rec);
        result.push_back(std::move(s));
    }

    std::sort(result.begin(), result.end());
    return result;
}

inline std::vector<stack_frame>
sample_parser::frames_from_pcs(backtrace_record const& rec)
{
    std::vector<stack_frame> frames;
    frames.reserve(rec.pc_count);
    for(uint8_t i = 0; i < rec.pc_count; ++i)
    {
        stack_frame f;
        f.address = rec.raw_pcs[i];
        // Symbol/DWARF resolution deferred to post_process (R-A1).
        frames.push_back(std::move(f));
    }
    return frames;
}

}  // namespace rocprofsys::sampling
