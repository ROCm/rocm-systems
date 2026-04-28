// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>

namespace rocprofsys::sampling
{

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

    uint64_t last_ts = init_rec.timestamp_ns;

    for(auto const& rec : raw)
    {
        uint64_t beg = last_ts;
        uint64_t end = rec.timestamp_ns;
        last_ts      = end;

        if(end <= beg) continue;  // clock skew / tie — skip

        // AC-13: filter samples that overlap a pause interval.
        if(pause_reg.spans_pause_interval(beg, end)) continue;

        timer_sample s;
        s.tid     = tid;
        s.beg_ns  = beg;
        s.end_ns  = end;
        s.metrics = rec.metrics;
        s.stack   = frames_from_pcs(rec);
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
