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

    uint64_t               last_ts      = init_rec.timestamp_ns;
    backtrace_metrics_data last_metrics = init_rec.metrics;

    for(auto const& rec : raw)
    {
        uint64_t beg = last_ts;
        uint64_t end = rec.timestamp_ns;

        last_ts = end;

        backtrace_metrics_data delta = rec.metrics;
        delta.cpu_ns                 = rec.metrics.cpu_ns - last_metrics.cpu_ns;
        delta.ctx_swch               = rec.metrics.ctx_swch - last_metrics.ctx_swch;
        delta.page_flt               = rec.metrics.page_flt - last_metrics.page_flt;
        // mem_peak_kb: keep absolute (peak, not cumulative)
        if(rec.metrics.valid.test(4) && last_metrics.valid.test(4))
        {
            for(size_t j = 0; j < delta.hw_counter.size(); ++j)
                delta.hw_counter[j] =
                    rec.metrics.hw_counter[j] - last_metrics.hw_counter[j];
        }

        last_metrics = rec.metrics;

        if(end <= beg) continue;

        // AC-13: filter samples that overlap a pause interval.
        if(pause_reg.spans_pause_interval(beg, end)) continue;

        timer_sample s;
        s.tid     = tid;
        s.beg_ns  = beg;
        s.end_ns  = end;
        s.stack   = frames_from_pcs(rec);
        s.metrics = delta;

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
    if(raw.empty()) return {};

    std::vector<overflow_sample> result;
    result.reserve(raw.size());

    uint64_t last_ts = raw.front().timestamp_ns;
    for(size_t i = 1; i < raw.size(); ++i)
    {
        auto const& rec = raw[i];
        uint64_t    beg = last_ts;
        uint64_t    end = rec.timestamp_ns;
        last_ts         = end;

        if(end <= beg) continue;

        // AC-13: filter samples that overlap a pause interval.
        if(pause_reg.spans_pause_interval(beg, end)) continue;

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
