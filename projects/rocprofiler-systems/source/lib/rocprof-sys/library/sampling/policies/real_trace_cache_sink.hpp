// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"
#include "sampling/data/stack_frame_json.hpp"
#include "sampling/data/track_traits.hpp"

#include <spdlog/fmt/fmt.h>

#include <cstdint>
#include <string>
#include <vector>

namespace rocprofsys::sampling
{

// Production TraceSinkPolicy: forwards parsed samples to
// trace_cache::buffer_storage so data_processor writes them to rocpd.db.
// store_timer / store_overflow are called from emit_resolved_to_trace_cache()
// (Variant 2 pipeline — per-tid at shutdown time).
class real_trace_cache_sink
{
public:
    void store_timer(int64_t tid, std::vector<timer_sample> const& samples)
    {
        // L44 — matches legacy: "[{}] Post-processing metrics for rocpd..."
        LOG_DEBUG("[{}] Post-processing metrics for rocpd...", tid);
        store_samples<timer_track_tag>(tid, samples);
    }

    void store_overflow(int64_t tid, std::vector<overflow_sample> const& samples)
    {
        store_samples<overflow_track_tag>(tid, samples);
    }

private:
    template <class Tag, class Sample>
    static void store_samples(int64_t tid, std::vector<Sample> const& samples)
    {
        using traits = track_traits<Tag>;

        // L01 — matches legacy: "[{}] Storing sampling data to trace cache..."
        LOG_DEBUG("[{}] Storing sampling data to trace cache...", tid);
        LOG_DEBUG("[real_trace_cache_sink] {}: tid={} samples={}", traits::label_str, tid,
                  samples.size());
        if(samples.empty()) return;

        const auto& info = thread_info::get(tid, SequentTID);
        LOG_DEBUG("[real_trace_cache_sink] {}: tid={} thread_info={}", traits::label_str,
                  tid, info ? "valid" : "null");
        if(!info) return;

        const size_t      sys_id       = info->index_data->system_value;
        const size_t      seq_id       = info->index_data->sequent_value;
        const std::string track_name   = make_thread_track_name(Tag{}, seq_id, sys_id);
        constexpr auto    category_id  = static_cast<uint32_t>(traits::category);
        constexpr auto    category_str = traits::label_str;

        for(auto const& sample : samples)
        {
            if(!info->is_valid_lifetime({ sample.beg_ns, sample.end_ns })) continue;

            int depth = 0;
            for(auto const& frame : sample.stack)
            {
                std::string name       = frame.name.empty()
                                             ? ("0x" + fmt::format("{:X}", frame.address))
                                             : frame.name;
                std::string call_stack = make_call_stack_json(frame);
                std::string line_info  = make_line_info_json(frame);
                std::string extdata    = make_extdata_json(depth);

                LOG_DEBUG("[real_trace_cache_sink] {}: tid={} frame='{}'",
                          traits::label_str, tid, name);
                trace_cache::get_buffer_storage().store(
                    trace_cache::backtrace_region_sample{
                        category_id, static_cast<uint64_t>(sys_id), track_name, name,
                        sample.beg_ns, sample.end_ns, category_str, std::move(call_stack),
                        std::move(line_info), std::move(extdata) });
                ++depth;
            }
        }
    }
};

}  // namespace rocprofsys::sampling
