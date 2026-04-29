// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"
#include "rocprofiler-systems/categories.h"
#include "sampling/data/backtrace_metrics_data.hpp"
#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/timer_sample.hpp"
#include "sampling/data/track_name.hpp"
#include "sampling/data/track_traits.hpp"
#include "sampling/policies/stack_frame_json.hpp"

#include <spdlog/fmt/fmt.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::sampling
{

// Production TraceSinkPolicy: forwards parsed samples to
// trace_cache::buffer_storage so data_processor writes them to rocpd.db.
//
// Single source of truth for sampling -> trace_cache emission. Owns three
// emission shapes:
//   - store_timer(tid, samples)            — wall-clock + cpu-time region samples
//   - store_overflow(tid, samples)         — overflow region samples
//   - store_thread_counters(tid, samples)  — per-thread metric tracks
//
// All three are called from real_production_hooks::emit_resolved at shutdown.
//
// thread_counter_prefixes is the matching list of per-thread metric track names
// that callers (real_production_hooks::setup_wiring) register with the trace
// cache so the metric values land on tracks that already exist.
class real_trace_cache_sink
{
public:
    // Track-name prefixes for the per-thread counter tracks emitted by
    // store_thread_counters. Order matches thread_metric_descriptors below.
    static constexpr std::array<char const*, 4> thread_counter_prefixes = {
        "thread_cpu_time", "thread_peak_memory", "thread_context_switch",
        "thread_page_fault"
    };

    void store_timer(int64_t tid, std::vector<timer_sample> const& samples)
    {
        if(samples.empty()) return;

        // L44 — matches legacy: "[{}] Post-processing metrics for rocpd..."
        LOG_DEBUG("[{}] Post-processing metrics for rocpd...", tid);

        const auto& info = resolve_tid_info(tid);
        if(!info) return;

        const std::size_t sys_id = info->index_data->system_value;
        const std::size_t seq_id = info->index_data->sequent_value;
        const std::string track_name =
            make_thread_track_name(timer_track_tag{}, seq_id, sys_id);
        constexpr auto category_id =
            static_cast<uint32_t>(ROCPROFSYS_CATEGORY_TIMER_SAMPLING);
        constexpr auto category_str     = "timer_sampling";
        constexpr auto cpu_category_str = "cputime_sampling";

        for(auto const& sample : samples)
        {
            if(!info->is_valid_lifetime({ sample.beg_ns, sample.end_ns })) continue;

            const bool has_cpu =
                sample.metrics.valid.test(0) && sample.metrics.cpu_ns > 0;

            int depth = 0;
            for(auto const& frame : sample.stack)
            {
                std::string name = frame.name.empty()
                                       ? ("0x" + fmt::format("{:X}", frame.address))
                                       : frame.name;

                trace_cache::get_buffer_storage().store(make_region_sample(
                    category_id, sys_id, track_name, name, sample.beg_ns, sample.end_ns,
                    category_str, frame, depth));

                // CPU-time sample: emitted only when cpu_ns delta is available.
                // Duration encoded as [0, cpu_ns] so tsv_processor computes the
                // correct cpu-time duration from end_timestamp - start_timestamp.
                if(has_cpu)
                {
                    trace_cache::get_buffer_storage().store(make_region_sample(
                        category_id, sys_id, track_name, name, std::uint64_t{ 0 },
                        static_cast<std::uint64_t>(sample.metrics.cpu_ns),
                        cpu_category_str, frame, depth));
                }

                ++depth;
            }
        }
    }

    void store_overflow(int64_t tid, std::vector<overflow_sample> const& samples)
    {
        if(samples.empty()) return;

        const auto& info = resolve_tid_info(tid);
        if(!info) return;

        const std::size_t sys_id = info->index_data->system_value;
        const std::size_t seq_id = info->index_data->sequent_value;
        const std::string track_name =
            make_thread_track_name(overflow_track_tag{}, seq_id, sys_id);
        constexpr auto category_id =
            static_cast<uint32_t>(ROCPROFSYS_CATEGORY_OVERFLOW_SAMPLING);
        constexpr auto category_str = "overflow_sampling";

        for(auto const& sample : samples)
        {
            if(!info->is_valid_lifetime({ sample.beg_ns, sample.end_ns })) continue;

            int depth = 0;
            for(auto const& frame : sample.stack)
            {
                std::string name = frame.name.empty()
                                       ? ("0x" + fmt::format("{:X}", frame.address))
                                       : frame.name;
                trace_cache::get_buffer_storage().store(make_region_sample(
                    category_id, sys_id, track_name, name, sample.beg_ns, sample.end_ns,
                    category_str, frame, depth));
                ++depth;
            }
        }
    }

    // Per-thread process-sampling counter tracks (TF-3): emits one
    // pmc_event_with_sample per descriptor per sample whose validity bit is set.
    void store_thread_counters(int64_t tid, std::vector<timer_sample> const& samples)
    {
        if(samples.empty()) return;

        const auto& info = resolve_tid_info(tid);
        if(!info) return;

        const std::size_t sys_id = info->index_data->system_value;
        const std::size_t seq_id = info->index_data->sequent_value;

        for(auto const& sample : samples)
        {
            if(!info->is_valid_lifetime({ sample.beg_ns, sample.end_ns })) continue;
            const std::uint64_t mid_ns = sample.end_ns;

            for(auto const& descriptor : thread_metric_descriptors)
            {
                if(!sample.metrics.valid.test(descriptor.valid_bit)) continue;
                std::string track = std::string{ descriptor.track_prefix } + " [" +
                                    std::to_string(seq_id) + "]";
                trace_cache::get_buffer_storage().store(
                    trace_cache::pmc_event_with_sample{
                        descriptor.category_enum, std::move(track),
                        static_cast<std::size_t>(mid_ns), std::string{ "{}" },
                        /*stack_id*/ 0, /*parent_stack_id*/ 0,
                        /*correlation_id*/ 0, /*call_stack*/ std::string{},
                        /*line_info*/ std::string{}, static_cast<uint32_t>(sys_id),
                        /*device_type*/ uint8_t{ 0 },
                        std::string{ descriptor.track_prefix },
                        descriptor.read(sample.metrics), std::optional<int64_t>{} });
            }
        }
    }

private:
    // tid may be either an internal_value (from tracing.cpp / library.cpp via
    // utility::get_thread_index()) or a sequent_value (from
    // pthread_create_gotcha which uses _info->sequent_value). Try sequent
    // first; fall back to internal so the resolved emission paths see a
    // consistent thread_info regardless of caller.
    static std::optional<thread_info> const& resolve_tid_info(int64_t tid)
    {
        auto const& info = thread_info::get(tid, SequentTID);
        if(info) return info;
        return thread_info::get(tid, InternalTID);
    }

    static trace_cache::backtrace_region_sample make_region_sample(
        uint32_t category_id, std::size_t sys_id, std::string const& track_name,
        std::string const& name, std::uint64_t beg_ns, std::uint64_t end_ns,
        char const* category_str, stack_frame const& frame, int depth)
    {
        return trace_cache::backtrace_region_sample{ category_id,
                                                     static_cast<uint64_t>(sys_id),
                                                     track_name,
                                                     name,
                                                     beg_ns,
                                                     end_ns,
                                                     category_str,
                                                     make_call_stack_json(frame),
                                                     make_line_info_json(frame),
                                                     make_extdata_json(depth) };
    }

    // Per-thread metric descriptor for the 4 process-sampling counter tracks
    // emitted by store_thread_counters. Single source of truth for the metric
    // name, perfetto category, valid-bit index, and value transform.
    struct thread_metric_descriptor
    {
        char const* track_prefix;
        std::size_t category_enum;
        std::size_t valid_bit;
        double (*read)(backtrace_metrics_data const&);
    };

    static constexpr std::array<thread_metric_descriptor, 4> thread_metric_descriptors = {
        thread_metric_descriptor{ "thread_cpu_time", ROCPROFSYS_CATEGORY_THREAD_CPU_TIME,
                                  0,
                                  [](backtrace_metrics_data const& m) {
                                      return static_cast<double>(m.cpu_ns) * 1.0e-9;
                                  } },
        thread_metric_descriptor{ "thread_peak_memory",
                                  ROCPROFSYS_CATEGORY_THREAD_PEAK_MEMORY, 1,
                                  [](backtrace_metrics_data const& m) {
                                      return static_cast<double>(m.mem_peak_kb) / 1024.0;
                                  } },
        thread_metric_descriptor{ "thread_context_switch",
                                  ROCPROFSYS_CATEGORY_THREAD_CONTEXT_SWITCH, 2,
                                  [](backtrace_metrics_data const& m) {
                                      return static_cast<double>(m.ctx_swch);
                                  } },
        thread_metric_descriptor{ "thread_page_fault",
                                  ROCPROFSYS_CATEGORY_THREAD_PAGE_FAULT, 3,
                                  [](backtrace_metrics_data const& m) {
                                      return static_cast<double>(m.page_flt);
                                  } },
    };
};

}  // namespace rocprofsys::sampling
