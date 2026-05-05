// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "logger/debug.hpp"
#include "rocprofiler-systems/categories.h"
#include "sampling/data/backtrace_metrics_data.hpp"
#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/timer_sample.hpp"
#include "sampling/data/track_name.hpp"
#include "sampling/data/track_traits.hpp"
#include "sampling/policies/stack_frame_json.hpp"
#include "sampling/thread_info_data.hpp"

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
// Template parameter ThreadInfoResolverT must satisfy:
//   std::optional<thread_info_data> resolve(int64_t tid) const;
//
// Production: real_thread_info_resolver (wraps thread_info::get()).
// Test: mock_thread_info_resolver (returns pre-configured data).
template <class ThreadInfoResolverT>
class trace_cache_sink
{
public:
    explicit trace_cache_sink(ThreadInfoResolverT&     resolver,
                              std::vector<std::string> hw_labels = {})
    : resolver_(resolver)
    , hw_labels_(std::move(hw_labels))
    {}

    void store_timer(int64_t tid, std::vector<timer_sample> const& samples)
    {
        if(samples.empty()) return;

        LOG_DEBUG("[{}] Post-processing metrics for rocpd...", tid);

        auto info = resolver_.resolve(tid);
        if(!info) return;

        const std::size_t sys_id = info->system_value;
        const std::size_t seq_id = info->sequent_value;
        const std::string track_name =
            make_thread_track_name(timer_track_tag{}, seq_id, sys_id);
        constexpr auto category_id =
            static_cast<uint32_t>(ROCPROFSYS_CATEGORY_TIMER_SAMPLING);
        constexpr auto category_str = "timer_sampling";

        for(auto const& sample : samples)
        {
            if(!info->is_valid_lifetime(sample.beg_ns, sample.end_ns)) continue;

            int depth = 0;
            for(auto const& frame : sample.stack)
            {
                std::string name = frame.name.empty()
                                       ? ("0x" + fmt::format("{:X}", frame.address))
                                       : frame.name;

                trace_cache::get_buffer_storage().store(make_region_sample(
                    category_id, sys_id, track_name, name, sample.beg_ns, sample.end_ns,
                    category_str, frame, depth, sample.metrics.cpu_ns));

                ++depth;
            }
        }
    }

    void store_overflow(int64_t tid, std::vector<overflow_sample> const& samples)
    {
        if(samples.empty()) return;

        auto info = resolver_.resolve(tid);
        if(!info) return;

        const std::size_t sys_id = info->system_value;
        const std::size_t seq_id = info->sequent_value;
        const std::string track_name =
            make_thread_track_name(overflow_track_tag{}, seq_id, sys_id);
        constexpr auto category_id =
            static_cast<uint32_t>(ROCPROFSYS_CATEGORY_OVERFLOW_SAMPLING);
        constexpr auto category_str = "overflow_sampling";

        for(auto const& sample : samples)
        {
            if(!info->is_valid_lifetime(sample.beg_ns, sample.end_ns)) continue;

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

    void store_thread_counters(int64_t tid, std::vector<timer_sample> const& samples)
    {
        if(samples.empty()) return;

        auto info = resolver_.resolve(tid);
        if(!info) return;

        const std::size_t sys_id = info->system_value;
        const std::size_t seq_id = info->sequent_value;

        for(auto const& sample : samples)
        {
            if(!info->is_valid_lifetime(sample.beg_ns, sample.end_ns)) continue;
            const std::uint64_t mid_ns = (sample.beg_ns + sample.end_ns) / 2;

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

            if(sample.metrics.valid.test(4) && !hw_labels_.empty())
            {
                constexpr auto hw_category =
                    static_cast<std::size_t>(ROCPROFSYS_CATEGORY_THREAD_HARDWARE_COUNTER);
                for(std::size_t idx = 0;
                    idx < hw_labels_.size() && idx < sample.metrics.hw_counter.size();
                    ++idx)
                {
                    std::string track =
                        hw_labels_[idx] + " [" + std::to_string(seq_id) + "]";
                    trace_cache::get_buffer_storage().store(
                        trace_cache::pmc_event_with_sample{
                            hw_category, std::move(track),
                            static_cast<std::size_t>(mid_ns), std::string{ "{}" },
                            /*stack_id*/ 0, /*parent_stack_id*/ 0,
                            /*correlation_id*/ 0, /*call_stack*/ std::string{},
                            /*line_info*/ std::string{}, static_cast<uint32_t>(sys_id),
                            /*device_type*/ uint8_t{ 0 }, hw_labels_[idx],
                            static_cast<double>(sample.metrics.hw_counter[idx]),
                            std::optional<int64_t>{} });
                }
            }
        }
    }

    void store_counter_termination(int64_t tid, uint64_t stop_ns)
    {
        auto info = resolver_.resolve(tid);
        if(!info) return;

        const std::size_t sys_id = info->system_value;
        const std::size_t seq_id = info->sequent_value;

        for(auto const& descriptor : thread_metric_descriptors)
        {
            std::string track = std::string{ descriptor.track_prefix } + " [" +
                                std::to_string(seq_id) + "]";
            trace_cache::get_buffer_storage().store(trace_cache::pmc_event_with_sample{
                descriptor.category_enum, std::move(track),
                static_cast<std::size_t>(stop_ns), std::string{ "{}" }, 0, 0, 0,
                std::string{}, std::string{}, static_cast<uint32_t>(sys_id), uint8_t{ 0 },
                std::string{ descriptor.track_prefix }, 0.0, std::optional<int64_t>{} });
        }

        if(!hw_labels_.empty())
        {
            constexpr auto hw_category =
                static_cast<std::size_t>(ROCPROFSYS_CATEGORY_THREAD_HARDWARE_COUNTER);
            for(auto const& label : hw_labels_)
            {
                std::string track = label + " [" + std::to_string(seq_id) + "]";
                trace_cache::get_buffer_storage().store(
                    trace_cache::pmc_event_with_sample{
                        hw_category, std::move(track), static_cast<std::size_t>(stop_ns),
                        std::string{ "{}" }, 0, 0, 0, std::string{}, std::string{},
                        static_cast<uint32_t>(sys_id), uint8_t{ 0 }, label, 0.0,
                        std::optional<int64_t>{} });
            }
        }
    }

private:
    ThreadInfoResolverT&     resolver_;
    std::vector<std::string> hw_labels_;

    static trace_cache::backtrace_region_sample make_region_sample(
        uint32_t category_id, std::size_t sys_id, std::string const& track_name,
        std::string const& name, std::uint64_t beg_ns, std::uint64_t end_ns,
        char const* category_str, stack_frame const& frame, int depth,
        int64_t cpu_ns = -1)
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
                                                     make_extdata_json(depth, cpu_ns) };
    }

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
