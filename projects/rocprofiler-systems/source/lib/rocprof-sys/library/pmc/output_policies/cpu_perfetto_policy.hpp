// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include "core/perfetto.hpp"
#include "library/pmc/cpu/types.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"

#include <timemory/units.hpp>

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

namespace rocprofsys
{
namespace pmc
{
namespace output_policies
{
namespace
{
struct perfetto_cpu_sample
{
    uint64_t     timestamp;
    cpu::metrics metrics;
};

inline std::vector<perfetto_cpu_sample>&
get_cpu_perfetto_samples()
{
    static std::vector<perfetto_cpu_sample> samples;
    return samples;
}
}  // namespace

struct cpu_perfetto_policy
{
    using freq_track    = perfetto_counter_track<category::cpu_freq>;
    using load_track    = perfetto_counter_track<category::cpu_load>;
    using page_track    = perfetto_counter_track<category::process_page>;
    using virt_track    = perfetto_counter_track<category::process_virt>;
    using peak_track    = perfetto_counter_track<category::process_peak>;
    using cswitch_track = perfetto_counter_track<category::process_context_switch>;
    using pfault_track  = perfetto_counter_track<category::process_page_fault>;
    using utime_track   = perfetto_counter_track<category::process_user_mode_time>;
    using ktime_track   = perfetto_counter_track<category::process_kernel_mode_time>;

    static void init_storage()
    {
        perfetto_counter_track<category::cpu_freq>::init();
        perfetto_counter_track<category::cpu_load>::init();
        perfetto_counter_track<category::process_page>::init();
        perfetto_counter_track<category::process_virt>::init();
        perfetto_counter_track<category::process_peak>::init();
        perfetto_counter_track<category::process_context_switch>::init();
        perfetto_counter_track<category::process_page_fault>::init();
        perfetto_counter_track<category::process_user_mode_time>::init();
        perfetto_counter_track<category::process_kernel_mode_time>::init();
    }

    static void setup_counter_tracks(const std::set<size_t>&    cpu_ids,
                                     const cpu::enabled_metrics& enabled_metrics)
    {
        // Per-CPU frequency tracks
        if(enabled_metrics.bits.frequency)
        {
            for(const auto& cpu_id : cpu_ids)
            {
                auto name = fmt::format("CPU Frequency [{}] (S)", cpu_id);
                freq_track::emplace(cpu_id, name.c_str(), "MHz");
            }
        }

        // Per-CPU load tracks
        if(enabled_metrics.bits.load)
        {
            for(const auto& cpu_id : cpu_ids)
            {
                auto name = fmt::format("CPU Load [{}] (S)", cpu_id);
                load_track::emplace(cpu_id, name.c_str(), "%");
            }
        }

        // Process-level tracks (single instance, device_id = 0)
        auto addendum = [](const char* label) {
            return fmt::format("CPU [{}] (S)", label);
        };

        if(enabled_metrics.bits.page_rss)
            page_track::emplace(0, addendum("Memory Usage"), "MB");
        if(enabled_metrics.bits.virt_mem)
            virt_track::emplace(0, addendum("Virtual Memory Usage"), "MB");
        if(enabled_metrics.bits.peak_rss)
            peak_track::emplace(0, addendum("Peak Memory"), "MB");
        if(enabled_metrics.bits.ctx_switches)
            cswitch_track::emplace(0, addendum("Context Switches"), "");
        if(enabled_metrics.bits.page_faults)
            pfault_track::emplace(0, addendum("Page Faults"), "");
        if(enabled_metrics.bits.user_time)
            utime_track::emplace(0, addendum("User Time"), "sec");
        if(enabled_metrics.bits.kernel_time)
            ktime_track::emplace(0, addendum("Kernel Time"), "sec");
    }

    static void store_sample(const cpu::metrics& metric_values, uint64_t timestamp)
    {
        get_cpu_perfetto_samples().emplace_back(
            perfetto_cpu_sample{timestamp, metric_values});
    }

    static void post_process(const std::set<size_t>&    cpu_ids,
                             const cpu::enabled_metrics& enabled_metrics)
    {
        auto& samples = get_cpu_perfetto_samples();

        LOG_DEBUG("Post-processing {} CPU PMC samples", samples.size());

        const auto& thread_info = thread_info::get(0, InternalTID);
        if(!thread_info) return;

        for(const auto& sample : samples)
        {
            const auto ts = sample.timestamp;
            if(!thread_info->is_valid_time(ts)) continue;

            // Per-CPU frequency
            if(enabled_metrics.bits.frequency)
            {
                for(const auto& cpu_data : sample.metrics.cpu_data)
                {
                    if(freq_track::exists(cpu_data.cpu_id))
                    {
                        TRACE_COUNTER(trait::name<category::cpu_freq>::value,
                                      freq_track::at(cpu_data.cpu_id, 0), ts,
                                      static_cast<double>(cpu_data.frequency));
                    }
                }
            }

            // Per-CPU load
            if(enabled_metrics.bits.load)
            {
                for(const auto& cpu_data : sample.metrics.cpu_data)
                {
                    if(load_track::exists(cpu_data.cpu_id))
                    {
                        TRACE_COUNTER(trait::name<category::cpu_load>::value,
                                      load_track::at(cpu_data.cpu_id, 0), ts,
                                      cpu_data.load);
                    }
                }
            }

            // Process-level metrics
            const auto& proc = sample.metrics.process_data;

            if(enabled_metrics.bits.page_rss && page_track::exists(0))
            {
                TRACE_COUNTER(trait::name<category::process_page>::value,
                              page_track::at(0, 0), ts,
                              proc.page_rss / static_cast<double>(units::megabyte));
            }
            if(enabled_metrics.bits.virt_mem && virt_track::exists(0))
            {
                TRACE_COUNTER(trait::name<category::process_virt>::value,
                              virt_track::at(0, 0), ts,
                              proc.virt_mem / static_cast<double>(units::megabyte));
            }
            if(enabled_metrics.bits.peak_rss && peak_track::exists(0))
            {
                TRACE_COUNTER(trait::name<category::process_peak>::value,
                              peak_track::at(0, 0), ts,
                              proc.peak_rss / static_cast<double>(units::megabyte));
            }
            if(enabled_metrics.bits.ctx_switches && cswitch_track::exists(0))
            {
                TRACE_COUNTER(
                    trait::name<category::process_context_switch>::value,
                    cswitch_track::at(0, 0), ts,
                    static_cast<uint64_t>(proc.context_switches));
            }
            if(enabled_metrics.bits.page_faults && pfault_track::exists(0))
            {
                TRACE_COUNTER(trait::name<category::process_page_fault>::value,
                              pfault_track::at(0, 0), ts,
                              static_cast<uint64_t>(proc.page_faults));
            }
            if(enabled_metrics.bits.user_time && utime_track::exists(0))
            {
                TRACE_COUNTER(
                    trait::name<category::process_user_mode_time>::value,
                    utime_track::at(0, 0), ts,
                    proc.user_mode_time / static_cast<double>(units::sec));
            }
            if(enabled_metrics.bits.kernel_time && ktime_track::exists(0))
            {
                TRACE_COUNTER(
                    trait::name<category::process_kernel_mode_time>::value,
                    ktime_track::at(0, 0), ts,
                    proc.kernel_mode_time / static_cast<double>(units::sec));
            }
        }

        // Write end markers
        auto end_ts = thread_info->get_stop();
        for(const auto& cpu_id : cpu_ids)
        {
            if(enabled_metrics.bits.frequency && freq_track::exists(cpu_id))
            {
                TRACE_COUNTER(trait::name<category::cpu_freq>::value,
                              freq_track::at(cpu_id, 0), end_ts, 0);
            }
            if(enabled_metrics.bits.load && load_track::exists(cpu_id))
            {
                TRACE_COUNTER(trait::name<category::cpu_load>::value,
                              load_track::at(cpu_id, 0), end_ts, 0.0);
            }
        }

        if(enabled_metrics.bits.page_rss && page_track::exists(0))
            TRACE_COUNTER(trait::name<category::process_page>::value,
                          page_track::at(0, 0), end_ts, 0.0);
        if(enabled_metrics.bits.virt_mem && virt_track::exists(0))
            TRACE_COUNTER(trait::name<category::process_virt>::value,
                          virt_track::at(0, 0), end_ts, 0.0);
        if(enabled_metrics.bits.peak_rss && peak_track::exists(0))
            TRACE_COUNTER(trait::name<category::process_peak>::value,
                          peak_track::at(0, 0), end_ts, 0.0);
        if(enabled_metrics.bits.ctx_switches && cswitch_track::exists(0))
            TRACE_COUNTER(trait::name<category::process_context_switch>::value,
                          cswitch_track::at(0, 0), end_ts, 0);
        if(enabled_metrics.bits.page_faults && pfault_track::exists(0))
            TRACE_COUNTER(trait::name<category::process_page_fault>::value,
                          pfault_track::at(0, 0), end_ts, 0);
        if(enabled_metrics.bits.user_time && utime_track::exists(0))
            TRACE_COUNTER(trait::name<category::process_user_mode_time>::value,
                          utime_track::at(0, 0), end_ts, 0.0);
        if(enabled_metrics.bits.kernel_time && ktime_track::exists(0))
            TRACE_COUNTER(trait::name<category::process_kernel_mode_time>::value,
                          ktime_track::at(0, 0), end_ts, 0.0);

        samples.clear();
    }
};

}  // namespace output_policies
}  // namespace pmc
}  // namespace rocprofsys
