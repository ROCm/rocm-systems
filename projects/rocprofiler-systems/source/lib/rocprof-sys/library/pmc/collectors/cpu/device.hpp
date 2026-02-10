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

#include "library/pmc/cpu/types.hpp"
#include "library/pmc/device_providers/procfs/drivers/driver.hpp"

#include <map>
#include <memory>
#include <set>

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace cpu
{

using ::rocprofsys::pmc::cpu::enabled_metrics;
using ::rocprofsys::pmc::cpu::metrics;
using ::rocprofsys::pmc::cpu::per_cpu_metrics;
using ::rocprofsys::pmc::cpu::process_metrics;

/**
 * @brief CPU device that manages metric collection for a set of monitored CPUs.
 *
 * Unlike the GPU device (one per GPU), a single CPU device manages all
 * monitored CPUs because /proc/stat and /proc/cpuinfo are read in a single
 * pass for all CPUs. The device maintains previous jiffies state for
 * delta-based CPU load computation.
 *
 * @tparam Driver The procfs driver type (real or mock).
 */
template <typename Driver>
class device
{
public:
    using cpu_jiffies     = drivers::procfs::cpu_jiffies;
    using rusage_snapshot  = drivers::procfs::rusage_snapshot;

    /**
     * @brief Construct a CPU device.
     * @param driver Shared pointer to the procfs driver.
     * @param monitored_cpus Set of CPU IDs to collect metrics for.
     */
    device(std::shared_ptr<Driver> driver, std::set<size_t> monitored_cpus)
    : m_driver(std::move(driver))
    , m_monitored_cpus(std::move(monitored_cpus))
    {
        initialize_supported_metrics();
    }

    /**
     * @brief Check if any metrics are supported.
     */
    [[nodiscard]] bool is_supported() const noexcept
    {
        return m_supported_metrics.value != 0;
    }

    /**
     * @brief Get the bitfield of supported metrics.
     */
    [[nodiscard]] enabled_metrics get_supported_metrics() const noexcept
    {
        return m_supported_metrics;
    }

    /**
     * @brief Get the set of monitored CPU IDs.
     */
    [[nodiscard]] const std::set<size_t>& get_monitored_cpus() const noexcept
    {
        return m_monitored_cpus;
    }

    /**
     * @brief Collect all CPU metrics in a single pass.
     *
     * Reads /proc/stat for load, /proc/cpuinfo for frequency, and getrusage
     * for process metrics. On the first call, load data will be empty because
     * there is no previous baseline for delta computation.
     *
     * @return Combined CPU metrics snapshot.
     */
    [[nodiscard]] metrics get_cpu_metrics()
    {
        metrics result;

        collect_load_metrics(result);
        collect_frequency_metrics(result);
        collect_process_metrics(result);

        return result;
    }

private:
    /**
     * @brief Probe which metrics are available on this system.
     */
    void initialize_supported_metrics()
    {
        m_supported_metrics.value = 0;

        // Check if we can read /proc/stat
        auto jiffies = m_driver->read_proc_stat();
        if(!jiffies.empty())
        {
            m_supported_metrics.bits.load = 1;
        }

        // Check if we can read /proc/cpuinfo frequencies
        auto freqs = m_driver->read_cpu_frequencies();
        if(!freqs.empty())
        {
            m_supported_metrics.bits.frequency = 1;
        }

        // Process metrics are always available via getrusage
        auto rusage                           = m_driver->read_rusage();
        m_supported_metrics.bits.page_rss     = 1;
        m_supported_metrics.bits.virt_mem      = 1;
        m_supported_metrics.bits.peak_rss     = (rusage.peak_rss > 0) ? 1 : 0;
        m_supported_metrics.bits.ctx_switches = 1;
        m_supported_metrics.bits.page_faults  = 1;
        m_supported_metrics.bits.user_time    = 1;
        m_supported_metrics.bits.kernel_time  = 1;
    }

    /**
     * @brief Collect CPU load from /proc/stat deltas.
     *
     * Computes load as: (delta_active / delta_total) * 100.0
     * Skips CPUs not in the monitored set. On first call per CPU,
     * stores baseline and produces no load entry.
     */
    void collect_load_metrics(metrics& result)
    {
        if(!m_supported_metrics.bits.load) return;

        auto current_jiffies = m_driver->read_proc_stat();

        for(const auto& cpu_id : m_monitored_cpus)
        {
            auto curr_it = current_jiffies.find(cpu_id);
            if(curr_it == current_jiffies.end()) continue;

            auto prev_it = m_prev_jiffies.find(cpu_id);
            if(prev_it == m_prev_jiffies.end())
            {
                // First sample: store baseline, no load data
                m_prev_jiffies[cpu_id] = curr_it->second;
                continue;
            }

            const auto& prev = prev_it->second;
            const auto& curr = curr_it->second;

            uint64_t total_delta  = curr.total() - prev.total();
            uint64_t active_delta = curr.active() - prev.active();

            double load_pct = 0.0;
            if(total_delta > 0)
            {
                load_pct = 100.0 * static_cast<double>(active_delta) /
                           static_cast<double>(total_delta);
            }

            // Find or create per_cpu_metrics entry for this CPU
            auto* cpu_entry = find_or_create_cpu_entry(result, cpu_id);
            cpu_entry->load = load_pct;

            // Update baseline
            m_prev_jiffies[cpu_id] = curr_it->second;
        }
    }

    /**
     * @brief Collect CPU frequencies from /proc/cpuinfo.
     */
    void collect_frequency_metrics(metrics& result)
    {
        if(!m_supported_metrics.bits.frequency) return;

        auto freqs = m_driver->read_cpu_frequencies();

        for(const auto& cpu_id : m_monitored_cpus)
        {
            auto freq_it = freqs.find(cpu_id);
            if(freq_it == freqs.end()) continue;

            auto* cpu_entry      = find_or_create_cpu_entry(result, cpu_id);
            cpu_entry->frequency = freq_it->second;
        }
    }

    /**
     * @brief Collect process-level resource usage metrics.
     */
    void collect_process_metrics(metrics& result)
    {
        auto snap = m_driver->read_rusage();

        if(m_supported_metrics.bits.page_rss)
            result.process_data.page_rss = snap.page_rss;
        if(m_supported_metrics.bits.virt_mem)
            result.process_data.virt_mem = snap.virt_mem;
        if(m_supported_metrics.bits.peak_rss)
            result.process_data.peak_rss = snap.peak_rss;
        if(m_supported_metrics.bits.ctx_switches)
            result.process_data.context_switches = snap.context_switches;
        if(m_supported_metrics.bits.page_faults)
            result.process_data.page_faults = snap.page_faults;
        if(m_supported_metrics.bits.user_time)
            result.process_data.user_mode_time = snap.user_mode_time;
        if(m_supported_metrics.bits.kernel_time)
            result.process_data.kernel_mode_time = snap.kernel_mode_time;
    }

    /**
     * @brief Find or create a per_cpu_metrics entry for the given CPU ID.
     */
    per_cpu_metrics* find_or_create_cpu_entry(metrics& result, size_t cpu_id)
    {
        for(auto& entry : result.cpu_data)
        {
            if(entry.cpu_id == cpu_id) return &entry;
        }
        result.cpu_data.push_back({cpu_id, 0.0f, 0.0});
        return &result.cpu_data.back();
    }

    std::shared_ptr<Driver>          m_driver;
    std::set<size_t>                 m_monitored_cpus;
    enabled_metrics                  m_supported_metrics;
    std::map<size_t, cpu_jiffies>    m_prev_jiffies;
};

}  // namespace cpu
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
