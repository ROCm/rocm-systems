// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/collectors/cpu/types.hpp"
#include "library/pmc/device_providers/procfs/drivers/driver.hpp"

#include <spdlog/fmt/fmt.h>

#include <map>
#include <memory>
#include <set>
#include <string>

namespace rocprofsys::pmc::collectors::cpu
{

/**
 * @brief CPU device that manages metric collection for a set of monitored CPUs.
 *
 * Unlike GPU (one device per GPU), a single CPU device manages all monitored
 * CPUs because /proc/stat and /proc/cpuinfo are read in a single pass.
 * Maintains previous jiffies state for delta-based CPU load computation.
 *
 * @tparam Driver The procfs driver type (real or mock).
 */
template <typename Driver>
class device
{
public:
    using cpu_jiffies     = drivers::procfs::cpu_jiffies;
    using rusage_snapshot = drivers::procfs::rusage_snapshot;

    /**
     * @param driver Shared pointer to the procfs driver.
     * @param socket_id Physical package (socket) ID for this device.
     * @param monitored_cpus Set of CPU IDs to collect metrics for.
     */
    device(std::shared_ptr<Driver> driver, size_t socket_id,
           std::set<size_t> monitored_cpus)
    : m_driver(std::move(driver))
    , m_socket_id(socket_id)
    , m_monitored_cpus(std::move(monitored_cpus))
    , m_device_name(fmt::format("CPU {}", socket_id))
    , m_product_name(fmt::format("CPU {}", socket_id))
    {
        initialize_supported_metrics();
    }

    [[nodiscard]] bool is_supported() const noexcept
    {
        return m_supported_metrics.value != 0;
    }

    [[nodiscard]] enabled_metrics get_supported_metrics() const noexcept
    {
        return m_supported_metrics;
    }

    [[nodiscard]] size_t get_index() const noexcept { return m_socket_id; }

    [[nodiscard]] const std::string& get_name() const noexcept { return m_device_name; }

    [[nodiscard]] const std::string& get_product_name() const noexcept
    {
        return m_product_name;
    }

    [[nodiscard]] const std::string& get_vendor_name() const noexcept
    {
        return m_vendor_name;
    }

    [[nodiscard]] const std::set<size_t>& get_monitored_cpus() const noexcept
    {
        return m_monitored_cpus;
    }

    /**
     * @brief Collect all CPU metrics in a single pass.
     *
     * Reads /proc/stat for load, /proc/cpuinfo for frequency, and getrusage
     * for process metrics. On the first call, load entries will have 0.0 because
     * there is no previous baseline for delta computation.
     *
     * @param enabled Enabled metrics bitfield — skips collection for disabled metrics.
     * @param timestamp Current timestamp in nanoseconds (unused by CPU).
     * @return Combined CPU metrics snapshot.
     */
    [[nodiscard]] metrics get_cpu_metrics(const enabled_metrics& enabled)
    {
        metrics result = make_empty_metrics();

        if(enabled.bits.load) collect_load_metrics(result);
        if(enabled.bits.frequency) collect_frequency_metrics(result);
        collect_process_metrics(result, enabled);

        return result;
    }

private:
    void initialize_supported_metrics()
    {
        m_supported_metrics.value = 0;

        auto jiffies = m_driver->read_proc_stat();
        if(!jiffies.empty())
        {
            m_supported_metrics.bits.load = 1;
        }

        auto freqs = m_driver->read_cpu_frequencies();
        if(!freqs.empty())
        {
            m_supported_metrics.bits.frequency = 1;
        }

        auto rusage                           = m_driver->read_rusage();
        m_supported_metrics.bits.page_rss     = 1;
        m_supported_metrics.bits.virt_mem     = 1;
        m_supported_metrics.bits.peak_rss     = (rusage.peak_rss > 0) ? 1u : 0u;
        m_supported_metrics.bits.ctx_switches = 1;
        m_supported_metrics.bits.page_faults  = 1;
        m_supported_metrics.bits.user_time    = 1;
        m_supported_metrics.bits.kernel_time  = 1;
    }

    /**
     * @brief Collect CPU load from /proc/stat deltas.
     *
     * Computes load as: (delta_active / delta_total) * 100.0
     * On first call per CPU, stores baseline and produces a 0.0 load entry.
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
                m_prev_jiffies[cpu_id] = curr_it->second;
                auto* entry            = find_or_create_cpu_entry(result, cpu_id);
                entry->load            = 0.0;
                continue;
            }

            const auto& prev = prev_it->second;
            const auto& curr = curr_it->second;

            // Guard against counter reset (e.g., CPU went offline/online)
            if(curr.total() < prev.total())
            {
                m_prev_jiffies[cpu_id] = curr_it->second;
                auto* entry            = find_or_create_cpu_entry(result, cpu_id);
                entry->load            = 0.0;
                continue;
            }

            const uint64_t total_delta  = curr.total() - prev.total();
            const uint64_t active_delta = curr.active() - prev.active();

            double load_pct = 0.0;
            if(total_delta > 0)
            {
                load_pct = 100.0 * static_cast<double>(active_delta) /
                           static_cast<double>(total_delta);
            }

            auto* entry = find_or_create_cpu_entry(result, cpu_id);
            entry->load = load_pct;

            m_prev_jiffies[cpu_id] = curr_it->second;
        }
    }

    void collect_frequency_metrics(metrics& result)
    {
        if(!m_supported_metrics.bits.frequency) return;

        auto freqs = m_driver->read_cpu_frequencies();

        for(const auto& cpu_id : m_monitored_cpus)
        {
            auto freq_it = freqs.find(cpu_id);
            if(freq_it == freqs.end()) continue;

            auto* entry      = find_or_create_cpu_entry(result, cpu_id);
            entry->frequency = freq_it->second;
        }
    }

    void collect_process_metrics(metrics& result, const enabled_metrics& enabled)
    {
        const bool any_process_metric =
            enabled.bits.page_rss || enabled.bits.virt_mem || enabled.bits.peak_rss ||
            enabled.bits.ctx_switches || enabled.bits.page_faults ||
            enabled.bits.user_time || enabled.bits.kernel_time;
        if(!any_process_metric) return;

        auto snap = m_driver->read_rusage();

        if(m_supported_metrics.bits.page_rss && enabled.bits.page_rss)
            result.process_data.page_rss = snap.page_rss;
        if(m_supported_metrics.bits.virt_mem && enabled.bits.virt_mem)
            result.process_data.virt_mem = snap.virt_mem;
        if(m_supported_metrics.bits.peak_rss && enabled.bits.peak_rss)
            result.process_data.peak_rss = snap.peak_rss;
        if(m_supported_metrics.bits.ctx_switches && enabled.bits.ctx_switches)
            result.process_data.context_switches = snap.context_switches;
        if(m_supported_metrics.bits.page_faults && enabled.bits.page_faults)
            result.process_data.page_faults = snap.page_faults;
        if(m_supported_metrics.bits.user_time && enabled.bits.user_time)
            result.process_data.user_mode_time = snap.user_mode_time;
        if(m_supported_metrics.bits.kernel_time && enabled.bits.kernel_time)
            result.process_data.kernel_mode_time = snap.kernel_mode_time;
    }

    /**
     * @brief Create a metrics struct pre-populated with zero entries for each
     * monitored CPU. This ensures that even a default/pause sample serializes
     * per-CPU entries, so Perfetto tracks drop to zero instead of retaining
     * their last value.
     */
    metrics make_empty_metrics() const
    {
        metrics result;
        result.cpu_data.reserve(m_monitored_cpus.size());
        for(auto cpu_id : m_monitored_cpus)
        {
            result.cpu_data.push_back({ cpu_id, 0.0f, 0.0 });
        }
        return result;
    }

    per_cpu_metrics* find_or_create_cpu_entry(metrics& result, size_t cpu_id)
    {
        for(auto& entry : result.cpu_data)
        {
            if(entry.cpu_id == cpu_id) return &entry;
        }
        result.cpu_data.push_back({ cpu_id, 0.0f, 0.0 });
        return &result.cpu_data.back();
    }

    std::shared_ptr<Driver>       m_driver;
    size_t                        m_socket_id = 0;
    std::set<size_t>              m_monitored_cpus;
    enabled_metrics               m_supported_metrics{};
    std::map<size_t, cpu_jiffies> m_prev_jiffies;
    std::string                   m_device_name;
    std::string                   m_product_name;
    std::string                   m_vendor_name = "AMD";
};

}  // namespace rocprofsys::pmc::collectors::cpu
