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

#include "core/agent.hpp"
#include "core/agent_manager.hpp"
#include "library/pmc/collectors/cpu/device.hpp"
#include "library/pmc/cpu/types.hpp"
#include "logger/debug.hpp"

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <set>
#include <vector>

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace cpu
{
using ::rocprofsys::pmc::device_filter;
using ::rocprofsys::pmc::device_selection_mode;
using ::rocprofsys::pmc::cpu::enabled_metrics;

using get_timestamp_t = std::function<unsigned long()>;

/**
 * @brief CPU metrics collector for performance monitoring.
 *
 * Manages the lifecycle of CPU performance monitoring, including
 * CPU enumeration, metric sampling, and data storage. Unlike the GPU
 * collector which manages multiple devices, the CPU collector manages
 * a single device that reads all monitored CPUs.
 *
 * @tparam Provider Type providing CPU enumeration and driver access
 * @tparam Config Configuration policy providing settings and output policies
 */
template <typename Provider, typename Config>
struct collector
{
    using provider_type = Provider;
    using SettingsApi   = typename Config::SettingsApi;
    using PerfettoApi   = typename Config::CpuPerfettoApi;
    using CacheApi      = typename Config::CpuCacheApi;
    using driver_t      = typename provider_type::driver_t;
    using device_t      = device<driver_t>;

    explicit collector(std::shared_ptr<provider_type> provider)
    : m_provider(std::move(provider))
    {}

    collector() = delete;

    void setup()
    {
        if(!m_provider)
        {
            throw std::runtime_error(
                "CPU provider not set. Use constructor.");
        }

        auto cpu_count = m_provider->get_cpu_count();
        LOG_INFO("Detected {} online CPUs for PMC sampling", cpu_count);

        // Determine which CPUs to monitor
        auto filter = SettingsApi::get_cpu_device_filter();
        m_monitored_cpus.clear();

        switch(filter.mode)
        {
            case device_selection_mode::ALL:
                for(size_t i = 0; i < cpu_count; ++i)
                    m_monitored_cpus.insert(i);
                break;
            case device_selection_mode::NONE:
                break;
            case device_selection_mode::SPECIFIC:
                for(auto idx : filter.indices)
                {
                    if(idx < cpu_count)
                        m_monitored_cpus.insert(idx);
                }
                break;
        }

        m_enabled_metrics = SettingsApi::get_cpu_enabled_metrics();

        // Create the CPU device
        auto driver = m_provider->get_driver();
        m_device    = std::make_shared<device_t>(driver, m_monitored_cpus);

        if(!m_device->is_supported())
        {
            LOG_WARNING("No CPU metrics are supported on this system");
        }

        LOG_INFO("Enabled {} CPUs for PMC sampling", m_monitored_cpus.size());

        if(SettingsApi::get_use_perfetto_legacy_metrics())
        {
            PerfettoApi::init_storage();
        }
    }

    void config()
    {
        CacheApi::initialize_category_metadata();
        CacheApi::initialize_cpu_tracks_metadata(m_monitored_cpus);

        // Register CPU PMC metadata once for all monitored CPUs
        // Use the CPU agent's base_id to match what RocPD processor uses
        if(!m_monitored_cpus.empty())
        {
            auto cpu_agent =
                get_agent_manager_instance().get_agent_by_type_index(0, agent_type::CPU);
            CacheApi::initialize_cpu_pmc_metadata(m_monitored_cpus, cpu_agent.base_id);
        }

        if(SettingsApi::get_use_perfetto_legacy_metrics())
        {
            PerfettoApi::setup_counter_tracks(m_monitored_cpus,
                                              m_enabled_metrics);
        }
    }

    void sample(const get_timestamp_t& get_timestamp)
    {
        if(!m_device || m_monitored_cpus.empty()) return;

        auto timestamp = get_timestamp();

        try
        {
            auto cpu_metrics = m_device->get_cpu_metrics();

            // Serialize per-CPU freq and load data for cache storage
            auto serialized_freqs = serialize_frequencies(cpu_metrics);
            auto serialized_loads = serialize_loads(cpu_metrics);

            CacheApi::store_sample(m_enabled_metrics, cpu_metrics, timestamp,
                                   serialized_freqs, serialized_loads);

            if(SettingsApi::get_use_perfetto_legacy_metrics())
            {
                PerfettoApi::store_sample(cpu_metrics, timestamp);
            }
        }
        catch(const std::runtime_error& e)
        {
            LOG_ERROR("Failed to sample CPU metrics: {}", e.what());
        }
    }

    void post_process()
    {
        if(SettingsApi::get_use_perfetto_legacy_metrics())
        {
            PerfettoApi::post_process(m_monitored_cpus, m_enabled_metrics);
        }
    }

    void shutdown()
    {
        m_device.reset();
        if(m_provider)
        {
            m_provider->shutdown();
            m_provider.reset();
        }
    }

    const std::set<size_t>& get_monitored_cpus() const noexcept
    {
        return m_monitored_cpus;
    }

    size_t get_cpu_count() const noexcept { return m_monitored_cpus.size(); }

private:
    /**
     * @brief Serialize per-CPU frequency data as pairs of (cpu_id, freq).
     */
    static std::vector<uint8_t>
    serialize_frequencies(const pmc::cpu::metrics& metrics)
    {
        constexpr size_t idx_size   = sizeof(size_t);
        constexpr size_t value_size = sizeof(float);

        std::vector<uint8_t> result;
        result.resize(metrics.cpu_data.size() * (idx_size + value_size));

        size_t offset = 0;
        for(const auto& cpu : metrics.cpu_data)
        {
            std::memcpy(result.data() + offset, &cpu.cpu_id, idx_size);
            offset += idx_size;
            std::memcpy(result.data() + offset, &cpu.frequency, value_size);
            offset += value_size;
        }
        return result;
    }

    /**
     * @brief Serialize per-CPU load data as pairs of (cpu_id, load).
     */
    static std::vector<uint8_t>
    serialize_loads(const pmc::cpu::metrics& metrics)
    {
        constexpr size_t idx_size   = sizeof(size_t);
        constexpr size_t value_size = sizeof(double);

        std::vector<uint8_t> result;
        result.resize(metrics.cpu_data.size() * (idx_size + value_size));

        size_t offset = 0;
        for(const auto& cpu : metrics.cpu_data)
        {
            std::memcpy(result.data() + offset, &cpu.cpu_id, idx_size);
            offset += idx_size;
            std::memcpy(result.data() + offset, &cpu.load, value_size);
            offset += value_size;
        }
        return result;
    }

    std::shared_ptr<provider_type> m_provider;
    std::shared_ptr<device_t>      m_device;
    std::set<size_t>               m_monitored_cpus;
    enabled_metrics                m_enabled_metrics;
};

}  // namespace cpu
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
