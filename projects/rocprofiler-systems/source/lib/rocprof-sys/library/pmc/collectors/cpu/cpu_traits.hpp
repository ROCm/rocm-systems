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
//   this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimers in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this Software without specific prior written permission.
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
#include "library/pmc/common/types.hpp"
#include "library/pmc/cpu/types.hpp"
#include "logger/debug.hpp"

#include <cstring>
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

/**
 * @brief Traits type for CPU collector configuration.
 *
 * Defines types, constants, and customization points for the base collector template
 * to work with CPU devices via procfs.
 *
 * Note: Unlike GPU which manages multiple device instances, CPU manages a single
 * device that monitors multiple CPU cores. The device filter applies to which cores
 * are monitored, not which device instances are created.
 *
 * @tparam Driver The procfs driver type (real or mock for testing)
 */
template <typename Driver>
struct cpu_traits
{
    // Required type aliases for base::collector
    using metrics_t         = pmc::cpu::metrics;
    using enabled_metrics_t = pmc::cpu::enabled_metrics;
    using device_t          = device<Driver>;
    using device_ptr_t      = std::shared_ptr<device_t>;
    using container_t       = std::vector<device_ptr_t>;  // Single-element for CPU
    using driver_t          = Driver;

    // Required constants
    static constexpr const char* device_name  = "CPU";
    static constexpr bool        multi_device = false;  // Single logical device

    // Settings customization points

    /**
     * @brief Get the device filter from settings.
     *
     * For CPU, this filter applies to which cores are monitored,
     * not device instances (since there's only one logical CPU device).
     */
    template <typename Settings>
    static device_filter get_device_filter()
    {
        return Settings::get_cpu_device_filter();
    }

    /**
     * @brief Get enabled metrics from settings.
     */
    template <typename Settings>
    static enabled_metrics_t get_enabled_metrics()
    {
        return Settings::get_cpu_enabled_metrics();
    }

    // Provider customization points

    /**
     * @brief Get CPU count from the provider.
     *
     * Note: This is NOT used for device enumeration (CPU has single device),
     * but for determining which cores to monitor.
     */
    template <typename Provider>
    static size_t get_cpu_count(Provider& provider)
    {
        return provider.get_cpu_count();
    }

    /**
     * @brief CPU doesn't filter by processor type (no multi-socket support yet).
     */
    static constexpr bool filter_by_processor_type() { return false; }

    // Cache API delegation

    /**
     * @brief Initialize category metadata in the cache.
     */
    template <typename Cache>
    static void init_category_metadata()
    {
        Cache::initialize_category_metadata();
    }

    /**
     * @brief Initialize CPU tracks metadata in the cache.
     *
     * Base::collector calls this with device_index, but CPU uses monitored cores.
     */
    template <typename Cache>
    static void init_tracks_metadata()
    {
        Cache::initialize_cpu_tracks_metadata(s_monitored_cpus);
    }

    /**
     * @brief Initialize PMC metadata for monitored CPUs.
     *
     * Base::collector calls this with device_index, but CPU uses monitored cores.
     */
    template <typename Cache>
    static void init_pmc_metadata(size_t /* device_index */)
    {
        if(s_monitored_cpus.empty()) return;
        auto cpu_agent =
            get_agent_manager_instance().get_agent_by_type_index(0, agent_type::CPU);
        Cache::initialize_cpu_pmc_metadata(s_monitored_cpus, cpu_agent.base_id);
    }

    /**
     * @brief Store a sample to the cache.
     *
     * Base::collector signature includes device_id and supported_metrics,
     * but CPU only has one device so we ignore those params.
     */
    template <typename Cache>
    static void store_sample(size_t /* device_id */, const enabled_metrics_t& enabled,
                             const enabled_metrics_t& /* supported */,
                             const metrics_t& metrics, uint64_t timestamp)
    {
        auto serialized_freqs = serialize_frequencies(metrics);
        auto serialized_loads = serialize_loads(metrics);
        Cache::store_sample(enabled, metrics, timestamp, serialized_freqs,
                            serialized_loads);
    }

    // Perfetto API delegation

    /**
     * @brief Initialize Perfetto storage for CPU.
     *
     * Note: CPU has different init_storage signature than GPU (no devices param).
     */
    template <typename Perfetto, typename DeviceVector>
    static void init_perfetto_storage(const DeviceVector&)
    {
        // CPU init_storage doesn't need device vector parameter
        Perfetto::init_storage();
    }

    /**
     * @brief Setup Perfetto counter tracks for monitored CPUs.
     *
     * Converts device_index to monitored CPUs set for CPU-specific API.
     */
    template <typename Perfetto>
    static void setup_counter_tracks(size_t /* device_index */,
                                     const enabled_metrics_t& enabled)
    {
        Perfetto::setup_counter_tracks(s_monitored_cpus, enabled);
    }

    /**
     * @brief Store a sample to Perfetto.
     *
     * Note: CPU uses different signature (no device_id param).
     */
    template <typename Perfetto>
    static void store_perfetto_sample(size_t /* device_id */, const metrics_t& metrics,
                                      uint64_t timestamp)
    {
        Perfetto::store_sample(metrics, timestamp);
    }

    /**
     * @brief Post-process Perfetto data.
     */
    template <typename Perfetto, typename DeviceVector>
    static void post_process_perfetto(const DeviceVector&,
                                      const enabled_metrics_t& enabled)
    {
        Perfetto::post_process(s_monitored_cpus, enabled);
    }

    // Device creation

    /**
     * @brief Create a new CPU device instance.
     *
     * Note: Unlike GPU, CPU creates a single device that monitors multiple cores.
     * Stores the monitored CPUs set in the device for later retrieval.
     */
    static device_ptr_t create_device(std::shared_ptr<driver_t> driver,
                                      const std::set<size_t>&   monitored_cpus)
    {
        s_monitored_cpus = monitored_cpus;
        auto device      = std::make_shared<device_t>(std::move(driver), monitored_cpus);
        return device;
    }

    /**
     * @brief Get the monitored CPUs set from the device.
     * Internal helper for traits methods that need access to monitored cores.
     */
    static const std::set<size_t>& get_monitored_cpus(const device_ptr_t&)
    {
        return s_monitored_cpus;
    }

    /**
     * @brief Get CPU metrics from the device.
     */
    static metrics_t get_metrics(const device_ptr_t& device, const enabled_metrics_t&)
    {
        // CPU doesn't use per-call enabled metrics filtering (done during sample storage)
        return device->get_cpu_metrics();
    }

    /**
     * @brief Check if the device is supported.
     */
    static bool is_device_supported(const device_ptr_t& device)
    {
        return device->is_supported();
    }

    /**
     * @brief Get device index (always 0 for CPU since there's only one logical device).
     */
    static size_t get_device_index(const device_ptr_t&)
    {
        return 0;  // CPU has single logical device
    }

    /**
     * @brief Get supported metrics (CPU supports all defined metrics).
     */
    static enabled_metrics_t get_supported_metrics(const device_ptr_t&)
    {
        // CPU procfs driver supports all defined metrics
        enabled_metrics_t supported;
        supported.bits.frequency    = true;
        supported.bits.load         = true;
        supported.bits.page_rss     = true;
        supported.bits.virt_mem     = true;
        supported.bits.peak_rss     = true;
        supported.bits.ctx_switches = true;
        supported.bits.page_faults  = true;
        supported.bits.user_time    = true;
        supported.bits.kernel_time  = true;
        return supported;
    }

private:
    // Static storage for monitored CPUs (shared across trait method calls)
    inline static std::set<size_t> s_monitored_cpus;

public:
    // Device enumeration

    /**
     * @brief Enumerate CPU device (single logical device with multiple cores).
     *
     * Unlike GPU which creates multiple device instances, CPU creates a single
     * device that monitors multiple cores based on filter.
     */
    template <typename Provider, typename Settings>
    static container_t enumerate_devices(Provider& provider, const device_filter& filter)
    {
        container_t devices;

        auto cpu_count = get_cpu_count(provider);
        LOG_INFO("Detected {} online CPUs for PMC sampling", cpu_count);

        // Determine which CPUs to monitor
        std::set<size_t> monitored_cpus;

        switch(filter.mode)
        {
            case device_selection_mode::ALL:
                for(size_t i = 0; i < cpu_count; ++i)
                    monitored_cpus.insert(i);
                break;
            case device_selection_mode::NONE:
                LOG_DEBUG("{} sampling disabled via configuration", device_name);
                return devices;
            case device_selection_mode::SPECIFIC:
                for(auto idx : filter.indices)
                {
                    if(idx < cpu_count)
                        monitored_cpus.insert(idx);
                    else
                        LOG_WARNING("CPU index {} out of range (max: {})", idx,
                                    cpu_count - 1);
                }
                break;
        }

        if(monitored_cpus.empty())
        {
            LOG_WARNING("No CPUs selected for monitoring");
            return devices;
        }

        // Create single CPU device that monitors selected cores
        auto driver = provider.get_driver();
        auto device = create_device(driver, monitored_cpus);

        if(!is_device_supported(device))
        {
            LOG_WARNING("No CPU metrics are supported on this system");
        }

        devices.emplace_back(std::move(device));
        LOG_INFO("Enabled {} CPUs for PMC sampling", monitored_cpus.size());

        return devices;
    }

    /**
     * @brief Serialize per-CPU frequency data as pairs of (cpu_id, freq).
     */
    static std::vector<uint8_t> serialize_frequencies(const metrics_t& metrics)
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
    static std::vector<uint8_t> serialize_loads(const metrics_t& metrics)
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
};

}  // namespace cpu
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
