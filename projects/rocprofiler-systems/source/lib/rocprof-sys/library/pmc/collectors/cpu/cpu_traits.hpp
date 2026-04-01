// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/collectors/cpu/device.hpp"
#include "library/pmc/collectors/cpu/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <vector>

namespace rocprofsys::pmc::collectors::cpu
{

using ::rocprofsys::pmc::device_filter;
using ::rocprofsys::pmc::device_selection_mode;

/**
 * @brief Traits type for CPU collector configuration.
 *
 * Unlike GPU which manages multiple device instances, CPU manages a single
 * device that monitors multiple CPU cores. The device filter applies to which
 * cores are monitored, not which device instances are created.
 *
 * @tparam DriverProvider The provider type (wraps procfs driver).
 */
template <typename DriverProvider>
struct cpu_traits
{
    using metrics_t         = cpu::metrics;
    using enabled_metrics_t = cpu::enabled_metrics;
    using device_t          = device<typename DriverProvider::driver_t>;
    using device_ptr_t      = std::shared_ptr<device_t>;
    using container_t       = std::vector<device_ptr_t>;
    using driver_t          = typename DriverProvider::driver_t;

    static constexpr const char* device_name = "CPU";

    struct device_entry
    {
        device_ptr_t      device;
        enabled_metrics_t supported_metrics;
    };

    template <typename Settings>
    [[nodiscard]] static device_filter get_device_filter()
    {
        return Settings::get_cpu_device_filter();
    }

    template <typename Settings>
    [[nodiscard]] static enabled_metrics_t get_enabled_metrics()
    {
        return Settings::get_cpu_enabled_metrics();
    }

    template <typename Cache>
    static void init_pmc_metadata(const device_ptr_t& dev)
    {
        Cache::initialize_pmc_metadata(dev->get_monitored_cpus());
    }

    template <typename Perfetto, typename DeviceVector>
    static void init_perfetto_storage(const DeviceVector& /*device_entries*/)
    {
        Perfetto::init_storage();
    }

    template <typename Perfetto>
    static void setup_counter_tracks(const device_ptr_t&      dev,
                                     const enabled_metrics_t& enabled)
    {
        Perfetto::setup_counter_tracks(dev->get_monitored_cpus(), enabled);
    }

    template <typename Perfetto, typename DeviceEntries>
    static void post_process_perfetto(const DeviceEntries&     entries,
                                      const enabled_metrics_t& enabled)
    {
        if(!entries.empty())
        {
            Perfetto::post_process(entries[0].device->get_monitored_cpus(), enabled);
        }
    }

    [[nodiscard]] static metrics_t get_metrics(const device_ptr_t&      dev,
                                               const enabled_metrics_t& enabled,
                                               uint64_t                 timestamp)
    {
        return dev->get_cpu_metrics(enabled, timestamp);
    }

    /**
     * @brief Enumerate CPU device (single logical device with multiple cores).
     *
     * Creates a single device that monitors cores based on the filter.
     * The provider's get_driver() and get_cpu_count() determine available cores.
     */
    template <typename Settings, typename Provider>
    [[nodiscard]] static std::vector<device_entry> enumerate_devices(
        std::shared_ptr<Provider> provider)
    {
        std::vector<device_entry> entries;
        auto                      filter = get_device_filter<Settings>();

        if(filter.mode == device_selection_mode::NONE)
        {
            LOG_DEBUG("{} sampling disabled via configuration", device_name);
            return entries;
        }

        auto cpu_count = provider->get_cpu_count();
        LOG_INFO("Detected {} online CPUs for PMC sampling", cpu_count);

        std::set<size_t> monitored_cpus;
        switch(filter.mode)
        {
            case device_selection_mode::ALL:
                for(size_t i = 0; i < cpu_count; ++i)
                    monitored_cpus.insert(i);
                break;
            case device_selection_mode::NONE: return entries;
            case device_selection_mode::SPECIFIC:
                for(auto idx : filter.indices)
                {
                    if(idx < cpu_count)
                    {
                        monitored_cpus.insert(idx);
                    }
                    else
                    {
                        LOG_WARNING("CPU index {} out of range (max: {})", idx,
                                    cpu_count - 1);
                    }
                }
                break;
        }

        if(monitored_cpus.empty())
        {
            LOG_WARNING("No CPUs selected for monitoring");
            return entries;
        }

        auto drv = provider->get_driver();
        auto dev = std::make_shared<device_t>(std::move(drv), monitored_cpus);

        if(!dev->is_supported())
        {
            LOG_WARNING("No CPU metrics are supported on this system");
        }

        auto supported = dev->get_supported_metrics();
        entries.push_back(device_entry{ std::move(dev), supported });
        LOG_INFO("Enabled {} CPUs for PMC sampling", monitored_cpus.size());

        return entries;
    }

    static std::vector<uint8_t> serialize_frequencies(const metrics_t& m)
    {
        constexpr size_t idx_size   = sizeof(size_t);
        constexpr size_t value_size = sizeof(float);

        std::vector<uint8_t> result;
        result.resize(m.cpu_data.size() * (idx_size + value_size));

        size_t offset = 0;
        for(const auto& cpu : m.cpu_data)
        {
            std::memcpy(result.data() + offset, &cpu.cpu_id, idx_size);
            offset += idx_size;
            std::memcpy(result.data() + offset, &cpu.frequency, value_size);
            offset += value_size;
        }
        return result;
    }

    static std::vector<uint8_t> serialize_loads(const metrics_t& m)
    {
        constexpr size_t idx_size   = sizeof(size_t);
        constexpr size_t value_size = sizeof(double);

        std::vector<uint8_t> result;
        result.resize(m.cpu_data.size() * (idx_size + value_size));

        size_t offset = 0;
        for(const auto& cpu : m.cpu_data)
        {
            std::memcpy(result.data() + offset, &cpu.cpu_id, idx_size);
            offset += idx_size;
            std::memcpy(result.data() + offset, &cpu.load, value_size);
            offset += value_size;
        }
        return result;
    }
};

}  // namespace rocprofsys::pmc::collectors::cpu
