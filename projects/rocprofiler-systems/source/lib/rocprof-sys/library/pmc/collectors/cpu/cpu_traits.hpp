// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/config.hpp"
#include "library/pmc/collectors/cpu/device.hpp"
#include "library/pmc/collectors/cpu/sample.hpp"
#include "library/pmc/collectors/cpu/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <cstdint>
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
 * Each CPU socket (physical package) is modeled as a separate device, aligned
 * with the GPU pattern where each GPU is a separate device. The device filter
 * selects socket IDs (not individual cores). All cores on a selected socket
 * are always monitored.
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
        return Settings::get_device_filter(rocprofsys::get_sampling_cpus());
    }

    template <typename Settings>
    [[nodiscard]] static enabled_metrics_t get_enabled_metrics()
    {
        return Settings::get_cpu_enabled_metrics();
    }

    template <typename Cache>
    static void init_pmc_metadata(const device_ptr_t& dev)
    {
        static bool first_socket_registered = false;
        const bool  is_first                = !first_socket_registered;
        first_socket_registered             = true;
        Cache::initialize_pmc_metadata(dev->get_index(), dev->get_monitored_cpus(),
                                       is_first);
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
        Perfetto::setup_counter_tracks(dev->get_index(), dev->get_monitored_cpus(),
                                       enabled);
    }

    template <typename Perfetto, typename DeviceEntries>
    static void post_process_perfetto(const DeviceEntries&     entries,
                                      const enabled_metrics_t& enabled)
    {
        for(const auto& entry : entries)
        {
            Perfetto::post_process(entry.device->get_index(),
                                   entry.device->get_monitored_cpus(), enabled);
        }
    }

    [[nodiscard]] static metrics_t get_metrics(const device_ptr_t&       dev,
                                               const enabled_metrics_t&  enabled,
                                               [[maybe_unused]] uint64_t timestamp)
    {
        return dev->get_cpu_metrics(enabled);
    }

    /**
     * @brief Enumerate CPU devices — one per socket (physical package).
     *
     * The provider constructs devices from socket topology. This method
     * applies the device filter and collects supported metrics, matching
     * the GPU traits pattern.
     */
    template <typename Settings, typename Provider>
    [[nodiscard]] static std::vector<device_entry> enumerate_devices(
        std::shared_ptr<Provider> provider)
    {
        std::vector<device_entry> entries;
        const auto                filter = get_device_filter<Settings>();

        if(filter.mode == device_selection_mode::NONE)
        {
            LOG_DEBUG("{} sampling disabled via configuration", device_name);
            return entries;
        }

        auto devices = provider->template get_devices<device_t>();

        for(auto& dev : devices)
        {
            const auto index = dev->get_index();

            const bool should_include = (filter.mode == device_selection_mode::ALL) ||
                                        (filter.mode == device_selection_mode::SPECIFIC &&
                                         filter.indices.count(index) > 0);

            if(should_include)
            {
                if(!dev->is_supported())
                {
                    LOG_WARNING("No CPU metrics supported on socket {}", index);
                }
                const auto supported = dev->get_supported_metrics();
                entries.push_back(device_entry{ std::move(dev), supported });
            }
        }

        warn_invalid_indices(filter, devices.size());
        LOG_INFO("Enabled {} CPU socket(s) for PMC sampling", entries.size());
        return entries;
    }

private:
    static void warn_invalid_indices(const device_filter& filter, size_t max_index)
    {
        if(filter.mode != device_selection_mode::SPECIFIC) return;
        for(const auto idx : filter.indices)
        {
            if(idx >= max_index)
            {
                LOG_WARNING("Requested CPU socket {} does not exist. "
                            "Available sockets: 0-{}",
                            idx, max_index > 0 ? max_index - 1 : 0);
            }
        }
    }
};

}  // namespace rocprofsys::pmc::collectors::cpu
