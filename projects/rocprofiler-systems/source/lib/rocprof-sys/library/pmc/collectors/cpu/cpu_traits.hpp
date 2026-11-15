// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/collectors/cpu/device.hpp"
#include "library/pmc/collectors/cpu/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <memory>
#include <set>
#include <vector>

namespace rocprofsys::pmc::collectors::cpu
{

using ::rocprofsys::pmc::device_filter;
using ::rocprofsys::pmc::device_selection_mode;

/**
 * @brief Traits type for CPU device enumeration and settings.
 *
 * Each CPU socket (physical package) is modeled as a separate device, aligned
 * with the GPU pattern where each GPU is a separate device. The device filter
 * selects socket IDs (not individual cores). All cores on a selected socket
 * are always monitored.
 *
 * @tparam BackendProvider The provider type (wraps procfs backend).
 */
template <typename BackendProvider>
struct cpu_traits
{
    using enabled_metrics_t = cpu::enabled_metrics;
    using device_t          = device<typename BackendProvider::backend_t>;
    using device_ptr_t      = std::shared_ptr<device_t>;

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

        if(filter.mode == device_selection_mode::none)
        {
            LOG_DEBUG("{} sampling disabled via configuration", device_name);
            return entries;
        }

        auto devices = provider->template get_devices<device_t>();

        for(auto& dev : devices)
        {
            const auto index = dev->get_index();

            const bool should_include = (filter.mode == device_selection_mode::all) ||
                                        (filter.mode == device_selection_mode::specific &&
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
        if(filter.mode != device_selection_mode::specific)
        {
            return;
        }
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
