// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/collectors/gpu/device.hpp"
#include "library/pmc/collectors/gpu/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu
{

using ::rocprofsys::pmc::device_filter;
using ::rocprofsys::pmc::device_selection_mode;
using ::rocprofsys::pmc::device_type;

/**
 * @brief Traits type for GPU collector configuration.
 *
 * Defines types, constants, and customization points for the base collector template
 * to work with GPU devices via AMD SMI.
 *
 * @tparam Driver The AMD SMI driver type (real or mock for testing)
 */
template <typename DriverProvider>
struct gpu_traits
{
    // Required type aliases for base::collector
    using metrics_t         = pmc::collectors::gpu::metrics;
    using enabled_metrics_t = pmc::collectors::gpu::enabled_metrics;
    using device_t          = device<typename DriverProvider::driver_t>;
    using device_ptr_t      = std::shared_ptr<device_t>;
    using container_t       = std::vector<device_ptr_t>;
    using driver_t          = typename DriverProvider::driver_t;

    // Required constants
    static constexpr const char* device_name = "GPU";
    // Settings customization points

    /**
     * @brief Get the device filter from settings.
     */
    template <typename Settings>
    [[nodiscard]] static device_filter get_device_filter()
    {
        return Settings::get_device_filter();
    }

    /**
     * @brief Get enabled metrics from settings.
     */
    template <typename Settings>
    [[nodiscard]] static enabled_metrics_t get_enabled_metrics()
    {
        return Settings::get_enabled_metrics();
    }

    // Cache API customization points

    /**
     * @brief Initialize PMC metadata for a specific device.
     */
    template <typename Cache>
    static void init_pmc_metadata(const device_ptr_t& device)
    {
        Cache::initialize_pmc_metadata(device->get_index());
    }

    /**
     * @brief Initialize Perfetto storage for devices.
     */
    template <typename Perfetto, typename DeviceVector>
    static void init_perfetto_storage(const DeviceVector& devices)
    {
        Perfetto::init_storage(devices);
    }

    /**
     * @brief Setup Perfetto counter tracks for a device.
     */
    template <typename Perfetto>
    static void setup_counter_tracks(const device_ptr_t&      device,
                                     const enabled_metrics_t& enabled)
    {
        Perfetto::setup_counter_tracks(device->get_index(), enabled);
    }

    /**
     * @brief Post-process Perfetto data.
     */
    template <typename Perfetto, typename DeviceEntries>
    static void post_process_perfetto(const DeviceEntries& /*device_entries*/,
                                      const enabled_metrics_t& enabled)
    {
        Perfetto::post_process(enabled);
    }

    /**
     * @brief Get metrics from a device.
     */
    [[nodiscard]] static metrics_t get_metrics(const device_ptr_t&      device,
                                               const enabled_metrics_t& enabled,
                                               uint64_t                 timestamp)
    {
        return device->get_gpu_metrics(enabled, timestamp);
    }

    // Device enumeration

    /**
     * @brief Entry holding a device and its cached supported metrics.
     *
     * This type is returned by enumerate_devices for the base collector to store.
     */
    struct device_entry
    {
        device_ptr_t      device;
        enabled_metrics_t supported_metrics;
    };

    /**
     * @brief Enumerate GPU devices using AMD SMI socket/processor iteration.
     *
     * This function implements GPU-specific enumeration:
     * - Gets device filter from settings
     * - Iterates through sockets and processors
     * - Filters by processor type (AMD GPU)
     * - Applies device filter (ALL, NONE, SPECIFIC indices)
     * - Creates device objects and queries supported metrics
     *
     * @tparam Settings Settings API type for device filter configuration
     * @tparam Provider Device provider type
     * @param provider Shared pointer to the device provider
     * @return Vector of device entries with cached supported metrics
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

        auto devices = provider->template get_devices<device_t>(device_type::GPU);

        for(auto& device : devices)
        {
            auto index = device->get_index();

            bool should_include = (filter.mode == device_selection_mode::ALL) ||
                                  (filter.mode == device_selection_mode::SPECIFIC &&
                                   filter.indices.count(index) > 0);

            if(should_include && device->is_supported())
            {
                auto supported = device->get_supported_metrics();
                entries.push_back(device_entry{ std::move(device), supported });
            }
        }

        warn_invalid_indices(filter, devices.size());
        return entries;
    }

    /**
     * @brief Warn about invalid device indices specified by the user.
     *
     * @param filter Device filter with requested indices
     * @param max_index Maximum valid device index + 1
     */
    static void warn_invalid_indices(const device_filter& filter, size_t max_index)
    {
        if(filter.mode != device_selection_mode::SPECIFIC)
        {
            return;
        }
        for(auto requested_index : filter.indices)
        {
            if(requested_index >= max_index)
            {
                LOG_WARNING("Requested GPU device index {} does not exist. "
                            "Available devices: 0-{}",
                            requested_index, max_index > 0 ? max_index - 1 : 0);
            }
        }
    }
};

}  // namespace rocprofsys::pmc::collectors::gpu
