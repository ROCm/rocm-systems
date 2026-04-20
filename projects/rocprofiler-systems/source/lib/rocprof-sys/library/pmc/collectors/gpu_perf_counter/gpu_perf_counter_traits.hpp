// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/gpu_perf_counter/device.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

using ::rocprofsys::pmc::device_filter;
using ::rocprofsys::pmc::device_selection_mode;
using ::rocprofsys::pmc::device_type;

/**
 * @brief Traits type for SDK PMC collector configuration.
 *
 * Defines types, constants, and customization points for the base collector template
 * to work with rocprofiler-sdk device_counting_service.
 *
 * @tparam DriverProvider The device provider type (rocprofiler_sdk::provider).
 */
template <typename DriverProvider>
struct gpu_perf_counter_traits
{
    // Required type aliases for base::collector
    using metrics_t         = pmc::collectors::gpu_perf_counter::metrics;
    using enabled_metrics_t = pmc::collectors::gpu_perf_counter::enabled_metrics;
    using device_t          = device<typename DriverProvider::driver_t>;
    using device_ptr_t      = std::shared_ptr<device_t>;
    using container_t       = std::vector<device_ptr_t>;
    using driver_t          = typename DriverProvider::driver_t;

    static constexpr const char* device_name = "GPU perf counter";

    /**
     * @brief Get the device filter from settings.
     */
    template <typename Settings>
    [[nodiscard]] static device_filter get_device_filter()
    {
        return Settings::get_gpu_perf_counter_device_filter();
    }

    /**
     * @brief Get enabled metrics from settings.
     */
    template <typename Settings>
    [[nodiscard]] static enabled_metrics_t get_enabled_metrics()
    {
        return Settings::get_gpu_perf_counter_enabled_metrics();
    }

    /**
     * @brief Initialize PMC metadata for a specific device.
     *
     * Registers per-device tracks and PMC info for each qualified counter.
     * Passes the device's qualified counter names (with dimension info) and
     * per-counter metadata (block, expression, is_constant, is_derived) to the
     * cache policy for pmc_info registration with real SDK counter properties.
     */
    template <typename Cache>
    static void init_pmc_metadata(const device_ptr_t& dev)
    {
        Cache::initialize_device_tracks(dev->get_index(), dev->get_qualified_names());
        Cache::initialize_pmc_metadata(dev->get_index(), dev->get_qualified_names(),
                                       dev->get_counter_metadata());
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
     *
     * SDK PMC tracks are created lazily during store_sample, so this is a no-op.
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
    [[nodiscard]] static metrics_t get_metrics(const device_ptr_t&      dev,
                                               const enabled_metrics_t& enabled,
                                               uint64_t                 timestamp)
    {
        return dev->get_gpu_perf_counter_metrics(enabled, timestamp);
    }

    /**
     * @brief Entry holding a device and its cached supported metrics.
     */
    struct device_entry
    {
        device_ptr_t      device;
        enabled_metrics_t supported_metrics;
    };

    /**
     * @brief Enumerate SDK PMC devices from the provider.
     *
     * Creates device objects from the provider's bridge-sourced agent list.
     * Applies device filter from settings.
     *
     * @tparam Settings Settings API type for device filter configuration.
     * @tparam Provider Device provider type.
     * @param provider Shared pointer to the device provider.
     * @return Vector of device entries with cached supported metrics.
     */
    template <typename Settings, typename Provider>
    [[nodiscard]] static std::vector<device_entry> enumerate_devices(
        std::shared_ptr<Provider> provider)
    {
        std::vector<device_entry> entries;
        auto                      filter = get_device_filter<Settings>();

        if(filter.mode == device_selection_mode::NONE)
        {
            LOG_DEBUG("SDK PMC sampling disabled via configuration");
            return entries;
        }

        auto devices = provider->template get_devices<device_t>(device_type::GPU);

        for(auto& dev : devices)
        {
            auto index = dev->get_index();

            bool should_include = (filter.mode == device_selection_mode::ALL) ||
                                  (filter.mode == device_selection_mode::SPECIFIC &&
                                   filter.indices.count(index) > 0);

            if(should_include && dev->is_supported())
            {
                auto supported = dev->get_supported_metrics();
                entries.push_back(device_entry{ std::move(dev), supported });
            }
        }

        return entries;
    }
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
