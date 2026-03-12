// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/collectors/gpu/device.hpp"
#include "library/pmc/collectors/gpu/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <memory>
#include <vector>

#include <amd_smi/amdsmi.h>

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace gpu
{

using ::rocprofsys::pmc::device_filter;
using ::rocprofsys::pmc::device_selection_mode;

/**
 * @brief Traits type for GPU collector configuration.
 *
 * Defines types, constants, and customization points for the base collector template
 * to work with GPU devices via AMD SMI.
 *
 * @tparam Driver The AMD SMI driver type (real or mock for testing)
 */
template <typename Driver>
struct gpu_traits
{
    // Required type aliases for base::collector
    using metrics_t         = pmc::collectors::gpu::metrics;
    using enabled_metrics_t = pmc::collectors::gpu::enabled_metrics;
    using device_t          = device<Driver>;
    using device_ptr_t      = std::shared_ptr<device_t>;
    using container_t       = std::vector<device_ptr_t>;
    using driver_t          = Driver;

    // Required constants
    static constexpr const char* device_name  = "GPU";
    static constexpr bool        multi_device = true;

    // Settings customization points

    /**
     * @brief Get the device filter from settings.
     */
    template <typename Settings>
    static device_filter get_device_filter()
    {
        return Settings::get_device_filter();
    }

    /**
     * @brief Get enabled metrics from settings.
     */
    template <typename Settings>
    static enabled_metrics_t get_enabled_metrics()
    {
        return Settings::get_enabled_metrics();
    }

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
     * @brief Initialize SMI tracks metadata in the cache.
     */
    template <typename Cache>
    static void init_tracks_metadata()
    {
        Cache::initialize_smi_tracks_metadata();
    }

    /**
     * @brief Initialize PMC metadata for a specific device.
     */
    template <typename Cache>
    static void init_pmc_metadata(size_t device_index)
    {
        Cache::initialize_smi_pmc_metadata(device_index);
    }

    /**
     * @brief Store a sample to the cache.
     */
    template <typename Cache>
    static void store_sample(size_t device_id, const enabled_metrics_t& enabled,
                             const enabled_metrics_t& supported, const metrics_t& metrics,
                             uint64_t timestamp)
    {
        Cache::store_sample(device_id, enabled, supported, metrics, timestamp);
    }

    // Perfetto API delegation

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
    static void setup_counter_tracks(size_t                   device_index,
                                     const enabled_metrics_t& enabled)
    {
        Perfetto::setup_counter_tracks(device_index, enabled);
    }

    /**
     * @brief Store a sample to Perfetto.
     */
    template <typename Perfetto>
    static void store_perfetto_sample(size_t device_id, const metrics_t& metrics,
                                      uint64_t timestamp)
    {
        Perfetto::store_sample(device_id, metrics, timestamp);
    }

    /**
     * @brief Post-process Perfetto data.
     */
    template <typename Perfetto>
    static void post_process_perfetto(const enabled_metrics_t& enabled)
    {
        Perfetto::post_process(enabled);
    }

    // Device creation

    /**
     * @brief Create a new GPU device instance.
     */
    static device_ptr_t create_device(std::shared_ptr<driver_t> driver,
                                      amdsmi_processor_handle   handle,
                                      processor_type_t type, size_t index)
    {
        return std::make_shared<device_t>(std::move(driver), handle, type, index);
    }

    /**
     * @brief Get the device index from a device pointer.
     */
    static size_t get_device_index(const device_ptr_t& device)
    {
        return device->get_index();
    }

    /**
     * @brief Get supported metrics from a device.
     */
    static enabled_metrics_t get_supported_metrics(const device_ptr_t& device)
    {
        return device->get_supported_metrics();
    }

    /**
     * @brief Get metrics from a device.
     */
    static metrics_t get_metrics(const device_ptr_t&      device,
                                 const enabled_metrics_t& enabled)
    {
        return device->get_gpu_metrics(enabled);
    }

    /**
     * @brief Check if a device is supported.
     */
    static bool is_device_supported(const device_ptr_t& device)
    {
        return device->is_supported();
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
    static std::vector<device_entry> enumerate_devices(std::shared_ptr<Provider> provider)
    {
        std::vector<device_entry> entries;
        auto                      filter = get_device_filter<Settings>();

        if(filter.mode == device_selection_mode::NONE)
        {
            LOG_DEBUG("{} sampling disabled via configuration", device_name);
            return entries;
        }

        auto   driver         = provider->get_driver();
        auto   socket_handles = provider->get_socket_handles();
        size_t index          = 0;

        auto process_handle = [&](amdsmi_processor_handle processor_handle) {
            processor_type_t processor_type;
            auto status = driver->get_processor_type(processor_handle, &processor_type);

            if(status != AMDSMI_STATUS_SUCCESS)
            {
                LOG_DEBUG("Failed to get processor type for handle at index {}", index);
                return;
            }

            if(processor_type != AMDSMI_PROCESSOR_TYPE_AMD_GPU) return;

            bool should_include = (filter.mode == device_selection_mode::ALL) ||
                                  (filter.mode == device_selection_mode::SPECIFIC &&
                                   filter.indices.count(index) > 0);

            if(!should_include) return;

            auto device = create_device(driver, processor_handle,
                                        AMDSMI_PROCESSOR_TYPE_AMD_GPU, index);

            if(is_device_supported(device))
            {
                auto supported = get_supported_metrics(device);
                entries.push_back(device_entry{ std::move(device), supported });
            }
        };

        for(auto& socket_handle : socket_handles)
        {
            for(auto& processor_handle : provider->get_processor_handles(socket_handle))
            {
                process_handle(processor_handle);
                index++;
            }
        }

        warn_invalid_indices(filter, index);
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
        if(filter.mode == device_selection_mode::SPECIFIC)
        {
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
    }
};

}  // namespace gpu
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
