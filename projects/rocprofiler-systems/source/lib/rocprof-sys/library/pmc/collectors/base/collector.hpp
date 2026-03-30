// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/collectors/base/traits_check.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace rocprofsys::pmc::collectors::base
{

/**
 * @brief Generic collector template for device performance monitoring.
 *
 * This collector provides a unified implementation for GPU, NIC, and CPU metrics
 * collection. Device-specific behavior is configured via the Traits template parameter.
 *
 * @tparam Traits Device-specific traits defining types and customization points
 * @tparam DeviceProvider Type providing device enumeration and management
 * @tparam Config Configuration policy providing settings and output policies
 */
template <typename Traits, typename DeviceProvider, typename Config>
struct collector
{
    // Validate traits at compile time
    static_assert(has_required_types_v<Traits>,
                  "Invalid traits: missing required type aliases");
    static_assert(has_device_name_v<Traits>, "Traits must define: device_name");
    static_assert(has_enumerate_devices_v<Traits>,
                  "Traits must define: enumerate_devices() and device_entry type");

    // Type aliases from traits
    using traits_t          = Traits;
    using metrics_t         = typename Traits::metrics_t;
    using enabled_metrics_t = typename Traits::enabled_metrics_t;
    using device_t          = typename Traits::device_t;
    using device_ptr_t      = typename Traits::device_ptr_t;
    using container_t       = typename Traits::container_t;
    using driver_t          = typename Traits::driver_t;

    // Type aliases from config
    using device_provider = DeviceProvider;
    using SettingsApi     = typename Config::SettingsApi;
    using PerfettoApi     = typename Config::PerfettoApi;
    using CacheApi        = typename Config::CacheApi;

    // Device entry type from traits (contains device + cached supported metrics)
    using device_entry     = typename Traits::device_entry;
    using device_entries_t = std::vector<device_entry>;

    /**
     * @brief Construct a collector with an injected device provider.
     *
     * @param provider Shared pointer to the device provider instance
     */
    explicit collector(std::shared_ptr<device_provider> provider)
    : m_device_provider(std::move(provider))
    {}

    collector() = delete;

    /**
     * @brief Initialize the collector and enumerate devices.
     *
     * Retrieves version information (for GPU), enumerates devices based on filter
     * settings, and initializes Perfetto storage if legacy metrics are enabled.
     *
     * @throws std::runtime_error If device provider is not set.
     */
    void setup()
    {
        if(!m_device_provider)
        {
            throw std::runtime_error(
                "Device provider not set. Use constructor or set_device_provider().");
        }

        m_device_entries =
            Traits::template enumerate_devices<SettingsApi>(m_device_provider);
        m_enabled_metrics = Traits::template get_enabled_metrics<SettingsApi>();

        LOG_INFO("Enabled {} {} devices for PMC sampling", m_device_entries.size(),
                 Traits::device_name);

        if(SettingsApi::get_use_perfetto_legacy_metrics())
        {
            Traits::template init_perfetto_storage<PerfettoApi>(m_device_entries);
        }
    }

    /**
     * @brief Configure metrics tracking and initialize metadata.
     *
     * Sets up category metadata, Perfetto counter tracks, and PMC tracks/metadata
     * for all enabled devices.
     */
    void config()
    {
        CacheApi::initialize_category_metadata();
        CacheApi::initialize_tracks_metadata();

        for(const auto& entry : m_device_entries)
        {
            if(SettingsApi::get_use_perfetto_legacy_metrics())
            {
                Traits::template setup_counter_tracks<PerfettoApi>(entry.device,
                                                                   m_enabled_metrics);
            }
            Traits::template init_pmc_metadata<CacheApi>(entry.device);
        }
    }

    /**
     * @brief Sample metrics from all enabled devices.
     *
     * Iterates through all devices, retrieves current metrics, and stores them
     * via the cache API and optionally Perfetto. Devices that fail to read metrics
     * are automatically disabled and removed from the device list.
     *
     * @param timestamp Current timestamp in nanoseconds for the sample.
     */
    void sample(int64_t timestamp)
    {
        auto new_end = std::remove_if(
            m_device_entries.begin(), m_device_entries.end(),
            [this, timestamp](const device_entry& entry) {
                auto _timestamp = static_cast<uint64_t>(timestamp);

                try
                {
                    auto _metrics =
                        Traits::get_metrics(entry.device, m_enabled_metrics, _timestamp);
                    auto _device_id   = entry.device->get_index();
                    auto _device_name = entry.device->get_name();

                    CacheApi::store_sample(_device_id, _device_name, m_enabled_metrics,
                                           entry.supported_metrics, _metrics, _timestamp);

                    if(SettingsApi::get_use_perfetto_legacy_metrics())
                    {
                        PerfettoApi::store_sample(_device_id, _metrics, _timestamp);
                    }
                    return false;  // Keep device
                } catch(const std::runtime_error& e)
                {
                    LOG_ERROR("Reading metrics failed for {} device {}. Error: {}. "
                              "Disabling device!",
                              Traits::device_name, entry.device->get_index(), e.what());
                    return true;  // Remove device
                }
            });
        m_device_entries.erase(new_end, m_device_entries.end());
    }

    /**
     * @brief Perform post-processing of collected metrics.
     *
     * Triggers Perfetto post-processing if legacy metrics mode is enabled.
     */
    void post_process()
    {
        if(SettingsApi::get_use_perfetto_legacy_metrics())
        {
            Traits::template post_process_perfetto<PerfettoApi>(m_device_entries,
                                                                m_enabled_metrics);
        }
    }

    /**
     * @brief Get the device entries (devices with cached supported metrics).
     * @return Const reference to the vector of device entries.
     */
    const device_entries_t& get_device_entries() const noexcept
    {
        return m_device_entries;
    }

    /**
     * @brief Get the number of enabled devices.
     * @return Number of devices currently enabled for sampling.
     */
    size_t get_device_count() const noexcept { return m_device_entries.size(); }

    /**
     * @brief Set the device provider (for backward compatibility).
     *
     * @param provider Shared pointer to the device provider instance
     */
    void set_device_provider(std::shared_ptr<device_provider> provider)
    {
        m_device_provider = std::move(provider);
    }

    /**
     * @brief Shutdown the device provider and release resources.
     *
     * @note This method does NOT clear m_device_entries because post_process()
     * may be called after shutdown() and needs access to device information.
     * The device entries are cleared in pmc::post_process() after post_process
     * is called on all collectors.
     */
    void shutdown()
    {
        if(m_device_provider)
        {
            m_device_provider->shutdown();
            m_device_provider.reset();
        }
    }

private:
    device_entries_t m_device_entries;  ///< Devices with cached supported metrics
    std::shared_ptr<device_provider> m_device_provider;  ///< Device provider instance
    enabled_metrics_t                m_enabled_metrics;  ///< Enabled metrics
};

}  // namespace rocprofsys::pmc::collectors::base
