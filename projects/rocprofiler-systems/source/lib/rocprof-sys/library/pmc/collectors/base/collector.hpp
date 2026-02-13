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

#include "library/pmc/collectors/base/traits_check.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace base
{

#if ROCPROFSYS_USE_ROCM > 0

using ::rocprofsys::pmc::device_filter;
using ::rocprofsys::pmc::device_selection_mode;

using get_timestamp_t = std::function<unsigned long()>;

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
    static_assert(has_multi_device_v<Traits>, "Traits must define: multi_device");

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

    /**
     * @brief Entry holding a device and its cached supported metrics.
     *
     * Supported metrics are queried once during device enumeration and cached here
     * to avoid repeated calls in the hot sampling path.
     */
    struct device_entry
    {
        device_ptr_t      device;
        enabled_metrics_t supported_metrics;
    };

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

        log_version_info();
        enumerate_devices();

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
        Traits::template init_category_metadata<CacheApi>();

        for(const auto& entry : m_device_entries)
        {
            auto device_index = Traits::get_device_index(entry.device);
            if(SettingsApi::get_use_perfetto_legacy_metrics())
            {
                Traits::template setup_counter_tracks<PerfettoApi>(device_index,
                                                                   m_enabled_metrics);
            }
            Traits::template init_tracks_metadata<CacheApi>();
            Traits::template init_pmc_metadata<CacheApi>(device_index);
        }
    }

    /**
     * @brief Sample metrics from all enabled devices.
     *
     * Iterates through all devices, retrieves current metrics, and stores them
     * via the cache API and optionally Perfetto. Devices that fail to read metrics
     * are automatically disabled and removed from the device list.
     *
     * @param get_timestamp Function to retrieve the current timestamp for the sample.
     */
    void sample(const get_timestamp_t& get_timestamp)
    {
        auto new_end = std::remove_if(
            m_device_entries.begin(), m_device_entries.end(),
            [this, &get_timestamp](const device_entry& entry) {
                auto _timestamp = get_timestamp();
                assert(_timestamp <
                       static_cast<unsigned long>(std::numeric_limits<int64_t>::max()));

                try
                {
                    auto _metrics = Traits::get_metrics(entry.device, m_enabled_metrics);
                    auto _device_id = Traits::get_device_index(entry.device);

                    Traits::template store_sample<CacheApi>(_device_id, m_enabled_metrics,
                                                            entry.supported_metrics,
                                                            _metrics, _timestamp);
                    if(SettingsApi::get_use_perfetto_legacy_metrics())
                    {
                        Traits::template store_perfetto_sample<PerfettoApi>(
                            _device_id, _metrics, _timestamp);
                    }
                    return false;  // Keep device
                } catch(const std::runtime_error& e)
                {
                    LOG_ERROR("Reading metrics failed for {} device {}. Error: {}. "
                              "Disabling device!",
                              Traits::device_name, Traits::get_device_index(entry.device),
                              e.what());
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
            Traits::template post_process_perfetto<PerfettoApi>(m_enabled_metrics);
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
     */
    void shutdown()
    {
        m_device_entries.clear();
        if(m_device_provider)
        {
            m_device_provider->shutdown();
            m_device_provider.reset();
        }
    }

private:
    /**
     * @brief Log version information if available.
     *
     * Only logs if the Traits class defines has_version_info = true.
     */
    void log_version_info()
    {
        if constexpr(traits_has_version_info_v<Traits>)
        {
            auto _version = m_device_provider->get_version();
            LOG_INFO("AMD SMI version: {}.{}.{} - str: {}.",
                     _version.numeric_representation.major,
                     _version.numeric_representation.minor,
                     _version.numeric_representation.release,
                     _version.string_representation);
        }
    }

    /**
     * @brief Enumerate devices from provider and create device objects.
     */
    void enumerate_devices()
    {
        auto filter = Traits::template get_device_filter<SettingsApi>();
        auto driver = m_device_provider->get_driver();

        if(filter.mode == device_selection_mode::NONE)
        {
            LOG_DEBUG("{} sampling disabled via configuration", Traits::device_name);
            return;
        }

        auto   socket_handles = m_device_provider->get_socket_handles();
        size_t index          = 0;

        for(auto& socket_handle : socket_handles)
        {
            auto processor_handles =
                Traits::get_processor_handles(*m_device_provider, socket_handle);

            for(auto& processor_handle : processor_handles)
            {
                if constexpr(Traits::filter_by_processor_type())
                {
                    processor_type_t processor_type;
                    auto             status =
                        driver->get_processor_type(processor_handle, &processor_type);

                    if(status != AMDSMI_STATUS_SUCCESS)
                    {
                        LOG_DEBUG("Failed to get processor type for handle at index {}",
                                  index);
                        index++;
                        continue;
                    }

                    if(processor_type != Traits::expected_processor_type())
                    {
                        index++;
                        continue;
                    }
                }

                bool should_include = false;
                switch(filter.mode)
                {
                    case device_selection_mode::ALL: should_include = true; break;
                    case device_selection_mode::NONE: should_include = false; break;
                    case device_selection_mode::SPECIFIC:
                        should_include = filter.indices.count(index) > 0;
                        break;
                }

                if(should_include)
                {
                    auto device =
                        Traits::create_device(driver, processor_handle,
                                              Traits::expected_processor_type(), index);
                    if(Traits::is_device_supported(device))
                    {
                        // Cache supported metrics at enumeration time to avoid
                        // repeated queries in the hot sampling path
                        auto supported = Traits::get_supported_metrics(device);
                        m_device_entries.push_back(
                            device_entry{ std::move(device), supported });
                    }
                }

                index++;
            }
        }

        warn_invalid_indices(filter, index);
    }

    /**
     * @brief Warn about invalid device indices specified by the user.
     */
    void warn_invalid_indices(const device_filter& filter, size_t max_index)
    {
        if(filter.mode == device_selection_mode::SPECIFIC)
        {
            for(auto requested_index : filter.indices)
            {
                if(requested_index >= max_index)
                {
                    LOG_WARNING("Requested {} device index {} does not exist. Available "
                                "devices: 0-{}",
                                Traits::device_name, requested_index,
                                max_index > 0 ? max_index - 1 : 0);
                }
            }
        }
    }

    device_entries_t m_device_entries;  ///< Devices with cached supported metrics
    std::shared_ptr<device_provider> m_device_provider;  ///< Device provider instance
    enabled_metrics_t                m_enabled_metrics;  ///< Enabled metrics
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace base
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
