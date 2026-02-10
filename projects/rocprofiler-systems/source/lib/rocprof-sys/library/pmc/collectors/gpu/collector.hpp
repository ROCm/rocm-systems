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

#include "library/pmc/collectors/gpu/device.hpp"
#include "library/pmc/gpu/types.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>
#include <memory>

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace gpu
{

#if ROCPROFSYS_USE_ROCM > 0

using ::rocprofsys::pmc::device_filter;
using ::rocprofsys::pmc::device_selection_mode;
using ::rocprofsys::pmc::gpu::check_status;
using ::rocprofsys::pmc::gpu::enabled_metrics;

using get_timestamp_t = std::function<unsigned long()>;

/**
 * @brief GPU metrics collector for performance monitoring.
 *
 * This collector manages the lifecycle of GPU performance monitoring, including
 * device enumeration, metric sampling, and data storage. It uses a policy-based design
 * pattern via template parameters to allow compile-time dependency injection.
 *
 * @tparam DeviceProvider Type providing GPU device enumeration and management
 * @tparam Config Configuration policy providing settings and output policies (Perfetto,
 * RocPD)
 */
template <typename DeviceProvider, typename Config>
struct collector
{
    using device_provider = DeviceProvider;
    using SettingsApi     = typename Config::SettingsApi;
    using PerfettoApi     = typename Config::PerfettoApi;
    using CacheApi        = typename Config::CacheApi;
    using driver_t        = typename device_provider::driver_t;
    using device_t        = device<driver_t>;
    using device_ptr_t    = std::shared_ptr<device_t>;
    using device_vector_t = std::vector<device_ptr_t>;

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
     * @brief Initialize the collector and enumerate GPU devices.
     *
     * Retrieves version information, enumerates GPU devices based on device filter
     * settings, and initializes Perfetto storage if legacy metrics are enabled.
     *
     * @note Device provider must be set (via constructor or manually) before calling
     * setup.
     * @throws std::runtime_error If device provider is not set.
     */
    void setup()
    {
        if(!m_device_provider)
        {
            throw std::runtime_error(
                "Device provider not set. Use constructor or set_device_provider().");
        }

        auto _version = m_device_provider->get_version();

        LOG_INFO("AMD SMI version: {}.{}.{} - str: {}.",
                 _version.numeric_representation.major,
                 _version.numeric_representation.minor,
                 _version.numeric_representation.release, _version.string_representation);

        // Enumerate devices from raw AMD SMI handles
        enumerate_devices();

        m_enabled_metrics = SettingsApi::get_enabled_metrics();

        LOG_INFO("Enabled {} GPU devices for PMC sampling", m_gpu_devices.size());

        if(SettingsApi::get_use_perfetto_legacy_metrics())
        {
            PerfettoApi::init_storage(m_gpu_devices);
        }
    }

    /**
     * @brief Configure metrics tracking and initialize metadata.
     *
     * Sets up category metadata, Perfetto counter tracks, and PMC tracks/metadata
     * for all enabled GPU devices.
     */
    void config()
    {
        CacheApi::initialize_category_metadata();

        for(const auto& device : m_gpu_devices)
        {
            auto device_index = device->get_index();
            if(SettingsApi::get_use_perfetto_legacy_metrics())
            {
                PerfettoApi::setup_counter_tracks(device_index, m_enabled_metrics);
            }
            CacheApi::initialize_smi_tracks_metadata();
            CacheApi::initialize_smi_pmc_metadata(device_index);
        }
    }

    /**
     * @brief Sample GPU metrics from all enabled devices.
     *
     * Iterates through all GPU devices, retrieves current metrics, and stores them
     * via the cache API and optionally Perfetto. Devices that fail to read metrics
     * are automatically disabled and removed from the device list.
     *
     * @param get_timestamp Function to retrieve the current timestamp for the sample.
     */
    void sample(const get_timestamp_t& get_timestamp)
    {
        auto new_end = std::remove_if(
            m_gpu_devices.begin(), m_gpu_devices.end(),
            [this, &get_timestamp](const device_ptr_t& device) {
                auto _timestamp = get_timestamp();
                assert(_timestamp <
                       static_cast<unsigned long>(std::numeric_limits<int64_t>::max()));

                try
                {
                    auto _supported_metrics = device->get_supported_metrics();
                    auto _gpu_metrics       = device->get_gpu_metrics(m_enabled_metrics);
                    auto _device_id         = device->get_index();

                    CacheApi::store_sample(_device_id, m_enabled_metrics,
                                           _supported_metrics, _gpu_metrics, _timestamp);
                    if(SettingsApi::get_use_perfetto_legacy_metrics())
                    {
                        PerfettoApi::store_sample(_device_id, _gpu_metrics, _timestamp);
                    }
                    return false;  // Keep device
                } catch(const std::runtime_error& e)
                {
                    LOG_ERROR("Reading metrics failed for device with ID %zu. Error: %s. "
                              "Disabling device!",
                              device->get_index(), e.what());
                    return true;  // Remove device
                }
            });
        m_gpu_devices.erase(new_end, m_gpu_devices.end());
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
            PerfettoApi::post_process(m_gpu_devices, m_enabled_metrics);
        }
    }

    /**
     * @brief Get the list of all GPU devices.
     * @return Const reference to the vector of GPU devices.
     */
    const device_vector_t& get_devices() const noexcept { return m_gpu_devices; }

    /**
     * @brief Get the number of enabled GPU devices.
     * @return Number of GPU devices currently enabled for sampling.
     */
    size_t get_device_count() const noexcept { return m_gpu_devices.size(); }

    /**
     * @brief Set the device provider (for backward compatibility with default
     * constructor).
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
     * Shuts down the device provider and resets the pointer.
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
    /**
     * @brief Enumerate devices from device provider and create device objects.
     *
     * Queries the device provider for raw AMD SMI handles, creates device objects,
     * and applies filtering based on settings.
     */
    void enumerate_devices()
    {
        auto filter = SettingsApi::get_device_filter();
        auto driver = m_device_provider->get_driver();

        auto   socket_handles = m_device_provider->get_socket_handles();
        size_t index          = 0;

        for(auto& socket_handle : socket_handles)
        {
            auto processor_handles =
                m_device_provider->get_processor_handles(socket_handle);

            for(auto& processor_handle : processor_handles)
            {
                processor_type_t processor_type;
                check_status(
                    driver->get_processor_type(processor_handle, &processor_type),
                    "Failed to get processor type!");

                // Only process AMD GPU devices, skip other processor types
                if(processor_type != AMDSMI_PROCESSOR_TYPE_AMD_GPU)
                {
                    index++;
                    continue;
                }

                auto device = std::make_shared<device_t>(driver, processor_handle,
                                                         processor_type, index);

                bool should_include = false;
                switch(filter.mode)
                {
                    case device_selection_mode::ALL: should_include = true; break;
                    case device_selection_mode::NONE: should_include = false; break;
                    case device_selection_mode::SPECIFIC:
                        should_include = filter.indices.count(index) > 0;
                        break;
                }

                if(should_include && device->is_supported())
                {
                    m_gpu_devices.emplace_back(std::move(device));
                }

                index++;
            }
        }
    }

    device_vector_t m_gpu_devices;  ///< List of enabled GPU devices for sampling
    std::shared_ptr<device_provider> m_device_provider;  ///< Device provider instance
    enabled_metrics m_enabled_metrics;  ///< Set of metrics enabled for collection
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace gpu
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
