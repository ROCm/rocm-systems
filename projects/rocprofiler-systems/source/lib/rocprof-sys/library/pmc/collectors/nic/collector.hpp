// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/agent_manager.hpp"
#include "library/pmc/collectors/nic/device.hpp"
#include "library/pmc/collectors/nic/types.hpp"
#include "library/pmc/common/types.hpp"
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
namespace nic
{

#if ROCPROFSYS_USE_ROCM > 0

using ::rocprofsys::pmc::device_selection_mode;
using ::rocprofsys::pmc::nic_device_filter;
using ::rocprofsys::pmc::collectors::nic::enabled_metrics;

using gettimestamp_t = std::function<unsigned long()>;

/**
 * @brief NIC RDMA metrics collector for performance monitoring.
 *
 * This collector manages the lifecycle of NIC performance monitoring, including
 * device enumeration, metric sampling, and data storage. It uses a policy-based design
 * pattern via template parameters to allow compile-time dependency injection.
 *
 * @tparam DeviceProvider Type providing device enumeration and management
 * @tparam Config Configuration policy providing settings and output policies
 */
template <typename DeviceProvider, typename Config>
struct collector
{
    using device_provider = DeviceProvider;
    using SettingsApi     = typename Config::SettingsApi;
    using CacheApi        = typename Config::CacheApi;
    using PerfettoApi     = typename Config::PerfettoApi;
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
     * @brief Initialize the collector and enumerate NIC devices.
     *
     * Retrieves version information, enumerates NIC devices based on device filter
     * settings, and logs initialization status.
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

        // Enumerate NIC devices
        enumerate_devices();

        m_enabled_metrics = SettingsApi::get_nic_enabled_metrics();

        if(SettingsApi::get_use_perfetto_legacy_metrics())
        {
            PerfettoApi::init_storage(m_nic_devices);
        }
    }

    /**
     * @brief Configure metrics tracking and initialize metadata.
     *
     * Sets up category metadata and per-device NIC metadata for all enabled devices.
     */
    void config()
    {
        CacheApi::initialize_category_metadata();

        for(const auto& device : m_nic_devices)
        {
            auto device_index = device->get_index();
            if(SettingsApi::get_use_perfetto_legacy_metrics())
            {
                PerfettoApi::setup_counter_tracks(device_index, device->get_name(),
                                                  m_enabled_metrics);
            }
            CacheApi::initialize_tracks_metadata();
            CacheApi::initialize_pmc_metadata(device_index, device->get_product_name());
        }
    }

    /**
     * @brief Sample NIC metrics from all enabled devices.
     *
     * Iterates through all NIC devices, retrieves current metrics, and stores them
     * via the cache API. Devices that fail to read metrics are automatically
     * disabled and removed from the device list.
     *
     * @param timestamp Current timestamp in nanoseconds for the sample.
     */
    void sample(int64_t timestamp)
    {
        auto new_end = std::remove_if(
            m_nic_devices.begin(), m_nic_devices.end(),
            [this, timestamp](const device_ptr_t& device) {
                try
                {
                    auto _supported_metrics = device->get_supported_metrics();
                    auto _nic_metrics       = device->get_nic_metrics();
                    auto _device_id         = device->get_index();
                    auto _device_name       = device->get_name();

                    CacheApi::store_sample(_device_id, _device_name, m_enabled_metrics,
                                           _supported_metrics, _nic_metrics, timestamp);
                    if(SettingsApi::get_use_perfetto_legacy_metrics())
                    {
                        PerfettoApi::store_sample(_device_id, _nic_metrics, timestamp);
                    }
                    return false;  // Keep device
                } catch(const std::runtime_error& e)
                {
                    LOG_ERROR("Reading NIC metrics failed for device %s (ID %zu). "
                              "Error: %s. Disabling device!",
                              device->get_name().c_str(), device->get_index(), e.what());
                    return true;  // Remove device
                }
            });
        m_nic_devices.erase(new_end, m_nic_devices.end());
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
            PerfettoApi::post_process(m_nic_devices, m_enabled_metrics);
        }
        m_nic_devices.clear();
    }

    /**
     * @brief Get the list of all NIC devices.
     * @return Const reference to the vector of NIC devices.
     */
    const device_vector_t& get_devices() const noexcept { return m_nic_devices; }

    /**
     * @brief Get the number of enabled NIC devices.
     * @return Number of NIC devices currently enabled for sampling.
     */
    size_t get_device_count() const noexcept { return m_nic_devices.size(); }

    /**
     * @brief Shutdown the device provider and release resources.
     */
    void shutdown() {}

private:
    /**
     * @brief Enumerate NIC devices from device provider and create device objects.
     *
     * Uses amdsmi_get_processor_handles_by_type() to directly query NIC devices,
     * creates device objects, and applies name-based filtering.
     *
     * @note amdsmi_get_processor_handles() only returns GPUs. For NICs, we must use
     * amdsmi_get_processor_handles_by_type() with AMDSMI_PROCESSOR_TYPE_AMD_NIC.
     */
    void enumerate_devices()
    {
        auto filter = SettingsApi::get_nic_device_filter();
        auto driver = m_device_provider->get_driver();

        if(filter.mode == device_selection_mode::NONE)
        {
            return;
        }

        auto   socket_handles = m_device_provider->get_socket_handles();
        size_t nic_index      = 0;

        for(size_t socket_idx = 0; socket_idx < socket_handles.size(); ++socket_idx)
        {
            auto& socket_handle = socket_handles[socket_idx];
            auto  nic_handles   = m_device_provider->get_processor_handles_by_type(
                socket_handle, AMDSMI_PROCESSOR_TYPE_AMD_NIC);

            for(auto& processor_handle : nic_handles)
            {
                auto nic_device = std::make_shared<device_t>(
                    driver, processor_handle, AMDSMI_PROCESSOR_TYPE_AMD_NIC, nic_index);

                if(!nic_device->is_supported())
                {
                    nic_index++;
                    continue;
                }

                bool should_include = false;
                switch(filter.mode)
                {
                    case device_selection_mode::ALL: should_include = true; break;
                    case device_selection_mode::NONE: should_include = false; break;
                    case device_selection_mode::SPECIFIC:
                        should_include = filter.names.count(nic_device->get_name()) > 0;
                        break;
                }

                if(should_include)
                {
                    m_nic_devices.emplace_back(std::move(nic_device));
                }

                nic_index++;
            }
        }

        // Register NIC agents with the agent manager
        nic_index = 0;
        for(const auto& device : m_nic_devices)
        {
            auto cur_agent = agent{ agent_type::NIC,
                                    0,
                                    nic_index,
                                    static_cast<uint32_t>(nic_index),
                                    static_cast<int32_t>(nic_index),
                                    static_cast<int32_t>(nic_index),
                                    device->get_product_name().c_str(),
                                    device->get_vendor_name().c_str(),
                                    "AI NIC",
                                    "AI NIC" };

            get_agent_manager_instance().insert_agent(cur_agent);
            nic_index++;
        }
    }

    device_vector_t m_nic_devices;  ///< List of enabled NIC devices for sampling
    std::shared_ptr<device_provider> m_device_provider;  ///< Device provider instance
    enabled_metrics m_enabled_metrics;  ///< Set of metrics enabled for collection
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace nic
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
