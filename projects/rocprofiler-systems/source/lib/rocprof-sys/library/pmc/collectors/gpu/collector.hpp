// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/collectors/gpu/device.hpp"
#include "library/pmc/collectors/gpu/types.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu
{

using ::rocprofsys::pmc::device_filter;
using ::rocprofsys::pmc::device_selection_mode;
using ::rocprofsys::pmc::collectors::gpu::check_status;
using ::rocprofsys::pmc::collectors::gpu::enabled_metrics;

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
            CacheApi::initialize_tracks_metadata();
            CacheApi::initialize_pmc_metadata(device_index);
        }
    }

    /**
     * @brief Sample GPU metrics from all enabled devices.
     *
     * Iterates through all GPU devices, retrieves current metrics, and stores them
     * via the cache API and optionally Perfetto. Devices that fail to read metrics
     * are automatically disabled and removed from the device list.
     *
     * @param timestamp Current timestamp in nanoseconds for the sample.
     */
    void sample(int64_t timestamp)
    {
        auto new_end = std::remove_if(
            m_gpu_devices.begin(), m_gpu_devices.end(),
            [this, timestamp](const device_ptr_t& device) {
                try
                {
                    auto _supported_metrics = device->get_supported_metrics();
                    auto _gpu_metrics       = device->get_gpu_metrics();
                    auto _device_id         = device->get_index();

                    compute_sdma_usage(device, _supported_metrics, _gpu_metrics,
                                       timestamp);

                    CacheApi::store_sample(_device_id, m_enabled_metrics,
                                           _supported_metrics, _gpu_metrics, timestamp);
                    if(SettingsApi::get_use_perfetto_legacy_metrics())
                    {
                        PerfettoApi::store_sample(_device_id, _gpu_metrics, timestamp);
                    }
                    return false;  // Keep device
                } catch(const std::runtime_error& e)
                {
                    LOG_ERROR("Reading metrics failed for GPU device {}. Error: {}. "
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
        m_gpu_devices.clear();
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

    /**
     * @brief Compute SDMA utilization percentage from raw cumulative data.
     *
     * Reads the raw cumulative SDMA usage from the device and computes a
     * percentage based on the delta from the previous sample. On the first
     * sample for a device, the percentage is 0 (no previous data for delta).
     *
     * @param device Device to read SDMA data from
     * @param supported_metrics Metrics supported by this device
     * @param gpu_metrics Metrics struct to write the SDMA percentage into
     * @param timestamp Current sample timestamp in nanoseconds
     */
    void compute_sdma_usage([[maybe_unused]] const device_ptr_t&    device,
                            [[maybe_unused]] const enabled_metrics& supported_metrics,
                            [[maybe_unused]] metrics&               gpu_metrics,
                            [[maybe_unused]] int64_t                timestamp)
    {
#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
        if(!m_enabled_metrics.bits.sdma_usage || !supported_metrics.bits.sdma_usage)
            return;

        auto     _device_id         = device->get_index();
        uint64_t current_cumulative = device->get_raw_sdma_usage();

        // Ensure vector is sized to accommodate device index
        if(_device_id >= m_sdma_states.size())
        {
            m_sdma_states.resize(_device_id + 1);
        }

        auto& state = m_sdma_states[_device_id];
        if(state.has_prev && timestamp > static_cast<int64_t>(state.prev_timestamp))
        {
            uint64_t delta_usage =
                current_cumulative - state.prev_cumulative;  // microseconds
            uint64_t delta_time =
                static_cast<uint64_t>(timestamp) - state.prev_timestamp;  // nanoseconds

            // percentage = (delta_usage_us * 1000 * 100) / delta_time_ns
            //            = (delta_usage_us * 100000) / delta_time_ns
            uint32_t pct = static_cast<uint32_t>((delta_usage * 100000ULL) / delta_time);
            gpu_metrics.sdma_usage = (pct > 100) ? 100 : pct;
        }

        state.prev_cumulative = current_cumulative;
        state.prev_timestamp  = static_cast<uint64_t>(timestamp);
        state.has_prev        = true;
#endif
    }

    /// Per-device state for SDMA delta computation
    struct sdma_state
    {
        uint64_t prev_cumulative = 0;
        uint64_t prev_timestamp  = 0;
        bool     has_prev        = false;
    };

    device_vector_t                  m_gpu_devices;  ///< Enabled GPU devices for sampling
    std::shared_ptr<device_provider> m_device_provider;  ///< Device provider instance
    enabled_metrics         m_enabled_metrics;  ///< Metrics enabled for collection
    std::vector<sdma_state> m_sdma_states;      ///< Per-device SDMA delta tracking
};

}  // namespace rocprofsys::pmc::collectors::gpu
