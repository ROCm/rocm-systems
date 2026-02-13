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

#include <memory>
#include <vector>

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace gpu
{

#if ROCPROFSYS_USE_ROCM > 0

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
    using metrics_t         = pmc::gpu::metrics;
    using enabled_metrics_t = pmc::gpu::enabled_metrics;
    using device_t          = device<Driver>;
    using device_ptr_t      = std::shared_ptr<device_t>;
    using container_t       = std::vector<device_ptr_t>;
    using driver_t          = Driver;

    // Required constants
    static constexpr const char* device_name      = "GPU";
    static constexpr bool        multi_device     = true;
    static constexpr bool        has_version_info = true;

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

    // Provider customization points

    /**
     * @brief Get processor handles from the device provider.
     */
    template <typename Provider>
    static std::vector<amdsmi_processor_handle> get_processor_handles(
        Provider& provider, amdsmi_socket_handle socket)
    {
        return provider.get_processor_handles(socket);
    }

    /**
     * @brief Whether to filter devices by processor type.
     */
    static constexpr bool filter_by_processor_type() { return true; }

    /**
     * @brief The expected processor type for GPU devices.
     */
    static constexpr processor_type_t expected_processor_type()
    {
        return AMDSMI_PROCESSOR_TYPE_AMD_GPU;
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
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace gpu
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
