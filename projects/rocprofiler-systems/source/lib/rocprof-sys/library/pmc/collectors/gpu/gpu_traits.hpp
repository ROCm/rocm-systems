// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/pci_bdf.hpp"
#include "library/pmc/collectors/gpu/device.hpp"
#include "library/pmc/collectors/gpu/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

namespace rocprofsys::pmc::collectors::gpu
{

using ::rocprofsys::pmc::device_filter;
using ::rocprofsys::pmc::device_selection_mode;
using ::rocprofsys::pmc::device_type;

/**
 * @brief Whether a device's PCIe BDF indicates it is visible to the ROCm runtime.
 *
 * A device is runtime-visible iff it reports a non-empty PCIe BDF that appears in
 * @p visible_bdfs (the set the runtime exposes via ROCR_VISIBLE_DEVICES /
 * HIP_VISIBLE_DEVICES). An empty BDF means the device identity could not be
 * determined, so it is treated as NOT visible.
 */
[[nodiscard]] inline bool
is_runtime_visible(const std::string& bdf, const std::set<std::string>& visible_bdfs)
{
    return !bdf.empty() && visible_bdfs.count(bdf) > 0;
}

/**
 * @brief Traits type for GPU collector configuration.
 *
 * Defines types, constants, and customization points for the base collector template
 * to work with GPU devices via AMD SMI.
 *
 * @tparam BackendProvider Device provider type.
 * @tparam DeviceType      Concrete device type; exposes @c backend_type so traits
 *                         stay decoupled from the AMD SMI backend headers.
 */
template <typename BackendProvider, typename DeviceType>
struct gpu_traits
{
    // Required type aliases for base::collector
    using metrics_t         = pmc::collectors::gpu::metrics;
    using enabled_metrics_t = pmc::collectors::gpu::enabled_metrics;
    using backend_t         = DeviceType::backend_type;
    using device_t          = DeviceType;
    using device_ptr_t      = std::shared_ptr<device_t>;
    using container_t       = std::vector<device_ptr_t>;

    // Required constants
    static constexpr const char* device_name = "GPU";
    // Settings customization points

    /**
     * @brief Get the device filter from settings.
     */
    template <typename Settings>
    [[nodiscard]] static device_filter get_device_filter()
    {
        return Settings::get_gpu_device_filter();
    }

    /**
     * @brief Get the PCIe BDFs of the GPUs the ROCm runtime exposes.
     *
     * Sourced from settings so the traits stay independent of the ROCm/AMD SMI
     * headers and can be exercised with a stub settings policy.
     *
     * @return std::nullopt when runtime visibility could not be determined.
     */
    template <typename Settings>
    [[nodiscard]] static std::optional<std::set<std::string>> get_visible_gpu_bdfs()
    {
        return Settings::get_visible_gpu_bdfs();
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
                                               std::uint64_t            timestamp)
    {
        return device->get_metrics(enabled, timestamp);
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
     * - Drops devices the ROCm runtime does not expose, correlating each device's
     *   PCIe BDF against the runtime-visible set (ROCR/HIP_VISIBLE_DEVICES)
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

        auto       devices      = provider->template get_gpu_devices<device_t>();
        const auto visible_bdfs = get_visible_gpu_bdfs<Settings>();

        size_t                                      selected_count = 0;
        std::vector<std::pair<size_t, std::string>> excluded;

        for(auto& device : devices)
        {
            auto index = device->get_index();

            bool should_include = (filter.mode == device_selection_mode::ALL) ||
                                  (filter.mode == device_selection_mode::SPECIFIC &&
                                   filter.indices.count(index) > 0);

            if(should_include) ++selected_count;

            if(should_include && visible_bdfs.has_value())
            {
                const auto& bdf = device->get_bdf();
                if(!is_runtime_visible(bdf, *visible_bdfs))
                {
                    excluded.emplace_back(index, bdf);
                    should_include = false;
                }
            }

            if(should_include && device->is_supported())
            {
                auto supported = device->get_supported_metrics();
                entries.push_back(device_entry{ std::move(device), supported });
            }
        }

        report_visibility(filter, visible_bdfs, devices.size(), selected_count, excluded);
        warn_invalid_indices(filter, devices.size());
        return entries;
    }

    /**
     * @brief Emit the runtime-visibility report once per process.
     *
     * enumerate_devices() re-runs on sampler-thread reinit after fork, but the masks (env
     * vars) and the device/agent set are fixed for the process, so every run produces the
     * same report. A single flag emits it once and stays quiet thereafter.
     */
    static void report_visibility(
        const device_filter& filter, const std::optional<std::set<std::string>>& visible,
        size_t num_devices, size_t selected_count,
        const std::vector<std::pair<size_t, std::string>>& excluded)
    {
        static std::atomic<bool> s_reported{ false };
        if(s_reported.exchange(true)) return;

        // Visibility unknown (the ROCm runtime reported no GPU agents at all) is not the
        // same as "nothing is visible". Stay quiet when AMD SMI found no GPUs either,
        // which is simply a machine without them.
        if(!visible.has_value())
        {
            if(num_devices > 0)
            {
                LOG_WARNING(
                    "Could not determine which GPUs the ROCm runtime exposes (no agents "
                    "reported); sampling the selected {} devices without applying "
                    "ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES filtering",
                    device_name);
            }
            return;
        }

        if(excluded.empty()) return;

        // Losing every selected device is either a deliberate mask or a failure to
        // correlate AMD SMI devices with ROCm agents. The runtime's own visible-BDF list
        // makes the two distinguishable, so a single summary is more useful here than one
        // line per masked device.
        if(selected_count > 0 && excluded.size() == selected_count)
        {
            // An empty set is reachable and meaningful here (agents exist, all masked)
            LOG_WARNING("None of the {} selected {} device(s) are visible to the ROCm "
                        "runtime; GPU sampling is disabled. Runtime-visible BDFs: {}",
                        selected_count, device_name,
                        visible->empty() ? std::string{ "<none>" }
                                         : fmt::format("{}", fmt::join(*visible, ", ")));
            return;
        }

        for(const auto& [index, bdf] : excluded)
            log_visibility_exclusion(filter, index, bdf);
    }

    /**
     * @brief Report a single device dropped because the ROCm runtime does not expose it.
     */
    static void log_visibility_exclusion(const device_filter& filter, size_t index,
                                         const std::string& bdf)
    {
        if(bdf.empty())
        {
            LOG_WARNING("{} device [{}] has no PCIe BDF; cannot verify runtime "
                        "visibility, excluding from sampling",
                        device_name, index);
            return;
        }

        const auto bdfid = ::rocprofsys::common::pci_bdfid_from_string(bdf);

        if(filter.mode == device_selection_mode::SPECIFIC)
        {
            LOG_WARNING("{} device [{}] (BDF {}, rocminfo BDFID {}) was requested via "
                        "ROCPROFSYS_SAMPLING_GPUS but is not visible to the ROCm "
                        "runtime; excluding from sampling",
                        device_name, index, bdf, bdfid);
        }
        else
        {
            LOG_DEBUG("{} device [{}] (BDF {}, rocminfo BDFID {}) not visible to ROCm "
                      "runtime; excluding from sampling",
                      device_name, index, bdf, bdfid);
        }
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
