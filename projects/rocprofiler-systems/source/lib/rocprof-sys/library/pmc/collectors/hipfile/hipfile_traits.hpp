// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/hipfile/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::collectors::hipfile
{

using ::rocprofsys::pmc::device_filter;
using ::rocprofsys::pmc::device_selection_mode;

/**
 * @brief Traits type for hipFile collector configuration.
 *
 * Bridges hipFile's GPU-direct storage I/O stats to base::collector.
 *
 * @note Unlike the AMD SMI collectors, enumeration does not probe the data source.
 * hipFile stats only become readable once the target performs hipFile I/O, which
 * generally happens after the sampler has started, so probing at setup() would find
 * nothing and permanently disable the collector. Devices are instead enumerated from
 * the ROCm-visible GPU set, and each sample decides for itself whether hipFile had
 * anything to report.
 *
 * @tparam BackendProvider Device provider type.
 * @tparam DeviceType      Concrete device type; exposes @c backend_type so the traits
 *                         stay decoupled from the hipFile backend headers.
 */
template <typename BackendProvider, typename DeviceType>
struct hipfile_traits
{
    using metrics_t         = pmc::collectors::hipfile::metrics;
    using enabled_metrics_t = pmc::collectors::hipfile::enabled_metrics;
    using backend_t         = DeviceType::backend_type;
    using device_t          = DeviceType;
    using device_ptr_t      = std::shared_ptr<device_t>;
    using container_t       = std::vector<device_ptr_t>;

    static constexpr const char* device_name = "hipFile";

    struct device_entry
    {
        device_ptr_t      device;
        enabled_metrics_t supported_metrics;
    };

    template <typename Settings>
    [[nodiscard]] static device_filter get_device_filter()
    {
        // hipFile telemetry is per GPU, so it honours the same GPU selection as the
        // AMD SMI GPU collector rather than introducing a second, divergent filter.
        return Settings::get_gpu_device_filter();
    }

    template <typename Settings>
    [[nodiscard]] static enabled_metrics_t get_enabled_metrics()
    {
        return Settings::get_hipfile_enabled_metrics();
    }

    template <typename Cache>
    static void init_pmc_metadata(const device_ptr_t& device)
    {
        Cache::initialize_pmc_metadata(device->get_index());
    }

    // Perfetto customization points are no-ops: hipFile counter tracks reach Perfetto
    // from the same hipfile_pmc_sample records that populate RocPD, not the legacy
    // per-collector Perfetto policy that the AMD SMI collectors still carry.
    template <typename Perfetto, typename DeviceEntries>
    static void init_perfetto_storage(const DeviceEntries& /*device_entries*/)
    {}

    template <typename Perfetto>
    static void setup_counter_tracks(const device_ptr_t& /*device*/,
                                     const enabled_metrics_t& /*enabled*/)
    {}

    template <typename Perfetto, typename DeviceEntries>
    static void post_process_perfetto(const DeviceEntries& /*device_entries*/,
                                      const enabled_metrics_t& /*enabled*/)
    {}

    [[nodiscard]] static metrics_t get_metrics(const device_ptr_t&      device,
                                               const enabled_metrics_t& enabled,
                                               std::uint64_t            timestamp)
    {
        return device->get_metrics(enabled, timestamp);
    }

    /**
     * @brief Enumerate one device per HIP-visible GPU.
     *
     * hipFile indexes @c per_gpu_stats by HIP ordinal. The settings mapping gives the
     * profiler @c device_type_index of each such ordinal so tracks, RocPD agents, and
     * @c ROCPROFSYS_SAMPLING_GPUS use the same GPU numbers as the AMD SMI collector.
     * Ordinals beyond the snapshot's capacity are dropped rather than silently aliased
     * onto a valid slot.
     */
    template <typename Settings, typename Provider>
    [[nodiscard]] static std::vector<device_entry> enumerate_devices(
        std::shared_ptr<Provider> provider)
    {
        std::vector<device_entry> entries;

        if(get_enabled_metrics<Settings>().value == 0)
        {
            LOG_DEBUG("{} sampling skipped: no metrics enabled", device_name);
            return entries;
        }

        const auto filter = get_device_filter<Settings>();
        if(filter.mode == device_selection_mode::NONE)
        {
            LOG_DEBUG("{} sampling disabled via configuration", device_name);
            return entries;
        }

        auto type_indices = Settings::get_visible_gpu_type_indices();
        if(type_indices.empty())
        {
            LOG_DEBUG("{} sampling disabled: no HIP-visible GPUs, or the visibility "
                      "mask cannot be mapped to profiler GPU indices",
                      device_name);
            return entries;
        }

        if(type_indices.size() > MAX_GPUS)
        {
            LOG_WARNING("{} GPUs are visible but hipFile reports stats for at most {}; "
                        "telemetry for the remaining GPUs is unavailable",
                        type_indices.size(), MAX_GPUS);
            type_indices.resize(MAX_GPUS);
        }

        auto devices = provider->template get_devices<device_t>(type_indices);

        for(auto& device : devices)
        {
            const auto index = device->get_index();

            const bool should_include = (filter.mode == device_selection_mode::ALL) ||
                                        (filter.mode == device_selection_mode::SPECIFIC &&
                                         filter.indices.count(index) > 0);

            if(!should_include)
            {
                LOG_DEBUG("{} GPU [{}] excluded by ROCPROFSYS_SAMPLING_GPUS filter",
                          device_name, index);
                continue;
            }

            auto supported = device->get_supported_metrics();
            entries.push_back(device_entry{ std::move(device), supported });
        }

        LOG_DEBUG("Enabled {} GPU(s) for {} telemetry", entries.size(), device_name);
        return entries;
    }
};

}  // namespace rocprofsys::pmc::collectors::hipfile
