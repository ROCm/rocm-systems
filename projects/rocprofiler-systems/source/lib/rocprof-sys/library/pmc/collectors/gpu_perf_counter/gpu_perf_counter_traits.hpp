// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/gpu_perf_counter/device.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

using ::rocprofsys::pmc::device_selection_mode;
using ::rocprofsys::pmc::device_type;

template <typename DriverProvider>
struct gpu_perf_counter_traits
{
    using metrics_t         = pmc::collectors::gpu_perf_counter::metrics;
    using enabled_metrics_t = pmc::collectors::gpu_perf_counter::enabled_metrics;
    using device_t          = device<typename DriverProvider::driver_t>;
    using device_ptr_t      = std::shared_ptr<device_t>;
    using container_t       = std::vector<device_ptr_t>;
    using driver_t          = typename DriverProvider::driver_t;

    static constexpr const char* device_name = "GPU";

    template <typename Settings>
    [[nodiscard]] static enabled_metrics_t get_enabled_metrics()
    {
        return to_enabled_metrics(Settings::get_gpu_perf_counter_enabled_metrics());
    }

    template <typename Cache>
    static void init_pmc_metadata(const device_ptr_t& dev)
    {
        Cache::initialize_pmc_metadata(dev->get_index(), dev->get_counter_metadata());
    }

    template <typename Perfetto, typename DeviceVector>
    static void init_perfetto_storage(const DeviceVector& devices)
    {
        Perfetto::init_storage(devices);
    }

    template <typename Perfetto>
    static void setup_counter_tracks(const device_ptr_t& device,
                                     const enabled_metrics_t& /*enabled*/)
    {
        Perfetto::setup_counter_tracks(device->get_index(),
                                       device->get_counter_metadata());
    }

    template <typename Perfetto, typename DeviceEntries>
    static void post_process_perfetto(const DeviceEntries& /*device_entries*/,
                                      const enabled_metrics_t& enabled)
    {
        Perfetto::post_process(enabled);
    }

    [[nodiscard]] static const metrics_t& get_metrics(const device_ptr_t&      dev,
                                                      const enabled_metrics_t& enabled,
                                                      std::uint64_t            timestamp)
    {
        return dev->get_gpu_perf_counter_metrics(enabled, timestamp);
    }

    struct device_entry
    {
        device_ptr_t      device;
        enabled_metrics_t supported_metrics;
    };

    // Device filtering is the caller's responsibility: the agent list passed into
    // register_gpu_perf_counter_source is already filtered upstream, so the
    // Settings parameter is unused here. It is retained because the base
    // collector's traits_check probe instantiates the two-parameter signature.
    template <typename Settings, typename Provider>
    [[nodiscard]] static std::vector<device_entry> enumerate_devices(
        std::shared_ptr<Provider> provider)
    {
        std::vector<device_entry> entries;

        auto devices = provider->template get_devices<device_t>(device_type::GPU);

        for(auto& dev : devices)
        {
            if(dev->is_supported())
            {
                entries.push_back(
                    device_entry{ std::move(dev), enabled_metrics_t{ {} } });
            }
        }

        return entries;
    }
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
