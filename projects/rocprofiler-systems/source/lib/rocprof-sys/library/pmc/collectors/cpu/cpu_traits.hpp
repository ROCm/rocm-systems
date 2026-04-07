// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/config.hpp"
#include "library/pmc/collectors/cpu/device.hpp"
#include "library/pmc/collectors/cpu/sample.hpp"
#include "library/pmc/collectors/cpu/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

namespace rocprofsys::pmc::collectors::cpu
{

using ::rocprofsys::pmc::device_filter;
using ::rocprofsys::pmc::device_selection_mode;

/**
 * @brief Traits type for CPU collector configuration.
 *
 * Each CPU socket (physical package) is modeled as a separate device, aligned
 * with the GPU pattern where each GPU is a separate device. The device filter
 * selects socket IDs (not individual cores). All cores on a selected socket
 * are always monitored.
 *
 * @tparam DriverProvider The provider type (wraps procfs driver).
 */
template <typename DriverProvider>
struct cpu_traits
{
    using metrics_t         = cpu::metrics;
    using enabled_metrics_t = cpu::enabled_metrics;
    using device_t          = device<typename DriverProvider::driver_t>;
    using device_ptr_t      = std::shared_ptr<device_t>;
    using container_t       = std::vector<device_ptr_t>;
    using driver_t          = typename DriverProvider::driver_t;

    static constexpr const char* device_name = "CPU";

    struct device_entry
    {
        device_ptr_t      device;
        enabled_metrics_t supported_metrics;
    };

    template <typename Settings>
    [[nodiscard]] static device_filter get_device_filter()
    {
        return Settings::get_device_filter(rocprofsys::get_sampling_cpus());
    }

    template <typename Settings>
    [[nodiscard]] static enabled_metrics_t get_enabled_metrics()
    {
        return Settings::get_cpu_enabled_metrics();
    }

    template <typename Cache>
    static void init_pmc_metadata(const device_ptr_t& dev)
    {
        static bool first_socket_registered = false;
        const bool  is_first                = !first_socket_registered;
        first_socket_registered             = true;
        Cache::initialize_pmc_metadata(dev->get_index(), dev->get_monitored_cpus(),
                                       is_first);
    }

    template <typename Perfetto, typename DeviceVector>
    static void init_perfetto_storage(const DeviceVector& /*device_entries*/)
    {
        Perfetto::init_storage();
    }

    template <typename Perfetto>
    static void setup_counter_tracks(const device_ptr_t&      dev,
                                     const enabled_metrics_t& enabled)
    {
        Perfetto::setup_counter_tracks(dev->get_index(), dev->get_monitored_cpus(),
                                       enabled);
    }

    template <typename Perfetto, typename DeviceEntries>
    static void post_process_perfetto(const DeviceEntries&     entries,
                                      const enabled_metrics_t& enabled)
    {
        for(const auto& entry : entries)
        {
            Perfetto::post_process(entry.device->get_index(),
                                   entry.device->get_monitored_cpus(), enabled);
        }
    }

    [[nodiscard]] static metrics_t get_metrics(const device_ptr_t&       dev,
                                               const enabled_metrics_t&  enabled,
                                               [[maybe_unused]] uint64_t timestamp)
    {
        return dev->get_cpu_metrics(enabled);
    }

    /**
     * @brief Enumerate CPU devices — one per socket (physical package).
     *
     * The filter selects socket IDs (not core IDs). All cores on a
     * selected socket are always monitored. All devices share a single
     * driver instance (one /proc/stat read serves all sockets).
     */
    template <typename Settings, typename Provider>
    [[nodiscard]] static std::vector<device_entry> enumerate_devices(
        std::shared_ptr<Provider> provider)
    {
        std::vector<device_entry> entries;
        const auto                filter = get_device_filter<Settings>();

        if(filter.mode == device_selection_mode::NONE)
        {
            LOG_DEBUG("{} sampling disabled via configuration", device_name);
            return entries;
        }

        const auto& topology     = provider->get_socket_topology();
        const auto  socket_count = topology.size();
        LOG_INFO("Detected {} CPU socket(s), {} online CPUs", socket_count,
                 provider->get_cpu_count());

        // Select which sockets to monitor
        std::set<size_t> selected_sockets;
        switch(filter.mode)
        {
            case device_selection_mode::ALL:
                for(const auto& [socket_id, cpus] : topology)
                    selected_sockets.insert(socket_id);
                break;
            case device_selection_mode::NONE: return entries;
            case device_selection_mode::SPECIFIC:
                for(const auto idx : filter.indices)
                {
                    if(topology.count(idx) > 0)
                    {
                        selected_sockets.insert(idx);
                    }
                    else if(!topology.empty())
                    {
                        LOG_WARNING("CPU socket {} not found (available: 0-{})", idx,
                                    topology.rbegin()->first);
                    }
                }
                break;
        }

        if(selected_sockets.empty())
        {
            LOG_WARNING("No CPU sockets selected for monitoring");
            return entries;
        }

        const auto& drv = provider->get_driver();

        for(const auto socket_id : selected_sockets)
        {
            const auto& cpu_set = topology.at(socket_id);
            auto        dev     = std::make_shared<device_t>(drv, socket_id, cpu_set);

            if(!dev->is_supported())
            {
                LOG_WARNING("No CPU metrics supported on socket {}", socket_id);
            }

            const auto supported = dev->get_supported_metrics();
            entries.push_back(device_entry{ std::move(dev), supported });
        }

        LOG_INFO("Enabled {} CPU socket(s) for PMC sampling", selected_sockets.size());
        return entries;
    }
};

}  // namespace rocprofsys::pmc::collectors::cpu
