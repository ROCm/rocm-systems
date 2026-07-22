// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent_manager.hpp"
#include "library/pmc/collectors/nic/device.hpp"
#include "library/pmc/collectors/nic/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>

namespace rocprofsys::pmc::collectors::nic
{

using ::rocprofsys::pmc::device_selection_mode;
using ::rocprofsys::pmc::device_type;
using ::rocprofsys::pmc::nic_device_filter;

/**
 * @brief Traits type for NIC collector configuration.
 *
 * Defines types, constants, and customization points for the base collector template
 * to work with NIC devices via AMD SMI.
 *
 * @note This traits class bridges the NIC-specific requirements to the base::collector:
 * - Name-based device filtering (vs GPU's index-based filtering)
 * - Device context storage for NIC-specific API signatures (device_name, product_name)
 * - Agent registration during device enumeration
 *
 * @tparam BackendProvider Device provider type.
 * @tparam DeviceType      Concrete device type; exposes @c backend_type so traits
 *                         stay decoupled from the AMD SMI backend headers.
 */
template <typename BackendProvider, typename DeviceType>
struct nic_traits
{
    using metrics_t         = pmc::collectors::nic::metrics;
    using enabled_metrics_t = pmc::collectors::nic::enabled_metrics;
    using backend_t         = DeviceType::backend_type;
    using device_t          = DeviceType;
    using device_ptr_t      = std::shared_ptr<device_t>;
    using container_t       = std::vector<device_ptr_t>;

    static constexpr const char* device_name = "NIC";
    struct device_entry
    {
        device_ptr_t      device;
        enabled_metrics_t supported_metrics;
    };

    template <typename Settings>
    [[nodiscard]] static nic_device_filter get_device_filter()
    {
        return Settings::get_nic_device_filter();
    }

    template <typename Settings>
    [[nodiscard]] static enabled_metrics_t get_enabled_metrics()
    {
        return Settings::get_nic_enabled_metrics();
    }

    template <typename Cache>
    static void init_pmc_metadata(const device_ptr_t& device)
    {
        Cache::initialize_pmc_metadata(device->get_index(), device->get_product_name());
    }

    template <typename DeviceEntries>
    static container_t extract_devices(const DeviceEntries& device_entries)
    {
        container_t devices;
        devices.reserve(device_entries.size());
        for(const auto& entry : device_entries)
        {
            devices.push_back(entry.device);
        }
        return devices;
    }

    template <typename Perfetto, typename DeviceEntries>
    static void init_perfetto_storage(const DeviceEntries& device_entries)
    {
        Perfetto::init_storage(extract_devices(device_entries));
    }

    template <typename Perfetto>
    static void setup_counter_tracks(const device_ptr_t&      device,
                                     const enabled_metrics_t& enabled)
    {
        Perfetto::setup_counter_tracks(device->get_index(), device->get_name(), enabled);
    }

    template <typename Perfetto, typename DeviceEntries>
    static void post_process_perfetto(const DeviceEntries&     device_entries,
                                      const enabled_metrics_t& enabled)
    {
        Perfetto::post_process(extract_devices(device_entries), enabled);
    }

    [[nodiscard]] static metrics_t get_metrics(
        const device_ptr_t& device, [[maybe_unused]] const enabled_metrics_t& enabled,
        [[maybe_unused]] std::uint64_t timestamp)
    {
        return device->get_nic_metrics();
    }

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

        auto devices = provider->template get_nic_devices<device_t>();

        LOG_DEBUG("Discovered {} AI NIC device(s) via AMD SMI", devices.size());

        std::set<std::string> available_names;
        for(const auto& device : devices)
            available_names.insert(device->get_name());

        for(auto& device : devices)
        {
            bool should_include = false;
            switch(filter.mode)
            {
                case device_selection_mode::ALL: should_include = true; break;
                // Unreachable (early return above), kept for switch exhaustiveness
                case device_selection_mode::NONE: should_include = false; break;
                case device_selection_mode::SPECIFIC:
                    should_include = filter.names.count(device->get_name()) > 0;
                    break;
            }

            if(should_include)
            {
                if(!device->is_supported())
                {
                    // Warn only when the user explicitly requested this device; under
                    // ALL an unsupported (e.g. non-RDMA) NIC is expected, not an error.
                    if(filter.mode == device_selection_mode::SPECIFIC)
                    {
                        LOG_WARNING("Requested NIC device [{}] ({}) has no supported "
                                    "RDMA metrics, skipping",
                                    device->get_index(), device->get_name());
                    }
                    else
                    {
                        LOG_DEBUG("NIC device [{}] ({}) has no supported RDMA metrics, "
                                  "skipping",
                                  device->get_index(), device->get_name());
                    }
                    continue;
                }
                LOG_INFO("NIC device [{}] ({}) enabled for AI NIC PMC sampling",
                         device->get_index(), device->get_name());
                auto supported = device->get_supported_metrics();
                entries.push_back(device_entry{ std::move(device), supported });
            }
            else
            {
                LOG_DEBUG(
                    "NIC device [{}] ({}) excluded by ROCPROFSYS_SAMPLING_AINICS filter",
                    device->get_index(), device->get_name());
            }
        }

        warn_invalid_names(filter, available_names);
        register_nic_agents(entries);

        return entries;
    }

    static void warn_invalid_names(const nic_device_filter&     filter,
                                   const std::set<std::string>& available_names)
    {
        if(filter.mode != device_selection_mode::SPECIFIC) return;
        if(available_names.empty())
        {
            LOG_WARNING("No AI NIC devices were discovered.");
            return;
        }
        for(const auto& requested : filter.names)
        {
            if(available_names.find(requested) == available_names.end())
            {
                LOG_WARNING("Requested AI NIC device '{}' not found. "
                            "Available device(s): [{}]",
                            requested, fmt::join(available_names, ", "));
            }
        }
    }

    static void register_nic_agents(const std::vector<device_entry>& entries)
    {
        size_t nic_index = 0;
        for(const auto& entry : entries)
        {
            agent cur_agent{ agent_type::NIC,
                             0,
                             nic_index,
                             static_cast<std::uint32_t>(nic_index),
                             static_cast<std::int32_t>(nic_index),
                             static_cast<std::int32_t>(nic_index),
                             entry.device->get_product_name().c_str(),
                             entry.device->get_vendor_name().c_str(),
                             "AI NIC",
                             "AI NIC",
                             0,
                             0,
                             {} };

            get_agent_manager_instance().insert_agent(cur_agent);
            nic_index++;
        }
    }
};

}  // namespace rocprofsys::pmc::collectors::nic
