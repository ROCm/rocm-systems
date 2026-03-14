// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/agent_manager.hpp"
#include "library/pmc/collectors/nic/device.hpp"
#include "library/pmc/collectors/nic/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <amd_smi/amdsmi.h>

namespace rocprofsys::pmc::collectors::nic
{

using ::rocprofsys::pmc::device_selection_mode;
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
 * @tparam Driver The AMD SMI driver type (real or mock for testing)
 */
template <typename Driver>
struct nic_traits
{
    using metrics_t         = pmc::collectors::nic::metrics;
    using enabled_metrics_t = pmc::collectors::nic::enabled_metrics;
    using device_t          = device<Driver>;
    using device_ptr_t      = std::shared_ptr<device_t>;
    using container_t       = std::vector<device_ptr_t>;
    using driver_t          = Driver;

    static constexpr const char* device_name  = "NIC";
    static constexpr bool        multi_device = true;

    struct device_entry
    {
        device_ptr_t      device;
        enabled_metrics_t supported_metrics;
    };

private:
    struct device_context
    {
        std::string       device_name;
        std::string       product_name;
        enabled_metrics_t supported_metrics;
    };

    static std::unordered_map<size_t, device_context>& get_device_contexts()
    {
        static std::unordered_map<size_t, device_context> contexts;
        return contexts;
    }

public:
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
    static void init_category_metadata()
    {
        Cache::initialize_category_metadata();
    }

    template <typename Cache>
    static void init_tracks_metadata()
    {
        Cache::initialize_tracks_metadata();
    }

    template <typename Cache>
    static void init_pmc_metadata(size_t device_index)
    {
        const auto& contexts = get_device_contexts();
        auto        it       = contexts.find(device_index);
        if(it != contexts.end())
        {
            Cache::initialize_pmc_metadata(device_index, it->second.product_name);
        }
    }

    template <typename Cache>
    static void store_sample(size_t device_id, const enabled_metrics_t& enabled,
                             const enabled_metrics_t& supported, const metrics_t& metrics,
                             uint64_t timestamp)
    {
        const auto& contexts = get_device_contexts();
        auto        it       = contexts.find(device_id);
        if(it != contexts.end())
        {
            Cache::store_sample(device_id, it->second.device_name, enabled, supported,
                                metrics, timestamp);
        }
    }

    template <typename Perfetto, typename DeviceEntries>
    static void init_perfetto_storage(const DeviceEntries& device_entries)
    {
        for(const auto& entry : device_entries)
        {
            auto idx                   = entry.device->get_index();
            get_device_contexts()[idx] = device_context{ entry.device->get_name(),
                                                         entry.device->get_product_name(),
                                                         entry.supported_metrics };
        }

        container_t devices;
        devices.reserve(device_entries.size());
        for(const auto& entry : device_entries)
        {
            devices.push_back(entry.device);
        }
        Perfetto::init_storage(devices);
    }

    template <typename Perfetto>
    static void setup_counter_tracks(size_t                   device_index,
                                     const enabled_metrics_t& enabled)
    {
        const auto& contexts = get_device_contexts();
        auto        it       = contexts.find(device_index);
        if(it != contexts.end())
        {
            Perfetto::setup_counter_tracks(device_index, it->second.device_name, enabled);
        }
    }

    template <typename Perfetto>
    static void store_perfetto_sample(size_t device_id, const metrics_t& metrics,
                                      uint64_t timestamp)
    {
        Perfetto::store_sample(device_id, metrics, timestamp);
    }

    template <typename Perfetto>
    static void post_process_perfetto(const enabled_metrics_t& enabled)
    {
        const auto& contexts = get_device_contexts();
        for(const auto& [device_index, ctx] : contexts)
        {
            Perfetto::post_process_device(device_index, enabled, ctx.supported_metrics);
        }
    }

    [[nodiscard]] static device_ptr_t create_device(std::shared_ptr<driver_t> driver,
                                                    amdsmi_processor_handle   handle,
                                                    processor_type_t type, size_t index)
    {
        return std::make_shared<device_t>(std::move(driver), handle, type, index);
    }

    [[nodiscard]] static size_t get_device_index(const device_ptr_t& device) noexcept
    {
        return device->get_index();
    }

    [[nodiscard]] static enabled_metrics_t get_supported_metrics(
        const device_ptr_t& device) noexcept
    {
        return device->get_supported_metrics();
    }

    [[nodiscard]] static metrics_t get_metrics(const device_ptr_t& device,
                                               const enabled_metrics_t& /*enabled*/,
                                               uint64_t /*timestamp*/)
    {
        return device->get_nic_metrics();
    }

    [[nodiscard]] static bool is_device_supported(const device_ptr_t& device) noexcept
    {
        return device->is_supported();
    }

    template <typename Settings, typename Provider>
    static std::vector<device_entry> enumerate_devices(std::shared_ptr<Provider> provider)
    {
        std::vector<device_entry> entries;
        auto                      filter = get_device_filter<Settings>();

        if(filter.mode == device_selection_mode::NONE)
        {
            LOG_DEBUG("{} sampling disabled via configuration", device_name);
            return entries;
        }

        auto   driver         = provider->get_driver();
        auto   socket_handles = provider->get_socket_handles();
        size_t nic_index      = 0;

        for(auto& socket_handle : socket_handles)
        {
            auto nic_handles = provider->get_processor_handles_by_type(
                socket_handle, AMDSMI_PROCESSOR_TYPE_AMD_NIC);

            for(auto& processor_handle : nic_handles)
            {
                auto nic_device = create_device(driver, processor_handle,
                                                AMDSMI_PROCESSOR_TYPE_AMD_NIC, nic_index);

                if(!is_device_supported(nic_device))
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
                    auto supported = get_supported_metrics(nic_device);
                    entries.push_back(device_entry{ std::move(nic_device), supported });
                }

                nic_index++;
            }
        }

        register_nic_agents(entries);

        return entries;
    }

    static void register_nic_agents(const std::vector<device_entry>& entries)
    {
        size_t nic_index = 0;
        for(const auto& entry : entries)
        {
            agent cur_agent{ agent_type::NIC,
                             0,
                             nic_index,
                             static_cast<uint32_t>(nic_index),
                             static_cast<int32_t>(nic_index),
                             static_cast<int32_t>(nic_index),
                             entry.device->get_product_name().c_str(),
                             entry.device->get_vendor_name().c_str(),
                             "AI NIC",
                             "AI NIC" };

            get_agent_manager_instance().insert_agent(cur_agent);
            nic_index++;
        }
    }
};

}  // namespace rocprofsys::pmc::collectors::nic
