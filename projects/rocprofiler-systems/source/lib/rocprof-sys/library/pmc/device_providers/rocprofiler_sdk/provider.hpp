// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/backend.hpp"
#include "core/agent.hpp"
#include "library/pmc/collectors/gpu_perf_counter/counter_definitions.hpp"
#include "library/pmc/collectors/gpu_perf_counter/derived_expression.hpp"
#include "library/pmc/collectors/gpu_perf_counter/device.hpp"
#include "library/pmc/collectors/gpu_perf_counter/expression_tree.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"
#include "policies/rocprofiler-sdk/gpu_perf_counters/backend.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::device_providers::rocprofiler_sdk
{

// Contract required of the factory type passed to provider<BackendFactory>: it must
// produce a backend satisfying policies::gpu_perf_counters::backend.
template <typename BackendFactory>
concept backend_factory_contract = requires {
    typename BackendFactory::backend_t;
    {
        BackendFactory::create_backend()
    } -> std::same_as<std::shared_ptr<typename BackendFactory::backend_t>>;
} && policies::gpu_perf_counters::backend<typename BackendFactory::backend_t>;

template <backend_factory_contract BackendFactory>
class provider
{
public:
    using backend_t = BackendFactory::backend_t;
    using device_t  = collectors::gpu_perf_counter::device<backend_t>;

    provider(const std::vector<std::shared_ptr<agent>>&           agent_list,
             const collectors::gpu_perf_counter::enabled_metrics& enabled)
    : m_backend_api{ BackendFactory::create_backend() }
    {
        configure_agents(agent_list, enabled);
    }

    // &m_profile_configs is passed as void* user_data to the SDK callback.
    // Moving this object after configure_agents() would relocate the map and
    // dangle that pointer. Delete copy/move operations to make this constraint
    // a compile-time guarantee rather than a comment.
    provider(const provider&)            = delete;
    provider& operator=(const provider&) = delete;
    provider(provider&&)                 = delete;
    provider& operator=(provider&&)      = delete;
    ~provider()                          = default;

    void start()
    {
        for(const auto& device : m_devices)
        {
            device->start();
        }
    }

    void stop()
    {
        for(const auto& device : m_devices)
        {
            device->stop();
        }
    }

    void shutdown() { stop(); }

    template <typename Device>
    [[nodiscard]] std::vector<std::shared_ptr<Device>> get_devices(device_type type)
    {
        return type != device_type::gpu
                   ? std::vector<std::shared_ptr<Device>>{}
                   : std::vector<std::shared_ptr<Device>>{ m_devices.begin(),
                                                           m_devices.end() };
    }

private:
    void configure_agents(const std::vector<std::shared_ptr<agent>>&           agent_list,
                          const collectors::gpu_perf_counter::enabled_metrics& enabled)
    {
        for(const auto& gpu_agent : agent_list)
        {
            const auto agent_id = backend_t::make_agent_id(gpu_agent->handle);

            auto const supported_ids = query_supported_counters(agent_id);
            LOG_INFO("Agent {} (device {}): {} supported counters", gpu_agent->name,
                     gpu_agent->device_id, supported_ids.size());
            if(supported_ids.empty())
            {
                continue;
            }

            auto [config_ids, counters] =
                resolve_counters(supported_ids, enabled, gpu_agent->device_type_index);
            LOG_INFO("Agent {}: {} counters after filtering", gpu_agent->handle,
                     config_ids.size());
            if(config_ids.empty())
            {
                continue;
            }

            auto profile = typename backend_t::counter_config_id_t{};
            auto status  = m_backend_api->create_counter_config(
                agent_id, config_ids.data(), config_ids.size(), &profile);
            if(status != backend_t::status_success)
            {
                LOG_WARNING("Failed to create profile config for agent {} (status={})",
                            gpu_agent->handle, static_cast<int>(status));
                continue;
            }

            m_profile_configs[gpu_agent->handle] = profile;

            typename backend_t::context_id_t counter_context{};
            try
            {
                m_backend_api->create_context(&counter_context);
            } catch(const std::exception& e)
            {
                LOG_WARNING("Failed to create context for agent {} ({})",
                            gpu_agent->handle, e.what());
                continue;
            }

            status = m_backend_api->configure_device_counting_service(
                counter_context, typename backend_t::buffer_id_t{ 0 }, agent_id,
                [](backend_t::context_id_t ctx, backend_t::agent_id_t agent_cb,
                   backend_t::device_counting_agent_cb_t set_config, void* user_data) {
                    auto* configs = static_cast<std::unordered_map<
                        std::uint64_t, typename backend_t::counter_config_id_t>*>(
                        user_data);
                    auto const iter = configs->find(agent_cb.handle);
                    if(iter != configs->end())
                    {
                        set_config(ctx, iter->second);
                    }
                },
                &m_profile_configs);
            if(status != backend_t::status_success)
            {
                LOG_WARNING(
                    "Failed to configure device counting for agent {} (status={})",
                    gpu_agent->handle, static_cast<int>(status));
                continue;
            }

            m_devices.push_back(std::make_shared<device_t>(
                m_backend_api, counter_context, gpu_agent, std::move(counters)));
        }
    }

    [[nodiscard]] std::vector<typename backend_t::counter_id_t> query_supported_counters(
        backend_t::agent_id_t agent_id) const
    {
        const auto collect_counters =
            [](backend_t::agent_id_t, backend_t::counter_id_t* counters,
               size_t num_counters, void* user_data) -> typename backend_t::status_t {
            auto* out =
                static_cast<std::vector<typename backend_t::counter_id_t>*>(user_data);
            out->insert(out->end(), counters, counters + num_counters);
            return backend_t::status_success;
        };

        auto       result = std::vector<typename backend_t::counter_id_t>{};
        const auto status = m_backend_api->iterate_agent_supported_counters(
            agent_id, collect_counters, &result);
        if(status != backend_t::status_success)
        {
            LOG_DEBUG("No counters found for agent {} (status={})", agent_id.handle,
                      static_cast<int>(status));
            return {};
        }
        return result;
    }

    // One SDK counter and the metadata of its dimension instances.
    struct catalog_entry
    {
        backend_t::counter_id_t                                     counter_id;
        std::vector<collectors::gpu_perf_counter::counter_metadata> details;
    };

    // Resolves the counter names used in derived counter expressions.
    struct catalog_lookup
    {
        collectors::gpu_perf_counter::counter_definitions     definitions;
        std::unordered_map<std::string, const catalog_entry*> entries;
    };

    // The SDK counter config and what the device reports and reads.
    struct resolved_counters
    {
        std::vector<typename backend_t::counter_id_t> config_ids;
        collectors::gpu_perf_counter::device_counters counters;
    };

    [[nodiscard]] std::vector<catalog_entry> query_catalog(
        const std::vector<typename backend_t::counter_id_t>& supported) const
    {
        auto catalog = std::vector<catalog_entry>{};
        catalog.reserve(supported.size());
        for(const auto& counter_id : supported)
        {
            auto details = m_backend_api->query_counter_details(counter_id);
            if(!details.empty())
            {
                catalog.push_back({ counter_id, std::move(details) });
            }
        }
        return catalog;
    }

    [[nodiscard]] resolved_counters resolve_counters(
        const std::vector<typename backend_t::counter_id_t>& supported,
        const collectors::gpu_perf_counter::enabled_metrics& enabled, size_t device_index)
    {
        auto const catalog = query_catalog(supported);
        auto all_details = std::vector<collectors::gpu_perf_counter::counter_metadata>{};
        auto lookup      = catalog_lookup{};
        for(const auto& entry : catalog)
        {
            all_details.insert(all_details.end(), entry.details.begin(),
                               entry.details.end());
            lookup.entries.try_emplace(entry.details.front().name, &entry);
        }
        lookup.definitions =
            collectors::gpu_perf_counter::counter_definitions{ all_details };

        auto result = resolved_counters{};
        for(const auto& entry : catalog)
        {
            if(enabled.is_counter_enabled({ entry.details.front().name, device_index }))
            {
                add_requested(entry, lookup, device_index, result);
            }
        }

        std::ranges::sort(result.config_ids, {}, &backend_t::counter_id_t::handle);
        auto const duplicates =
            std::ranges::unique(result.config_ids, {}, &backend_t::counter_id_t::handle);
        result.config_ids.erase(duplicates.begin(), duplicates.end());
        return result;
    }

    // A derived counter that is non-linear in totals since start_context is evaluated
    // by the device on its inputs, so the SDK collects the inputs instead of it.
    static void add_requested(const catalog_entry& entry, const catalog_lookup& lookup,
                              size_t device_index, resolved_counters& result)
    {
        const auto& meta        = entry.details.front();
        auto&       counters    = result.counters;
        auto        is_interval = false;
        if(meta.is_derived)
        {
            try
            {
                collectors::gpu_perf_counter::derived_expression expression =
                    collectors::gpu_perf_counter::expression_parser::parse(
                        meta.expression, lookup.definitions, counters.slots);
                is_interval = expression.evaluation ==
                              collectors::gpu_perf_counter::derived_evaluation::interval;
                counters.expressions.insert_or_assign(meta.name, std::move(expression));
            } catch(const std::invalid_argument& e)
            {
                LOG_ERROR("GPU perf counter {} on device {} is not collected: {}. "
                          "ROCPROFSYS_ROCM_EVENTS collects it per kernel.",
                          meta.name, device_index, e.what());
                return;
            }
        }

        counters.requested.insert(counters.requested.end(), entry.details.begin(),
                                  entry.details.end());
        if(!is_interval)
        {
            result.config_ids.push_back(entry.counter_id);
            return;
        }

        for(const auto& name : counters.expressions.at(meta.name).referenced_names)
        {
            const auto* input = lookup.entries.at(name);
            if(!input->details.front().is_derived)
            {
                counters.inputs.insert(counters.inputs.end(), input->details.begin(),
                                       input->details.end());
                result.config_ids.push_back(input->counter_id);
            }
        }
    }

    std::shared_ptr<backend_t>             m_backend_api;
    std::vector<std::shared_ptr<device_t>> m_devices;
    // Must outlive the context: the SDK callback set via
    // configure_device_counting_service fires during start_context(), reading from this
    // map via the user_data pointer.
    std::unordered_map<std::uint64_t, typename backend_t::counter_config_id_t>
        m_profile_configs;
};

}  // namespace rocprofsys::pmc::device_providers::rocprofiler_sdk
