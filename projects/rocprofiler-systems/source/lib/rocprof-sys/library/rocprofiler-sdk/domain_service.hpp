// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/string_utility.hpp"
#include "library/rocprofiler-sdk/buffered_domain.hpp"
#include "library/rocprofiler-sdk/callback_domain.hpp"
#include "library/rocprofiler-sdk/domain_registry.hpp"
#include "library/rocprofiler-sdk/domain_selection.hpp"
#include "library/rocprofiler-sdk/types.hpp"
#include "logger/debug.hpp"
#include "policies/rocprofiler-sdk/domain_service.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocprofsys
{

template <policies::rocprofiler_sdk::domain_service_backend   SdkBackend,
          policies::rocprofiler_sdk::domain_service_externals Externals>
class domain_service
{
public:
    explicit domain_service()
    {
        auto callback_domains = filter_supported_domains(
            SdkBackend::get_callback_tracing_names(), domains::collection_mode::callback);
        auto buffered_domains = filter_supported_domains(
            SdkBackend::get_buffer_tracing_names(), domains::collection_mode::buffered);

        m_available_domains = std::move(buffered_domains);
        m_available_domains.insert(m_available_domains.end(),
                                   std::make_move_iterator(callback_domains.begin()),
                                   std::make_move_iterator(callback_domains.end()));

        LOG_DEBUG("SDK reports {} available domains", m_available_domains.size());
        for(const auto& domain : m_available_domains)
        {
            LOG_DEBUG("Available domain: {}", domain);
        }
    }

    [[nodiscard]] std::span<const domains::domain_info> available_domains() const noexcept
    {
        return m_available_domains;
    }

    void configure(std::span<const domain_selection> selections)
    {
        LOG_DEBUG("Configuring {} domain selection(s)", selections.size());

        m_configuration = resolve_configuration(selections);
        LOG_DEBUG("Resolved {} domain configuration(s)", m_configuration.size());

        m_buffered_domains.reserve(m_configuration.size());
        m_callback_domains.reserve(m_configuration.size());
        for(const auto& config : m_configuration)
        {
            configure_domain(config);
        }

        SdkBackend::start_context(context());
    }

    void flush() const
    {
        LOG_DEBUG("Flushing {} buffered domain(s)", m_buffered_domains.size());
        for(const auto& domain : m_buffered_domains)
        {
            domain.flush();
        }
    }

    [[nodiscard]] std::span<const domains::domain_configuration> configuration()
        const noexcept
    {
        return m_configuration;
    }

private:
    [[nodiscard]] std::vector<domains::domain_configuration> resolve_configuration(
        std::span<const domain_selection> selections) const
    {
        std::vector<domains::domain_configuration> resolved;

        for(const auto& selection : selections)
        {
            for(const auto* domain : match_domains(m_available_domains, selection))
            {
                merge_domain(resolved, *domain,
                             resolve_operations(*domain, selection.operations));
            }
        }

        return resolved;
    }

    void configure_domain(const domains::domain_configuration& domain)
    {
        std::vector<typename SdkBackend::tracing_operation_t> operations;
        operations.reserve(domain.operations.size());
        for(const auto operation : domain.operations)
        {
            operations.push_back(static_cast<SdkBackend::tracing_operation_t>(operation));
        }

        switch(domain.key.mode)
        {
            case domains::collection_mode::buffered:
                configure_buffered(domain, std::move(operations));
                break;
            case domains::collection_mode::callback:
                configure_callback(domain, std::move(operations));
                break;
            default:
                throw std::runtime_error{ fmt::format(
                    "unsupported collection mode: {}",
                    static_cast<std::underlying_type_t<domains::collection_mode>>(
                        domain.key.mode)) };
        }
    }

    void configure_buffered(
        const domains::domain_configuration&                  domain,
        std::vector<typename SdkBackend::tracing_operation_t> operations)
    {
        const auto& definition =
            domains::registry<SdkBackend, Externals>::get_buffered(domain.key.value);

        LOG_DEBUG("Configuring buffered domain '{}' ({} operation(s))",
                  definition.meta.name, operations.size());

        m_buffered_domains.emplace_back(definition, context(), std::move(operations));
        m_buffered_domains.back().configure();

        if(definition.on_configure)
        {
            definition.on_configure();
        }
    }

    void configure_callback(
        const domains::domain_configuration&                  domain,
        std::vector<typename SdkBackend::tracing_operation_t> operations)
    {
        const auto& definition =
            domains::registry<SdkBackend, Externals>::get_callback(domain.key.value);

        LOG_DEBUG("Configuring callback domain '{}' ({} operation(s))",
                  definition.meta.name, operations.size());

        m_callback_domains.emplace_back(definition, context(), std::move(operations));
        m_callback_domains.back().configure();

        if(definition.on_configure)
        {
            definition.on_configure();
        }
    }

    SdkBackend::context_id_t context()
    {
        if(m_context.handle != 0)
        {
            return m_context;
        }

        LOG_DEBUG("Creating new SDK context");
        SdkBackend::create_context(&m_context);

        return m_context;
    }

private:
    std::vector<domains::domain_info>                 m_available_domains;
    std::vector<domains::domain_configuration>        m_configuration;
    std::vector<domains::buffered_domain<SdkBackend>> m_buffered_domains;
    std::vector<domains::callback_domain<SdkBackend>> m_callback_domains;
    SdkBackend::context_id_t                          m_context{};

    std::vector<domains::domain_info> filter_supported_domains(
        const auto& table, domains::collection_mode mode)
    {
        std::vector<domains::domain_info> output{};
        for(const auto& entry : table)
        {
            const auto domain_dsc =
                domains::registry<SdkBackend, Externals>::find_descriptor(entry.name);
            if(domain_dsc == nullptr || domain_dsc->mode != mode)
            {
                continue;
            }

            std::vector<domains::operation_info> operations;
            operations.reserve(entry.operations.size());

            for(std::size_t index = 0; index < entry.operations.size(); index++)
            {
                operations.push_back(
                    { .id = index, .name = std::string(entry.operations[index]) });
            }

            std::optional<std::string_view> group;
            if(domain_dsc->group.has_value())
            {
                group = domain_dsc->group->name;
            }

            output.push_back(domains::domain_info{
                .key        = { .mode = domain_dsc->mode, .value = entry.value },
                .name       = domain_dsc->name,
                .operations = std::move(operations),
                .group      = group });
        }
        return output;
    }

    void validate(const domain_selection& selection) const
    {
        if(selection.name.has_value() && selection.group.has_value())
        {
            throw std::runtime_error{ fmt::format(
                "selection sets both name '{}' and group '{}'; use one or the other",
                *selection.name, *selection.group) };
        }

        if(selection.operations.has_value() && !selection.name.has_value())
        {
            throw std::runtime_error{ "selection sets operations without a domain name" };
        }
    }

    [[nodiscard]] std::vector<const domains::domain_info*> match_by_name(
        std::span<const domains::domain_info> available, std::string_view name) const
    {
        const auto found =
            std::ranges::find_if(available, [name](const domains::domain_info& domain) {
                return rocprofsys::utility::string::equals_ignore_case(domain.name, name);
            });
        if(found == available.end())
        {
            throw std::runtime_error{ fmt::format("unknown domain '{}'", name) };
        }
        return { &*found };
    }

    [[nodiscard]] std::vector<const domains::domain_info*> match_by_group(
        std::span<const domains::domain_info> available, std::string_view group) const
    {
        std::vector<const domains::domain_info*> matched;
        for(const auto& domain : available)
        {
            if(domain.group.has_value() &&
               rocprofsys::utility::string::equals_ignore_case(*domain.group, group))
            {
                matched.push_back(&domain);
            }
        }
        if(matched.empty())
        {
            throw std::runtime_error{ fmt::format("unknown domain group '{}'", group) };
        }
        return matched;
    }

    [[nodiscard]] std::vector<const domains::domain_info*> match_domains(
        std::span<const domains::domain_info> available,
        const domain_selection&               selection) const
    {
        validate(selection);

        if(selection.name.has_value())
        {
            return match_by_name(available, *selection.name);
        }
        if(selection.group.has_value())
        {
            return match_by_group(available, *selection.group);
        }

        std::vector<const domains::domain_info*> matched;
        matched.reserve(available.size());
        for(const auto& domain : available)
        {
            matched.push_back(&domain);
        }
        return matched;
    }

    [[nodiscard]] std::vector<domains::operation_id_t> resolve_operations(
        const domains::domain_info&                    domain,
        const std::optional<std::vector<std::string>>& requested) const
    {
        std::vector<domains::operation_id_t> resolved;

        if(!requested.has_value())
        {
            resolved.reserve(domain.operations.size());
            for(const auto& operation : domain.operations)
            {
                resolved.push_back(operation.id);
            }
            return resolved;
        }

        resolved.reserve(requested->size());
        for(const auto& name : *requested)
        {
            const auto found = std::ranges::find_if(
                domain.operations, [&](const domains::operation_info& operation) {
                    return rocprofsys::utility::string::equals_ignore_case(operation.name,
                                                                           name);
                });
            if(found == domain.operations.end())
            {
                throw std::runtime_error{ fmt::format(
                    "unknown operation '{}' in domain '{}'", name, domain.name) };
            }
            resolved.push_back(found->id);
        }
        return resolved;
    }

    void merge_domain(std::vector<domains::domain_configuration>& resolved,
                      const domains::domain_info&                 domain,
                      std::vector<domains::operation_id_t>        operations) const
    {
        const auto existing =
            std::ranges::find(resolved, domain.key, &domains::domain_configuration::key);
        if(existing == resolved.end())
        {
            resolved.push_back(
                { .key = domain.key, .operations = std::move(operations) });
            return;
        }

        for(const auto operation : operations)
        {
            if(std::ranges::find(existing->operations, operation) ==
               existing->operations.end())
            {
                existing->operations.push_back(operation);
            }
        }
    }
};

}  // namespace rocprofsys
