// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/string_utility.hpp"
#include "common/version.hpp"

#include "library/rocprofiler-sdk/buffered/kfd_event_dropped_events.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_event_page_fault.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_event_page_migrate.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_event_queue.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_event_unmap_from_gpu.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_page_fault.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_page_migrate.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_queue.hpp"

#include "library/rocprofiler-sdk/callback/code_object.hpp"

#include "library/rocprofiler-sdk/types.hpp"
#include "policies/rocprofiler-sdk/domain_service.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace rocprofsys::domains
{

template <typename T, std::size_t Capacity>
struct simple_static_vector
{
    std::array<T, Capacity> elements{};
    std::size_t             used = 0;

    constexpr void add(T value) { elements[used++] = value; }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return used; }

    [[nodiscard]] constexpr const T* begin() const noexcept { return elements.data(); }

    [[nodiscard]] constexpr const T* end() const noexcept
    {
        return elements.data() + used;
    }
};

template <policies::rocprofiler_sdk::domain_service_backend   SdkBackend,
          policies::rocprofiler_sdk::domain_service_externals Externals>
struct registry
{
    [[nodiscard]] static const domain_descriptor* find_descriptor(
        std::string_view name) noexcept
    {
        const auto buffered_match = std::ranges::find_if(
            k_buffered_domains_definitions,
            [name](const buffered_domain_definition<SdkBackend>& definition) {
                return rocprofsys::utility::string::equals_ignore_case(
                    name, definition.meta.name);
            });
        if(buffered_match != k_buffered_domains_definitions.end())
        {
            return &buffered_match->meta;
        }

        const auto callback_match = std::ranges::find_if(
            k_callback_domains_definitions,
            [name](const callback_domain_definition<SdkBackend>& definition) {
                return rocprofsys::utility::string::equals_ignore_case(
                    name, definition.meta.name);
            });

        return callback_match != k_callback_domains_definitions.end()
                   ? &callback_match->meta
                   : nullptr;
    }

    [[nodiscard]] static const buffered_domain_definition<SdkBackend>& get_buffered(
        domain_id_t domain_id)
    {
        const auto result = std::ranges::find_if(
            k_buffered_domains_definitions,
            [domain_id](const buffered_domain_definition<SdkBackend>& definition) {
                return definition.meta.id == domain_id;
            });

        if(result == k_buffered_domains_definitions.end())
        {
            throw std::runtime_error{ fmt::format(
                "no buffered definition for domain id {}", domain_id) };
        }
        return *result;
    }

    [[nodiscard]] static const callback_domain_definition<SdkBackend>& get_callback(
        domain_id_t domain_id)
    {
        const auto result = std::ranges::find_if(
            k_callback_domains_definitions,
            [domain_id](const callback_domain_definition<SdkBackend>& definition) {
                return definition.meta.id == domain_id;
            });

        if(result == k_callback_domains_definitions.end())
        {
            throw std::runtime_error{ fmt::format(
                "no callback definition for domain id {}", domain_id) };
        }
        return *result;
    }

private:
    consteval static auto collect_buffered_domains()
    {
        constexpr auto k_buffered_domains_size = 8;
        simple_static_vector<buffered_domain_definition<SdkBackend>,
                             k_buffered_domains_size>
            result;

        if constexpr(version::from_formatted(SdkBackend::compile_time_version) >=
                     version{ .major = 1, .minor = 2, .patch = 2 })
        {
            result.add(buffered::k_kfd_event_dropped_events<SdkBackend, Externals>);
            result.add(buffered::k_kfd_event_page_fault<SdkBackend, Externals>);
            result.add(buffered::k_kfd_event_page_migrate<SdkBackend, Externals>);
            result.add(buffered::k_kfd_event_queue<SdkBackend, Externals>);
            result.add(buffered::k_kfd_event_unmap_from_gpu<SdkBackend, Externals>);
            result.add(buffered::k_kfd_page_fault<SdkBackend, Externals>);
            result.add(buffered::k_kfd_page_migrate<SdkBackend, Externals>);
            result.add(buffered::k_kfd_queue<SdkBackend, Externals>);
        }

        return result;
    }

    consteval static auto collect_callback_domains()
    {
        constexpr auto k_callback_domains_size = 1;
        simple_static_vector<callback_domain_definition<SdkBackend>,
                             k_callback_domains_size>
            result;

        result.add(callback::k_code_object<SdkBackend, Externals>);

        return result;
    }

    constexpr static auto k_buffered_domains_definitions{ collect_buffered_domains() };
    constexpr static auto k_callback_domains_definitions{ collect_callback_domains() };
};

}  // namespace rocprofsys::domains
