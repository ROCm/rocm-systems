// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/string_utility.hpp"

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

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string_view>

namespace rocprofsys::domains
{

template <typename SdkBackend, typename Externals>
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
    constexpr static std::array k_buffered_domains_definitions{
        buffered::k_kfd_event_dropped_events<SdkBackend, Externals>,
        buffered::k_kfd_event_page_fault<SdkBackend, Externals>,
        buffered::k_kfd_event_page_migrate<SdkBackend, Externals>,
        buffered::k_kfd_event_queue<SdkBackend, Externals>,
        buffered::k_kfd_event_unmap_from_gpu<SdkBackend, Externals>,
        buffered::k_kfd_page_fault<SdkBackend, Externals>,
        buffered::k_kfd_page_migrate<SdkBackend, Externals>,
        buffered::k_kfd_queue<SdkBackend, Externals>,
    };

    constexpr static std::array k_callback_domains_definitions{
        callback::k_code_object<SdkBackend, Externals>
    };
};

}  // namespace rocprofsys::domains
