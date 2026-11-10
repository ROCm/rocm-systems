// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/delimit.hpp"
#include "common/string_utility.hpp"
#include "common/version.hpp"
#include "logger/debug.hpp"
#include "policies/rocprofiler-sdk/tracing_config/backend.hpp"
#include "policies/rocprofiler-sdk/tracing_config/externals.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <iterator>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocprofsys::rocprofiler_sdk
{

namespace concepts
{

template <typename TracingKind, typename SdkBackend>
concept tracing_kind =
    std::same_as<TracingKind, typename SdkBackend::callback_tracing_kind_t> ||
    std::same_as<TracingKind, typename SdkBackend::buffer_tracing_kind_t>;
}  // namespace concepts

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
class tracing_config
{
public:
    struct operation_options_env_names
    {
        std::string operations_include_env_name;
        std::string operations_exclude_env_name;
        std::string operations_annotate_backtrace_env_name;

        operation_options_env_names() = default;

        // NOLINTBEGIN(misc-include-cleaner) -- fmt::format is provided by the
        // <spdlog/fmt/fmt.h> wrapper above; include-cleaner doesn't see through it.
        explicit operation_options_env_names(const std::string_view domain_name)
        : operations_include_env_name(
              fmt::format("ROCPROFSYS_ROCM_{}_OPERATIONS", domain_name))
        , operations_exclude_env_name(
              fmt::format("ROCPROFSYS_ROCM_{}_OPERATIONS_EXCLUDE", domain_name))
        , operations_annotate_backtrace_env_name(fmt::format(
              "ROCPROFSYS_ROCM_{}_OPERATIONS_ANNOTATE_BACKTRACE", domain_name))
        {}
        // NOLINTEND(misc-include-cleaner)

        [[nodiscard]] bool is_empty() const
        {
            return operations_annotate_backtrace_env_name.empty() &&
                   operations_exclude_env_name.empty() &&
                   operations_include_env_name.empty();
        }
    };
    struct operation_setting_spec
    {
        operation_options_env_names env_names;
        std::vector<std::string>    operation_choices;
    };

    /// @brief One domain the user asked for, as domain_service names it.
    struct domain_request
    {
        /// Domain name (lowercased SDK name, e.g. "hip_runtime_api").
        std::string name;
        /// Operation names to trace; std::nullopt means every operation.
        std::optional<std::vector<std::string>> operations = std::nullopt;
        /// SDK ids of the operations that get a backtrace annotation.
        std::unordered_set<std::int32_t> backtrace_operations;
    };

    static common::version get_version();

    static std::vector<std::string>            get_domain_choices();
    static std::string                         get_domain_defaults();
    static std::vector<operation_setting_spec> get_operation_settings();

    /// @brief Domains selected by ROCPROFSYS_ROCM_DOMAINS (aliases expanded, plus
    /// the implicitly enabled ones), ordered by name. Domains traced outside
    /// domain_service (marker API, legacy page_migration) are omitted.
    /// @throws std::runtime_error on an unsupported domain name.
    static std::vector<domain_request> get_domains();

private:
    using operation_item = std::pair<std::int32_t, std::string>;

    static std::vector<std::string> expand_domain_names();

    /// @brief Alias name -> the domains it stands for.
    static const std::unordered_map<std::string, std::vector<std::string>>&
    get_domain_aliases();

    /// @brief Named (id, name) operations of an SDK table entry, "none" excluded.
    template <typename Entry>
    static std::vector<operation_item> collect_operation_items(const Entry& entry);

    /// @brief Fills the operation selection of request from the SDK table entry named
    /// request.name. @return false when the table has no such entry.
    template <typename Table>
    static bool apply_operation_settings(const Table& table, domain_request& request);

    /// @brief Names of the operations left after the include/exclude settings;
    /// std::nullopt when every operation is left.
    static std::optional<std::vector<std::string>> select_operation_names(
        const operation_options_env_names& env_names,
        const std::vector<operation_item>& items);

    /// @brief Items matching any pattern of the setting; std::nullopt when the
    /// setting holds no pattern.
    static std::optional<std::vector<operation_item>> matching_operations(
        const std::string&                 operations_setting_env_name,
        const std::vector<operation_item>& items);

    [[noreturn]] static void finalize_and_throw(std::string_view exception_message);

    template <concepts::tracing_kind<SdkBackend> TracingKind>
    static operation_options_env_names assemble_operation_env_names_for_kind(
        TracingKind kind);

    static operation_options_env_names assemble_operation_env_names(std::string_view name,
                                                                    bool has_operations);

    static const std::unordered_set<std::string_view>&
    get_domains_to_skip_for_operation_options();

    static const std::unordered_set<std::string_view>&
    get_domains_to_skip_for_domain_choices();

    static constexpr common::version s_compile_time_sdk_version =
        common::version::from_formatted(SdkBackend::compile_time_version);

    /// @brief Named SDK version gates used by if constexpr feature checks below,
    /// in place of raw common::version{...} literals at each call site.
    static constexpr common::version s_sdk_version_1_0_0{ .major = 1,
                                                          .minor = 0,
                                                          .patch = 0 };
    static constexpr common::version s_sdk_version_0_6_0{ .major = 0,
                                                          .minor = 6,
                                                          .patch = 0 };
    static constexpr common::version s_sdk_version_0_7_0{ .major = 0,
                                                          .minor = 7,
                                                          .patch = 0 };
    static constexpr common::version s_sdk_version_1_2_2{ .major = 1,
                                                          .minor = 2,
                                                          .patch = 2 };
};

/// @brief Return the version of the rocprofiler-sdk
template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
common::version
tracing_config<SdkBackend, Externals>::get_version()
{
    auto version = common::version{};
    SdkBackend::get_version(&version.major, &version.minor, &version.patch);
    return version;
}

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
std::vector<std::string>
tracing_config<SdkBackend, Externals>::get_domain_choices()
{
    const auto& buffered_tracing_info = SdkBackend::get_buffer_tracing_names();
    const auto& callback_tracing_info = SdkBackend::get_callback_tracing_names();

    auto choices = std::unordered_set<std::string>{};
    choices.reserve(buffered_tracing_info.size() + callback_tracing_info.size());

    auto const add_domain_f = [&choices](std::string_view domain_to_add) {
        const auto domain_lowercase = utility::string::to_lower(domain_to_add);
        if(get_domains_to_skip_for_domain_choices().contains(domain_lowercase))
        {
            return;
        }

        choices.emplace(domain_lowercase);
    };

    add_domain_f("hip_api");
    add_domain_f("hsa_api");
    add_domain_f("marker_api");
    add_domain_f("roctx");

    if constexpr(s_compile_time_sdk_version >= s_sdk_version_1_2_2)
    {
        add_domain_f("kfd_events");
    }
    const auto name_projection = [](const auto& info) { return info.name; };

    std::ranges::for_each(buffered_tracing_info, add_domain_f, name_projection);
    std::ranges::for_each(callback_tracing_info, add_domain_f, name_projection);

    std::vector<std::string> choices_vec{ choices.begin(), choices.end() };
    std::ranges::sort(choices_vec);

    return choices_vec;
}

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
std::string
tracing_config<SdkBackend, Externals>::get_domain_defaults()
{
    auto defaults = std::string{ "hip_runtime_api,marker_api,kernel_dispatch,"
                                 "memory_copy,scratch_memory" };

    if constexpr(s_compile_time_sdk_version < s_sdk_version_1_0_0)
    {
        defaults.append(",page_migration");
    }

    return defaults;
}

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
std::vector<typename tracing_config<SdkBackend, Externals>::operation_setting_spec>
tracing_config<SdkBackend, Externals>::get_operation_settings()
{
    const auto& buffered_tracing_info = SdkBackend::get_buffer_tracing_names();
    const auto& callback_tracing_info = SdkBackend::get_callback_tracing_names();

    auto result = std::vector<operation_setting_spec>{};

    auto const gather_domain_f = [&result](auto kind, const auto& domain_operations) {
        const auto env_names = assemble_operation_env_names_for_kind(kind);
        if(env_names.is_empty())
        {
            return;
        }

        result.push_back(
            operation_setting_spec{ .env_names         = env_names,
                                    .operation_choices = { domain_operations.begin(),
                                                           domain_operations.end() } });
    };

    for(const auto& itr : callback_tracing_info)
    {
        gather_domain_f(itr.value, itr.operations);
    }

    for(const auto& itr : buffered_tracing_info)
    {
        gather_domain_f(itr.value, itr.operations);
    }

    return result;
}

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
void
tracing_config<SdkBackend, Externals>::finalize_and_throw(
    std::string_view exception_message)
{
    Externals::ProcessState::set(Externals::ProcessState::Finalized);
    throw std::runtime_error(std::string{ exception_message });
}

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
template <concepts::tracing_kind<SdkBackend> TracingKind>
tracing_config<SdkBackend, Externals>::operation_options_env_names
tracing_config<SdkBackend, Externals>::assemble_operation_env_names_for_kind(
    TracingKind kind)
{
    if constexpr(std::same_as<TracingKind, typename SdkBackend::callback_tracing_kind_t>)
    {
        const auto& entry = SdkBackend::get_callback_tracing_names()[kind];
        return assemble_operation_env_names(
            (kind == SdkBackend::CALLBACK_TRACING_MARKER_CORE_API)
                ? std::string_view{ "MARKER_API" }
                : std::string_view{ entry.name },
            !entry.operations.empty());
    }
    else
    {
        const auto& entry = SdkBackend::get_buffer_tracing_names()[kind];
        return assemble_operation_env_names(entry.name, !entry.operations.empty());
    }
}

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
tracing_config<SdkBackend, Externals>::operation_options_env_names
tracing_config<SdkBackend, Externals>::assemble_operation_env_names(std::string_view name,
                                                                    bool has_operations)
{
    if(!has_operations || get_domains_to_skip_for_operation_options().contains(
                              utility::string::to_lower(name)))
    {
        return {};
    }

    return operation_options_env_names{ name };
}

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
std::vector<typename tracing_config<SdkBackend, Externals>::domain_request>
tracing_config<SdkBackend, Externals>::get_domains()
{
    auto requests = std::vector<domain_request>{};

    for(auto& name : expand_domain_names())
    {
        // Pre-1.0 SDK domain that domain_service has no definition for.
        if(name == "page_migration")
        {
            continue;
        }

        auto request =
            domain_request{ .name = std::move(name), .backtrace_operations = {} };
        if(!apply_operation_settings(SdkBackend::get_callback_tracing_names(), request))
        {
            apply_operation_settings(SdkBackend::get_buffer_tracing_names(), request);
        }
        requests.push_back(std::move(request));
    }

    return requests;
}

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
std::vector<std::string>
tracing_config<SdkBackend, Externals>::expand_domain_names()
{
    const auto user_domains =
        rocprofsys::delimit(Externals::get_rocm_domains(), " ,;:\t\n");
    const auto  valid_choices = get_domain_choices();
    const auto& aliases       = get_domain_aliases();

    auto names = std::set<std::string>{};
    for(const auto& domain : user_domains)
    {
        if(std::ranges::find(valid_choices, domain) == valid_choices.end())
        {
            throw std::runtime_error(
                fmt::format("unsupported ROCPROFSYS_ROCM_DOMAINS value: {}", domain));
        }

        if(const auto alias = aliases.find(domain); alias != aliases.end())
        {
            names.insert(alias->second.begin(), alias->second.end());
        }
        else
        {
            names.insert(domain);
        }
    }

    if constexpr(s_compile_time_sdk_version >= s_sdk_version_0_6_0)
    {
        if(Externals::get_use_rcclp())
        {
            names.emplace("rccl_api");
        }
        if(Externals::get_use_ompt())
        {
            names.emplace("ompt");
        }
    }

    if constexpr(s_compile_time_sdk_version >= s_sdk_version_1_2_2)
    {
        if(Externals::get_use_unified_memory_profiling())
        {
            LOG_INFO("ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON: implicitly enabling "
                     "KFD page_fault and page_migrate domains.");
            names.emplace("kfd_page_fault");
            names.emplace("kfd_page_migrate");
        }
    }

    // hip_stream has no user-facing switch: it is derived from the domains whose
    // records it correlates with streams.
    if constexpr(s_compile_time_sdk_version >= s_sdk_version_0_7_0)
    {
        if(names.contains("kernel_dispatch") || names.contains("memory_copy"))
        {
            names.emplace("hip_stream");
        }
    }

    return { names.begin(), names.end() };
}

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
const std::unordered_map<std::string, std::vector<std::string>>&
tracing_config<SdkBackend, Externals>::get_domain_aliases()
{
    static const auto s_aliases = [] {
        // marker_api and roctx expand to nothing: roctx_client traces them on its own
        // context, outside the domain service.
        auto aliases = std::unordered_map<std::string, std::vector<std::string>>{
            { "hsa_api",
              { "hsa_core_api", "hsa_amd_ext_api", "hsa_image_ext_api",
                "hsa_finalize_ext_api" } },
            { "hip_api", { "hip_runtime_api", "hip_compiler_api" } },
            { "marker_api", {} },
            { "roctx", {} },
        };

        if constexpr(s_compile_time_sdk_version >= s_sdk_version_1_2_2)
        {
            aliases.emplace("kfd_events",
                            std::vector<std::string>{
                                "kfd_page_fault", "kfd_page_migrate", "kfd_queue",
                                "kfd_event_page_fault", "kfd_event_page_migrate",
                                "kfd_event_queue", "kfd_event_unmap_from_gpu",
                                "kfd_event_dropped_events" });
        }
        return aliases;
    }();
    return s_aliases;
}

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
template <typename Entry>
std::vector<typename tracing_config<SdkBackend, Externals>::operation_item>
tracing_config<SdkBackend, Externals>::collect_operation_items(const Entry& entry)
{
    auto items = std::vector<operation_item>{};
    for(const auto& item : entry.items())
    {
        if(item.second && *item.second != "none")
        {
            items.emplace_back(item.first, std::string{ *item.second });
        }
    }
    return items;
}

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
template <typename Table>
bool
tracing_config<SdkBackend, Externals>::apply_operation_settings(const Table&    table,
                                                                domain_request& request)
{
    for(const auto& entry : table)
    {
        if(utility::string::to_lower(entry.name) != request.name)
        {
            continue;
        }

        const auto env_names =
            assemble_operation_env_names(entry.name, !entry.operations.empty());
        if(env_names.is_empty())
        {
            return true;
        }

        const auto items = collect_operation_items(entry);

        if(const auto annotated = matching_operations(
               env_names.operations_annotate_backtrace_env_name, items))
        {
            std::ranges::transform(*annotated,
                                   std::inserter(request.backtrace_operations,
                                                 request.backtrace_operations.end()),
                                   [](const auto& item) { return item.first; });
        }

        request.operations = select_operation_names(env_names, items);
        return true;
    }
    return false;
}

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
std::optional<std::vector<std::string>>
tracing_config<SdkBackend, Externals>::select_operation_names(
    const operation_options_env_names& env_names,
    const std::vector<operation_item>& items)
{
    auto selected = items;
    if(const auto included =
           matching_operations(env_names.operations_include_env_name, items))
    {
        selected = *included;
    }
    if(const auto excluded =
           matching_operations(env_names.operations_exclude_env_name, items))
    {
        std::erase_if(selected, [&excluded](const auto& item) {
            return std::ranges::find(*excluded, item) != excluded->end();
        });
    }

    // Everything selected means no narrowing: leave operations unset so the domain
    // is traced in full.
    if(selected.size() == items.size())
    {
        return std::nullopt;
    }

    std::ranges::sort(selected);
    auto names = std::vector<std::string>{};
    names.reserve(selected.size());
    for(auto& item : selected)
    {
        names.push_back(std::move(item.second));
    }
    return names;
}

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
std::optional<std::vector<typename tracing_config<SdkBackend, Externals>::operation_item>>
tracing_config<SdkBackend, Externals>::matching_operations(
    const std::string&                 operations_setting_env_name,
    const std::vector<operation_item>& items)
{
    const auto setting_value = Externals::get_setting_value(operations_setting_env_name);
    // NOLINTBEGIN(misc-include-cleaner) -- see operation_options_env_names ctor
    if(!setting_value)
    {
        finalize_and_throw(
            fmt::format("tracing_config::matching_operations: no registered setting '{}'",
                        operations_setting_env_name));
    }
    // NOLINTEND(misc-include-cleaner)

    const auto pattern_texts = rocprofsys::delimit(*setting_value, " ,;:\n\t");
    if(pattern_texts.empty())
    {
        return std::nullopt;
    }

    auto patterns = std::vector<std::regex>{};
    for(const auto& text : pattern_texts)
    {
        patterns.emplace_back(text, std::regex_constants::icase);
    }

    auto matched = std::vector<operation_item>{};
    for(const auto& item : items)
    {
        if(std::ranges::any_of(patterns, [&](const auto& pattern) {
               return std::regex_search(item.second, pattern);
           }))
        {
            LOG_DEBUG("{} matched: {}", operations_setting_env_name, item.second);
            matched.push_back(item);
        }
    }
    return matched;
}

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
const std::unordered_set<std::string_view>&
tracing_config<SdkBackend, Externals>::get_domains_to_skip_for_operation_options()
{
    static const auto s_domains = std::unordered_set<std::string_view>{
        "none",
        "correlation_id_retirement",
        "marker_core_api",
        "marker_control_api",
        "marker_name_api",
        "code_object",
        "kernel_dispatch",
        "page_migration",
    };
    return s_domains;
}

template <policies::tracing_config::backend   SdkBackend,
          policies::tracing_config::externals Externals>
const std::unordered_set<std::string_view>&
tracing_config<SdkBackend, Externals>::get_domains_to_skip_for_domain_choices()
{
    static const auto s_domains = std::unordered_set<std::string_view>{
        "none",
        "correlation_id_retirement",
        "marker_core_api",
        "marker_control_api",
        "marker_name_api",
        "code_object",
    };
    return s_domains;
}
}  // namespace rocprofsys::rocprofiler_sdk
