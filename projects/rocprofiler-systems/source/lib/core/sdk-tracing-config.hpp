// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/delimit.hpp"
#include "logger/debug.hpp"

#include <cctype>
#include <cstddef>
#include <iterator>
#include <ranges>

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocprofsys::rocprofiler_sdk
{

struct version_info
{
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
    std::uint32_t patch = 0;

    [[nodiscard]] auto formatted() const
    {
        constexpr auto major_multiplier = 10000U;
        constexpr auto minor_multiplier = 100U;
        return (major * major_multiplier) + (minor * minor_multiplier) + patch;
    }

    [[nodiscard]] static constexpr version_info from_formatted(std::uint32_t formatted)
    {
        constexpr auto major_multiplier          = 10000u;
        constexpr auto minor_multiplier          = 100u;
        constexpr auto version_component_modulus = 100u;  // keep 2 digits
        return version_info{ .major = formatted / major_multiplier,
                             .minor = (formatted / minor_multiplier) %
                                      version_component_modulus,
                             .patch = formatted % version_component_modulus };
    }

    constexpr auto operator<=>(const version_info&) const = default;
};

namespace concepts
{

template <typename SdkBackend, typename TracingKind>
concept tracing_kind_for =
    std::same_as<TracingKind, typename SdkBackend::callback_tracing_kind_t> ||
    std::same_as<TracingKind, typename SdkBackend::buffer_tracing_kind_t>;

template <typename Externals>
concept sdk_tracing_config_externals = requires(std::string_view setting_name) {
    typename Externals::ProcessState;
    typename Externals::ProcessState::State;
    {
        Externals::ProcessState::Finalized
    } -> std::convertible_to<typename Externals::ProcessState::State>;
    { Externals::ProcessState::set(Externals::ProcessState::Finalized) };
    { Externals::get_use_rcclp() } -> std::convertible_to<bool>;
    { Externals::get_use_ompt() } -> std::convertible_to<bool>;
    { Externals::get_use_unified_memory_profiling() } -> std::convertible_to<bool>;
    { Externals::get_rocm_domains() } -> std::convertible_to<std::string>;
    {
        Externals::get_setting_value(setting_name)
    } -> std::same_as<std::optional<std::string>>;
};
}  // namespace concepts

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
class sdk_tracing_config
{
public:
    struct operation_options_env_names
    {
        std::string operations_include_env_name            = {};
        std::string operations_exclude_env_name            = {};
        std::string operations_annotate_backtrace_env_name = {};

        operation_options_env_names() = default;

        explicit operation_options_env_names(const std::string_view domain_name)
        : operations_include_env_name(
              fmt::format("ROCPROFSYS_ROCM_{}_OPERATIONS", domain_name))
        , operations_exclude_env_name(
              fmt::format("ROCPROFSYS_ROCM_{}_OPERATIONS_EXCLUDE", domain_name))
        , operations_annotate_backtrace_env_name(fmt::format(
              "ROCPROFSYS_ROCM_{}_OPERATIONS_ANNOTATE_BACKTRACE", domain_name))
        {}

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

    static version_info get_version();

    static std::vector<std::string>            domain_choices();
    static std::string                         domain_defaults();
    static std::vector<operation_setting_spec> operation_settings();

    static std::unordered_set<typename SdkBackend::callback_tracing_kind_t>
    get_callback_domains();
    static std::unordered_set<typename SdkBackend::buffer_tracing_kind_t>
    get_buffered_domains();

    template <typename TracingKind>
        requires concepts::tracing_kind_for<SdkBackend, TracingKind>
    static std::vector<std::int32_t> get_operations(TracingKind kindv);
    template <typename TracingKind>
        requires concepts::tracing_kind_for<SdkBackend, TracingKind>
    static std::unordered_set<std::int32_t> get_backtrace_operations(TracingKind kindv);

private:
    template <typename Tp>
    static std::string to_lower(const Tp& val);

    template <typename TracingKind>
        requires concepts::tracing_kind_for<SdkBackend, TracingKind>
    static std::vector<std::pair<std::int32_t, std::string>> all_operation_items_for_kind(
        TracingKind tracing_kind);

    static std::vector<std::pair<std::string, std::regex>> compile_operation_patterns(
        const std::string& operations_setting_env_name);

    static bool matches_any_operation_pattern(
        std::string_view operations_setting_env_name, const std::string& operation_name,
        const std::vector<std::pair<std::string, std::regex>>& patterns);

    [[noreturn]] static void finalize_and_throw(std::string_view exception_message);

    template <typename TracingKind>
        requires concepts::tracing_kind_for<SdkBackend, TracingKind>
    static operation_options_env_names assemble_operation_env_names_for_kind(
        TracingKind kind);

    static constexpr std::unordered_set<typename SdkBackend::callback_tracing_kind_t>
    get_supported_callback_domains();

    static constexpr std::unordered_set<typename SdkBackend::buffer_tracing_kind_t>
    get_supported_buffer_domains();

    /// @brief Domain-name -> kind(s) lookup table for get_callback_domains(),
    /// covering the alias names plus every compile-time-supported callback domain.
    static std::unordered_map<std::string,
                              std::vector<typename SdkBackend::callback_tracing_kind_t>>
    get_callback_domain_map();

    /// @brief Domain-name -> kind(s) lookup table for get_buffered_domains(),
    /// covering the alias names plus every buffered domain.
    static std::unordered_map<std::string,
                              std::vector<typename SdkBackend::buffer_tracing_kind_t>>
    get_buffered_domain_map();

    static constexpr version_info s_compile_time_sdk_version =
        version_info::from_formatted(SdkBackend::compile_time_version);

    const static std::unordered_set<std::string_view>
        s_domains_to_skip_for_operation_options;
};

}  // namespace rocprofsys::rocprofiler_sdk

namespace rocprofsys::rocprofiler_sdk
{

/// @brief Return the version of the rocprofiler-sdk
template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
version_info
sdk_tracing_config<SdkBackend, Externals>::get_version()
{
    auto version = version_info{};
    SdkBackend::get_version(&version.major, &version.minor, &version.patch);
    return version;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::vector<std::string>
sdk_tracing_config<SdkBackend, Externals>::domain_choices()
{
    const auto& buffered_tracing_info = SdkBackend::get_buffer_tracing_names();
    const auto& callback_tracing_info = SdkBackend::get_callback_tracing_names();

    const auto domains_to_skip =
        std::unordered_set<std::string_view>{ "none",
                                              "correlation_id_retirement",
                                              "marker_core_api",
                                              "marker_control_api",
                                              "marker_name_api",
                                              "code_object" };

    auto choices = std::unordered_set<std::string>{};
    choices.reserve(buffered_tracing_info.size() + callback_tracing_info.size());

    auto add_domain_f = [&choices, &domains_to_skip](std::string_view domain_to_add) {
        const auto domain_lowercase = to_lower(domain_to_add);
        if(domains_to_skip.contains(domain_lowercase))
        {
            return;
        }

        choices.emplace(domain_lowercase);
    };

    add_domain_f("hip_api");
    add_domain_f("hsa_api");
    add_domain_f("marker_api");
    add_domain_f("roctx");

    if constexpr(s_compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 2, .patch = 2 })
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

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::string
sdk_tracing_config<SdkBackend, Externals>::domain_defaults()
{
    auto defaults = std::string{ "hip_runtime_api,marker_api,kernel_dispatch,"
                                 "memory_copy,scratch_memory" };

    if constexpr(s_compile_time_sdk_version <
                 version_info{ .major = 1, .minor = 0, .patch = 0 })
    {
        defaults.append(",page_migration");
    }

    return defaults;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::vector<typename sdk_tracing_config<SdkBackend, Externals>::operation_setting_spec>
sdk_tracing_config<SdkBackend, Externals>::operation_settings()
{
    const auto& buffered_tracing_info = SdkBackend::get_buffer_tracing_names();
    const auto& callback_tracing_info = SdkBackend::get_callback_tracing_names();

    auto result = std::vector<operation_setting_spec>{};

    auto gather_domain_f = [&result](auto kind, const auto& domain_operations) {
        const auto env_names = assemble_operation_env_names_for_kind(kind);
        if(env_names.is_empty())
        {
            return;
        }

        result.push_back(operation_setting_spec{
            env_names, { domain_operations.begin(), domain_operations.end() } });
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

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::unordered_set<typename SdkBackend::callback_tracing_kind_t>
sdk_tracing_config<SdkBackend, Externals>::get_callback_domains()
{
    using kind_t = typename SdkBackend::callback_tracing_kind_t;

    const auto sdk_runtime_version = get_version();
    if(sdk_runtime_version == version_info{})
    {
        LOG_WARNING("rocprofiler-sdk version not initialized");
    }

    auto       callback_domains = std::unordered_set<kind_t>{};
    const auto domains_input =
        rocprofsys::delimit(Externals::get_rocm_domains(), " ,;:\t\n");

    if constexpr(s_compile_time_sdk_version >=
                 version_info{ .major = 0, .minor = 6, .patch = 0 })
    {
        if(Externals::get_use_rcclp())
        {
            callback_domains.emplace(SdkBackend::CALLBACK_TRACING_RCCL_API);
        }

        if(Externals::get_use_ompt())
        {
            callback_domains.emplace(SdkBackend::CALLBACK_TRACING_OMPT);
        }
    }

    const auto domain_map = get_callback_domain_map();

    // Check that the domains are valid
    const auto valid_choices = domain_choices();

    const auto invalid_domain_fn = [&valid_choices](const auto& domainv) {
        return !std::ranges::any_of(
            valid_choices, [&domainv](const auto& choice) { return choice == domainv; });
    };
    if(const auto invalid_domain_itr =
           std::ranges::find_if(domains_input, invalid_domain_fn);
       invalid_domain_itr != domains_input.end())
    {
        throw std::runtime_error(fmt::format(
            "unsupported ROCPROFSYS_ROCM_DOMAINS value: {}", *invalid_domain_itr));
    }

    for(const auto& itr : domains_input)
    {
        // Domains that pass validation but aren't in the compile-time supported
        // set (e.g. an older SDK header) have no domain_map entry; skip silently.
        if(const auto domain_map_itr = domain_map.find(itr);
           domain_map_itr != domain_map.end())
        {
            callback_domains.insert(domain_map_itr->second.begin(),
                                    domain_map_itr->second.end());
        }
    }

    return callback_domains;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::unordered_set<typename SdkBackend::buffer_tracing_kind_t>
sdk_tracing_config<SdkBackend, Externals>::get_buffered_domains()
{
    using kind_t = typename SdkBackend::buffer_tracing_kind_t;

    auto data    = std::unordered_set<kind_t>{};
    auto domains = rocprofsys::delimit(Externals::get_rocm_domains(), " ,;:\t\n");
    // Check that the domains are valid

    const auto valid_choices = domain_choices();
    const auto domain_map    = get_buffered_domain_map();

    const auto invalid_domain_fn = [&valid_choices](const auto& domainv) {
        return !std::ranges::any_of(
            valid_choices, [&domainv](const auto& choice) { return choice == domainv; });
    };
    if(const auto invalid_domain_itr = std::ranges::find_if(domains, invalid_domain_fn);
       invalid_domain_itr != domains.end())
    {
        throw std::runtime_error(fmt::format(
            "unsupported ROCPROFSYS_ROCM_DOMAINS value: {}", *invalid_domain_itr));
    }

    for(const auto& itr : domains)
    {
        const auto& domain_map_itr = domain_map.find(itr);
        if(domain_map_itr != domain_map.end())
        {
            const auto& domain_list = domain_map_itr->second;
            data.insert(domain_list.begin(), domain_list.end());
        }
    }

    if constexpr(s_compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 2, .patch = 2 })
    {
        // Automatically enable KFD domains when unified memory profiling is enabled
        if(Externals::get_use_unified_memory_profiling())
        {
            LOG_INFO("ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON: implicitly enabling "
                     "KFD page_fault and page_migrate buffered tracing domains.");
            data.emplace(SdkBackend::BUFFER_TRACING_KFD_PAGE_FAULT);
            data.emplace(SdkBackend::BUFFER_TRACING_KFD_PAGE_MIGRATE);
        }
    }

    return data;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
template <typename TracingKind>
    requires concepts::tracing_kind_for<SdkBackend, TracingKind>
std::vector<std::int32_t>
sdk_tracing_config<SdkBackend, Externals>::get_operations(TracingKind kind)
{
    const auto names = assemble_operation_env_names_for_kind(kind);
    if(names.is_empty())
    {
        if constexpr(std::same_as<TracingKind,
                                  typename SdkBackend::callback_tracing_kind_t>)
        {
            finalize_and_throw(fmt::format(
                "sdk_tracing_config::get_operations: no options registered for "
                "callback tracing kind {}",
                static_cast<int>(kind)));
        }
        else
        {
            finalize_and_throw(fmt::format(
                "sdk_tracing_config::get_operations: no options registered for "
                "buffer tracing kind {}",
                static_cast<int>(kind)));
        }
    }

    const auto include_patterns =
        compile_operation_patterns(names.operations_include_env_name);
    const auto exclude_patterns =
        compile_operation_patterns(names.operations_exclude_env_name);

    auto operations = all_operation_items_for_kind(kind);

    if(!include_patterns.empty())
    {
        std::erase_if(operations, [&](const auto& operation) {
            return !matches_any_operation_pattern(names.operations_include_env_name,
                                                  operation.second, include_patterns);
        });
    }
    if(!exclude_patterns.empty())
    {
        std::erase_if(operations, [&](const auto& operation) {
            return matches_any_operation_pattern(names.operations_exclude_env_name,
                                                 operation.second, exclude_patterns);
        });
    }

    std::vector<std::int32_t> operation_ids{};
    operation_ids.reserve(operations.size());
    std::ranges::transform(operations, std::back_inserter(operation_ids),
                           [](const auto& operation) { return operation.first; });
    std::ranges::sort(operation_ids);

    return operation_ids;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
template <typename TracingKind>
    requires concepts::tracing_kind_for<SdkBackend, TracingKind>
std::unordered_set<std::int32_t>
sdk_tracing_config<SdkBackend, Externals>::get_backtrace_operations(TracingKind kind)
{
    const auto names = assemble_operation_env_names_for_kind(kind);
    if(names.is_empty())
    {
        if constexpr(std::same_as<TracingKind,
                                  typename SdkBackend::callback_tracing_kind_t>)
        {
            finalize_and_throw(fmt::format("sdk_tracing_config::get_backtrace_"
                                           "operations: no options registered for "
                                           "callback tracing kind {}",
                                           static_cast<int>(kind)));
        }
        else
        {
            finalize_and_throw(fmt::format("sdk_tracing_config::get_backtrace_"
                                           "operations: no options registered for "
                                           "buffer tracing kind {}",
                                           static_cast<int>(kind)));
        }
    }

    const auto patterns =
        compile_operation_patterns(names.operations_annotate_backtrace_env_name);

    std::unordered_set<std::int32_t> matched_operation_ids{};
    if(!patterns.empty())
    {
        for(const auto& [operation_id, operation_name] :
            all_operation_items_for_kind(kind))
        {
            if(matches_any_operation_pattern(names.operations_annotate_backtrace_env_name,
                                             operation_name, patterns))
            {
                matched_operation_ids.insert(operation_id);
            }
        }
    }

    return matched_operation_ids;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
template <typename Tp>
std::string
sdk_tracing_config<SdkBackend, Externals>::to_lower(const Tp& value)
{
    auto str_copy = std::string{ value };

    for(auto& itr : str_copy)
    {
        itr = static_cast<char>(::tolower(itr));
    }

    return str_copy;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
template <typename TracingKind>
    requires concepts::tracing_kind_for<SdkBackend, TracingKind>
std::vector<std::pair<std::int32_t, std::string>>
sdk_tracing_config<SdkBackend, Externals>::all_operation_items_for_kind(
    TracingKind tracing_kind)
{
    auto items = [&] {
        if constexpr(std::same_as<TracingKind,
                                  typename SdkBackend::callback_tracing_kind_t>)
        {
            return SdkBackend::get_callback_tracing_names()[tracing_kind].items();
        }
        else
        {
            return SdkBackend::get_buffer_tracing_names()[tracing_kind].items();
        }
    }();

    auto named_items =
        items | std::views::filter([](const auto& item) {
            return item.second && *item.second != "none";
        }) |
        std::views::transform([](const auto& item) {
            return std::pair<std::int32_t, std::string>{ item.first,
                                                         std::string{ *item.second } };
        });

    return std::vector<std::pair<std::int32_t, std::string>>{ named_items.begin(),
                                                              named_items.end() };
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::vector<std::pair<std::string, std::regex>>
sdk_tracing_config<SdkBackend, Externals>::compile_operation_patterns(
    const std::string& operations_setting_env_name)
{
    const auto setting_value = Externals::get_setting_value(operations_setting_env_name);
    if(!setting_value)
    {
        finalize_and_throw(fmt::format(
            "sdk_tracing_config::compile_operation_patterns: no registered setting '{}'",
            operations_setting_env_name));
    }

    constexpr std::string_view operation_filter_delimiters{ " ,;:\n\t" };

    const auto delimited_patterns =
        rocprofsys::delimit(*setting_value, operation_filter_delimiters);

    auto compiled_patterns =
        delimited_patterns | std::views::transform([](const std::string& pattern_text) {
            return std::pair<std::string, std::regex>{
                pattern_text, std::regex{ pattern_text, std::regex_constants::icase }
            };
        });

    return std::vector<std::pair<std::string, std::regex>>{ compiled_patterns.begin(),
                                                            compiled_patterns.end() };
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
bool
sdk_tracing_config<SdkBackend, Externals>::matches_any_operation_pattern(
    std::string_view operations_setting_env_name, const std::string& operation_name,
    const std::vector<std::pair<std::string, std::regex>>& patterns)
{
    return std::ranges::any_of(patterns, [&](const auto& pattern_entry) {
        const auto& [pattern_text, pattern] = pattern_entry;
        const bool matched                  = std::regex_search(operation_name, pattern);
        if(matched)
        {
            LOG_DEBUG("{} ('{}') matched: {}", operations_setting_env_name, pattern_text,
                      operation_name);
        }
        return matched;
    });
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
void
sdk_tracing_config<SdkBackend, Externals>::finalize_and_throw(
    std::string_view exception_message)
{
    Externals::ProcessState::set(Externals::ProcessState::Finalized);
    throw std::runtime_error(std::string{ exception_message });
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
template <typename TracingKind>
    requires concepts::tracing_kind_for<SdkBackend, TracingKind>
typename sdk_tracing_config<SdkBackend, Externals>::operation_options_env_names
sdk_tracing_config<SdkBackend, Externals>::assemble_operation_env_names_for_kind(
    TracingKind kind)
{
    std::string_view name{};
    bool             has_operations{};

    if constexpr(std::same_as<TracingKind, typename SdkBackend::callback_tracing_kind_t>)
    {
        const auto& entry = SdkBackend::get_callback_tracing_names()[kind];
        name              = (kind == SdkBackend::CALLBACK_TRACING_MARKER_CORE_API)
                                ? std::string_view{ "MARKER_API" }
                                : std::string_view{ entry.name };
        has_operations    = !entry.operations.empty();
    }
    else
    {
        const auto& entry = SdkBackend::get_buffer_tracing_names()[kind];
        name              = entry.name;
        has_operations    = !entry.operations.empty();
    }

    if(!has_operations ||
       s_domains_to_skip_for_operation_options.contains(to_lower(name)))
    {
        return {};
    }

    return operation_options_env_names{ name };
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
constexpr std::unordered_set<typename SdkBackend::callback_tracing_kind_t>
sdk_tracing_config<SdkBackend, Externals>::get_supported_callback_domains()
{
    auto supported = std::unordered_set<typename SdkBackend::callback_tracing_kind_t>{
        SdkBackend::CALLBACK_TRACING_HSA_CORE_API,
        SdkBackend::CALLBACK_TRACING_HSA_AMD_EXT_API,
        SdkBackend::CALLBACK_TRACING_HSA_IMAGE_EXT_API,
        SdkBackend::CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
        SdkBackend::CALLBACK_TRACING_HIP_RUNTIME_API,
        SdkBackend::CALLBACK_TRACING_HIP_COMPILER_API,
        SdkBackend::CALLBACK_TRACING_MARKER_CORE_API,
        SdkBackend::CALLBACK_TRACING_CODE_OBJECT,
    };

    if constexpr(s_compile_time_sdk_version >=
                 version_info{ .major = 0, .minor = 6, .patch = 0 })
    {
        supported.insert({ SdkBackend::CALLBACK_TRACING_RCCL_API,
                           SdkBackend::CALLBACK_TRACING_OMPT,
                           SdkBackend::CALLBACK_TRACING_ROCDECODE_API });
    }
    if constexpr(s_compile_time_sdk_version >=
                 version_info{ .major = 0, .minor = 7, .patch = 0 })
    {
        supported.emplace(SdkBackend::CALLBACK_TRACING_ROCJPEG_API);
    }

    if constexpr(s_compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 3, .patch = 4 })
    {
        supported.emplace(SdkBackend::CALLBACK_TRACING_ROCSHMEM_API);
    }
    if constexpr(s_compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 3, .patch = 5 })
    {
        supported.emplace(SdkBackend::CALLBACK_TRACING_HIPFILE_API);
    }
    return supported;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
constexpr std::unordered_set<typename SdkBackend::buffer_tracing_kind_t>
sdk_tracing_config<SdkBackend, Externals>::get_supported_buffer_domains()
{
    auto supported = std::unordered_set<typename SdkBackend::buffer_tracing_kind_t>{
        SdkBackend::BUFFER_TRACING_KERNEL_DISPATCH,
        SdkBackend::BUFFER_TRACING_MEMORY_COPY,
        SdkBackend::BUFFER_TRACING_SCRATCH_MEMORY,
    };

    if constexpr(s_compile_time_sdk_version >=
                 version_info{ .major = 0, .minor = 6, .patch = 0 })
    {
        supported.emplace(SdkBackend::BUFFER_TRACING_MEMORY_ALLOCATION);
    }

    if constexpr(s_compile_time_sdk_version <
                 version_info{ .major = 1, .minor = 0, .patch = 0 })
    {
        supported.emplace(SdkBackend::BUFFER_TRACING_PAGE_MIGRATION);
    }

    // rocprofiler-sdk < 1.2.2 has a fatal bug parsing KFD events with undefined
    // node IDs (0xFFFFFFFF), so the KFD domains are gated on the bugfix
    // version rather than the version that first declared the enums (1.0.0).
    if constexpr(s_compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 2, .patch = 2 })
    {
        supported.emplace(SdkBackend::BUFFER_TRACING_KFD_PAGE_FAULT);
        supported.emplace(SdkBackend::BUFFER_TRACING_KFD_PAGE_MIGRATE);
        supported.emplace(SdkBackend::BUFFER_TRACING_KFD_QUEUE);
        supported.emplace(SdkBackend::BUFFER_TRACING_KFD_EVENT_QUEUE);
        supported.emplace(SdkBackend::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU);
        supported.emplace(SdkBackend::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS);
    }

    return supported;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::unordered_map<std::string, std::vector<typename SdkBackend::callback_tracing_kind_t>>
sdk_tracing_config<SdkBackend, Externals>::get_callback_domain_map()
{
    using kind_t              = typename SdkBackend::callback_tracing_kind_t;
    const auto& callback_info = SdkBackend::get_callback_tracing_names();
    const auto  supported     = get_supported_callback_domains();

    std::unordered_map<std::string, std::vector<kind_t>> domain_map{
        {
            "hsa_api",
            {
                SdkBackend::CALLBACK_TRACING_HSA_CORE_API,
                SdkBackend::CALLBACK_TRACING_HSA_AMD_EXT_API,
                SdkBackend::CALLBACK_TRACING_HSA_IMAGE_EXT_API,
                SdkBackend::CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
            },
        },
        {
            "hip_api",
            {
                SdkBackend::CALLBACK_TRACING_HIP_RUNTIME_API,
                SdkBackend::CALLBACK_TRACING_HIP_COMPILER_API,
            },
        },
        {
            "marker_api",
            {
                SdkBackend::CALLBACK_TRACING_MARKER_CORE_API,
            },
        },
        {
            "roctx",
            {
                SdkBackend::CALLBACK_TRACING_MARKER_CORE_API,
            },
        },
    };

    for(size_t idx = 0; idx < callback_info.size(); ++idx)
    {
        const auto& ditr = callback_info[idx];
        if(supported.contains(ditr.value))
        {
            domain_map.emplace(to_lower(ditr.name), std::vector<kind_t>{ ditr.value });
        }
    }

    return domain_map;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::unordered_map<std::string, std::vector<typename SdkBackend::buffer_tracing_kind_t>>
sdk_tracing_config<SdkBackend, Externals>::get_buffered_domain_map()
{
    using kind_t            = typename SdkBackend::buffer_tracing_kind_t;
    const auto& buffer_info = SdkBackend::get_buffer_tracing_names();
    const auto  supported   = get_supported_buffer_domains();

    std::unordered_map<std::string, std::vector<kind_t>> domain_map{
        {
            "hsa_api",
            {
                SdkBackend::BUFFER_TRACING_HSA_CORE_API,
                SdkBackend::BUFFER_TRACING_HSA_AMD_EXT_API,
                SdkBackend::BUFFER_TRACING_HSA_IMAGE_EXT_API,
                SdkBackend::BUFFER_TRACING_HSA_FINALIZE_EXT_API,
            },
        },
        {
            "hip_api",
            {
                SdkBackend::BUFFER_TRACING_HIP_COMPILER_API,
                SdkBackend::BUFFER_TRACING_HIP_RUNTIME_API,
            },
        },
        {
            "marker_api",
            {
                SdkBackend::BUFFER_TRACING_MARKER_CORE_API,
            },
        },
        {
            "roctx",
            {
                SdkBackend::BUFFER_TRACING_MARKER_CORE_API,
            },
        },
        {
            "memory_copy",
            {
                SdkBackend::BUFFER_TRACING_MEMORY_COPY,
            },
        },
    };

    if constexpr(s_compile_time_sdk_version >=
                 version_info{ .major = 0, .minor = 6, .patch = 0 })
    {
        domain_map.emplace(
            "memory_allocation",
            std::vector<kind_t>{ SdkBackend::BUFFER_TRACING_MEMORY_ALLOCATION });
    }

    if constexpr(s_compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 2, .patch = 2 })
    {
        domain_map.insert({
            { "kfd_events",
              {
                  SdkBackend::BUFFER_TRACING_KFD_PAGE_FAULT,
                  SdkBackend::BUFFER_TRACING_KFD_PAGE_MIGRATE,
                  SdkBackend::BUFFER_TRACING_KFD_QUEUE,
                  SdkBackend::BUFFER_TRACING_KFD_EVENT_QUEUE,
                  SdkBackend::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                  SdkBackend::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS,
              } },
            {
                "kfd_page_fault",
                { SdkBackend::BUFFER_TRACING_KFD_PAGE_FAULT },
            },
            {
                "kfd_page_migrate",
                { SdkBackend::BUFFER_TRACING_KFD_PAGE_MIGRATE },
            },
            {
                "kfd_queue",
                { SdkBackend::BUFFER_TRACING_KFD_QUEUE },
            },
            {
                "kfd_event_queue",
                { SdkBackend::BUFFER_TRACING_KFD_EVENT_QUEUE },
            },
            {
                "kfd_event_unmap_from_gpu",
                { SdkBackend::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU },
            },
            {
                "kfd_event_dropped_events",
                { SdkBackend::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS },
            },
        });
    }

    for(size_t index = 0; index < buffer_info.size(); ++index)
    {
        const auto& domain = buffer_info[index];
        if(supported.contains(domain.value))
        {
            domain_map.emplace(to_lower(domain.name),
                               std::vector<kind_t>{ domain.value });
        }
    }

    return domain_map;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
const std::unordered_set<std::string_view> sdk_tracing_config<
    SdkBackend, Externals>::s_domains_to_skip_for_operation_options{

    "none",        "correlation_id_retirement", "marker_control_api", "marker_name_api",
    "code_object", "kernel_dispatch",           "page_migration",
};
}  // namespace rocprofsys::rocprofiler_sdk
