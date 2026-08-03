// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/delimit.hpp"
#include "common/env_vars.hpp"
#include "logger/debug.hpp"

#include <cctype>
#include <cstddef>
#include <initializer_list>
#include <spdlog/fmt/ranges.h>

#include <algorithm>
#include <array>
#include <compare>
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
    typename Externals::Settings;
    typename Externals::ProcessState;
    typename Externals::ProcessState::State;
    {
        Externals::ProcessState::Finalized
    } -> std::convertible_to<typename Externals::ProcessState::State>;
    { Externals::ProcessState::set(Externals::ProcessState::Finalized) };
    { Externals::get_settings() } -> std::same_as<typename Externals::Settings*>;
    { Externals::get_use_rcclp() } -> std::convertible_to<bool>;
    { Externals::get_use_ompt() } -> std::convertible_to<bool>;
    { Externals::get_use_unified_memory_profiling() } -> std::convertible_to<bool>;
    { Externals::get_rocm_domains() } -> std::convertible_to<std::string>;
    { Externals::get_rocm_events_setting() } -> std::convertible_to<std::string>;
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
    struct domain_choice_settings
    {
        std::vector<std::string> domain_choices;
        std::string              domain_description;
        std::string              domain_defaults;
    };

    static domain_choice_settings domain_settings();

    struct operation_setting_spec
    {
        std::string              all_operations_env;
        std::string              exclude_operations_env;
        std::string              annotate_backtrace_env;
        std::vector<std::string> operation_choices;
    };

    static std::vector<operation_setting_spec> operation_settings();

    static version_info get_version();

    static std::unordered_set<typename SdkBackend::callback_tracing_kind_t>
    get_callback_domains();

    static std::unordered_set<typename SdkBackend::buffer_tracing_kind_t>
    get_buffered_domains();

    static std::vector<std::int32_t> get_operations(
        typename SdkBackend::callback_tracing_kind_t kindv);

    static std::vector<std::int32_t> get_operations(
        typename SdkBackend::buffer_tracing_kind_t kindv);

    static std::unordered_set<std::int32_t> get_backtrace_operations(
        typename SdkBackend::callback_tracing_kind_t kindv);

    static std::unordered_set<std::int32_t> get_backtrace_operations(
        typename SdkBackend::buffer_tracing_kind_t kindv);

private:
    static constexpr version_info compile_time_sdk_version =
        version_info::from_formatted(SdkBackend::compile_time_version);

    static std::vector<std::int32_t> filter_operations(
        const std::unordered_set<std::int32_t>& complete,
        const std::unordered_set<std::int32_t>& include,
        const std::unordered_set<std::int32_t>& exclude);

    template <typename Tp>
    static std::string to_lower(const Tp& val);

    template <typename TracingKind>
        requires concepts::tracing_kind_for<SdkBackend, TracingKind>
    static std::unordered_set<std::int32_t> parse_operation_string(
        TracingKind tracing_kind, const std::string& operations_setting = {});

    template <typename OperationItems>
    static std::unordered_set<std::int32_t> operation_ids_for_tracing_kind(
        const std::string& operations_setting, const OperationItems& operation_items);

    [[noreturn]] static void finalize_and_throw(std::string_view exception_message);

    struct operation_options_env_names
    {
        std::string operations_include_env_name            = {};
        std::string operations_exclude_env_name            = {};
        std::string operations_annotate_backtrace_env_name = {};
    };

    /// @brief Pure: nullopt if not filterable, else the three formatted env names.
    [[nodiscard]] static std::optional<operation_options_env_names>
    assemble_operation_env_names_for_domain(std::string_view domain_name,
                                            bool             has_operations);

    /// @brief Env-var names for one kind, via SdkBackend's cached tracing-name
    /// table, applying the MARKER_CORE_API -> "MARKER_API" alias.
    template <typename TracingKind>
        requires concepts::tracing_kind_for<SdkBackend, TracingKind>
    static std::optional<operation_options_env_names>
    assemble_operation_env_names_for_kind(TracingKind kind);

    struct kfd_runtime_support
    {
        version_info version{};
        bool         supported = false;
    };

    // rocprofiler-sdk < 1.2.2 has a fatal bug parsing KFD events with undefined
    // node IDs (0xFFFFFFFF). The compile-time gate confirms the SDK headers declare
    // the KFD enums; the loaded runtime library must be checked separately since it
    // can be older than the headers this binary was built against.
    static kfd_runtime_support get_kfd_runtime_support();

    static bool is_kfd_domain_name(std::string_view name);

    static std::vector<typename SdkBackend::buffer_tracing_kind_t> kfd_kinds_for_name(
        std::string_view name);

    static void warn_kfd_disabled_once(std::string_view    domain_name,
                                       const version_info& kfd_version);

    const static std::unordered_set<std::string_view> s_domains_to_skip;
};

}  // namespace rocprofsys::rocprofiler_sdk

namespace rocprofsys::rocprofiler_sdk
{

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
const std::unordered_set<std::string_view>
    sdk_tracing_config<SdkBackend, Externals>::s_domains_to_skip{

        "none",
        "correlation_id_retirement",
        "marker_core_api",
        "marker_control_api",
        "marker_name_api",
        "code_object",
        "kernel_dispatch",
        "page_migration",
    };

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
void
sdk_tracing_config<SdkBackend, Externals>::finalize_and_throw(
    std::string_view exception_message)
{
    Externals::ProcessState::set(Externals::ProcessState::Finalized);
    throw std::runtime_error(std::string{ exception_message });
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::optional<
    typename sdk_tracing_config<SdkBackend, Externals>::operation_options_env_names>
sdk_tracing_config<SdkBackend, Externals>::assemble_operation_env_names_for_domain(
    std::string_view domain_name, bool has_operations)
{
    if(!has_operations || s_domains_to_skip.contains(to_lower(domain_name)))
    {
        return std::nullopt;
    }

    return operation_options_env_names{
        fmt::format("ROCPROFSYS_ROCM_{}_OPERATIONS", domain_name),
        fmt::format("ROCPROFSYS_ROCM_{}_OPERATIONS_EXCLUDE", domain_name),
        fmt::format("ROCPROFSYS_ROCM_{}_OPERATIONS_ANNOTATE_BACKTRACE", domain_name),
    };
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
template <typename TracingKind>
    requires concepts::tracing_kind_for<SdkBackend, TracingKind>
std::optional<
    typename sdk_tracing_config<SdkBackend, Externals>::operation_options_env_names>
sdk_tracing_config<SdkBackend, Externals>::assemble_operation_env_names_for_kind(
    TracingKind kind)
{
    if constexpr(std::same_as<TracingKind, typename SdkBackend::callback_tracing_kind_t>)
    {
        const auto& entry       = SdkBackend::get_callback_tracing_names()[kind];
        const auto  domain_name = (kind == SdkBackend::CALLBACK_TRACING_MARKER_CORE_API)
                                      ? std::string_view{ "MARKER_API" }
                                      : std::string_view{ entry.name };
        return assemble_operation_env_names_for_domain(domain_name,
                                                       !entry.operations.empty());
    }
    else
    {
        const auto& entry = SdkBackend::get_buffer_tracing_names()[kind];
        return assemble_operation_env_names_for_domain(std::string_view{ entry.name },
                                                       !entry.operations.empty());
    }
}

// ─── operation_ids_for_tracing_kind (tracing kind + optional setting) ────────
template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
template <typename OperationItems>
std::unordered_set<std::int32_t>
sdk_tracing_config<SdkBackend, Externals>::operation_ids_for_tracing_kind(
    const std::string& operations_setting_env_name, const OperationItems& operation_items)
{
    // maybe delete this
    if(operations_setting_env_name.empty())
    {
        std::unordered_set<std::int32_t> all_operation_ids{};
        for(const auto& [operation_id, operation_name] : operation_items)
        {
            if(operation_name && *operation_name != "none")
            {
                all_operation_ids.insert(operation_id);
            }
        }
        return all_operation_ids;
    }

    auto operations_filter = Externals::get_setting_value(operations_setting_env_name);
    if(!operations_filter)
    {
        finalize_and_throw(fmt::format(
            "sdk_tracing_config::operation_ids_for_tracing_kind: no registered setting "
            "'{}'",
            operations_setting_env_name));
    }

    if(operations_filter->empty())
    {
        return {};
    }

    std::vector<std::pair<std::int32_t, std::string_view>> operations_by_name{};
    for(const auto& [operation_id, operation_name] : operation_items)
    {
        if(operation_name)
        {
            operations_by_name.emplace_back(operation_id,
                                            std::string_view{ *operation_name });
        }
    }

    std::unordered_set<std::int32_t> matched_operation_ids{};
    matched_operation_ids.reserve(operations_by_name.size());

    constexpr std::string_view operation_filter_delimiters{ " ,;:\n\t" };
    for(const auto& pattern :
        rocprofsys::delimit(*operations_filter, operation_filter_delimiters))
    {
        const std::regex case_insensitive_pattern{ pattern, std::regex_constants::icase };
        for(const auto& [operation_id, operation_label] : operations_by_name)
        {
            if(!std::regex_search(operation_label.begin(), operation_label.end(),
                                  case_insensitive_pattern))
            {
                continue;
            }

            LOG_DEBUG("{} ('{}') matched: {}", operations_setting_env_name, pattern,
                      operation_label);
            matched_operation_ids.insert(operation_id);
        }
    }
    return matched_operation_ids;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
template <typename TracingKind>
    requires concepts::tracing_kind_for<SdkBackend, TracingKind>
std::unordered_set<std::int32_t>
sdk_tracing_config<SdkBackend, Externals>::parse_operation_string(
    TracingKind tracing_kind, const std::string& operations_setting_env_name)
{
    if constexpr(std::same_as<TracingKind, typename SdkBackend::callback_tracing_kind_t>)
    {
        return operation_ids_for_tracing_kind(
            operations_setting_env_name,
            SdkBackend::get_callback_tracing_names()[tracing_kind].items());
    }
    else
    {
        return operation_ids_for_tracing_kind(
            operations_setting_env_name,
            SdkBackend::get_buffer_tracing_names()[tracing_kind].items());
    }
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::vector<std::int32_t>
sdk_tracing_config<SdkBackend, Externals>::filter_operations(
    const std::unordered_set<std::int32_t>& complete_set,
    const std::unordered_set<std::int32_t>& to_include,
    const std::unordered_set<std::int32_t>& to_exclude)
{
    auto convert_to_vector = [](const std::unordered_set<std::int32_t>& set_to_convert) {
        auto result_vector =
            std::vector<std::int32_t>(set_to_convert.begin(), set_to_convert.end());
        std::ranges::sort(result_vector);
        return result_vector;
    };

    if(to_include.empty() && to_exclude.empty())
    {
        return convert_to_vector(complete_set);
    }

    auto result = to_include.empty() ? complete_set : to_include;
    for(auto itr : to_exclude)
    {
        result.erase(itr);
    }

    return convert_to_vector(result);
}

// ─── Public method implementations ───────────────────────────────────────────

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
typename sdk_tracing_config<SdkBackend, Externals>::domain_choice_settings
sdk_tracing_config<SdkBackend, Externals>::domain_settings()
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

    auto domain_choices = std::unordered_set<std::string>{};
    domain_choices.reserve(buffered_tracing_info.size() + callback_tracing_info.size());

    auto add_domain_f = [&domain_choices,
                         &domains_to_skip](std::string_view domain_to_add) {
        const auto domain_lowercase = to_lower(domain_to_add);
        if(domains_to_skip.contains(domain_lowercase))
        {
            return;
        }

        domain_choices.emplace(domain_lowercase);
    };

    add_domain_f("hip_api");
    add_domain_f("hsa_api");
    add_domain_f("marker_api");
    add_domain_f("roctx");

    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 0, .patch = 0 })
    {
        add_domain_f("kfd_events");
    }
    const auto name_projection = [](const auto& info) { return info.name; };

    std::ranges::for_each(buffered_tracing_info, add_domain_f, name_projection);
    std::ranges::for_each(callback_tracing_info, add_domain_f, name_projection);

    std::vector<std::string> domain_choices_vec{ domain_choices.begin(),
                                                 domain_choices.end() };
    std::ranges::sort(domain_choices_vec);

    const auto domain_description =
        fmt::format("Specification of ROCm domains to trace/profile. Choices: {}",
                    fmt::join(domain_choices_vec, ", "));
    auto domain_defaults = std::string{ "hip_runtime_api,marker_api,kernel_dispatch,"
                                        "memory_copy,scratch_memory" };

    if constexpr(compile_time_sdk_version <
                 version_info{ .major = 1, .minor = 0, .patch = 0 })
    {
        domain_defaults.append(",page_migration");
    }

    return domain_choice_settings{ std::move(domain_choices_vec),
                                   std::move(domain_description),
                                   std::move(domain_defaults) };
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::vector<typename sdk_tracing_config<SdkBackend, Externals>::operation_setting_spec>
sdk_tracing_config<SdkBackend, Externals>::operation_settings()
{
    const auto& buffered_tracing_info = SdkBackend::get_buffer_tracing_names();
    const auto& callback_tracing_info = SdkBackend::get_callback_tracing_names();

    auto result = std::vector<operation_setting_spec>{};

    auto gather_domain_f = [&result](std::string_view domain_name,
                                     const auto&      domain_operations) {
        const auto names = assemble_operation_env_names_for_domain(
            domain_name, !domain_operations.empty());
        if(!names)
        {
            return;
        }

        result.push_back(operation_setting_spec{
            names->operations_include_env_name,
            names->operations_exclude_env_name,
            names->operations_annotate_backtrace_env_name,
            { domain_operations.begin(), domain_operations.end() } });
    };

    gather_domain_f(
        "MARKER_API",
        callback_tracing_info[SdkBackend::CALLBACK_TRACING_MARKER_CORE_API].operations);

    for(const auto& itr : callback_tracing_info)
    {
        gather_domain_f(itr.name, itr.operations);
    }

    for(const auto& itr : buffered_tracing_info)
    {
        gather_domain_f(itr.name, itr.operations);
    }

    return result;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::unordered_set<typename SdkBackend::callback_tracing_kind_t>
sdk_tracing_config<SdkBackend, Externals>::get_callback_domains()
{
    using kind_t              = typename SdkBackend::callback_tracing_kind_t;
    const auto& callback_info = SdkBackend::get_callback_tracing_names();
    auto        supported     = std::unordered_set<kind_t>{
        SdkBackend::CALLBACK_TRACING_HSA_CORE_API,
        SdkBackend::CALLBACK_TRACING_HSA_AMD_EXT_API,
        SdkBackend::CALLBACK_TRACING_HSA_IMAGE_EXT_API,
        SdkBackend::CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
        SdkBackend::CALLBACK_TRACING_HIP_RUNTIME_API,
        SdkBackend::CALLBACK_TRACING_HIP_COMPILER_API,
        SdkBackend::CALLBACK_TRACING_MARKER_CORE_API,
        SdkBackend::CALLBACK_TRACING_CODE_OBJECT,
    };

    const auto sdk_runtime_version = get_version();
    if(sdk_runtime_version == version_info{})
    {
        LOG_WARNING("rocprofiler-sdk version not initialized");
    }

    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 0, .minor = 6, .patch = 0 })
    {
        if(sdk_runtime_version >= version_info{ .major = 0, .minor = 6, .patch = 0 })
        {
            // Argument tracing is supported in rocprofiler-sdk 0.6.0 and later
            supported.emplace(SdkBackend::CALLBACK_TRACING_RCCL_API);
            supported.emplace(SdkBackend::CALLBACK_TRACING_OMPT);
            supported.emplace(SdkBackend::CALLBACK_TRACING_ROCDECODE_API);
        }
    }
    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 0, .minor = 7, .patch = 0 })
    {
        if(sdk_runtime_version >= version_info{ .major = 0, .minor = 7, .patch = 0 })
        {
            supported.emplace(SdkBackend::CALLBACK_TRACING_ROCJPEG_API);
        }
    }

    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 3, .patch = 4 })
    {
        if(sdk_runtime_version >= version_info{ .major = 1, .minor = 3, .patch = 4 })
        {
            supported.emplace(SdkBackend::CALLBACK_TRACING_ROCSHMEM_API);
        }
    }
    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 3, .patch = 5 })
    {
        if(sdk_runtime_version >= version_info{ .major = 1, .minor = 3, .patch = 5 })
        {
            supported.emplace(SdkBackend::CALLBACK_TRACING_HIPFILE_API);
        }
    }

    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 3, .patch = 4 })
    {
        if(sdk_runtime_version >= version_info{ .major = 1, .minor = 3, .patch = 4 })
        {
            supported.emplace(SdkApi::CALLBACK_TRACING_ROCSHMEM_API);
        }
    }
    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 3, .patch = 5 })
    {
        if(sdk_runtime_version >= version_info{ .major = 1, .minor = 3, .patch = 5 })
        {
            supported.emplace(SdkApi::CALLBACK_TRACING_HIPFILE_API);
        }
    }

    auto       callback_domains = std::unordered_set<kind_t>{};
    const auto domains_input =
        rocprofsys::delimit(Externals::get_rocm_domains(), " ,;:\t\n");

    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 0, .minor = 6, .patch = 0 })
    {
        if(Externals::get_use_rcclp() &&
           sdk_runtime_version >= version_info{ .major = 0, .minor = 6, .patch = 0 })
        {
            // Translate ROCPROFSYS_USE_RCCLP to entry in ROCPROFSYS_ROCM_DOMAINS
            callback_domains.emplace(SdkBackend::CALLBACK_TRACING_RCCL_API);
        }
        if(Externals::get_use_ompt() &&
           sdk_runtime_version >= version_info{ .major = 0, .minor = 6, .patch = 0 })
        {
            // Translate some configuration settings to rocprofiler domains
            callback_domains.emplace(SdkBackend::CALLBACK_TRACING_OMPT);
        }
    }

    // Check that the domains are valid
    const auto valid_choices  = Externals::get_settings()
                                    ->at(std::string{ env_vars::ROCM_DOMAINS })
                                    ->get_choices();
    auto       invalid_domain = [&valid_choices](const auto& domainv) {
        return !std::ranges::any_of(
            valid_choices, [&domainv](const auto& choice) { return choice == domainv; });
    };

    for(const auto& itr : domains_input)
    {
        if(invalid_domain(itr))
        {
            throw std::runtime_error(
                fmt::format("unsupported ROCPROFSYS_ROCM_DOMAINS value: {}", itr));
        }

        if(itr == "hsa_api")
        {
            for(auto eitr : { SdkBackend::CALLBACK_TRACING_HSA_CORE_API,
                              SdkBackend::CALLBACK_TRACING_HSA_AMD_EXT_API,
                              SdkBackend::CALLBACK_TRACING_HSA_IMAGE_EXT_API,
                              SdkBackend::CALLBACK_TRACING_HSA_FINALIZE_EXT_API })
                callback_domains.emplace(eitr);
        }
        else if(itr == "hip_api")
        {
            for(auto eitr : { SdkBackend::CALLBACK_TRACING_HIP_RUNTIME_API,
                              SdkBackend::CALLBACK_TRACING_HIP_COMPILER_API })
                callback_domains.emplace(eitr);
        }
        else if(itr == "marker_api" || itr == "roctx")
        {
            callback_domains.emplace(SdkBackend::CALLBACK_TRACING_MARKER_CORE_API);
        }
        else
        {
            for(size_t idx = 0; idx < callback_info.size(); ++idx)
            {
                const auto& ditr = callback_info[idx];
                auto        dval = static_cast<kind_t>(idx);
                if(itr == to_lower(ditr.name) && supported.contains(dval))
                {
                    callback_domains.emplace(dval);
                    break;
                }
            }
        }
    }

    return callback_domains;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
typename sdk_tracing_config<SdkBackend, Externals>::kfd_runtime_support
sdk_tracing_config<SdkBackend, Externals>::get_kfd_runtime_support()
{
    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 0, .patch = 0 })
    {
        constexpr auto kfd_min_version =
            version_info{ .major = 1, .minor = 2, .patch = 2 };
        const auto kfd_version = get_version();
        return kfd_runtime_support{ kfd_version, kfd_version >= kfd_min_version };
    }
    else
    {
        return kfd_runtime_support{};
    }
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
bool
sdk_tracing_config<SdkBackend, Externals>::is_kfd_domain_name(std::string_view name)
{
    static constexpr auto kfd_domain_names = std::array<std::string_view, 7>{
        "kfd_events",
        "kfd_page_fault",
        "kfd_page_migrate",
        "kfd_queue",
        "kfd_event_queue",
        "kfd_event_unmap_from_gpu",
        "kfd_event_dropped_events",
    };
    return std::ranges::find(kfd_domain_names, name) != kfd_domain_names.end();
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::vector<typename SdkBackend::buffer_tracing_kind_t>
sdk_tracing_config<SdkBackend, Externals>::kfd_kinds_for_name(std::string_view name)
{
    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 0, .patch = 0 })
    {
        using kind_t = typename SdkBackend::buffer_tracing_kind_t;
        static const auto kfd_kind_table =
            std::unordered_map<std::string_view, std::vector<kind_t>>{
                { "kfd_events",
                  { SdkBackend::BUFFER_TRACING_KFD_PAGE_FAULT,
                    SdkBackend::BUFFER_TRACING_KFD_PAGE_MIGRATE,
                    SdkBackend::BUFFER_TRACING_KFD_QUEUE,
                    SdkBackend::BUFFER_TRACING_KFD_EVENT_QUEUE,
                    SdkBackend::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                    SdkBackend::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS } },
                { "kfd_page_fault", { SdkBackend::BUFFER_TRACING_KFD_PAGE_FAULT } },
                { "kfd_page_migrate", { SdkBackend::BUFFER_TRACING_KFD_PAGE_MIGRATE } },
                { "kfd_queue", { SdkBackend::BUFFER_TRACING_KFD_QUEUE } },
                { "kfd_event_queue", { SdkBackend::BUFFER_TRACING_KFD_EVENT_QUEUE } },
                { "kfd_event_unmap_from_gpu",
                  { SdkBackend::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU } },
                { "kfd_event_dropped_events",
                  { SdkBackend::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS } },
            };

        const auto itr = kfd_kind_table.find(name);
        return itr != kfd_kind_table.end() ? itr->second : std::vector<kind_t>{};
    }
    else
    {
        return {};
    }
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
void
sdk_tracing_config<SdkBackend, Externals>::warn_kfd_disabled_once(
    std::string_view domain_name, const version_info& kfd_version)
{
    static bool warned = false;
    if(warned)
    {
        return;
    }

    LOG_WARNING("KFD tracing domain '{}' disabled: rocprofiler-sdk {}.{}.{} has a bug "
                "with undefined KFD node IDs (fixed in >= 1.2.2)",
                domain_name, kfd_version.major, kfd_version.minor, kfd_version.patch);
    warned = true;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::unordered_set<typename SdkBackend::buffer_tracing_kind_t>
sdk_tracing_config<SdkBackend, Externals>::get_buffered_domains()
{
    using kind_t            = typename SdkBackend::buffer_tracing_kind_t;
    const auto& buffer_info = SdkBackend::get_buffer_tracing_names();

    auto supported = std::unordered_set<kind_t>{
        SdkBackend::BUFFER_TRACING_KERNEL_DISPATCH,
        SdkBackend::BUFFER_TRACING_MEMORY_COPY,
        SdkBackend::BUFFER_TRACING_SCRATCH_MEMORY,
    };

    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 0, .minor = 6, .patch = 0 })
    {
        supported.emplace(SdkBackend::BUFFER_TRACING_MEMORY_ALLOCATION);
    }

    if constexpr(compile_time_sdk_version <
                 version_info{ .major = 1, .minor = 0, .patch = 0 })
    {
        supported.emplace(SdkBackend::BUFFER_TRACING_PAGE_MIGRATION);
    }

    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 0, .patch = 0 })
    {
        supported.emplace(SdkBackend::BUFFER_TRACING_KFD_PAGE_FAULT);
        supported.emplace(SdkBackend::BUFFER_TRACING_KFD_PAGE_MIGRATE);
        supported.emplace(SdkBackend::BUFFER_TRACING_KFD_QUEUE);
        supported.emplace(SdkBackend::BUFFER_TRACING_KFD_EVENT_QUEUE);
        supported.emplace(SdkBackend::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU);
        supported.emplace(SdkBackend::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS);
    }

    const auto kfd_support = get_kfd_runtime_support();

    auto data    = std::unordered_set<kind_t>{};
    auto domains = rocprofsys::delimit(Externals::get_rocm_domains(), " ,;:\t\n");
    // Check that the domains are valid
    const auto valid_choices  = Externals::get_settings()
                                    ->at(std::string{ env_vars::ROCM_DOMAINS })
                                    ->get_choices();
    auto       invalid_domain = [&valid_choices](const auto& domainv) {
        return !std::ranges::any_of(
            valid_choices, [&domainv](const auto& choice) { return choice == domainv; });
    };

    for(const auto& itr : domains)
    {
        if(invalid_domain(itr))
        {
            throw std::runtime_error(
                fmt::format("unsupported ROCPROFSYS_ROCM_DOMAINS value: {}", itr));
        }

        if(itr == "hsa_api")
        {
            for(const auto& eitr : { SdkBackend::BUFFER_TRACING_HSA_CORE_API,
                                     SdkBackend::BUFFER_TRACING_HSA_AMD_EXT_API,
                                     SdkBackend::BUFFER_TRACING_HSA_IMAGE_EXT_API,
                                     SdkBackend::BUFFER_TRACING_HSA_FINALIZE_EXT_API })
            {
                data.emplace(eitr);
            }
        }
        else if(itr == "hip_api")
        {
            for(const auto& eitr : { SdkBackend::BUFFER_TRACING_HIP_COMPILER_API,
                                     SdkBackend::BUFFER_TRACING_HIP_RUNTIME_API })
            {
                data.emplace(eitr);
            }
        }
        else if(itr == "marker_api" || itr == "roctx")
        {
            data.emplace(SdkBackend::BUFFER_TRACING_MARKER_CORE_API);
        }
        else if(itr == "memory_allocation")
        {
            if constexpr(compile_time_sdk_version >=
                         version_info{ .major = 0, .minor = 6, .patch = 0 })
            {
                data.emplace(SdkBackend::BUFFER_TRACING_MEMORY_ALLOCATION);
            }
        }
        else if(itr == "memory_copy")
        {
            data.emplace(SdkBackend::BUFFER_TRACING_MEMORY_COPY);
        }
        else if(is_kfd_domain_name(itr))
        {
            if constexpr(compile_time_sdk_version >=
                         version_info{ .major = 1, .minor = 0, .patch = 0 })
            {
                if(!kfd_support.supported)
                {
                    warn_kfd_disabled_once(itr, kfd_support.version);
                    continue;
                }
                for(auto eitr : kfd_kinds_for_name(itr))
                {
                    data.emplace(eitr);
                }
            }
        }
        else
        {
            for(size_t idx = 0; idx < buffer_info.size(); ++idx)
            {
                const auto& ditr = buffer_info[idx];
                auto        dval = static_cast<kind_t>(idx);
                if(itr == to_lower(ditr.name) && supported.contains(dval))
                {
                    data.emplace(dval);
                    break;
                }
            }
        }
    }

    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 0, .patch = 0 })
    {
        // Automatically enable KFD domains when unified memory profiling is enabled
        if(Externals::get_use_unified_memory_profiling())
        {
            if(kfd_support.supported)
            {
                LOG_INFO(
                    "ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON: implicitly enabling "
                    "KFD page_fault and page_migrate buffered tracing domains");
                data.emplace(SdkBackend::BUFFER_TRACING_KFD_PAGE_FAULT);
                data.emplace(SdkBackend::BUFFER_TRACING_KFD_PAGE_MIGRATE);
            }
            else
            {
                LOG_WARNING("ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON requested KFD "
                            "page_fault/page_migrate tracing, but rocprofiler-sdk "
                            "{}.{}.{} is too old (requires >= 1.2.2)",
                            kfd_support.version.major, kfd_support.version.minor,
                            kfd_support.version.patch);
            }
        }
    }

    return data;
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::vector<std::int32_t>
sdk_tracing_config<SdkBackend, Externals>::get_operations(
    typename SdkBackend::callback_tracing_kind_t kind)
{
    const auto names = assemble_operation_env_names_for_kind(kind);
    if(!names)
    {
        finalize_and_throw(
            fmt::format("sdk_tracing_config::get_operations: no options registered for "
                        "callback tracing kind {}",
                        static_cast<int>(kind)));
    }

    const auto complete_set = parse_operation_string(kind);
    const auto include_operations =
        parse_operation_string(kind, names->operations_include_env_name);
    const auto exclude_operations =
        parse_operation_string(kind, names->operations_exclude_env_name);

    return filter_operations(complete_set, include_operations, exclude_operations);
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::vector<std::int32_t>
sdk_tracing_config<SdkBackend, Externals>::get_operations(
    typename SdkBackend::buffer_tracing_kind_t kind)
{
    const auto names = assemble_operation_env_names_for_kind(kind);
    if(!names)
    {
        finalize_and_throw(
            fmt::format("sdk_tracing_config::get_operations: no options registered for "
                        "buffer tracing kind {}",
                        static_cast<int>(kind)));
    }

    const auto complete_set = parse_operation_string(kind);
    const auto include_operations =
        parse_operation_string(kind, names->operations_include_env_name);
    const auto exclude_operations =
        parse_operation_string(kind, names->operations_exclude_env_name);

    return filter_operations(complete_set, include_operations, exclude_operations);
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::unordered_set<std::int32_t>
sdk_tracing_config<SdkBackend, Externals>::get_backtrace_operations(
    typename SdkBackend::callback_tracing_kind_t kind)
{
    const auto names = assemble_operation_env_names_for_kind(kind);
    if(!names)
    {
        finalize_and_throw(fmt::format(
            "sdk_tracing_config::get_backtrace_operations: no options registered for "
            "callback tracing kind {}",
            static_cast<int>(kind)));
    }

    const auto result =
        parse_operation_string(kind, names->operations_annotate_backtrace_env_name);
    return { result.begin(), result.end() };
}

template <typename SdkBackend, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::unordered_set<std::int32_t>
sdk_tracing_config<SdkBackend, Externals>::get_backtrace_operations(
    typename SdkBackend::buffer_tracing_kind_t kind)
{
    const auto names = assemble_operation_env_names_for_kind(kind);
    if(!names)
    {
        finalize_and_throw(fmt::format(
            "sdk_tracing_config::get_backtrace_operations: no options registered for "
            "buffer tracing kind {}",
            static_cast<int>(kind)));
    }

    const auto result =
        parse_operation_string(kind, names->operations_annotate_backtrace_env_name);
    return { result.begin(), result.end() };
}

}  // namespace rocprofsys::rocprofiler_sdk
