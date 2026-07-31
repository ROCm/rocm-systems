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

template <typename SdkApi, typename TracingKind>
concept tracing_kind_for =
    std::same_as<TracingKind, typename SdkApi::callback_tracing_kind> ||
    std::same_as<TracingKind, typename SdkApi::buffer_tracing_kind>;

// Constrains Externals to sdk_tracing_config's DI surface (settings, feature
// flags, ROCm domain/event config, finalize-on-error state). A mismatch fails
// at class declaration, not deep in a method body. `default_sdk_externals`
// is the production policy; tests supply a mock.
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

template <typename SdkApi, typename Externals>
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

    /// @brief Gather the ROCm domain choices/description/default value that
    /// should be registered as the ROCPROFSYS_ROCM_DOMAINS setting. Pure data
    /// gathering — performs no insertion into any settings store.
    static domain_choice_settings domain_settings();

    struct operation_setting_spec
    {
        std::string              all_operations_env;
        std::string              exclude_operations_env;
        std::string              annotate_backtrace_env;
        std::vector<std::string> operation_choices;
    };

    /// @brief Gather the per-domain operation-filter settings (inclusive,
    /// exclusive, and backtrace-annotation env vars) that should be
    /// registered for every domain with named operations. As a side effect,
    /// populates the operation-name lookup used by get_operations()/
    /// get_backtrace_operations(). Performs no insertion into any settings
    /// store.
    static std::vector<operation_setting_spec> operation_settings();

    static version_info& get_version();

    static std::unordered_set<typename SdkApi::callback_tracing_kind>
    get_callback_domains();

    static std::unordered_set<typename SdkApi::buffer_tracing_kind>
    get_buffered_domains();

    static std::vector<std::int32_t> get_operations(
        typename SdkApi::callback_tracing_kind kindv);

    static std::vector<std::int32_t> get_operations(
        typename SdkApi::buffer_tracing_kind kindv);

    static std::vector<std::string> get_rocm_events();

    static std::unordered_set<std::int32_t> get_backtrace_operations(
        typename SdkApi::callback_tracing_kind kindv);

    static std::unordered_set<std::int32_t> get_backtrace_operations(
        typename SdkApi::buffer_tracing_kind kindv);

private:
    static constexpr version_info compile_time_sdk_version =
        version_info::from_formatted(SdkApi::compile_time_version);

    static std::vector<std::int32_t> filter_operations(
        const std::unordered_set<std::int32_t>& complete,
        const std::unordered_set<std::int32_t>& include,
        const std::unordered_set<std::int32_t>& exclude);

    template <typename Tp>
    static std::string to_lower(const Tp& val);
    static std::string get_setting_name(const std::string& val);

    template <typename TracingKind>
        requires concepts::tracing_kind_for<SdkApi, TracingKind>
    static std::unordered_set<std::int32_t> parse_operation_string(
        TracingKind tracing_kind, const std::string& operations_setting = {});

    template <typename TracingKind, typename TracingNameTable,
              typename LoadTracingNamesFn>
    static std::unordered_set<std::int32_t> operation_ids_for_tracing_kind(
        TracingKind tracing_kind, const std::string& operations_setting,
        std::optional<TracingNameTable>& cached_tracing_names,
        LoadTracingNamesFn&&             load_tracing_names);

    [[noreturn]] static void finalize_and_throw(std::string_view exception_message);

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

    static std::vector<typename SdkApi::buffer_tracing_kind> kfd_kinds_for_name(
        std::string_view name);

    static void warn_kfd_disabled_once(std::string_view    domain_name,
                                       const version_info& kfd_version);

    struct operation_options_env_names
    {
        std::string operations_include_env_name            = {};
        std::string operations_exclude_env_name            = {};
        std::string operations_annotate_backtrace_env_name = {};
    };

    static std::unordered_map<typename SdkApi::callback_tracing_kind,
                              operation_options_env_names>
        s_callback_operation_option_env_names;
    static std::unordered_map<typename SdkApi::buffer_tracing_kind,
                              operation_options_env_names>
        s_buffered_operation_option_env_names;

    static version_info s_version;

    static std::optional<typename SdkApi::callback_name_info_t> s_callback_names;
    static std::optional<typename SdkApi::buffer_name_info_t>   s_buffer_names;
};

}  // namespace rocprofsys::rocprofiler_sdk

// ─── Template Implementations ────────────────────────────────────────────────

namespace rocprofsys::rocprofiler_sdk
{

// ─── Private helpers ─────────────────────────────────────────────────────────

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
template <typename Tp>
std::string
sdk_tracing_config<SdkApi, Externals>::to_lower(const Tp& value)
{
    auto str_copy = std::string{ value };

    for(auto& itr : str_copy)
    {
        itr = static_cast<char>(::tolower(itr));
    }

    return str_copy;
}

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::string
sdk_tracing_config<SdkApi, Externals>::get_setting_name(const std::string& value)
{
    constexpr auto prefix             = std::string_view{ "rocprofsys_" };
    const auto     lower_setting_name = to_lower(value);

    if(lower_setting_name.starts_with(prefix))
    {
        return lower_setting_name.substr(prefix.length());
    }

    return lower_setting_name;
}

// ─── Static data members ─────────────────────────────────────────────────────

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::unordered_map<
    typename SdkApi::callback_tracing_kind,
    typename sdk_tracing_config<SdkApi, Externals>::operation_options_env_names>
    sdk_tracing_config<SdkApi, Externals>::s_callback_operation_option_env_names{};

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::unordered_map<
    typename SdkApi::buffer_tracing_kind,
    typename sdk_tracing_config<SdkApi, Externals>::operation_options_env_names>
    sdk_tracing_config<SdkApi, Externals>::s_buffered_operation_option_env_names{};

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::optional<typename SdkApi::callback_name_info_t>
    sdk_tracing_config<SdkApi, Externals>::s_callback_names{};

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::optional<typename SdkApi::buffer_name_info_t>
    sdk_tracing_config<SdkApi, Externals>::s_buffer_names{};

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
version_info sdk_tracing_config<SdkApi, Externals>::s_version{};

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
[[noreturn]] void
sdk_tracing_config<SdkApi, Externals>::finalize_and_throw(
    std::string_view exception_message)
{
    Externals::ProcessState::set(Externals::ProcessState::Finalized);
    throw std::runtime_error(std::string{ exception_message });
}

// ─── get_operations_impl (tracing kind + optional setting) ───────────────────
template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
template <typename TracingKind, typename TracingNameTable, typename LoadTracingNamesFn>
std::unordered_set<std::int32_t>
sdk_tracing_config<SdkApi, Externals>::operation_ids_for_tracing_kind(
    TracingKind tracing_kind, const std::string& operations_setting,
    std::optional<TracingNameTable>& cached_tracing_names,
    LoadTracingNamesFn&&             load_tracing_names)
{
    if(!cached_tracing_names)
    {
        cached_tracing_names = load_tracing_names();
    }

    const auto& operation_items = (*cached_tracing_names)[tracing_kind].items();

    if(operations_setting.empty())
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

    auto operations_filter = Externals::get_setting_value(operations_setting);
    if(!operations_filter)
    {
        finalize_and_throw(fmt::format(
            "sdk_tracing_config::get_operations_impl: no registered setting '{}'",
            operations_setting));
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

            LOG_DEBUG("{} ('{}') matched: {}", operations_setting, pattern,
                      operation_label);
            matched_operation_ids.insert(operation_id);
        }
    }
    return matched_operation_ids;
}

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
template <typename TracingKind>
    requires concepts::tracing_kind_for<SdkApi, TracingKind>
std::unordered_set<std::int32_t>
sdk_tracing_config<SdkApi, Externals>::parse_operation_string(
    TracingKind tracing_kind, const std::string& operations_setting)
{
    if constexpr(std::same_as<TracingKind, typename SdkApi::callback_tracing_kind>)
    {
        return operation_ids_for_tracing_kind(
            tracing_kind, operations_setting, s_callback_names,
            [] { return SdkApi::get_callback_tracing_names(); });
    }
    else
    {
        return operation_ids_for_tracing_kind(
            tracing_kind, operations_setting, s_buffer_names,
            [] { return SdkApi::get_buffer_tracing_names(); });
    }
}

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::vector<std::int32_t>
sdk_tracing_config<SdkApi, Externals>::filter_operations(
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
/// @return The version of the rocprofiler-sdk or 0 if not initialized
template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
version_info&
sdk_tracing_config<SdkApi, Externals>::get_version()
{
    if(s_version.formatted() == 0)
    {
        SdkApi::get_version(&s_version.major, &s_version.minor, &s_version.patch);
    }

    return s_version;
}

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
typename sdk_tracing_config<SdkApi, Externals>::domain_choice_settings
sdk_tracing_config<SdkApi, Externals>::domain_settings()
{
    const auto buffered_tracing_info = SdkApi::get_buffer_tracing_names();
    const auto callback_tracing_info = SdkApi::get_callback_tracing_names();

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

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::vector<typename sdk_tracing_config<SdkApi, Externals>::operation_setting_spec>
sdk_tracing_config<SdkApi, Externals>::operation_settings()
{
    const auto buffered_tracing_info = SdkApi::get_buffer_tracing_names();
    const auto callback_tracing_info = SdkApi::get_callback_tracing_names();

    const auto domains_to_skip =
        std::unordered_set<std::string_view>{ "none",
                                              "correlation_id_retirement",
                                              "marker_core_api",
                                              "marker_control_api",
                                              "marker_name_api",
                                              "code_object",
                                              "kernel_dispatch",
                                              "page_migration" };

    auto result = std::vector<operation_setting_spec>{};

    auto gather_domain_f = [&domains_to_skip, &result](std::string_view domain_name,
                                                       const auto&      concrete_domain,
                                                       auto& operation_option_names) {
        const auto domain_lowercase = to_lower(domain_name);

        if(domains_to_skip.contains(domain_lowercase))
        {
            return;
        }

        auto operation_choices = std::vector<std::string>{};
        operation_choices.insert(operation_choices.end(),
                                 concrete_domain.operations.begin(),
                                 concrete_domain.operations.end());

        if(operation_choices.empty())
        {
            return;
        }

        const auto all_operation_options_env_name =
            fmt::format("ROCPROFSYS_ROCM_{}_OPERATIONS", domain_name);
        const auto exclude_operation_options_env_name =
            fmt::format("ROCPROFSYS_ROCM_{}_OPERATIONS_EXCLUDE", domain_name);
        const auto annotate_backtrace_operation_options_env_name =
            fmt::format("ROCPROFSYS_ROCM_{}_OPERATIONS_ANNOTATE_BACKTRACE", domain_name);

        operation_option_names.emplace(
            concrete_domain.value,
            operation_options_env_names{ all_operation_options_env_name,
                                         exclude_operation_options_env_name,
                                         annotate_backtrace_operation_options_env_name });

        result.push_back(operation_setting_spec{
            std::move(all_operation_options_env_name),
            std::move(exclude_operation_options_env_name),
            std::move(annotate_backtrace_operation_options_env_name),
            std::move(operation_choices) });
    };

    gather_domain_f("MARKER_API",
                    callback_tracing_info[SdkApi::CALLBACK_TRACING_MARKER_CORE_API],
                    s_callback_operation_option_env_names);

    for(const auto& itr : callback_tracing_info)
    {
        gather_domain_f(itr.name, itr, s_callback_operation_option_env_names);
    }

    for(const auto& itr : buffered_tracing_info)
    {
        gather_domain_f(itr.name, itr, s_buffered_operation_option_env_names);
    }

    return result;
}

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::unordered_set<typename SdkApi::callback_tracing_kind>
sdk_tracing_config<SdkApi, Externals>::get_callback_domains()
{
    using kind_t             = typename SdkApi::callback_tracing_kind;
    const auto callback_info = SdkApi::get_callback_tracing_names();
    auto       supported     = std::unordered_set<kind_t>{
        SdkApi::CALLBACK_TRACING_HSA_CORE_API,
        SdkApi::CALLBACK_TRACING_HSA_AMD_EXT_API,
        SdkApi::CALLBACK_TRACING_HSA_IMAGE_EXT_API,
        SdkApi::CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
        SdkApi::CALLBACK_TRACING_HIP_RUNTIME_API,
        SdkApi::CALLBACK_TRACING_HIP_COMPILER_API,
        SdkApi::CALLBACK_TRACING_MARKER_CORE_API,
        SdkApi::CALLBACK_TRACING_CODE_OBJECT,
    };

    const auto& sdk_runtime_version = get_version();
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
            supported.emplace(SdkApi::CALLBACK_TRACING_RCCL_API);
            supported.emplace(SdkApi::CALLBACK_TRACING_OMPT);
            supported.emplace(SdkApi::CALLBACK_TRACING_ROCDECODE_API);
        }
    }
    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 0, .minor = 7, .patch = 0 })
    {
        if(sdk_runtime_version >= version_info{ .major = 0, .minor = 7, .patch = 0 })
        {
            supported.emplace(SdkApi::CALLBACK_TRACING_ROCJPEG_API);
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
            callback_domains.emplace(SdkApi::CALLBACK_TRACING_RCCL_API);
        }
        if(Externals::get_use_ompt() &&
           sdk_runtime_version >= version_info{ .major = 0, .minor = 6, .patch = 0 })
        {
            // Translate some configuration settings to rocprofiler domains
            callback_domains.emplace(SdkApi::CALLBACK_TRACING_OMPT);
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
            for(auto eitr : { SdkApi::CALLBACK_TRACING_HSA_CORE_API,
                              SdkApi::CALLBACK_TRACING_HSA_AMD_EXT_API,
                              SdkApi::CALLBACK_TRACING_HSA_IMAGE_EXT_API,
                              SdkApi::CALLBACK_TRACING_HSA_FINALIZE_EXT_API })
                callback_domains.emplace(eitr);
        }
        else if(itr == "hip_api")
        {
            for(auto eitr : { SdkApi::CALLBACK_TRACING_HIP_RUNTIME_API,
                              SdkApi::CALLBACK_TRACING_HIP_COMPILER_API })
                callback_domains.emplace(eitr);
        }
        else if(itr == "marker_api" || itr == "roctx")
        {
            callback_domains.emplace(SdkApi::CALLBACK_TRACING_MARKER_CORE_API);
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

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
typename sdk_tracing_config<SdkApi, Externals>::kfd_runtime_support
sdk_tracing_config<SdkApi, Externals>::get_kfd_runtime_support()
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

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
bool
sdk_tracing_config<SdkApi, Externals>::is_kfd_domain_name(std::string_view name)
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

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::vector<typename SdkApi::buffer_tracing_kind>
sdk_tracing_config<SdkApi, Externals>::kfd_kinds_for_name(std::string_view name)
{
    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 0, .patch = 0 })
    {
        using kind_t = typename SdkApi::buffer_tracing_kind;
        static const auto kfd_kind_table =
            std::unordered_map<std::string_view, std::vector<kind_t>>{
                { "kfd_events",
                  { SdkApi::BUFFER_TRACING_KFD_PAGE_FAULT,
                    SdkApi::BUFFER_TRACING_KFD_PAGE_MIGRATE,
                    SdkApi::BUFFER_TRACING_KFD_QUEUE,
                    SdkApi::BUFFER_TRACING_KFD_EVENT_QUEUE,
                    SdkApi::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                    SdkApi::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS } },
                { "kfd_page_fault", { SdkApi::BUFFER_TRACING_KFD_PAGE_FAULT } },
                { "kfd_page_migrate", { SdkApi::BUFFER_TRACING_KFD_PAGE_MIGRATE } },
                { "kfd_queue", { SdkApi::BUFFER_TRACING_KFD_QUEUE } },
                { "kfd_event_queue", { SdkApi::BUFFER_TRACING_KFD_EVENT_QUEUE } },
                { "kfd_event_unmap_from_gpu",
                  { SdkApi::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU } },
                { "kfd_event_dropped_events",
                  { SdkApi::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS } },
            };

        const auto itr = kfd_kind_table.find(name);
        return itr != kfd_kind_table.end() ? itr->second : std::vector<kind_t>{};
    }
    else
    {
        return {};
    }
}

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
void
sdk_tracing_config<SdkApi, Externals>::warn_kfd_disabled_once(
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

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::unordered_set<typename SdkApi::buffer_tracing_kind>
sdk_tracing_config<SdkApi, Externals>::get_buffered_domains()
{
    using kind_t           = typename SdkApi::buffer_tracing_kind;
    const auto buffer_info = SdkApi::get_buffer_tracing_names();

    auto supported = std::unordered_set<kind_t>{
        SdkApi::BUFFER_TRACING_KERNEL_DISPATCH,
        SdkApi::BUFFER_TRACING_MEMORY_COPY,
        SdkApi::BUFFER_TRACING_SCRATCH_MEMORY,
    };

    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 0, .minor = 6, .patch = 0 })
    {
        supported.emplace(SdkApi::BUFFER_TRACING_MEMORY_ALLOCATION);
    }

    if constexpr(compile_time_sdk_version <
                 version_info{ .major = 1, .minor = 0, .patch = 0 })
    {
        supported.emplace(SdkApi::BUFFER_TRACING_PAGE_MIGRATION);
    }

    if constexpr(compile_time_sdk_version >=
                 version_info{ .major = 1, .minor = 0, .patch = 0 })
    {
        supported.emplace(SdkApi::BUFFER_TRACING_KFD_PAGE_FAULT);
        supported.emplace(SdkApi::BUFFER_TRACING_KFD_PAGE_MIGRATE);
        supported.emplace(SdkApi::BUFFER_TRACING_KFD_QUEUE);
        supported.emplace(SdkApi::BUFFER_TRACING_KFD_EVENT_QUEUE);
        supported.emplace(SdkApi::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU);
        supported.emplace(SdkApi::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS);
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
            for(const auto& eitr : { SdkApi::BUFFER_TRACING_HSA_CORE_API,
                                     SdkApi::BUFFER_TRACING_HSA_AMD_EXT_API,
                                     SdkApi::BUFFER_TRACING_HSA_IMAGE_EXT_API,
                                     SdkApi::BUFFER_TRACING_HSA_FINALIZE_EXT_API })
            {
                data.emplace(eitr);
            }
        }
        else if(itr == "hip_api")
        {
            for(const auto& eitr : { SdkApi::BUFFER_TRACING_HIP_COMPILER_API,
                                     SdkApi::BUFFER_TRACING_HIP_RUNTIME_API })
            {
                data.emplace(eitr);
            }
        }
        else if(itr == "marker_api" || itr == "roctx")
        {
            data.emplace(SdkApi::BUFFER_TRACING_MARKER_CORE_API);
        }
        else if(itr == "memory_allocation")
        {
            if constexpr(compile_time_sdk_version >=
                         version_info{ .major = 0, .minor = 6, .patch = 0 })
            {
                data.emplace(SdkApi::BUFFER_TRACING_MEMORY_ALLOCATION);
            }
        }
        else if(itr == "memory_copy")
        {
            data.emplace(SdkApi::BUFFER_TRACING_MEMORY_COPY);
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
                data.emplace(SdkApi::BUFFER_TRACING_KFD_PAGE_FAULT);
                data.emplace(SdkApi::BUFFER_TRACING_KFD_PAGE_MIGRATE);
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

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::vector<std::string>
sdk_tracing_config<SdkApi, Externals>::get_rocm_events()
{
    return rocprofsys::delimit(Externals::get_rocm_events_setting(), " ,;\t\n");
}

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::vector<std::int32_t>
sdk_tracing_config<SdkApi, Externals>::get_operations(
    typename SdkApi::callback_tracing_kind kind)
{
    if(s_callback_operation_option_env_names.count(kind) == 0)
    {
        finalize_and_throw(
            fmt::format("sdk_tracing_config::get_operations: no options registered for "
                        "callback tracing kind {}",
                        static_cast<int>(kind)));
    }

    const auto complete_set = parse_operation_string(kind);

    const auto& opts = s_callback_operation_option_env_names.at(kind);
    // Empty option string means "no filter" — produce an empty set so the
    // three-argument overload falls through to the complete set / removes nothing.
    const auto include_operations =
        opts.operations_include.empty()
            ? std::unordered_set<std::int32_t>{}
            : parse_operation_string(kind, opts.operations_include);
    const auto exclude_operations =
        opts.operations_exclude.empty()
            ? std::unordered_set<std::int32_t>{}
            : parse_operation_string(kind, opts.operations_exclude);

    return filter_operations(complete_set, include_operations, exclude_operations);
}

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::vector<std::int32_t>
sdk_tracing_config<SdkApi, Externals>::get_operations(
    typename SdkApi::buffer_tracing_kind kind)
{
    if(!s_buffered_operation_option_env_names.contains(kind))
    {
        finalize_and_throw(
            fmt::format("sdk_tracing_config::get_operations: no options registered for "
                        "buffer tracing kind {}",
                        static_cast<int>(kind)));
    }

    const auto& opts         = s_buffered_operation_option_env_names.at(kind);
    const auto  complete_set = parse_operation_string(kind);
    const auto  include_operations =
        opts.operations_include.empty()
            ? std::unordered_set<std::int32_t>{}
            : parse_operation_string(kind, opts.operations_include);
    const auto exclude_operations =
        opts.operations_exclude.empty()
            ? std::unordered_set<std::int32_t>{}
            : parse_operation_string(kind, opts.operations_exclude);

    return filter_operations(complete_set, include_operations, exclude_operations);
}

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::unordered_set<std::int32_t>
sdk_tracing_config<SdkApi, Externals>::get_backtrace_operations(
    typename SdkApi::callback_tracing_kind kind)
{
    if(!s_callback_operation_option_env_names.contains(kind))
    {
        finalize_and_throw(fmt::format(
            "sdk_tracing_config::get_backtrace_operations: no options registered for "
            "callback tracing kind {}",
            static_cast<int>(kind)));
    }

    const auto& annotate_backtrace_operations =
        s_callback_operation_option_env_names.at(kind).operations_annotate_backtrace;
    if(annotate_backtrace_operations.empty())
    {
        return {};
    }

    const auto result = parse_operation_string(kind, annotate_backtrace_operations);
    return { result.begin(), result.end() };
}

template <typename SdkApi, typename Externals>
    requires concepts::sdk_tracing_config_externals<Externals>
std::unordered_set<std::int32_t>
sdk_tracing_config<SdkApi, Externals>::get_backtrace_operations(
    typename SdkApi::buffer_tracing_kind kind)
{
    if(!s_buffered_operation_option_env_names.contains(kind))
    {
        finalize_and_throw(fmt::format(
            "sdk_tracing_config::get_backtrace_operations: no options registered for "
            "buffer tracing kind {}",
            static_cast<int>(kind)));
    }

    const auto& annotate_backtrace_operations =
        s_buffered_operation_option_env_names.at(kind).operations_annotate_backtrace;
    if(annotate_backtrace_operations.empty())
    {
        return {};
    }

    const auto result = parse_operation_string(kind, annotate_backtrace_operations);
    return { result.begin(), result.end() };
}

}  // namespace rocprofsys::rocprofiler_sdk
