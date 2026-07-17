// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/wrapper.hpp"
#include "common/delimit.hpp"
#include "common/env_vars.hpp"
#include "core/config.hpp"
#include "core/timemory.hpp"
#include "logger/debug.hpp"

#include <spdlog/fmt/ranges.h>

#include <algorithm>
#include <cstdint>
#include <memory>
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

struct version_info
{
    std::uint32_t major     = 0;
    std::uint32_t minor     = 0;
    std::uint32_t patch     = 0;
    std::uint32_t formatted = 0;  // major * 10000 + minor * 100 + patch
};

// Default Externals for sdk_core: reads settings and config from the global singletons.
// Replaced by a mock in tests.
struct default_sdk_externals
{
    using Settings = tim::settings;
    static Settings* get_settings() { return ::rocprofsys::settings::instance(); }
    static bool      get_use_rcclp() { return ::rocprofsys::config::get_use_rcclp(); }
    static bool      get_use_ompt() { return ::rocprofsys::config::get_use_ompt(); }
    static bool      get_use_unified_memory_profiling()
    {
        return ::rocprofsys::config::get_use_unified_memory_profiling();
    }
    static bool get_use_process_sampling()
    {
        return ::rocprofsys::config::get_use_process_sampling();
    }
    static std::string get_trace_region()
    {
        return ::rocprofsys::config::get_trace_region();
    }
    static std::string get_rocm_domains()
    {
        return ::rocprofsys::get_setting_value<std::string>(
                   std::string{ ::rocprofsys::env_vars::ROCM_DOMAINS })
            .value_or(std::string{});
    }
    static std::string get_rocm_events_setting()
    {
        return ::rocprofsys::get_setting_value<std::string>(
                   std::string{ ::rocprofsys::env_vars::ROCM_EVENTS })
            .value_or(std::string{});
    }
    static std::string get_gpu_perf_counters()
    {
        return ::rocprofsys::get_gpu_perf_counters();
    }

    static std::optional<std::string> get_setting_value(std::string_view s)
    {
        return ::rocprofsys::get_setting_value<std::string>(std::string{ s });
    }
};

template <typename Wrapper, typename Externals = default_sdk_externals>
class sdk_core
{
public:
    static void config_settings(const std::shared_ptr<typename Externals::Settings>&);

    static version_info& get_version();

    // Reset cached state so the next tool_init call re-queries every Wrapper
    // method exactly once.  For unit tests only — allows pinning Times(N).
    // Direct assignment to s_version avoids calling get_version() (which would
    // call Wrapper::get_version() via the cache-miss path before any EXPECT_CALL
    // is registered, generating spurious "uninteresting" GMock warnings).
    static void reset_version_cache() noexcept { s_version = version_info{ 0 }; }

    static void reset_tracing_names_cache() noexcept
    {
        s_callback_names.reset();
        s_buffer_names.reset();
    }

    static std::unordered_set<typename Wrapper::callback_tracing_kind>
    get_callback_domains();

    static std::unordered_set<typename Wrapper::buffer_tracing_kind>
    get_buffered_domains();

    static std::vector<std::int32_t> get_operations(
        typename Wrapper::callback_tracing_kind kindv);

    static std::vector<std::int32_t> get_operations(
        typename Wrapper::buffer_tracing_kind kindv);

    static std::vector<std::string> get_rocm_events();

    static std::unordered_set<std::int32_t> get_backtrace_operations(
        typename Wrapper::callback_tracing_kind kindv);

    static std::unordered_set<std::int32_t> get_backtrace_operations(
        typename Wrapper::buffer_tracing_kind kindv);

    // Pure filtering logic — public so it can be unit-tested directly.
    static std::vector<std::int32_t> get_operations_impl(
        const std::unordered_set<std::int32_t>& complete,
        const std::unordered_set<std::int32_t>& include,
        const std::unordered_set<std::int32_t>& exclude);

    // Helper functions — public for direct unit testing.
    template <typename Tp>
    static std::string to_lower(const Tp& val);
    static std::string get_setting_name(std::string val);

    // Test-only: directly inject operation options for a tracing kind without
    // running config_settings(). Enables unit-testing get_operations() and
    // get_backtrace_operations() without the settings infrastructure.
    static void set_operation_options(typename Wrapper::callback_tracing_kind kind,
                                      std::string include, std::string exclude,
                                      std::string backtrace);
    static void set_operation_options(typename Wrapper::buffer_tracing_kind kind,
                                      std::string include, std::string exclude,
                                      std::string backtrace);

    struct operation_options
    {
        std::string operations_include            = {};
        std::string operations_exclude            = {};
        std::string operations_annotate_backtrace = {};
    };

    static std::unordered_map<typename Wrapper::callback_tracing_kind, operation_options>
        callback_operation_option_names;
    static std::unordered_map<typename Wrapper::buffer_tracing_kind, operation_options>
        buffered_operation_option_names;

    // Cached version — class-level so reset_version_cache() can clear it without
    // going through the cache-miss path (which would call Wrapper::get_version()).
    static version_info s_version;

    // Cached copies used by get_operations_impl; populated on first call and
    // reset by reset_tracing_names_cache() so unit tests can pin Times(N).
    static std::optional<typename Wrapper::callback_name_info_t> s_callback_names;
    static std::optional<typename Wrapper::buffer_name_info_t>   s_buffer_names;

    static std::unordered_set<std::int32_t> get_operations_impl(
        typename Wrapper::callback_tracing_kind tracing_kind,
        const std::string&                      operations_setting = {});

    static std::unordered_set<std::int32_t> get_operations_impl(
        typename Wrapper::buffer_tracing_kind tracing_kind,
        const std::string&                    operations_setting = {});

    template <typename Tp>
    static auto insert_config_setting(

        const std::shared_ptr<typename Externals::Settings>& config,
        std::string_view env_name, std::string_view description, Tp initial_value,
        std::initializer_list<std::string_view> extra_categories);

private:
    template <typename TracingKind, typename TracingNameTable,
              typename LoadTracingNamesFn>
    static std::unordered_set<std::int32_t> operation_ids_for_tracing_kind(
        TracingKind tracing_kind, const std::string& operations_setting,
        std::optional<TracingNameTable>& cached_tracing_names,
        LoadTracingNamesFn&&             load_tracing_names);
};

using core_sdk = sdk_core<backend>;

}  // namespace rocprofsys::rocprofiler_sdk

// ─── Template Implementations ────────────────────────────────────────────────

namespace rocprofsys::rocprofiler_sdk
{

// ─── Private helpers ─────────────────────────────────────────────────────────

template <typename Wrapper, typename Externals>
template <typename Tp>
std::string
sdk_core<Wrapper, Externals>::to_lower(const Tp& _val)
{
    auto _v = std::string{ _val };
    for(auto& itr : _v)
        itr = ::tolower(itr);
    return _v;
}

template <typename Wrapper, typename Externals>
std::string
sdk_core<Wrapper, Externals>::get_setting_name(std::string _v)
{
    constexpr auto _prefix = std::string_view{ "rocprofsys_" };
    for(auto& itr : _v)
        itr = tolower(itr);
    if(_v.starts_with(_prefix)) return _v.substr(_prefix.length());
    return _v;
}

// ─── set_operation_options ────────────────────────────────────────────────────

template <typename Wrapper, typename Externals>
void
sdk_core<Wrapper, Externals>::set_operation_options(
    typename Wrapper::callback_tracing_kind kind, std::string include,
    std::string exclude, std::string backtrace)
{
    callback_operation_option_names[kind] = { std::move(include), std::move(exclude),
                                              std::move(backtrace) };
}

template <typename Wrapper, typename Externals>
void
sdk_core<Wrapper, Externals>::set_operation_options(
    typename Wrapper::buffer_tracing_kind kind, std::string include, std::string exclude,
    std::string backtrace)
{
    buffered_operation_option_names[kind] = { std::move(include), std::move(exclude),
                                              std::move(backtrace) };
}

// ─── Static data members ─────────────────────────────────────────────────────

template <typename Wrapper, typename Externals>
std::unordered_map<typename Wrapper::callback_tracing_kind,
                   typename sdk_core<Wrapper, Externals>::operation_options>
    sdk_core<Wrapper, Externals>::callback_operation_option_names{};

template <typename Wrapper, typename Externals>
std::unordered_map<typename Wrapper::buffer_tracing_kind,
                   typename sdk_core<Wrapper, Externals>::operation_options>
    sdk_core<Wrapper, Externals>::buffered_operation_option_names{};

template <typename Wrapper, typename Externals>
std::optional<typename Wrapper::callback_name_info_t>
    sdk_core<Wrapper, Externals>::s_callback_names{};

template <typename Wrapper, typename Externals>
std::optional<typename Wrapper::buffer_name_info_t>
    sdk_core<Wrapper, Externals>::s_buffer_names{};

template <typename Wrapper, typename Externals>
version_info sdk_core<Wrapper, Externals>::s_version{};

// ─── get_operations_impl (tracing kind + optional setting) ───────────────────

template <typename Wrapper, typename Externals>
template <typename TracingKind, typename TracingNameTable, typename LoadTracingNamesFn>
std::unordered_set<std::int32_t>
sdk_core<Wrapper, Externals>::operation_ids_for_tracing_kind(
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
        ::rocprofsys::set_state(::rocprofsys::State::Finalized);
        throw std::runtime_error(
            fmt::format("sdk_core::get_operations_impl: no registered setting '{}'",
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

template <typename Wrapper, typename Externals>
std::unordered_set<std::int32_t>
sdk_core<Wrapper, Externals>::get_operations_impl(
    typename Wrapper::callback_tracing_kind tracing_kind,
    const std::string&                      operations_setting)
{
    return operation_ids_for_tracing_kind(
        tracing_kind, operations_setting, s_callback_names,
        [] { return Wrapper::get_callback_tracing_names(); });
}

template <typename Wrapper, typename Externals>
std::unordered_set<std::int32_t>
sdk_core<Wrapper, Externals>::get_operations_impl(
    typename Wrapper::buffer_tracing_kind tracing_kind,
    const std::string&                    operations_setting)
{
    return operation_ids_for_tracing_kind(
        tracing_kind, operations_setting, s_buffer_names,
        [] { return Wrapper::get_buffer_tracing_names(); });
}

template <typename Wrapper, typename Externals>
std::vector<std::int32_t>
sdk_core<Wrapper, Externals>::get_operations_impl(
    const std::unordered_set<std::int32_t>& _complete,
    const std::unordered_set<std::int32_t>& _include,
    const std::unordered_set<std::int32_t>& _exclude)
{
    auto _convert = [](const auto& _dset) {
        auto _dret = std::vector<std::int32_t>{};
        _dret.reserve(_dset.size());
        for(auto itr : _dset)
            _dret.emplace_back(itr);
        std::ranges::sort(_dret);
        return _dret;
    };

    if(_include.empty() && _exclude.empty()) return _convert(_complete);

    auto _ret = (_include.empty()) ? _complete : _include;
    for(auto itr : _exclude)
        _ret.erase(itr);

    return _convert(_ret);
}

template <typename Wrapper, typename Externals>
template <typename Tp>
auto
sdk_core<Wrapper, Externals>::insert_config_setting(
    const std::shared_ptr<typename Externals::Settings>& config,
    std::string_view env_name, std::string_view description, Tp initial_value,
    std::initializer_list<std::string_view> extra_categories)
{
    auto env_str    = std::string{ env_name };
    auto categories = std::set<std::string>{ "custom", "rocprofsys", "librocprof-sys" };
    for(auto cat : extra_categories)
        categories.emplace(cat);

    auto [it, inserted] = config->template insert<Tp, Tp>(
        env_str, get_setting_name(env_str), std::string{ description },
        Tp{ std::move(initial_value) }, std::move(categories));

    if(!inserted)
        LOG_WARNING("Duplicate setting: {} / {}", get_setting_name(env_str), env_str);

    return config->find(env_str)->second;
}

// ─── Public method implementations ───────────────────────────────────────────

/// @brief Return the version of the rocprofiler-sdk
/// @return The version of the rocprofiler-sdk or 0 if not initialized
template <typename Wrapper, typename Externals>
version_info&
sdk_core<Wrapper, Externals>::get_version()
{
    if(s_version.formatted == 0)
    {
        std::uint32_t _major = 0;
        std::uint32_t _minor = 0;
        std::uint32_t _patch = 0;

        Wrapper::get_version(&_major, &_minor, &_patch);

        s_version.major     = _major;
        s_version.minor     = _minor;
        s_version.patch     = _patch;
        s_version.formatted = _major * 10000u + _minor * 100u + _patch;
    }

    return s_version;
}

template <typename Wrapper, typename Externals>
void
sdk_core<Wrapper, Externals>::config_settings(
    const std::shared_ptr<typename Externals::Settings>& _config)
{
    const auto buffered_tracing_info = Wrapper::get_buffer_tracing_names();
    const auto callback_tracing_info = Wrapper::get_callback_tracing_names();

    auto _skip_domains =
        std::unordered_set<std::string_view>{ "none",
                                              "correlation_id_retirement",
                                              "marker_core_api",
                                              "marker_control_api",
                                              "marker_name_api",
                                              "code_object" };

    auto _domain_choices = std::vector<std::string>{};
    auto _add_domain     = [&_domain_choices, &_skip_domains](std::string_view _domain) {
        auto _v = to_lower(_domain);
        if(!_skip_domains.contains(_v) &&
           std::ranges::find(_domain_choices, _v) == _domain_choices.end())
            _domain_choices.emplace_back(_v);
    };

    static auto _option_names           = std::unordered_set<std::string>{};
    auto        _add_operation_settings = [&_config, &_skip_domains](
                                       std::string_view _domain_name, const auto& _domain,
                                       auto& _operation_option_names) {
        auto _v = to_lower(_domain_name);

        if(_skip_domains.contains(_v)) return;

        auto _op_option_name = fmt::format("ROCPROFSYS_ROCM_{}_OPERATIONS", _domain_name);
        auto _eop_option_name =
            fmt::format("ROCPROFSYS_ROCM_{}_OPERATIONS_EXCLUDE", _domain_name);
        auto _bt_option_name =
            fmt::format("ROCPROFSYS_ROCM_{}_OPERATIONS_ANNOTATE_BACKTRACE", _domain_name);

        auto _op_choices = std::vector<std::string>{};
        for(auto itr : _domain.operations)
            _op_choices.emplace_back(std::string{ itr });

        if(_op_choices.empty()) return;

        _operation_option_names.emplace(
            _domain.value,
            operation_options{ _op_option_name, _eop_option_name, _bt_option_name });

        if(_option_names.emplace(_op_option_name).second)
        {
            insert_config_setting<std::string>(
                _config, _op_option_name,
                "Inclusive filter for domain operations (for API domains, this selects "
                "the functions to trace) [regex supported]",
                std::string{}, { "rocm", "rocprofiler-sdk", "advanced" })
                ->set_choices(_op_choices);
        }

        if(_option_names.emplace(_eop_option_name).second)
        {
            insert_config_setting<std::string>(
                _config, _eop_option_name,
                "Exclusive filter for domain operations applied after the inclusive "
                "filter (for API domains, removes function from trace) [regex supported]",
                std::string{}, { "rocm", "rocprofiler-sdk", "advanced" })
                ->set_choices(_op_choices);
        }

        if(_option_names.emplace(_bt_option_name).second)
        {
            insert_config_setting<std::string>(
                _config, _bt_option_name,
                "Specification of domain operations which will record a backtrace (for "
                "API domains, this is a list of function names) [regex supported]",
                std::string{}, { "rocm", "rocprofiler-sdk", "advanced" })
                ->set_choices(_op_choices);
        }
    };

    _domain_choices.reserve(buffered_tracing_info.size());
    _domain_choices.reserve(callback_tracing_info.size());
    _add_domain("hip_api");
    _add_domain("hsa_api");
    _add_domain("marker_api");
    _add_domain("roctx");
    if constexpr(Wrapper::compile_time_version >= 10000) _add_domain("kfd_events");

    for(const auto& itr : buffered_tracing_info)
        _add_domain(itr.name);

    for(const auto& itr : callback_tracing_info)
        _add_domain(itr.name);

    std::ranges::sort(_domain_choices);

    auto _domain_description =
        fmt::format("Specification of ROCm domains to trace/profile. Choices: {}",
                    fmt::join(_domain_choices, ", "));
    auto _domain_defaults = std::string{ "hip_runtime_api,marker_api,kernel_dispatch,"
                                         "memory_copy,scratch_memory" };
    if constexpr(Wrapper::compile_time_version < 10000)
        _domain_defaults.append(",page_migration");

    insert_config_setting<std::string>(_config, env_vars::ROCM_DOMAINS,
                                       _domain_description, _domain_defaults,
                                       { "rocm", "rocprofiler-sdk" })
        ->set_choices(_domain_choices);

    insert_config_setting<std::string>(
        _config, env_vars::ROCM_EVENTS,
        "ROCm hardware counters. Use ':device=N' syntax to specify collection on device "
        "number N, e.g. ':device=0'. If no device specification is provided, the event "
        "is collected on every available device",
        std::string{}, { "rocm", "hardware_counters" });

    _skip_domains.emplace("kernel_dispatch");
    _skip_domains.emplace("page_migration");

    _add_operation_settings(
        "MARKER_API", callback_tracing_info[Wrapper::CALLBACK_TRACING_MARKER_CORE_API],
        callback_operation_option_names);

    for(const auto& itr : callback_tracing_info)
        _add_operation_settings(itr.name, itr, callback_operation_option_names);

    for(const auto& itr : buffered_tracing_info)
        _add_operation_settings(itr.name, itr, buffered_operation_option_names);

    // Add the ROCPROFSYS_ROCM_GROUP_BY_QUEUE setting if the hip_stream domain is present
    // in supported ROCProfiler-SDK domains.
    auto _has_hip_stream =
        std::ranges::find(_domain_choices, std::string{ "hip_stream" }) !=
        _domain_choices.end();

    if(_has_hip_stream)
    {
        insert_config_setting<bool>(
            _config, env_vars::ROCM_GROUP_BY_QUEUE,
            "By default, Perfetto trace will show the HIP streams to which kernel "
            "and memory copy operations submitted. With the "
            "`ROCPROFSYS_ROCM_GROUP_BY_QUEUE` option, the trace will display HSA queues "
            "to which these kernel and memory operations were submitted.",
            false, { "rocm", "perfetto" });
    }
}

template <typename Wrapper, typename Externals>
std::unordered_set<typename Wrapper::callback_tracing_kind>
sdk_core<Wrapper, Externals>::get_callback_domains()
{
    using kind_t             = typename Wrapper::callback_tracing_kind;
    const auto callback_info = Wrapper::get_callback_tracing_names();
    auto       supported     = std::unordered_set<kind_t>{
        Wrapper::CALLBACK_TRACING_HSA_CORE_API,
        Wrapper::CALLBACK_TRACING_HSA_AMD_EXT_API,
        Wrapper::CALLBACK_TRACING_HSA_IMAGE_EXT_API,
        Wrapper::CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
        Wrapper::CALLBACK_TRACING_HIP_RUNTIME_API,
        Wrapper::CALLBACK_TRACING_HIP_COMPILER_API,
        Wrapper::CALLBACK_TRACING_MARKER_CORE_API,
        Wrapper::CALLBACK_TRACING_CODE_OBJECT,
    };

    auto _version = get_version();
    if(_version.formatted == 0)
    {
        LOG_WARNING("rocprofiler-sdk version not initialized");
    }

    if constexpr(Wrapper::compile_time_version >= 600)
    {
        if(_version.formatted >= 600)
        {
            // Argument tracing is supported in rocprofiler-sdk 0.6.0 and later
            supported.emplace(Wrapper::CALLBACK_TRACING_RCCL_API);
            supported.emplace(Wrapper::CALLBACK_TRACING_OMPT);
            supported.emplace(Wrapper::CALLBACK_TRACING_ROCDECODE_API);
        }
    }
    if constexpr(Wrapper::compile_time_version >= 700)
    {
        if(_version.formatted >= 700)
            supported.emplace(Wrapper::CALLBACK_TRACING_ROCJPEG_API);
    }

    auto _data    = std::unordered_set<kind_t>{};
    auto _domains = rocprofsys::delimit(Externals::get_rocm_domains(), " ,;:\t\n");

    if constexpr(Wrapper::compile_time_version >= 600)
    {
        if(Externals::get_use_rcclp() && _version.formatted >= 600)
        {
            // Translate ROCPROFSYS_USE_RCCLP to entry in ROCPROFSYS_ROCM_DOMAINS
            _data.emplace(Wrapper::CALLBACK_TRACING_RCCL_API);
        }
        if(Externals::get_use_ompt() && _version.formatted >= 600)
        {
            // Translate some configuration settings to rocprofiler domains
            _data.emplace(Wrapper::CALLBACK_TRACING_OMPT);
        }
    }

    // Check that the domains are valid
    const auto valid_choices = Externals::get_settings()
                                   ->at(std::string{ env_vars::ROCM_DOMAINS })
                                   ->get_choices();
    auto invalid_domain = [&valid_choices](const auto& domainv) {
        return !std::ranges::any_of(
            valid_choices, [&domainv](const auto& choice) { return choice == domainv; });
    };

    for(const auto& itr : _domains)
    {
        if(invalid_domain(itr))
            throw std::runtime_error(
                fmt::format("unsupported ROCPROFSYS_ROCM_DOMAINS value: {}", itr));

        if(itr == "hsa_api")
        {
            for(auto eitr : { Wrapper::CALLBACK_TRACING_HSA_CORE_API,
                              Wrapper::CALLBACK_TRACING_HSA_AMD_EXT_API,
                              Wrapper::CALLBACK_TRACING_HSA_IMAGE_EXT_API,
                              Wrapper::CALLBACK_TRACING_HSA_FINALIZE_EXT_API })
                _data.emplace(eitr);
        }
        else if(itr == "hip_api")
        {
            for(auto eitr : { Wrapper::CALLBACK_TRACING_HIP_RUNTIME_API,
                              Wrapper::CALLBACK_TRACING_HIP_COMPILER_API })
                _data.emplace(eitr);
        }
        else if(itr == "marker_api" || itr == "roctx")
        {
            _data.emplace(Wrapper::CALLBACK_TRACING_MARKER_CORE_API);
        }
        else
        {
            for(size_t idx = 0; idx < callback_info.size(); ++idx)
            {
                const auto& ditr = callback_info[idx];
                auto        dval = static_cast<kind_t>(idx);
                if(itr == to_lower(ditr.name) && supported.count(dval) > 0)
                {
                    _data.emplace(dval);
                    break;
                }
            }
        }
    }

    return _data;
}

template <typename Wrapper, typename Externals>
std::unordered_set<typename Wrapper::buffer_tracing_kind>
sdk_core<Wrapper, Externals>::get_buffered_domains()
{
    using kind_t           = typename Wrapper::buffer_tracing_kind;
    const auto buffer_info = Wrapper::get_buffer_tracing_names();
    auto       supported   = std::unordered_set<kind_t>{
        Wrapper::BUFFER_TRACING_KERNEL_DISPATCH,
        Wrapper::BUFFER_TRACING_MEMORY_COPY,
        Wrapper::BUFFER_TRACING_SCRATCH_MEMORY,
    };
    if constexpr(Wrapper::compile_time_version >= 600)
        supported.emplace(Wrapper::BUFFER_TRACING_MEMORY_ALLOCATION);
    if constexpr(Wrapper::compile_time_version < 10000)
        supported.emplace(Wrapper::BUFFER_TRACING_PAGE_MIGRATION);
    if constexpr(Wrapper::compile_time_version >= 10000)
    {
        supported.emplace(Wrapper::BUFFER_TRACING_KFD_PAGE_FAULT);
        supported.emplace(Wrapper::BUFFER_TRACING_KFD_PAGE_MIGRATE);
        supported.emplace(Wrapper::BUFFER_TRACING_KFD_QUEUE);
        supported.emplace(Wrapper::BUFFER_TRACING_KFD_EVENT_QUEUE);
        supported.emplace(Wrapper::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU);
        supported.emplace(Wrapper::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS);
    }

    // rocprofiler-sdk < 1.2.2 has a fatal bug parsing KFD events with
    // undefined node IDs (0xFFFFFFFF). Guard at compile time via compile_time_version
    // (= ROCPROFILER_VERSION for production, mock-defined for tests) so the mock can
    // enable KFD paths without fighting the function-local-static version cache.
    version_info kfd_version{};
    bool         kfd_supported_by_runtime = false;
    if constexpr(Wrapper::compile_time_version >= 10000)
    {
        constexpr std::uint32_t kfd_min_version = 10202;  // 1.2.2
        kfd_version                             = get_version();
        kfd_supported_by_runtime = (Wrapper::compile_time_version >= kfd_min_version);
    }

    auto _data    = std::unordered_set<kind_t>{};
    auto _domains = rocprofsys::delimit(Externals::get_rocm_domains(), " ,;:\t\n");
    // Check that the domains are valid
    const auto valid_choices = Externals::get_settings()
                                   ->at(std::string{ env_vars::ROCM_DOMAINS })
                                   ->get_choices();
    auto invalid_domain = [&valid_choices](const auto& domainv) {
        return !std::ranges::any_of(
            valid_choices, [&domainv](const auto& choice) { return choice == domainv; });
    };

    for(const auto& itr : _domains)
    {
        if(invalid_domain(itr))
            throw std::runtime_error(
                fmt::format("unsupported ROCPROFSYS_ROCM_DOMAINS value: {}", itr));

        if(itr == "hsa_api")
        {
            for(auto eitr : { Wrapper::BUFFER_TRACING_HSA_CORE_API,
                              Wrapper::BUFFER_TRACING_HSA_AMD_EXT_API,
                              Wrapper::BUFFER_TRACING_HSA_IMAGE_EXT_API,
                              Wrapper::BUFFER_TRACING_HSA_FINALIZE_EXT_API })
                _data.emplace(eitr);
        }
        else if(itr == "hip_api")
        {
            for(auto eitr : { Wrapper::BUFFER_TRACING_HIP_COMPILER_API,
                              Wrapper::BUFFER_TRACING_HIP_RUNTIME_API })
                _data.emplace(eitr);
        }
        else if(itr == "marker_api" || itr == "roctx")
        {
            _data.emplace(Wrapper::BUFFER_TRACING_MARKER_CORE_API);
        }
        else if(itr == "memory_allocation")
        {
            if constexpr(Wrapper::compile_time_version >= 600)
                _data.emplace(Wrapper::BUFFER_TRACING_MEMORY_ALLOCATION);
        }
        else if(itr == "memory_copy")
        {
            _data.emplace(Wrapper::BUFFER_TRACING_MEMORY_COPY);
        }
        else if(itr == "kfd_events" || itr == "kfd_page_fault" ||
                itr == "kfd_page_migrate" || itr == "kfd_queue" ||
                itr == "kfd_event_queue" || itr == "kfd_event_unmap_from_gpu" ||
                itr == "kfd_event_dropped_events")
        {
            if constexpr(Wrapper::compile_time_version >= 10000)
            {
                if(!kfd_supported_by_runtime)
                {
                    static bool _warned = false;
                    if(!_warned)
                    {
                        LOG_WARNING(
                            "KFD tracing domain '{}' disabled: rocprofiler-sdk "
                            "{}.{}.{} has a bug with undefined KFD node IDs (fixed in "
                            ">= 1.2.2)",
                            itr, kfd_version.major, kfd_version.minor, kfd_version.patch);
                        _warned = true;
                    }
                    continue;
                }
                if(itr == "kfd_events")
                {
                    for(auto eitr : { Wrapper::BUFFER_TRACING_KFD_PAGE_FAULT,
                                      Wrapper::BUFFER_TRACING_KFD_PAGE_MIGRATE,
                                      Wrapper::BUFFER_TRACING_KFD_QUEUE,
                                      Wrapper::BUFFER_TRACING_KFD_EVENT_QUEUE,
                                      Wrapper::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                                      Wrapper::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS })
                        _data.emplace(eitr);
                }
                else if(itr == "kfd_page_fault")
                    _data.emplace(Wrapper::BUFFER_TRACING_KFD_PAGE_FAULT);
                else if(itr == "kfd_page_migrate")
                    _data.emplace(Wrapper::BUFFER_TRACING_KFD_PAGE_MIGRATE);
                else if(itr == "kfd_queue")
                    _data.emplace(Wrapper::BUFFER_TRACING_KFD_QUEUE);
                else if(itr == "kfd_event_queue")
                    _data.emplace(Wrapper::BUFFER_TRACING_KFD_EVENT_QUEUE);
                else if(itr == "kfd_event_unmap_from_gpu")
                    _data.emplace(Wrapper::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU);
                else if(itr == "kfd_event_dropped_events")
                    _data.emplace(Wrapper::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS);
            }
        }
        else
        {
            for(size_t idx = 0; idx < buffer_info.size(); ++idx)
            {
                const auto& ditr = buffer_info[idx];
                auto        dval = static_cast<kind_t>(idx);
                if(itr == to_lower(ditr.name) && supported.count(dval) > 0)
                {
                    _data.emplace(dval);
                    break;
                }
            }
        }
    }

    if constexpr(Wrapper::compile_time_version >= 10000)
    {
        // Automatically enable KFD domains when unified memory profiling is enabled
        if(Externals::get_use_unified_memory_profiling())
        {
            if(kfd_supported_by_runtime)
            {
                LOG_INFO(
                    "ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON: implicitly enabling "
                    "KFD page_fault and page_migrate buffered tracing domains");
                _data.emplace(Wrapper::BUFFER_TRACING_KFD_PAGE_FAULT);
                _data.emplace(Wrapper::BUFFER_TRACING_KFD_PAGE_MIGRATE);
            }
            else
            {
                LOG_WARNING("ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON requested KFD "
                            "page_fault/page_migrate tracing, but rocprofiler-sdk "
                            "{}.{}.{} is too old (requires >= 1.2.2)",
                            kfd_version.major, kfd_version.minor, kfd_version.patch);
            }
        }
    }

    return _data;
}

template <typename Wrapper, typename Externals>
std::vector<std::string>
sdk_core<Wrapper, Externals>::get_rocm_events()
{
    return rocprofsys::delimit(Externals::get_rocm_events_setting(), " ,;\t\n");
}

template <typename Wrapper, typename Externals>
std::vector<std::int32_t>
sdk_core<Wrapper, Externals>::get_operations(
    typename Wrapper::callback_tracing_kind kindv)
{
    if(callback_operation_option_names.count(kindv) == 0)
    {
        ::rocprofsys::set_state(::rocprofsys::State::Finalized);
        throw std::runtime_error(
            fmt::format("sdk_core::get_operations: no options registered for "
                        "callback tracing kind {}",
                        static_cast<int>(kindv)));
    }

    const auto& opts      = callback_operation_option_names.at(kindv);
    auto        _complete = get_operations_impl(kindv);
    // Empty option string means "no filter" — produce an empty set so the
    // three-argument overload falls through to the complete set / removes nothing.
    auto _include = opts.operations_include.empty()
                        ? std::unordered_set<std::int32_t>{}
                        : get_operations_impl(kindv, opts.operations_include);
    auto _exclude = opts.operations_exclude.empty()
                        ? std::unordered_set<std::int32_t>{}
                        : get_operations_impl(kindv, opts.operations_exclude);

    return get_operations_impl(_complete, _include, _exclude);
}

template <typename Wrapper, typename Externals>
std::vector<std::int32_t>
sdk_core<Wrapper, Externals>::get_operations(typename Wrapper::buffer_tracing_kind kindv)
{
    if(buffered_operation_option_names.count(kindv) == 0)
    {
        ::rocprofsys::set_state(::rocprofsys::State::Finalized);
        throw std::runtime_error(
            fmt::format("sdk_core::get_operations: no options registered for "
                        "buffer tracing kind {}",
                        static_cast<int>(kindv)));
    }

    const auto& opts      = buffered_operation_option_names.at(kindv);
    auto        _complete = get_operations_impl(kindv);
    auto        _include  = opts.operations_include.empty()
                                ? std::unordered_set<std::int32_t>{}
                                : get_operations_impl(kindv, opts.operations_include);
    auto        _exclude  = opts.operations_exclude.empty()
                                ? std::unordered_set<std::int32_t>{}
                                : get_operations_impl(kindv, opts.operations_exclude);

    return get_operations_impl(_complete, _include, _exclude);
}

template <typename Wrapper, typename Externals>
std::unordered_set<std::int32_t>
sdk_core<Wrapper, Externals>::get_backtrace_operations(
    typename Wrapper::callback_tracing_kind kindv)
{
    if(callback_operation_option_names.count(kindv) == 0)
    {
        ::rocprofsys::set_state(::rocprofsys::State::Finalized);
        throw std::runtime_error(
            fmt::format("sdk_core::get_backtrace_operations: no options registered for "
                        "callback tracing kind {}",
                        static_cast<int>(kindv)));
    }

    const auto& bt =
        callback_operation_option_names.at(kindv).operations_annotate_backtrace;
    if(bt.empty()) return {};

    auto _vec = get_operations_impl(kindv, bt);
    return { _vec.begin(), _vec.end() };
}

template <typename Wrapper, typename Externals>
std::unordered_set<std::int32_t>
sdk_core<Wrapper, Externals>::get_backtrace_operations(
    typename Wrapper::buffer_tracing_kind kindv)
{
    if(buffered_operation_option_names.count(kindv) == 0)
    {
        ::rocprofsys::set_state(::rocprofsys::State::Finalized);
        throw std::runtime_error(
            fmt::format("sdk_core::get_backtrace_operations: no options registered for "
                        "buffer tracing kind {}",
                        static_cast<int>(kindv)));
    }

    const auto& bt =
        buffered_operation_option_names.at(kindv).operations_annotate_backtrace;
    if(bt.empty()) return {};

    auto _vec = get_operations_impl(kindv, bt);
    return { _vec.begin(), _vec.end() };
}

}  // namespace rocprofsys::rocprofiler_sdk
