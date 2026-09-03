// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/env_vars.hpp"
#include "core/config.hpp"
#include "core/gpu_visibility.hpp"
#include "library/pmc/collectors/cpu/types.hpp"
#include "library/pmc/collectors/gpu/types.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/collectors/hipfile/types.hpp"
#include "library/pmc/collectors/nic/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::collectors
{

// Import GPU types into collectors namespace
namespace gpu
{
using ::rocprofsys::pmc::device_filter;
using ::rocprofsys::pmc::device_selection_mode;
using ::rocprofsys::pmc::collectors::gpu::enabled_metrics;
}  // namespace gpu

// Import NIC types into collectors namespace
namespace nic
{
using ::rocprofsys::pmc::device_selection_mode;
using ::rocprofsys::pmc::nic_device_filter;
using ::rocprofsys::pmc::collectors::nic::enabled_metrics;
}  // namespace nic

// Import CPU types into collectors namespace
namespace cpu
{
using ::rocprofsys::pmc::device_filter;
using ::rocprofsys::pmc::device_selection_mode;
using ::rocprofsys::pmc::collectors::cpu::enabled_metrics;
}  // namespace cpu

// Import hipFile types into collectors namespace
namespace hipfile
{
using ::rocprofsys::pmc::collectors::hipfile::ALL_HIPFILE_METRICS;
using ::rocprofsys::pmc::collectors::hipfile::enabled_metrics;
using ::rocprofsys::pmc::collectors::hipfile::METRIC_TABLE;
}  // namespace hipfile

// GPU metric bitfield helpers: ENABLE_ALL_METRICS sets bits 0..NUM_GPU_METRIC_BITS-1
inline constexpr std::uint32_t NUM_GPU_METRIC_BITS = 17;
inline constexpr std::uint32_t ENABLE_ALL_METRICS  = (1U << NUM_GPU_METRIC_BITS) - 1U;
inline constexpr std::uint32_t DISABLE_ALL_METRICS = 0x0000;

struct settings_policy
{
    /**
     * @brief Build a device filter from a setting string.
     *
     * Parses numeric range (e.g., "0-3", "0,2,4") or special values
     * "all"/"on" (enable all), "none"/"off" (disable), empty (enable all).
     * Used by both GPU and CPU traits.
     */
    static device_filter get_device_filter(const std::string& filter_str)
    {
        if(filter_str == "all" || filter_str == "on" || filter_str.empty())
        {
            device_filter result;
            result.mode = device_selection_mode::ALL;
            return result;
        }
        if(filter_str == "none" || filter_str == "off")
        {
            device_filter result;
            result.mode = device_selection_mode::NONE;
            return result;
        }
        device_filter result;
        result.mode    = device_selection_mode::SPECIFIC;
        result.indices = parse_numeric_range(filter_str);
        return result;
    }

    /**
     * @brief Get the GPU device filter from ROCPROFSYS_SAMPLING_GPUS.
     */
    static device_filter get_gpu_device_filter()
    {
        return get_device_filter(rocprofsys::get_sampling_gpus());
    }

    /**
     * @brief PCIe BDFs of the GPUs the ROCm runtime exposes.
     *
     * Honors ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES. Returns std::nullopt when
     * visibility could not be determined; see rocprofsys::gpu::get_visible_gpu_bdfs.
     */
    static std::optional<std::set<std::string>> get_visible_gpu_bdfs()
    {
        return rocprofsys::gpu::get_visible_gpu_bdfs();
    }

    /**
     * @brief Profiler GPU indices of the GPUs the HIP runtime exposes, in HIP ordinal
     *        order.
     *
     * Element k is the device_type_index of HIP ordinal k, which is how hipFile indexes
     * per_gpu_stats. Empty when the visibility mask is a permutation or UUID list, so
     * hipFile telemetry is not recorded under the wrong GPU. See
     * rocprofsys::gpu::get_visible_gpu_type_indices.
     */
    static std::vector<std::size_t> get_visible_gpu_type_indices()
    {
        return rocprofsys::gpu::get_visible_gpu_type_indices();
    }

    /**
     * @brief hipFile metrics selected by ROCPROFSYS_HIPFILE_METRICS.
     *
     * Accepts "all"/"on", "none"/"off", or a comma or semicolon separated list of group
     * keys (e.g. "fastpath,fallback,bandwidth,bytes,errors"), case-insensitively. Each
     * key selects a read/write pair. The setting's own default
     * (@c env_vars::HIPFILE_METRICS_DEFAULT) is registered in config.cpp; the "all"
     * below applies only when the setting is absent entirely, as it is in unit tests.
     */
    static hipfile::enabled_metrics get_hipfile_enabled_metrics()
    {
        static auto _enabled_metrics = []() {
            auto setting =
                get_setting_value<std::string>(std::string{ env_vars::HIPFILE_METRICS });
            return parse_hipfile_enabled_metrics(setting.value_or("all"));
        }();
        return _enabled_metrics;
    }

    static hipfile::enabled_metrics parse_hipfile_enabled_metrics(
        const std::string& setting)
    {
        // Case and stray whitespace are folded away exactly as ROCPROFSYS_AMD_SMI_METRICS
        // does it, so "All" and " fastpath, fallback " behave as written.
        std::string normalized;
        normalized.reserve(setting.size());
        for(const char character : setting)
        {
            if(character == ' ' || character == '\t')
            {
                continue;
            }
            normalized.push_back(
                static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
        }

        hipfile::enabled_metrics metrics;

        if(normalized.empty() || normalized == "all" || normalized == "on")
        {
            metrics.value = hipfile::ALL_HIPFILE_METRICS;
            return metrics;
        }
        if(normalized == "none" || normalized == "off")
        {
            metrics.value = 0U;
            return metrics;
        }

        for(const auto& token : parse_name_list(normalized))
        {
            const auto mask = hipfile::metric_group_mask(token);
            if(mask == 0U)
            {
                LOG_WARNING("Unknown hipFile metric group '{}' in {}, ignoring", token,
                            env_vars::HIPFILE_METRICS);
                continue;
            }
            metrics.value |= mask;
        }
        return metrics;
    }

    static gpu::enabled_metrics get_enabled_metrics() noexcept
    {
        static auto _enabled_metrics = []() noexcept {
            try
            {
                auto setting = get_setting_value<std::string>(
                    std::string{ env_vars::AMD_SMI_METRICS });
                auto value_str = setting.has_value() ? setting.value() : "all";
                return parse_enabled_metrics(value_str);
            } catch(const std::exception& ex)
            {
                LOG_ERROR("Failed to apply {}: {}. Using default AMD SMI metrics "
                          "(busy, temp, power, mem_usage).",
                          env_vars::AMD_SMI_METRICS, ex.what());
            } catch(...)
            {
                LOG_ERROR("Failed to apply {}: unknown exception. Using default AMD SMI "
                          "metrics (busy, temp, power, mem_usage).",
                          env_vars::AMD_SMI_METRICS);
            }
            return parse_enabled_metrics("busy, temp, power, mem_usage");
        }();
        return _enabled_metrics;
    }

    static bool get_use_perfetto_legacy_metrics() { return get_use_perfetto(); }

    /**
     * @brief Get NIC device filter based on ROCPROFSYS_SAMPLING_AINICS setting.
     *
     * Parses comma-separated list of NIC device names (e.g., "enp226s0,eth0").
     * Special values: "all" enables all NICs, "none" disables NIC sampling.
     */
    static nic::nic_device_filter get_nic_device_filter() noexcept
    {
        try
        {
            auto filter =
                get_setting_value<std::string>(std::string{ env_vars::SAMPLING_AINICS });
            if(!filter.has_value())
            {
                // NIC sampling disabled by default
                nic::nic_device_filter result;
                result.mode = nic::device_selection_mode::NONE;
                return result;
            }

            const auto& filter_str = filter.value();
            if(filter_str == "all" || filter_str == "on")
            {
                nic::nic_device_filter result;
                result.mode = nic::device_selection_mode::ALL;
                return result;
            }

            if(filter_str == "none" || filter_str == "off" || filter_str.empty())
            {
                nic::nic_device_filter result;
                result.mode = nic::device_selection_mode::NONE;
                return result;
            }

            // Parse comma-separated names
            nic::nic_device_filter result;
            result.mode  = nic::device_selection_mode::SPECIFIC;
            result.names = parse_name_list(filter_str);
            return result;
        } catch(const std::exception& ex)
        {
            LOG_ERROR("Failed to apply {}: {}. Disabling NIC sampling only.",
                      env_vars::SAMPLING_AINICS, ex.what());
        } catch(...)
        {
            LOG_ERROR("Failed to apply {}: unknown exception. Disabling NIC sampling "
                      "only.",
                      env_vars::SAMPLING_AINICS);
        }
        nic::nic_device_filter result;
        result.mode = nic::device_selection_mode::NONE;
        return result;
    }

    /**
     * @brief Get NIC enabled metrics.
     *
     * For NIC, all RDMA metrics are enabled when NIC sampling is active.
     */
    static nic::enabled_metrics get_nic_enabled_metrics() noexcept
    {
        nic::enabled_metrics result;
        result.value = nic::ALL_NIC_METRICS;
        return result;
    }

    /**
     * @brief Get CPU enabled metrics based on ROCPROFSYS_CPU_METRICS setting.
     *
     * Parses token list (e.g., "frequency,load,memory") or "all"/"none".
     * Cached on first call.
     */
    static cpu::enabled_metrics get_cpu_enabled_metrics()
    {
        static auto _result = []() {
            auto setting =
                get_setting_value<std::string>(std::string{ env_vars::CPU_METRICS });
            const auto value_str = setting.has_value() ? setting.value() : "all";
            return parse_cpu_enabled_metrics(value_str);
        }();
        return _result;
    }

    static gpu_perf_counter::gpu_perf_counter_settings
    get_gpu_perf_counter_enabled_metrics() noexcept
    {
        auto value_str = rocprofsys::get_gpu_perf_counters();
        if(value_str.empty())
        {
            return gpu_perf_counter::gpu_perf_counter_settings{};
        }

        std::string trimmed;
        trimmed.reserve(value_str.size());
        for(auto chr : value_str)
        {
            if(chr != '\t' && chr != ' ') trimmed.push_back(chr);
        }

        gpu_perf_counter::gpu_perf_counter_settings result;

        constexpr auto device_qualifier = std::string_view{ ":device=" };

        std::stringstream stream(trimmed);
        std::string       token;
        while(std::getline(stream, token, ','))
        {
            std::stringstream sub_stream(token);
            std::string       subtoken;
            while(std::getline(sub_stream, subtoken, ';'))
            {
                if(subtoken.empty()) continue;
                auto pos = subtoken.find(device_qualifier);
                if(pos == std::string::npos)
                {
                    result.broadcast_names.push_back(subtoken);
                }
                else
                {
                    auto name       = subtoken.substr(0, pos);
                    auto device_str = subtoken.substr(pos + device_qualifier.size());
                    if(name.empty()) continue;
                    if(device_str.empty() ||
                       !std::all_of(device_str.begin(), device_str.end(), ::isdigit))
                    {
                        LOG_ERROR("Invalid :device= value in "
                                  "ROCPROFSYS_GPU_PERF_COUNTERS: '{}'",
                                  subtoken);
                        continue;
                    }
                    try
                    {
                        result.explicit_counters.push_back(
                            { name, std::stoull(device_str) });
                    } catch(const std::exception&)
                    {
                        LOG_ERROR("Invalid :device= value in "
                                  "ROCPROFSYS_GPU_PERF_COUNTERS: '{}'",
                                  subtoken);
                    }
                }
            }
        }

        return result;
    }

private:
    static cpu::enabled_metrics parse_cpu_enabled_metrics(const std::string& input)
    {
        std::string trimmed;
        trimmed.reserve(input.size());
        std::for_each(input.begin(), input.end(), [&trimmed](char ch) {
            if(ch != '\t' && ch != ' ')
                trimmed.push_back(static_cast<char>(std::tolower(ch)));
        });

        if(trimmed.empty() || trimmed == "all")
        {
            cpu::enabled_metrics result;
            result.value = cpu::ALL_CPU_METRICS;
            return result;
        }
        if(trimmed == "none")
        {
            cpu::enabled_metrics result;
            result.value = DISABLE_ALL_METRICS;
            return result;
        }

        auto make_bits =
            [](std::initializer_list<std::uint8_t> positions) -> std::uint32_t {
            std::uint32_t v = 0;
            for(auto b : positions)
                v |= (1u << b);
            return v;
        };

        const std::unordered_map<std::string, std::uint32_t> mapper{
            { "frequency", make_bits({ 0 }) },    { "load", make_bits({ 1 }) },
            { "memory", make_bits({ 2, 3, 4 }) }, { "page_rss", make_bits({ 2 }) },
            { "virt_mem", make_bits({ 3 }) },     { "peak_rss", make_bits({ 4 }) },
            { "ctx_switches", make_bits({ 5 }) }, { "page_faults", make_bits({ 6 }) },
            { "cpu_time", make_bits({ 7, 8 }) },  { "user_time", make_bits({ 7 }) },
            { "kernel_time", make_bits({ 8 }) },
        };

        cpu::enabled_metrics metrics;
        metrics.value = DISABLE_ALL_METRICS;

        const std::regex           tokenizer{ R"(\w+)" };
        std::sregex_iterator       it(trimmed.begin(), trimmed.end(), tokenizer);
        const std::sregex_iterator end;

        for(; it != end; ++it)
        {
            const auto found = mapper.find(it->str());
            if(found != mapper.end()) metrics.value |= found->second;
        }

        if(metrics.value == DISABLE_ALL_METRICS)
        {
            LOG_INFO("Invalid CPU metrics settings '{}'. Enabling all metrics.", input);
            metrics.value = cpu::ALL_CPU_METRICS;
        }

        return metrics;
    }

    static gpu::enabled_metrics parse_enabled_metrics(const std::string& input)
    {
        std::string settings_trimmed;
        settings_trimmed.reserve(input.size());
        std::for_each(input.begin(), input.end(), [&settings_trimmed](char ch) {
            if(ch != '\t' && ch != ' ')
            {
                settings_trimmed.push_back(static_cast<char>(std::tolower(ch)));
            }
        });

        if(settings_trimmed.empty() || settings_trimmed == "all")
        {
            gpu::enabled_metrics result;
            result.value = ENABLE_ALL_METRICS;
            return result;
        }

        if(settings_trimmed == "none")
        {
            gpu::enabled_metrics result;
            result.value = DISABLE_ALL_METRICS;
            return result;
        }

        const std::regex validator{
            R"(^(?:temp|power|busy|mem_usage|vcn_activity|jpeg_activity|xgmi|pcie|sdma_usage|gfx_clock|mem_clock)"
            R"()(?:[,;](?:temp|power|busy|mem_usage|vcn_activity|jpeg_activity|xgmi|pcie|sdma_usage|gfx_clock|mem_clock))*$)"
        };

        if(!std::regex_match(settings_trimmed, validator))
        {
            LOG_INFO("Invalid metrics settings '{}'. Enabling all metrics.", input);
            gpu::enabled_metrics result;
            result.value = ENABLE_ALL_METRICS;
            return result;
        }

        auto make_metric = [](std::initializer_list<std::uint8_t> bit_positions) {
            std::uint32_t value = 0;
            for(auto bit : bit_positions)
            {
                value |= (1u << bit);
            }
            gpu::enabled_metrics result;
            result.value = value;
            return result.value;
        };

        // See enabled_metrics definition in common.hpp for bit position documentation
        const std::unordered_map<std::string, std::uint32_t> mapper{
            { "power", make_metric({ 0, 1 }) },           // current, average
            { "mem_usage", make_metric({ 2 }) },          // memory_usage
            { "temp", make_metric({ 3, 4 }) },            // hotspot, edge
            { "busy", make_metric({ 5, 6, 7 }) },         // gfx, umc, mm
            { "vcn_activity", make_metric({ 8, 10 }) },   // vcn_activity
            { "jpeg_activity", make_metric({ 9, 11 }) },  // jpeg_activity
            { "xgmi", make_metric({ 12 }) },              // xgmi
            { "pcie", make_metric({ 13 }) },              // pcie
            { "sdma_usage", make_metric({ 14 }) },        // sdma_usage
            { "gfx_clock", make_metric({ 15 }) },         // gfx_clock
            { "mem_clock", make_metric({ 16 }) },         // mem_clock
        };

        gpu::enabled_metrics metrics;
        metrics.value = DISABLE_ALL_METRICS;
        const std::regex           tokenizer{ R"(\w+)" };
        std::sregex_iterator       it(settings_trimmed.begin(), settings_trimmed.end(),
                                      tokenizer);
        const std::sregex_iterator end;

        for(; it != end; ++it)
        {
            auto found = mapper.find(it->str());
            if(found != mapper.end())
            {
                metrics.value |= found->second;
            }
        }

        return metrics;
    }

    static std::set<size_t> parse_numeric_range(const std::string& input_range)
    {
        std::set<size_t> result;

        const std::regex validator{ R"(^\d+(?:-\d+)?(?:[;,]\d+(?:[-:]\d+)?)*$)" };

        if(!std::regex_match(input_range, validator))
        {
            LOG_ERROR("Failed to parse device index list: {}", input_range);
            return result;
        }

        const std::regex           tokenizer{ R"(\d+(?:[-:]\d+)*)" };
        std::sregex_iterator       it(input_range.begin(), input_range.end(), tokenizer);
        const std::sregex_iterator end;

        for(; it != end; ++it)
        {
            auto token              = it->str();
            auto delimiter_position = std::find_if(
                token.begin(), token.end(), [](char c) { return c == ':' || c == '-'; });

            if(delimiter_position != token.end())
            {
                size_t begin =
                    std::stoul(std::string{ token.begin(), delimiter_position });
                size_t range_end =
                    std::stoul(std::string{ delimiter_position + 1, token.end() });

                if(begin > range_end)
                {
                    std::swap(begin, range_end);
                }

                for(auto i = begin; i <= range_end; ++i)
                {
                    result.insert(i);
                }
            }
            else
            {
                result.insert(std::stoul(token));
            }
        }

        return result;
    }

    /**
     * @brief Parse comma or semicolon-separated list of names.
     */
    static std::set<std::string> parse_name_list(const std::string& input)
    {
        std::set<std::string> result;
        std::stringstream     ss(input);
        std::string           token;

        while(std::getline(ss, token, ','))
        {
            // Also handle semicolons
            std::stringstream ss2(token);
            std::string       subtoken;
            while(std::getline(ss2, subtoken, ';'))
            {
                // Trim whitespace
                auto start = subtoken.find_first_not_of(" \t");
                auto end   = subtoken.find_last_not_of(" \t");
                if(start != std::string::npos && end != std::string::npos)
                {
                    result.insert(subtoken.substr(start, end - start + 1));
                }
            }
        }
        return result;
    }
};

}  // namespace rocprofsys::pmc::collectors
