// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/config.hpp"
#include "library/pmc/collectors/cpu/types.hpp"
#include "library/pmc/collectors/gpu/types.hpp"
#include "library/pmc/collectors/nic/types.hpp"
#include "logger/debug.hpp"
#include <cstdint>

#include <algorithm>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

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

// GPU metric bitfield: 0x7FFF sets bits 0-14 (all 15 GPU metrics enabled)
inline constexpr std::uint32_t ENABLE_ALL_METRICS  = 0x7FFF;
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

    static gpu::enabled_metrics get_enabled_metrics() noexcept
    {
        static auto _enabled_metrics = []() {
            auto setting   = get_setting_value<std::string>("ROCPROFSYS_AMD_SMI_METRICS");
            auto value_str = setting.has_value() ? setting.value() : "all";
            auto result    = parse_enabled_metrics(value_str);
            return result;
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
        auto filter = get_setting_value<std::string>("ROCPROFSYS_SAMPLING_AINICS");
        if(!filter.has_value())
        {
            // NIC sampling disabled by default
            nic::nic_device_filter result;
            result.mode = nic::device_selection_mode::NONE;
            return result;
        }

        auto filter_str = filter.value();
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
    }

    /**
     * @brief Get NIC enabled metrics.
     *
     * For NIC, all 6 RDMA metrics are enabled when NIC sampling is active.
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
            auto       setting = get_setting_value<std::string>("ROCPROFSYS_CPU_METRICS");
            const auto value_str = setting.has_value() ? setting.value() : "all";
            return parse_cpu_enabled_metrics(value_str);
        }();
        return _result;
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

        std::regex           tokenizer{ R"(\w+)" };
        std::sregex_iterator it(trimmed.begin(), trimmed.end(), tokenizer);
        std::sregex_iterator end;

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

        std::regex validator{
            R"(^(?:temp|power|busy|mem_usage|vcn_activity|jpeg_activity|xgmi|pcie|sdma_usage)"
            R"()(?:[,;](?:temp|power|busy|mem_usage|vcn_activity|jpeg_activity|xgmi|pcie|sdma_usage))*$)"
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
        const std::unordered_map<std::string, std::uint16_t> mapper{
            { "power", make_metric({ 0, 1 }) },           // current, average
            { "mem_usage", make_metric({ 2 }) },          // memory_usage
            { "temp", make_metric({ 3, 4 }) },            // hotspot, edge
            { "busy", make_metric({ 5, 6, 7 }) },         // gfx, umc, mm
            { "vcn_activity", make_metric({ 8, 10 }) },   // vcn_activity
            { "jpeg_activity", make_metric({ 9, 11 }) },  // jpeg_activity
            { "xgmi", make_metric({ 12 }) },              // xgmi
            { "pcie", make_metric({ 13 }) },              // pcie
            { "sdma_usage", make_metric({ 14 }) },        // sdma_usage
        };

        gpu::enabled_metrics metrics;
        metrics.value = DISABLE_ALL_METRICS;
        std::regex           tokenizer{ R"(\w+)" };
        std::sregex_iterator it(settings_trimmed.begin(), settings_trimmed.end(),
                                tokenizer);
        std::sregex_iterator end;

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

        std::regex           tokenizer{ R"(\d+(?:[-:]\d+)*)" };
        std::sregex_iterator it(input_range.begin(), input_range.end(), tokenizer);
        std::sregex_iterator end;

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
