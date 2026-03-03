// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/config.hpp"
#include "library/pmc/gpu/types.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{

// Import GPU types into collectors namespace
namespace gpu
{
using ::rocprofsys::pmc::device_filter;
using ::rocprofsys::pmc::device_selection_mode;
using ::rocprofsys::pmc::gpu::enabled_metrics;
}  // namespace gpu

namespace
{
// Bitfield values for enabling/disabling all metrics at once
// 0xffff sets all 16 metric bits to 1 (all enabled)
// 0x0000 sets all bits to 0 (all disabled)
constexpr uint16_t ENABLE_ALL_METRICS  = 0xffff;
constexpr uint16_t DISABLE_ALL_METRICS = 0x0000;
}  // namespace

struct settings_policy
{
    static gpu::device_filter get_device_filter() noexcept
    {
        auto filter = rocprofsys::get_sampling_gpus();
        if(filter == "all" || filter == "on" || filter.empty())
        {
            gpu::device_filter result;
            result.mode = gpu::device_selection_mode::ALL;
            return result;
        }

        if(filter == "none" || filter == "off")
        {
            gpu::device_filter result;
            result.mode = gpu::device_selection_mode::NONE;
            return result;
        }

        auto               enabled_devices = parse_numeric_range(filter);
        gpu::device_filter result;
        result.mode    = gpu::device_selection_mode::SPECIFIC;
        result.indices = enabled_devices;
        return result;
    }

    static gpu::enabled_metrics get_enabled_metrics() noexcept
    {
        static auto _enabled_metrics = []() {
            auto setting = get_setting_value<std::string>("ROCPROFSYS_AMD_SMI_METRICS");
            return parse_enabled_metrics(setting.has_value() ? setting.value() : "all");
        }();
        return _enabled_metrics;
    }

    static bool get_use_perfetto_legacy_metrics() { return get_use_perfetto(); }

private:
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
            R"(^(?:temp|power|busy|mem_usage|vcn_activity|jpeg_activity|xgmi|pcie)"
            R"()(?:[,;](?:temp|power|busy|mem_usage|vcn_activity|jpeg_activity|xgmi|pcie))*$)"
        };

        if(!std::regex_match(settings_trimmed, validator))
        {
            LOG_INFO("Invalid metrics settings '{}'. Enabling all metrics.", input);
            gpu::enabled_metrics result;
            result.value = ENABLE_ALL_METRICS;
            return result;
        }

        auto make_metric = [](std::initializer_list<uint8_t> bit_positions) {
            uint32_t value = 0;
            for(auto bit : bit_positions)
            {
                value |= (1u << bit);
            }
            gpu::enabled_metrics result;
            result.value = value;
            return result.value;
        };

        // See enabled_metrics definition in common.hpp for bit position documentation
        const std::unordered_map<std::string, uint16_t> mapper{
            { "temp", make_metric({ 3, 4 }) },        // hotspot, edge
            { "power", make_metric({ 0, 1 }) },       // current, average
            { "busy", make_metric({ 5, 6, 7 }) },     // gfx, umc, mm
            { "mem_usage", make_metric({ 2 }) },      // memory_usage
            { "vcn_activity", make_metric({ 8 }) },   // vcn_activity
            { "jpeg_activity", make_metric({ 9 }) },  // jpeg_activity
            { "xgmi", make_metric({ 12 }) },          // xgmi
            { "pcie", make_metric({ 13 }) },          // pcie
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
            LOG_ERROR("Failed to parse gpu input list: {}", input_range);
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
};

}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
