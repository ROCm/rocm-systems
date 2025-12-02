// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include "core/config.hpp"
#include "core/debug.hpp"
#include "library/amd_smi/common.hpp"

#include <algorithm>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>

namespace rocprofsys
{
namespace amd_smi
{

namespace
{
constexpr uint16_t ENABLE_ALL_METRICS  = 0xffff;
constexpr uint16_t DISABLE_ALL_METRICS = 0x0000;
}  // namespace

struct settings_policy
{
    static device_filter get_device_filter()
    {
        auto filter = rocprofsys::get_sampling_gpus();
        if(filter == "all" || filter == "on" || filter.empty())
        {
            return { .mode = device_selection_mode::ALL, .indices = {} };
        }

        if(filter == "none" || filter == "off")
        {
            return { .mode = device_selection_mode::NONE, .indices = {} };
        }

        auto enabled_devices = parse_numeric_range(filter);
        return { .mode = device_selection_mode::SPECIFIC, .indices = enabled_devices };
    }

    static enabled_metric get_enabled_metrics()
    {
        static auto _enabled_metrics = []() {
            auto setting = get_setting_value<std::string>("ROCPROFSYS_AMD_SMI_METRICS");
            return parse_enabled_metrics(setting.has_value() ? setting.value() : "all");
        }();
        return _enabled_metrics;
    }

private:
    static enabled_metric parse_enabled_metrics(const std::string& input)
    {
        // Trim whitespace
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
            return enabled_metric{ .value = ENABLE_ALL_METRICS };
        }

        if(settings_trimmed == "none")
        {
            return enabled_metric{ .value = DISABLE_ALL_METRICS };
        }

        // Validate input
        std::regex validator{
            R"(^(?:temp|power|busy|mem_usage|vcn_activity|jpeg_activity|xgmi|pcie)"
            R"()(?:[,;](?:temp|power|busy|mem_usage|vcn_activity|jpeg_activity|xgmi|pcie))*$)"
        };

        if(!std::regex_match(settings_trimmed, validator))
        {
            ROCPROFSYS_VERBOSE(0,
                               "Invalid metrics settings '%s'. Enabling all metrics.\n",
                               input.c_str());
            return { .value = ENABLE_ALL_METRICS };
        }

        // Map metric names to bitfield values
        const std::unordered_map<std::string, uint16_t> mapper{
            { "temp",
              (enabled_metric{ .bits{ .hotspot_temperature = 1, .edge_temperature = 1 } })
                  .value },
            { "power", (enabled_metric{ .bits{ .current_socket_power = 1,
                                               .average_socket_power = 1 } })
                           .value },
            { "busy", (enabled_metric{ .bits{
                           .gfx_activity = 1, .umc_activity = 1, .mm_activity = 1 } })
                          .value },
            { "mem_usage", (enabled_metric{ .bits{ .memory_usage = 1 } }).value },
            { "vcn_activity", (enabled_metric{ .bits{ .vcn_activity = 1 } }).value },
            { "jpeg_activity", (enabled_metric{ .bits{ .jpeg_activity = 1 } }).value },
            { "xgmi", (enabled_metric{ .bits{ .xgmi = 1 } }).value },
            { "pcie", (enabled_metric{ .bits{ .pcie = 1 } }).value },
        };

        enabled_metric       metrics{ .value = DISABLE_ALL_METRICS };
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

        // Validate input string
        const std::regex validator{ R"(^\d+(?:-\d+)?(?:[;,]\d+(?:[-:]\d+)?)*$)" };

        if(!std::regex_match(input_range, validator))
        {
            ROCPROFSYS_VERBOSE(0, "Failed to parse gpu input list: %s\n",
                               input_range.c_str());
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

}  // namespace amd_smi
}  // namespace rocprofsys
