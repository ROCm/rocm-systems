// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/config.hpp"
#include "library/pmc/gpu/metric_descriptors.hpp"
#include "library/pmc/gpu/types.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <regex>
#include <set>
#include <string>

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{

// Type aliases using fully qualified names
using device_filter_t         = ::rocprofsys::pmc::device_filter;
using device_selection_mode_t = ::rocprofsys::pmc::device_selection_mode;
using enabled_metrics_t       = ::rocprofsys::pmc::gpu::enabled_metrics;

struct settings_policy
{
    static device_filter_t get_device_filter() noexcept
    {
        auto filter = rocprofsys::get_sampling_gpus();
        if(filter == "all" || filter == "on" || filter.empty())
        {
            device_filter_t result;
            result.mode = device_selection_mode_t::ALL;
            return result;
        }

        if(filter == "none" || filter == "off")
        {
            device_filter_t result;
            result.mode = device_selection_mode_t::NONE;
            return result;
        }

        auto            enabled_devices = parse_numeric_range(filter);
        device_filter_t result;
        result.mode    = device_selection_mode_t::SPECIFIC;
        result.indices = enabled_devices;
        return result;
    }

    static enabled_metrics_t get_enabled_metrics() noexcept
    {
        static auto _enabled_metrics = []() {
            auto setting = get_setting_value<std::string>("ROCPROFSYS_AMD_SMI_METRICS");
            return parse_enabled_metrics(setting.has_value() ? setting.value() : "all");
        }();
        return _enabled_metrics;
    }

    static bool get_use_perfetto_legacy_metrics() { return get_use_perfetto(); }

private:
    static enabled_metrics_t parse_enabled_metrics(const std::string& input)
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
            return enabled_metrics_t(gpu::metric_masks::all);
        }

        if(settings_trimmed == "none")
        {
            return enabled_metrics_t(gpu::metric_masks::none);
        }

        // Use validation pattern from metric_descriptors.hpp
        std::regex validator{ std::string(gpu::metric_validation_pattern) };

        if(!std::regex_match(settings_trimmed, validator))
        {
            LOG_INFO("Invalid metrics settings '{}'. Enabling all metrics.", input);
            return enabled_metrics_t(gpu::metric_masks::all);
        }

        // Parse metric aliases using the centralized alias table
        enabled_metrics_t    metrics(gpu::metric_masks::none);
        std::regex           tokenizer{ R"(\w+)" };
        std::sregex_iterator it(settings_trimmed.begin(), settings_trimmed.end(),
                                tokenizer);
        std::sregex_iterator end_iter;

        for(; it != end_iter; ++it)
        {
            const auto& token = it->str();
            // Look up alias in the centralized table
            for(const auto& alias : gpu::user_metric_aliases)
            {
                if(alias.name == token)
                {
                    metrics.set_value(metrics.value() | alias.mask);
                    break;
                }
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
