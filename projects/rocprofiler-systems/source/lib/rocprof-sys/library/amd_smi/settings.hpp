#pragma once

#include "core/config.hpp"
#include "core/debug.hpp"
#include "library/amd_smi/common.hpp"
#include <regex.h>

namespace rocprofsys
{
namespace amd_smi
{

namespace
{
constexpr uint16_t enable_all_metrics  = 0xffff;
constexpr uint16_t disable_all_metrics = 0x0000;

}  // namespace

struct settings
{
    static device_filter get_device_filter()
    {
        auto filter = rocprofsys::get_sampling_gpus();
        if(filter == "all" || filter == "on")
        {
            return { .mode = device_selection_mode::all, .indices = {} };
        }

        if(filter == "none" || filter == "off")
        {
            return { .mode = device_selection_mode::none, .indices = {} };
        }

        auto enabled_devices = parse_numeric_range(filter);

        return { .mode = device_selection_mode::specific, .indices = enabled_devices };
    }

    static smi_metric_options get_enabled_metrics()
    {
        static auto _enabled_metrics = [] {
            auto settings = get_setting_value<std::string>("ROCPROFSYS_AMD_SMI_METRICS");
            return parse_enabled_metrics(settings.has_value() ? settings.value() : "");
        }();
        return _enabled_metrics;
    };

private:
    static smi_metric_options parse_enabled_metrics(const std::string& settings)
    {
        const auto settings_trimmed = [](const std::string& settings) {
            std::string str;
            str.reserve(settings.size());
            std::for_each(settings.begin(), settings.end(), [&str](auto ch) {
                if(ch != '\t' && ch != ' ')
                {
                    str.push_back(ch);
                }
            });
            return str;
        }(settings);

        if(settings_trimmed == "none")
        {
            return smi_metric_options{ .value = disable_all_metrics };
        }

        if(settings_trimmed == "all")
        {
            return smi_metric_options{ .value = enable_all_metrics };
        }

        std::regex validator{
            R"(^(?:temp|power|busy|mem_usage|vcn_activity|jpeg_activity)(?:[,;](?:temp|power|busy|mem_usage|vcn_activity|jpeg_activity))*$)"
        };

        if(!std::regex_match(settings_trimmed, validator))
        {
            printf("Invalid metrics settings '%s'. Disabling all SMI metrics!\n",
                   settings.c_str());
            return { .value = disable_all_metrics };
        }

        const std::unordered_map<std::string, uint16_t> mapper{
            { "temp", (smi_metric_options{
                           .bits{ .hotspot_temperature = 1, .edge_temperature = 1 } })
                          .value },
            { "power", (smi_metric_options{ .bits{ .current_socket_power = 1,
                                                   .average_socket_power = 1 } })
                           .value },
            { "busy", (smi_metric_options{ .bits{
                           .gfx_activity = 1, .umc_activity = 1, .mm_activity = 1 } })
                          .value },
            { "mem_usage", (smi_metric_options{ .bits{ .memory_usage = 1 } }).value },
            { "vcn_activity", (smi_metric_options{ .bits{ .vcn_activity = 1 } }).value },
            { "jpeg_activity",
              (smi_metric_options{ .bits{ .jpeg_activity = 1 } }).value },
        };

        smi_metric_options   metrics{ .value = disable_all_metrics };
        std::regex           tokenizer{ R"(\w+)" };
        std::sregex_iterator it(settings_trimmed.begin(), settings_trimmed.end(),
                                tokenizer);
        std::sregex_iterator end;

        for(; it != end; ++it)
        {
            metrics.value |= mapper.at(it->str());
        }

        return metrics;
    }

    static std::set<size_t> parse_numeric_range(const std::string& input_range)
    {
        std::set<size_t> result{};

        auto get_range_values = [](auto& token, const auto& range_delimiter_position) {
            size_t begin =
                std::stoi(std::string{ token.begin(), range_delimiter_position });
            size_t end =
                std::stoi(std::string{ range_delimiter_position + 1, token.end() });

            if(begin > end)
            {
                std::swap(begin, end);
            }

            return std::pair<size_t, size_t>{ begin, end };
        };

        // validate input string
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
            auto token = it->str();
            auto delimiter_position =
                std::find_if(token.begin(), token.end(),
                             [](const auto& c) { return c == ':' || c == '-'; });

            if(delimiter_position != token.end())
            {
                auto [begining, end] = get_range_values(token, delimiter_position);
                for(auto i = begining; i <= end; ++i)
                {
                    result.insert(i);
                }
            }
            else
            {
                size_t value = std::stoi(token);
                result.insert(value);
            }
        }

        return result;
    }
};

}  // namespace amd_smi
}  // namespace rocprofsys
