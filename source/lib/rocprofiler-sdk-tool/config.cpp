// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include "config.hpp"

#include "lib/common/defines.hpp"
#include "lib/common/demangle.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/units.hpp"
#include "lib/common/utility.hpp"
#include "lib/output/output_key.hpp"

#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

#include <fmt/core.h>

#include <linux/limits.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofiler
{
namespace tool
{
namespace
{
const auto env_regexes =
    new std::array<std::regex, 3>{std::regex{"(.*)%(env|ENV)\\{([A-Z0-9_]+)\\}%(.*)"},
                                  std::regex{"(.*)\\$(env|ENV)\\{([A-Z0-9_]+)\\}(.*)"},
                                  std::regex{"(.*)%q\\{([A-Z0-9_]+)\\}(.*)"}};
// env regex examples:
//  - %env{USER}%       Consistent with other output key formats (start+end with %)
//  - $ENV{USER}        Similar to CMake
//  - %q{USER}          Compatibility with NVIDIA
//

inline bool
not_is_space(int ch)
{
    return std::isspace(ch) == 0;
}

inline std::string
ltrim(std::string s, bool (*f)(int) = not_is_space)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), f));
    return s;
}

inline std::string
rtrim(std::string s, bool (*f)(int) = not_is_space)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), f).base(), s.end());
    return s;
}

inline std::string
trim(std::string s, bool (*f)(int) = not_is_space)
{
    ltrim(s, f);
    rtrim(s, f);
    return s;
}

// replace unsuported specail chars with space
void
handle_special_chars(std::string& str)
{
    // Iterate over the string and replace any special characters with a space.
    auto pos = std::string::npos;
    while((pos = str.find_first_of("!@#$%&(),*+-./;<>?@{}^`~|")) != std::string::npos)
        str.at(pos) = ' ';
}

bool
has_counter_format(std::string const& str)
{
    return std::find_if(str.begin(), str.end(), [](unsigned char ch) {
               return (isalnum(ch) != 0 || ch == '_');
           }) != str.end();
}

// validate kernel names
std::unordered_set<size_t>
get_kernel_filter_range(const std::string& kernel_filter)
{
    if(kernel_filter.empty()) return {};

    auto delim     = rocprofiler::sdk::parse::tokenize(kernel_filter, "[], ");
    auto range_set = std::unordered_set<size_t>{};
    for(const auto& itr : delim)
    {
        if(itr.find('-') != std::string::npos)
        {
            auto drange = rocprofiler::sdk::parse::tokenize(itr, "- ");

            ROCP_FATAL_IF(drange.size() != 2)
                << "bad range format for '" << itr << "'. Expected [A-B] where A and B are numbers";

            size_t start_range = std::stoul(drange.front());
            size_t end_range   = std::stoul(drange.back());
            for(auto i = start_range; i <= end_range; i++)
                range_set.emplace(i);
        }
        else
        {
            ROCP_FATAL_IF(itr.find_first_not_of("0123456789") != std::string::npos)
                << "expected integer for " << itr << ". Non-integer value detected";
            range_set.emplace(std::stoul(itr));
        }
    }
    return range_set;
}

std::vector<att_perfcounter>
parse_att_counters(std::string line)
{
    auto counters = std::vector<att_perfcounter>{};

    if(line.empty()) return counters;

    // strip the comment
    if(auto pos = line.find('#'); pos != std::string::npos) line = line.substr(0, pos);

    // trim line for any white spaces after comment strip
    trim(line);

    // check to see if comment stripping + trim resulted in empty line
    if(line.empty()) return counters;

    handle_special_chars(line);

    auto extract_counter_name_and_simd_mask = [](std::string& input) {
        auto ret = att_perfcounter{};

        size_t pos = input.find(':');

        if(pos != std::string::npos)
        {
            ret.counter_name = input.substr(0, pos);
            ret.simd_mask    = std::stoi(input.substr(pos + 1), nullptr, 16);
        }
        else
        {
            ret.counter_name = input;
        }
        return ret;
    };

    // regex to check if string is of the form "counter_name:simd_mask"
    std::set<std::string> unique_counters;

    auto input_ss = std::stringstream{line};
    while(true)
    {
        auto counter = std::string{};
        input_ss >> counter;
        if(counter.empty()) break;

        // Consider only those counters where combination of counter name and simd mask is unique
        if(unique_counters.insert(counter).second == false) continue;

        auto res = extract_counter_name_and_simd_mask(counter);
        counters.emplace_back(res);
    }

    return counters;
}

std::set<std::string>
parse_counters(std::string line)
{
    auto counters = std::set<std::string>{};

    if(line.empty()) return counters;

    // strip the comment
    if(auto pos = std::string::npos; (pos = line.find('#')) != std::string::npos)
        line = line.substr(0, pos);

    // trim line for any white spaces after comment strip
    trim(line);

    // check to see if comment stripping + trim resulted in empty line
    if(line.empty()) return counters;

    constexpr auto pmc_qualifier = std::string_view{"pmc:"};
    auto           pos           = std::string::npos;

    // should we handle an "pmc:" not being present? Seems like it should be a fatal error
    if((pos = line.find(pmc_qualifier)) != std::string::npos)
    {
        // strip out pmc qualifier
        line = line.substr(pos + pmc_qualifier.length());

        handle_special_chars(line);

        auto input_ss = std::stringstream{line};
        while(true)
        {
            auto counter = std::string{};
            input_ss >> counter;
            if(counter.empty())
                break;
            else if(counter != pmc_qualifier && has_counter_format(counter))
                counters.emplace(counter);
        }
    }

    return counters;
}

std::vector<std::set<std::string>>
parse_counter_envs()
{
    if(auto single_counter = get_env("ROCPROF_COUNTERS", std::string{}); !single_counter.empty())
    {
        return {parse_counters(single_counter)};
    }

    if(auto group_counters = get_env("ROCPROF_COUNTER_GROUPS", std::string{});
       !group_counters.empty())
    {
        auto counters = std::vector<std::set<std::string>>{};
        for(const auto& group : rocprofiler::sdk::parse::tokenize(group_counters, "\n"))
        {
            counters.emplace_back(parse_counters(group));
        }
        return counters;
    }
    return {};
}
}  // namespace

config::config()
: base_type{base_type::load_from_env()}
, kernel_filter_range{get_kernel_filter_range(
      get_env("ROCPROF_KERNEL_FILTER_RANGE", std::string{}))}
, counters{parse_counter_envs()}
, att_param_perfcounters{
      parse_att_counters(get_env("ROCPROF_ATT_PARAM_PERFCOUNTERS", std::string{}))}
{
    if(kernel_filter_include.empty()) kernel_filter_include = std::string{".*"};

    std::unordered_map<std::string_view, rocprofiler_pc_sampling_unit_t> pc_sampling_unit_map = {
        {"none", ROCPROFILER_PC_SAMPLING_UNIT_NONE},
        {"instructions", ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS},
        {"cycles", ROCPROFILER_PC_SAMPLING_UNIT_CYCLES},
        {"time", ROCPROFILER_PC_SAMPLING_UNIT_TIME}};

    std::unordered_map<std::string_view, rocprofiler_pc_sampling_method_t> pc_sampling_method_map =
        {{"none", ROCPROFILER_PC_SAMPLING_METHOD_NONE},
         {"stochastic", ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC},
         {"host_trap", ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP}};

    try
    {
        pc_sampling_method_value = pc_sampling_method_map.at(pc_sampling_method);
    } catch(...)
    {
        ROCP_FATAL << "Invalid value for ROCPROF_PC_SAMPLING_METHOD: " << pc_sampling_method << "."
                   << "Valid choices are stochastic and host_trap\n";
    }

    if(pc_sampling_method_value == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP)
        pc_sampling_host_trap = true;
    else if(pc_sampling_method_value == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC)
        pc_sampling_stochastic = true;

    try
    {
        pc_sampling_unit_value = pc_sampling_unit_map.at(pc_sampling_unit);
    } catch(...)
    {
        ROCP_FATAL << "Invalid value for ROCPROF_PC_SAMPLING_UNIT: " << pc_sampling_unit << "."
                   << "Valid choices are instructions, cycles and time\n";
    }

    if(auto _collection_period = get_env("ROCPROF_COLLECTION_PERIOD", "");
       !_collection_period.empty())
    {
        for(const auto& _config : sdk::parse::tokenize(_collection_period, ";"))
        {
            auto _config_params = sdk::parse::tokenize(_config, ":");
            collection_periods.emplace(CollectionPeriod{std::stoull(_config_params.at(0)),
                                                        std::stoull(_config_params.at(1)),
                                                        std::stoull(_config_params.at(2))});
        }
    }

    // Benchmarking Enable/Disable
    if(!benchmark_mode_env.empty())
    {
        const auto valid_options = std::unordered_map<std::string_view, config::benchmark>{
            {"disabled-sdk-contexts", benchmark::disabled_contexts_overhead},
            {"sdk-buffer-overhead", benchmark::sdk_buffered_overhead},
            {"sdk-callback-overhead", benchmark::sdk_callback_overhead},
            {"tool-runtime-overhead", benchmark::tool_runtime_overhead},
            {"execution-profile", benchmark::execution_profile},
        };

        ROCP_FATAL_IF(valid_options.count(benchmark_mode_env) == 0)
            << fmt::format("Invalid value for ROCPROF_BENCHMARK_MODE: {}", benchmark_mode_env);

        benchmark_mode = valid_options.at(benchmark_mode_env);
    }
}

std::string
format_name(std::string_view _name, const config& _cfg)
{
    if(!_cfg.demangle && !_cfg.truncate) return std::string{_name};

    // truncating requires demangling first so always demangle
    auto _demangled_name =
        common::cxx_demangle(std::regex_replace(_name.data(), std::regex{"(\\.kd)$"}, ""));

    if(_cfg.truncate) return common::truncate_name(_demangled_name);

    return _demangled_name;
}

void
initialize()
{
    (void) get_config<config_context::global>();
}
}  // namespace tool
}  // namespace rocprofiler
