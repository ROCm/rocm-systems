// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "library/components/cpu_load.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unistd.h>  // for sysconf(_SC_CLK_TCK)
#include <vector>

namespace rocprofsys
{
namespace component
{
namespace
{
// Helper: Get environment variable value
std::string
get_sampling_cpus_env()
{
    const char* env = std::getenv("ROCPROFSYS_SAMPLING_CPUS");
    return env ? std::string(env) : "all";
}

// Helper: Parse delimited string
std::vector<std::string>
delimit(const std::string& str, const std::string& delims)
{
    std::vector<std::string> result;
    size_t                   start = 0;
    size_t                   end   = str.find_first_of(delims);

    while(end != std::string::npos)
    {
        if(end != start)
        {
            result.push_back(str.substr(start, end - start));
        }
        start = end + 1;
        end   = str.find_first_of(delims, start);
    }

    if(start < str.length())
    {
        result.push_back(str.substr(start));
    }

    return result;
}

// Helper: Count CPUs from /proc/stat
size_t
count_cpus()
{
    std::ifstream ifs("/proc/stat");
    if(!ifs.is_open()) return 0;

    size_t      count = 0;
    std::string line;
    while(std::getline(ifs, line))
    {
        if(line.size() >= 4 && line.substr(0, 3) == "cpu" && std::isdigit(line[3]))
        {
            count++;
        }
    }
    return count;
}

}  // anonymous namespace

// Static member definitions
std::map<uint64_t, cpu_load::cpu_times> cpu_load::s_previous_times;
cpu_load::cpu_id_set_t                  cpu_load::s_enabled_cpus;
uint64_t                                cpu_load::s_ns_per_jiffy = 0;

void
cpu_load::configure()
{
    auto ncpu        = count_cpus();
    auto enabled_val = get_sampling_cpus_env();

    // Normalize to lowercase
    std::transform(enabled_val.begin(),
                   enabled_val.end(),
                   enabled_val.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // Handle aliases
    if(enabled_val == "off") enabled_val = "none";
    if(enabled_val == "on") enabled_val = "all";

    s_enabled_cpus.clear();

    if(enabled_val == "all")
    {
        for(size_t i = 0; i < ncpu; ++i)
        {
            s_enabled_cpus.insert(i);
        }
    }
    else if(enabled_val != "none")
    {
        // Parse comma/semicolon/space delimited list
        auto tokens = delimit(enabled_val, ",; \t");
        for(const auto& token : tokens)
        {
            // Check for range (e.g., "0-7")
            if(token.find('-') != std::string::npos)
            {
                auto parts = delimit(token, "-");
                if(parts.size() == 2)
                {
                    uint64_t start = std::stoull(parts[0]);
                    uint64_t end   = std::stoull(parts[1]);
                    for(uint64_t i = start; i <= end; ++i)
                    {
                        if(i < ncpu)
                        {
                            s_enabled_cpus.insert(i);
                        }
                    }
                }
            }
            else
            {
                // Single CPU ID
                uint64_t cpu_id = std::stoull(token);
                if(cpu_id < ncpu)
                {
                    s_enabled_cpus.insert(cpu_id);
                }
            }
        }
    }

    // Initialize conversion factor
    s_ns_per_jiffy = 1000000000ull / static_cast<uint64_t>(sysconf(_SC_CLK_TCK));

    // Clear previous times on reconfiguration
    s_previous_times.clear();
}

const cpu_load::cpu_id_set_t&
cpu_load::get_enabled_cpus()
{
    return s_enabled_cpus;
}

void
cpu_load::sample()
{
    m_loads.clear();

    std::ifstream ifs("/proc/stat");
    if(!ifs.is_open())
    {
        return;
    }

    std::string line;
    while(std::getline(ifs, line))
    {
        // Look for lines like "cpu0 ..."
        if(line.size() < 4 || line.substr(0, 3) != "cpu" || !std::isdigit(line[3]))
        {
            continue;
        }

        std::istringstream iss(line);
        std::string        cpu_label;
        cpu_times          current;

        iss >> cpu_label >> current.user >> current.nice >> current.system >> current.idle >>
            current.iowait >> current.irq >> current.softirq;

        // Extract CPU ID from "cpuN"
        uint64_t cpu_id = std::stoull(cpu_label.substr(3));

        // Skip if not in enabled set
        if(s_enabled_cpus.find(cpu_id) == s_enabled_cpus.end())
        {
            continue;
        }

        // Check if we have previous data for this CPU
        auto prev_it = s_previous_times.find(cpu_id);
        if(prev_it == s_previous_times.end())
        {
            // First sample - just store baseline
            s_previous_times[cpu_id] = current;
            continue;
        }

        // Calculate delta
        const cpu_times& prev         = prev_it->second;
        uint64_t         total_delta  = current.total() - prev.total();
        uint64_t         active_delta = current.active() - prev.active();

        // Compute load percentage
        double load_pct = 0.0;
        if(total_delta > 0)
        {
            load_pct =
                100.0 * static_cast<double>(active_delta) / static_cast<double>(total_delta);
        }

        // Store result
        m_loads[cpu_id] = load_pct;

        // Update previous
        s_previous_times[cpu_id] = current;
    }
}

double
cpu_load::get_load(uint64_t cpu_id) const
{
    auto it = m_loads.find(cpu_id);
    if(it != m_loads.end())
    {
        return it->second;
    }
    return 0.0;
}

}  // namespace component
}  // namespace rocprofsys
