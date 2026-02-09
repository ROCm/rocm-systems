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

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>

namespace rocprofsys
{
namespace component
{
class cpu_load
{
public:
    using cpu_id_set_t = std::set<uint64_t>;

    struct cpu_times
    {
        uint64_t user    = 0;
        uint64_t nice    = 0;
        uint64_t system  = 0;
        uint64_t idle    = 0;
        uint64_t iowait  = 0;
        uint64_t irq     = 0;
        uint64_t softirq = 0;

        uint64_t total() const
        {
            return user + nice + system + idle + iowait + irq + softirq;
        }

        uint64_t active() const { return user + nice + system + irq + softirq; }
    };

    cpu_load() = default;

    // Static metadata
    static std::string label() { return "cpu_load"; }
    static std::string description() { return "CPU load percentage"; }
    static std::string unit() { return "%"; }
    static std::string display_unit() { return "%"; }

    // Configuration
    static void                configure();
    static const cpu_id_set_t& get_enabled_cpus();

    // Data collection
    void sample();

    // Data access
    const std::map<uint64_t, double>& get_loads() const { return m_loads; }
    double                            get_load(uint64_t cpu_id) const;

private:
    std::map<uint64_t, double> m_loads;  // CPU ID -> load percentage

    static std::map<uint64_t, cpu_times> s_previous_times;
    static cpu_id_set_t                  s_enabled_cpus;
    static uint64_t                      s_ns_per_jiffy;
};

}  // namespace component
}  // namespace rocprofsys
