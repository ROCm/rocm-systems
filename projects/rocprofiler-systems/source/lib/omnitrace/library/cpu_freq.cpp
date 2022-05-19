// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/cpu_freq.hpp"
#include "library/common.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"
#include "library/perfetto.hpp"
#include "library/timemory.hpp"

#include <timemory/components/rusage/backends.hpp>
#include <timemory/utility/procfs/cpuinfo.hpp>

#include <cstdlib>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cpuinfo = tim::procfs::cpuinfo;

namespace omnitrace
{
namespace cpu_freq
{
namespace
{
struct cpu_freq
{};
struct cpu_page
{};
struct cpu_virt
{};
using cpu_data_tuple_t = std::tuple<size_t, int64_t, int64_t, std::vector<double>>;

std::set<size_t>             enabled_cpu_freqs = {};
std::deque<cpu_data_tuple_t> cpu_data          = {};
int64_t                      ncpu              = threading::affinity::hw_concurrency();
}  // namespace

void
setup()
{
    perfetto_counter_track<cpu_freq>::init();
    perfetto_counter_track<cpu_page>::init();
    perfetto_counter_track<cpu_virt>::init();
}

void
config()
{
    auto _ncpu          = cpuinfo::freq::size();
    auto _enabled_freqs = std::set<size_t>{};

    auto _enabled_val = get_sampling_cpus();
    for(auto& itr : _enabled_val)
        itr = tolower(itr);
    if(_enabled_val == "off")
        _enabled_val = "none";
    else if(_enabled_val == "on")
        _enabled_val = "all";
    if(_enabled_val != "none" && _enabled_val != "all")
    {
        auto _enabled = tim::delimit(_enabled_val, ",; \t");
        if(_enabled.empty())
        {
            for(size_t i = 0; i < _ncpu; ++i)
                _enabled_freqs.emplace(i);
        }
        for(auto&& _v : _enabled)
        {
            if(_v.find_first_not_of("0123456789-") != std::string::npos)
            {
                OMNITRACE_VERBOSE_F(
                    0,
                    "Invalid CPU specification. Only numerical values (e.g., 0) or "
                    "ranges (e.g., 0-7) are permitted. Ignoring %s...",
                    _v.c_str());
                continue;
            }
            if(_v.find('-') != std::string::npos)
            {
                auto _vv = tim::delimit(_v, "-");
                OMNITRACE_CONDITIONAL_THROW(
                    _vv.size() != 2,
                    "Invalid CPU range specification: %s. Required format N-M, e.g. 0-4",
                    _v.c_str());
                for(size_t i = std::stoull(_vv.at(0)); i < std::stoull(_vv.at(1)); ++i)
                    _enabled_freqs.emplace(i);
            }
            else
            {
                _enabled_freqs.emplace(std::stoull(_v));
            }
        }
    }
    else if(_enabled_val == "all")
    {
        for(size_t i = 0; i < _ncpu; ++i)
            _enabled_freqs.emplace(i);
    }
    else if(_enabled_val == "none")
    {
        _enabled_freqs.clear();
    }

    for(auto itr : _enabled_freqs)
    {
        if(itr < cpuinfo::freq::size())
            _enabled_freqs.emplace(itr);
        else
        {
            OMNITRACE_VERBOSE(
                0, "[cpu_freq::config] Warning! Removing invalid cpu %zu...\n", itr);
            OMNITRACE_CI_FAIL(true, "[cpu_freq::config] CPU frequencies are disabled "
                                    ":: unable to open /proc/cpuinfo");
        }
    }

    if(!cpuinfo::freq{})
    {
        OMNITRACE_VERBOSE(0, "[cpu_freq::config] Warning! CPU frequencies are disabled "
                             ":: unable to open /proc/cpuinfo");
        _enabled_freqs.clear();
    }

    OMNITRACE_CI_FAIL(!cpuinfo::freq{}, "[cpu_freq::config] CPU frequencies are disabled "
                                        ":: unable to open /proc/cpuinfo");

    enabled_cpu_freqs = _enabled_freqs;
}

void
sample()
{
    std::vector<double> _freqs{};
    if(!enabled_cpu_freqs.empty())
    {
        _freqs.reserve(enabled_cpu_freqs.size());
        auto&& _freq = cpuinfo::freq{};
        for(const auto& itr : enabled_cpu_freqs)
        {
            _freqs.emplace_back(_freq(itr));
        }
    }

    auto _ts = tim::get_clock_real_now<size_t, std::nano>();

    cpu_data.emplace_back(_ts, tim::get_page_rss(), tim::get_virt_mem(),
                          std::move(_freqs));
}

void
shutdown()
{}

void
post_process()
{
    OMNITRACE_PRINT("Post-processing %zu cpu frequency and memory usage entries...\n",
                    cpu_data.size());
    auto _process_frequencies = [](size_t _idx, size_t _offset) {
        using freq_track = perfetto_counter_track<cpu_freq>;
        if(!freq_track::exists(_idx))
        {
            auto _devname = TIMEMORY_JOIN("", "[CPU ", _idx, "] ");
            auto addendum = [&](const char* _v) { return _devname + std::string{ _v }; };
            freq_track::emplace(_idx, addendum("Frequency (S)"), "MHz");
        }

        for(auto& itr : cpu_data)
        {
            uint64_t _ts   = std::get<0>(itr);
            double   _freq = std::get<3>(itr).at(_offset);
            TRACE_COUNTER("sampling", freq_track::at(_idx, 0), _ts, _freq);
        }
    };

    auto _process_cpu_mem_usage = []() {
        using page_track = perfetto_counter_track<cpu_page>;
        using virt_track = perfetto_counter_track<cpu_virt>;

        if(!page_track::exists(0))
        {
            auto _devname = TIMEMORY_JOIN("", "[CPU] ");
            auto addendum = [&](const char* _v) { return _devname + std::string{ _v }; };
            page_track::emplace(0, addendum("Memory Usage (S)"), "MB");
        }

        if(!virt_track::exists(0))
        {
            auto _devname = TIMEMORY_JOIN("", "[CPU] ");
            auto addendum = [&](const char* _v) { return _devname + std::string{ _v }; };
            virt_track::emplace(0, addendum("Virtual Memory Usage (S)"), "MB");
        }

        for(auto& itr : cpu_data)
        {
            uint64_t _ts   = std::get<0>(itr);
            double   _page = std::get<1>(itr);
            double   _virt = std::get<2>(itr);
            TRACE_COUNTER("sampling", page_track::at(0, 0), _ts, _page / units::megabyte);
            TRACE_COUNTER("sampling", virt_track::at(0, 0), _ts, _virt / units::megabyte);
        }
    };

    _process_cpu_mem_usage();
    for(auto itr = enabled_cpu_freqs.begin(); itr != enabled_cpu_freqs.end(); ++itr)
    {
        auto _idx    = *itr;
        auto _offset = std::distance(enabled_cpu_freqs.begin(), itr);
        _process_frequencies(_idx, _offset);
    }

    enabled_cpu_freqs.clear();
}
}  // namespace cpu_freq
}  // namespace omnitrace
