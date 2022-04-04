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

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace omnitrace
{
namespace cpu_freq
{
namespace
{
struct cpu_freq
{};
using freq_pair_t                                            = std::pair<size_t, double>;
std::vector<std::deque<freq_pair_t>> cpu_frequencies         = {};
std::set<size_t>                     enabled_cpu_frequencies = {};

struct cpu_mem
{};
using cpu_mem_usage_pair_t                     = std::pair<size_t, int64_t>;
std::deque<cpu_mem_usage_pair_t> cpu_mem_usage = {};

int64_t                        ncpu        = threading::affinity::hw_concurrency();
std::unique_ptr<std::ifstream> ifs         = {};
std::vector<size_t>            cpu_mhz_pos = {};

}  // namespace

void
setup()
{
    perfetto_counter_track<cpu_freq>::init();
    perfetto_counter_track<cpu_mem>::init();
}

void
config()
{
    auto                _ncpu = threading::affinity::hw_concurrency();
    std::vector<size_t> _cpu_mhz_pos{};
    std::ifstream       _ifs{ "/proc/cpuinfo" };
    if(_ifs)
    {
        for(size_t i = 0; i < _ncpu; ++i)
        {
            short       _n = 0;
            std::string _st{};
            while(_ifs && _ifs.good())
            {
                std::string _s{};
                _ifs >> _s;
                if(!_ifs.good() || !_ifs) break;

                if(_s == "cpu" || _s == "MHz" || _s == ":")
                {
                    ++_n;
                    _st += _s + " ";
                }
                else
                {
                    _n  = 0;
                    _st = {};
                }

                if(_n == 3)
                {
                    size_t _pos = _ifs.tellg();
                    _cpu_mhz_pos.emplace_back(_pos + 1);
                    _ifs >> _s;
                    if(!_ifs.good() || !_ifs) break;
                    OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug() || get_verbose() > 1,
                                                      "[%zu] %s %s (pos = %zu)\n", i,
                                                      _st.c_str(), _s.c_str(), _pos + 1);
                    break;
                }
            }
        }
    }

    _ifs.close();

    auto _enabled_val = get_sampling_cpus();
    if(_enabled_val != "none" && _enabled_val != "all")
    {
        auto _enabled = tim::delimit(_enabled_val, ",; \t");
        if(_enabled.empty())
        {
            for(size_t i = 0; i < _ncpu; ++i)
                enabled_cpu_frequencies.emplace(i);
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
                    enabled_cpu_frequencies.insert(i);
            }
            else
            {
                enabled_cpu_frequencies.insert(std::stoull(_v));
            }
        }
    }
    cpu_frequencies.resize(_ncpu);
    cpu_mhz_pos = _cpu_mhz_pos;
    ifs         = std::make_unique<std::ifstream>("/proc/cpuinfo", std::ifstream::binary);
}

void
sample()
{
    cpu_mem_usage.emplace_back(tim::get_clock_real_now<size_t, std::nano>(),
                               tim::get_page_rss());

    if(!ifs) return;

    auto _read_cpu_freq = [](size_t _idx) {
        double _freq = 0;
        ifs->seekg(cpu_mhz_pos.at(_idx), ifs->beg);
        (*ifs) >> _freq;
        return _freq;
    };

    auto _ts = tim::get_clock_real_now<size_t, std::nano>();
    for(int64_t i = 0; i < ncpu; ++i)
    {
        if(!enabled_cpu_frequencies.empty() && enabled_cpu_frequencies.count(i) == 0)
            continue;
        cpu_frequencies.at(i).emplace_back(_ts, _read_cpu_freq(i));
    }
}

void
shutdown()
{}

void
post_process()
{
    OMNITRACE_PRINT("Post-processing %zu cpu freqs and %zu memory usage entries\n",
                    cpu_frequencies.size(), cpu_mem_usage.size());
    auto _process_frequencies = [](size_t _idx) {
        using counter_track = perfetto_counter_track<cpu_freq>;
        if(!counter_track::exists(_idx))
        {
            auto _devname = TIMEMORY_JOIN("", "[CPU ", _idx, "] ");
            auto addendum = [&](const char* _v) { return _devname + std::string{ _v }; };
            counter_track::emplace(_idx, addendum("Frequency (S)"), "MHz");
        }

        for(auto& itr : cpu_frequencies.at(_idx))
        {
            uint64_t _ts   = itr.first;
            double   _freq = itr.second;
            TRACE_COUNTER("sampling", counter_track::at(_idx, 0), _ts, _freq);
        }
    };

    auto _process_cpu_mem_usage = []() {
        using counter_track = perfetto_counter_track<cpu_mem>;
        if(!counter_track::exists(0))
        {
            auto _devname = TIMEMORY_JOIN("", "[CPU] ");
            auto addendum = [&](const char* _v) { return _devname + std::string{ _v }; };
            counter_track::emplace(0, addendum("Memory Usage (S)"), "MB");
        }

        for(auto& itr : cpu_mem_usage)
        {
            uint64_t _ts        = itr.first;
            double   _mem_usage = itr.second;
            TRACE_COUNTER("sampling", counter_track::at(0, 0), _ts,
                          _mem_usage / units::megabyte);
        }
    };

    _process_cpu_mem_usage();
    for(size_t i = 0; i < cpu_frequencies.size(); ++i)
        _process_frequencies(i);
}
}  // namespace cpu_freq
}  // namespace omnitrace
