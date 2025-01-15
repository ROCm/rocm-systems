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

#include "perf.hpp"
#include "debug.hpp"

#include <timemory/units.hpp>

namespace rocprofsys
{
namespace perf
{
namespace units = ::tim::units;

std::vector<std::string>
get_config_choices()
{
    namespace regex_const = ::std::regex_constants;

    auto       _data        = std::vector<std::string>{};
    auto       _papi_events = tim::papi::available_events_info();
    const auto _prefix      = std::string_view{ "perf::" };
    auto       _regex =
        std::regex{ "^(perf::|)PERF_COUNT_(HW|SW|HW_CACHE)_([A-Z_]+)(|:[A-Z]+)$",
                    regex_const::optimize };

    for(const auto& itr : _papi_events)
    {
        if(std::regex_match(itr.symbol(), _regex))
        {
            auto _symbol = itr.symbol();
            auto _pos    = _symbol.find(_prefix);
            if(_pos == 0) _symbol = _symbol.substr(_prefix.length());
            _data.emplace_back(_symbol);
        }
    }

    std::sort(_data.begin(), _data.end());
    _data.erase(std::unique(_data.begin(), _data.end()), _data.end());

    return _data;
}

event_type
get_event_type(std::string_view _v)
{
    if(_v.find("PERF_COUNT_SW_") != std::string_view::npos)
        return event_type::software;
    else if(_v.find("PERF_COUNT_SW_") != std::string_view::npos &&
            !std::regex_search(_v.data(),
                               std::regex{ "PERF_COUNT_HW_CACHE_(MISSES|REFERENCES)$" }))
        return event_type::hw_cache;
    else if(_v.find("PERF_COUNT_HW_") != std::string_view::npos)
        return event_type::hardware;
    return event_type::max;
}

hw_config
get_hw_config(std::string_view _v)
{
#define HW_CONFIG_REGEX(KEY) std::regex_search(_v.data(), std::regex{ "(HW_" KEY ")$" })

    if(HW_CONFIG_REGEX("CPU_CYCLES"))
        return hw_config::cpu_cycles;
    else if(HW_CONFIG_REGEX("INSTRUCTIONS"))
        return hw_config::instructions;
    else if(HW_CONFIG_REGEX("CACHE_REFERENCES"))
        return hw_config::cache_references;
    else if(HW_CONFIG_REGEX("CACHE_MISSES"))
        return hw_config::cache_misses;
    else if(HW_CONFIG_REGEX("BRANCH_INSTRUCTIONS"))
        return hw_config::branch_instructions;
    else if(HW_CONFIG_REGEX("BRANCH_MISSES"))
        return hw_config::branch_misses;
    else if(HW_CONFIG_REGEX("BUS_CYCLES"))
        return hw_config::bus_cycles;
    else if(HW_CONFIG_REGEX("STALLED_CYCLES_FRONTEND"))
        return hw_config::stalled_cycles_frontend;
    else if(HW_CONFIG_REGEX("STALLED_CYCLES_BACKEND"))
        return hw_config::stalled_cycles_backend;
    else if(HW_CONFIG_REGEX("REF_CPU_CYCLES"))
        return hw_config::reference_cpu_cycles;
    else
    {
        ROCPROFSYS_THROW("Unknown perf hardware config: %s", _v.data());
    }

#undef HW_CONFIG_REGEX

    return hw_config::max;
}

sw_config
get_sw_config(std::string_view _v)
{
#define SW_CONFIG_REGEX(KEY) std::regex_search(_v.data(), std::regex{ "(SW_" KEY ")$" })

    if(SW_CONFIG_REGEX("CPU_CLOCK"))
        return sw_config::cpu_clock;
    else if(SW_CONFIG_REGEX("TASK_CLOCK"))
        return sw_config::task_clock;
    else if(SW_CONFIG_REGEX("PAGE_FAULTS"))
        return sw_config::page_faults;
    else if(SW_CONFIG_REGEX("CONTEXT_SWITCHES"))
        return sw_config::context_switches;
    else if(SW_CONFIG_REGEX("CPU_MIGRATIONS"))
        return sw_config::cpu_migrations;
    else if(SW_CONFIG_REGEX("PAGE_FAULTS_MIN"))
        return sw_config::page_faults_minor;
    else if(SW_CONFIG_REGEX("PAGE_FAULTS_MAJ"))
        return sw_config::page_faults_major;
    else if(SW_CONFIG_REGEX("ALIGNMENT_FAULTS"))
        return sw_config::alignment_faults;
    else if(SW_CONFIG_REGEX("EMULATION_FAULTS"))
        return sw_config::emulation_faults;
    else
    {
        ROCPROFSYS_THROW("Unknown perf hw cache config: %s", _v.data());
    }

#undef SW_CONFIG_REGEX

    return sw_config::max;
}

int
get_hw_cache_config(std::string_view _v)
{
    int _value = 0;

#define HW_CACHE_CONFIG_REGEX(KEY)                                                       \
    std::regex_search(_v.data(), std::regex{ "(HW_CACHE_" KEY ")" })

    if(HW_CACHE_CONFIG_REGEX("L1D"))
        _value |= static_cast<int>(hw_cache_config::l1d);
    else if(HW_CACHE_CONFIG_REGEX("L1I"))
        _value |= static_cast<int>(hw_cache_config::l1i);
    else if(HW_CACHE_CONFIG_REGEX("LL"))
        _value |= static_cast<int>(hw_cache_config::ll);
    else if(HW_CACHE_CONFIG_REGEX("DTLB"))
        _value |= static_cast<int>(hw_cache_config::dtlb);
    else if(HW_CACHE_CONFIG_REGEX("ITLB"))
        _value |= static_cast<int>(hw_cache_config::itlb);
    else if(HW_CACHE_CONFIG_REGEX("BPU"))
        _value |= static_cast<int>(hw_cache_config::bpu);
    else if(HW_CACHE_CONFIG_REGEX("NODE"))
        _value |= static_cast<int>(hw_cache_config::node);
    else
        ROCPROFSYS_THROW("Unknown perf software config: %s", _v.data());

#undef HW_CACHE_CONFIG_REGEX
#define HW_CACHE_OP_REGEX(KEY)                                                           \
    std::regex_search(_v.data(), std::regex{ "(HW_CACHE_([A-Z1]+):" KEY ")" })

    if(HW_CACHE_OP_REGEX("READ"))
        _value |= (static_cast<int>(hw_cache_op::read) << 8);
    else if(HW_CACHE_OP_REGEX("WRITE"))
        _value |= (static_cast<int>(hw_cache_op::write) << 8);
    else if(HW_CACHE_OP_REGEX("PREFETCH"))
        _value |= (static_cast<int>(hw_cache_op::prefetch) << 8);
    else
        _value |= (static_cast<int>(hw_cache_op::read) << 8);

#undef HW_CACHE_OP_REGEX
#define HW_CACHE_OP_RESULT_REGEX(KEY)                                                    \
    std::regex_search(_v.data(), std::regex{ "(HW_CACHE_([A-Z1]+):" KEY ")" })

    if(HW_CACHE_OP_RESULT_REGEX("READ"))
        _value |= (static_cast<int>(hw_cache_op_result::access) << 16);
    else if(HW_CACHE_OP_RESULT_REGEX("WRITE"))
        _value |= (static_cast<int>(hw_cache_op_result::miss) << 16);
    else
        _value |= (static_cast<int>(hw_cache_op_result::access) << 16);

#undef HW_CACHE_OP_RESULT_REGEX

    return _value;
}

void
config_overflow_sampling(struct perf_event_attr& _pe, std::string_view _event,
                         double _freq)
{
    auto _period = (1.0 / _freq) * units::sec;

    _pe.type = static_cast<int>(perf::get_event_type(_event));
    switch(_pe.type)
    {
        case PERF_TYPE_HARDWARE:
        {
            _pe.config = static_cast<int>(perf::get_hw_config(_event));
            break;
        }
        case PERF_TYPE_SOFTWARE:
        {
            _pe.config = static_cast<int>(perf::get_sw_config(_event));
            break;
        }
        case PERF_TYPE_HW_CACHE:
        {
            _pe.config = static_cast<int>(perf::get_hw_cache_config(_event));
            break;
        }
        case PERF_TYPE_BREAKPOINT:
        case PERF_TYPE_TRACEPOINT:
        case PERF_TYPE_RAW:
        case PERF_TYPE_MAX:
        default:
        {
            ROCPROFSYS_THROW("unsupported perf type");
        }
    };

    if(_pe.type == PERF_TYPE_SOFTWARE &&
       (_pe.config == PERF_COUNT_SW_CPU_CLOCK || _pe.config == PERF_COUNT_SW_TASK_CLOCK))
    {
        _pe.sample_period = static_cast<uint64_t>(_period);
    }
    else
    {
        _pe.sample_period = static_cast<uint64_t>(_freq);
    }
}
}  // namespace perf
}  // namespace rocprofsys
