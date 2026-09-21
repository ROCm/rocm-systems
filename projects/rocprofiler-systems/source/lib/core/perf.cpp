// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "perf.hpp"
#include "logger/debug.hpp"

#include <chrono>
#include <cstdint>
#include <linux/perf_event.h>

using namespace std::chrono_literals;

namespace rocprofsys::perf
{

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
        throw std::runtime_error(
            fmt::format("Unknown perf hardware config: {}", _v.data()));
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
        throw std::runtime_error(
            fmt::format("Unknown perf hw cache config: {}", _v.data()));
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
        throw std::runtime_error(
            fmt::format("Unknown perf software config: {}", _v.data()));

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
config_overflow_sampling(struct perf_event_attr& perf_event_att, std::string_view event,
                         double freq)
{
    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>{ 1.0 / freq });

    perf_event_att.type = static_cast<int>(perf::get_event_type(event));
    switch(perf_event_att.type)
    {
        case PERF_TYPE_HARDWARE:
        {
            perf_event_att.config = static_cast<int>(perf::get_hw_config(event));
            break;
        }
        case PERF_TYPE_SOFTWARE:
        {
            perf_event_att.config = static_cast<int>(perf::get_sw_config(event));
            break;
        }
        case PERF_TYPE_HW_CACHE:
        {
            perf_event_att.config = static_cast<int>(perf::get_hw_cache_config(event));
            break;
        }
        case PERF_TYPE_BREAKPOINT:
        case PERF_TYPE_TRACEPOINT:
        case PERF_TYPE_RAW:
        case PERF_TYPE_MAX:
        default:
        {
            throw std::runtime_error("Unsupported perf type");
        }
    };

    if(perf_event_att.type == PERF_TYPE_SOFTWARE &&
       (perf_event_att.config == PERF_COUNT_SW_CPU_CLOCK ||
        perf_event_att.config == PERF_COUNT_SW_TASK_CLOCK))
    {
        perf_event_att.sample_period = static_cast<std::uint64_t>(period.count());
    }
    else
    {
        perf_event_att.sample_period = static_cast<std::uint64_t>(freq);
    }
}
}  // namespace rocprofsys::perf
