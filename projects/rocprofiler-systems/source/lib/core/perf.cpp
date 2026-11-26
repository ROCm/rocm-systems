// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "perf.hpp"
#include "common/clocks.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <fmt/format.h>
#include <linux/perf_event.h>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

namespace rocprofsys::perf
{

namespace
{
constexpr std::string_view k_hw_cache_prefix    = "PERF_COUNT_HW_CACHE_";
constexpr char             k_modifier_separator = ':';

// Bit positions of the operation and result ids in a PERF_TYPE_HW_CACHE config,
// see perf_event_open(2).
constexpr int k_hw_cache_op_shift     = 8;
constexpr int k_hw_cache_result_shift = 16;

template <typename Tp>
using named_value = std::pair<std::string_view, Tp>;

// Names follow the libpfm4 perf event table, which is what rocprof-sys-avail lists.
constexpr auto k_hw_cache_ids = std::to_array<named_value<hw_cache_config>>({
    { "L1D", hw_cache_config::l1d },
    { "L1I", hw_cache_config::l1i },
    { "LL", hw_cache_config::ll },
    { "DTLB", hw_cache_config::dtlb },
    { "ITLB", hw_cache_config::itlb },
    { "BPU", hw_cache_config::bpu },
    { "NODE", hw_cache_config::node },
});

constexpr auto k_hw_cache_ops = std::to_array<named_value<hw_cache_op>>({
    { "READ", hw_cache_op::read },
    { "WRITE", hw_cache_op::write },
    { "PREFETCH", hw_cache_op::prefetch },
});

constexpr auto k_hw_cache_results = std::to_array<named_value<hw_cache_op_result>>({
    { "ACCESS", hw_cache_op_result::access },
    { "MISS", hw_cache_op_result::miss },
});

template <typename Tp, std::size_t N>
constexpr std::optional<Tp>
find_named(const std::array<named_value<Tp>, N>& table, std::string_view name)
{
    const auto* const itr = std::ranges::find(table, name, &named_value<Tp>::first);
    if(itr == table.end())
    {
        return std::nullopt;
    }
    return itr->second;
}
}  // namespace

event_type
get_event_type(std::string_view name)
{
    if(name.find("PERF_COUNT_SW_") != std::string_view::npos)
    {
        return event_type::software;
    }
    // These two share the cache prefix but are plain hardware events.
    const bool is_plain_hardware_event = name.ends_with("PERF_COUNT_HW_CACHE_MISSES") ||
                                         name.ends_with("PERF_COUNT_HW_CACHE_REFERENCES");
    if(name.find(k_hw_cache_prefix) != std::string_view::npos && !is_plain_hardware_event)
    {
        return event_type::hw_cache;
    }
    if(name.find("PERF_COUNT_HW_") != std::string_view::npos)
    {
        return event_type::hardware;
    }
    return event_type::max;
}

hw_config
get_hw_config(std::string_view _v)
{
#define HW_CONFIG_REGEX(KEY) std::regex_search(_v.data(), std::regex{ "(HW_" KEY ")$" })

    if(HW_CONFIG_REGEX("CPU_CYCLES"))
    {
        return hw_config::cpu_cycles;
    }
    if(HW_CONFIG_REGEX("INSTRUCTIONS"))
    {
        return hw_config::instructions;
    }
    if(HW_CONFIG_REGEX("CACHE_REFERENCES"))
    {
        return hw_config::cache_references;
    }
    if(HW_CONFIG_REGEX("CACHE_MISSES"))
    {
        return hw_config::cache_misses;
    }
    if(HW_CONFIG_REGEX("BRANCH_INSTRUCTIONS"))
    {
        return hw_config::branch_instructions;
    }
    if(HW_CONFIG_REGEX("BRANCH_MISSES"))
    {
        return hw_config::branch_misses;
    }
    if(HW_CONFIG_REGEX("BUS_CYCLES"))
    {
        return hw_config::bus_cycles;
    }
    if(HW_CONFIG_REGEX("STALLED_CYCLES_FRONTEND"))
    {
        return hw_config::stalled_cycles_frontend;
    }
    if(HW_CONFIG_REGEX("STALLED_CYCLES_BACKEND"))
    {
        return hw_config::stalled_cycles_backend;
    }
    if(HW_CONFIG_REGEX("REF_CPU_CYCLES"))
    {
        return hw_config::reference_cpu_cycles;
    }

    throw std::runtime_error(fmt::format("Unknown perf hardware config: {}", _v.data()));

#undef HW_CONFIG_REGEX

    return hw_config::max;
}

sw_config
get_sw_config(std::string_view _v)
{
#define SW_CONFIG_REGEX(KEY) std::regex_search(_v.data(), std::regex{ "(SW_" KEY ")$" })

    if(SW_CONFIG_REGEX("CPU_CLOCK"))
    {
        return sw_config::cpu_clock;
    }
    if(SW_CONFIG_REGEX("TASK_CLOCK"))
    {
        return sw_config::task_clock;
    }
    if(SW_CONFIG_REGEX("PAGE_FAULTS"))
    {
        return sw_config::page_faults;
    }
    if(SW_CONFIG_REGEX("CONTEXT_SWITCHES"))
    {
        return sw_config::context_switches;
    }
    if(SW_CONFIG_REGEX("CPU_MIGRATIONS"))
    {
        return sw_config::cpu_migrations;
    }
    if(SW_CONFIG_REGEX("PAGE_FAULTS_MIN"))
    {
        return sw_config::page_faults_minor;
    }
    if(SW_CONFIG_REGEX("PAGE_FAULTS_MAJ"))
    {
        return sw_config::page_faults_major;
    }
    if(SW_CONFIG_REGEX("ALIGNMENT_FAULTS"))
    {
        return sw_config::alignment_faults;
    }
    if(SW_CONFIG_REGEX("EMULATION_FAULTS"))
    {
        return sw_config::emulation_faults;
    }

    throw std::runtime_error(fmt::format("Unknown perf software config: {}", _v.data()));

#undef SW_CONFIG_REGEX

    return sw_config::max;
}

int
get_hw_cache_config(std::string_view name)
{
    const auto make_error = [name](std::string_view reason) {
        return std::runtime_error(
            fmt::format("Unknown perf hw cache config: {} ({})", name, reason));
    };

    const auto prefix_pos = name.find(k_hw_cache_prefix);
    if(prefix_pos == std::string_view::npos)
    {
        throw make_error("missing PERF_COUNT_HW_CACHE_ prefix");
    }

    // Split by hand: the pieces of std::views::split are not contiguous in GCC 11, so a
    // string_view cannot be built from them. Always yields at least one token.
    const auto spec = name.substr(prefix_pos + k_hw_cache_prefix.size());
    std::vector<std::string_view> tokens;
    for(std::size_t token_pos = 0; token_pos <= spec.size();)
    {
        const auto separator_pos = spec.find(k_modifier_separator, token_pos);
        tokens.emplace_back(spec.substr(token_pos, separator_pos - token_pos));
        if(separator_pos == std::string_view::npos)
        {
            break;
        }
        token_pos = separator_pos + 1;
    }

    const auto cache = find_named(k_hw_cache_ids, tokens.front());
    if(!cache)
    {
        throw make_error("unknown cache id");
    }

    std::optional<hw_cache_op>        cache_op;
    std::optional<hw_cache_op_result> cache_result;
    for(const auto modifier : tokens | std::views::drop(1))
    {
        const auto named_op     = find_named(k_hw_cache_ops, modifier);
        const auto named_result = find_named(k_hw_cache_results, modifier);
        if(!named_op && !named_result)
        {
            throw make_error("unknown modifier");
        }
        if((named_op && cache_op) || (named_result && cache_result))
        {
            throw make_error("more than one operation or result modifier");
        }
        if(named_op)
        {
            cache_op = named_op;
        }
        else
        {
            cache_result = named_result;
        }
    }

    // Unspecified groups take the libpfm4 defaults (READ, MISS), so a name means the
    // same thing here as it does as a PAPI counter.
    return static_cast<int>(*cache) |
           (static_cast<int>(cache_op.value_or(hw_cache_op::read))
            << k_hw_cache_op_shift) |
           (static_cast<int>(cache_result.value_or(hw_cache_op_result::miss))
            << k_hw_cache_result_shift);
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

    // perf_event_open() fails with EINVAL when a hardware event asks for the timeline
    // clock (CLOCK_BOOTTIME), so only software events sample on it. Hardware samples are
    // stamped in CLOCK_MONOTONIC and shifted onto the timeline when they are parsed.
    perf_event_att.use_clockid = 1;
    perf_event_att.clockid     = (perf_event_att.type == PERF_TYPE_SOFTWARE)
                                     ? clocks::timeline::k_timeline_clock_id
                                     : CLOCK_MONOTONIC;
}
}  // namespace rocprofsys::perf
