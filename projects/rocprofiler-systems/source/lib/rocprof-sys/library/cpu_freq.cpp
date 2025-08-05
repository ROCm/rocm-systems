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

#include "library/cpu_freq.hpp"
#include "core/agent.hpp"
#include "core/agent_manager.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/node_info.hpp"
#include "core/perfetto.hpp"
#include "core/rocpd/data_processor.hpp"
#include "core/timemory.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "library/components/cpu_freq.hpp"
#include "library/thread_info.hpp"

#include <timemory/components/rusage/backends.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/procfs/cpuinfo.hpp>
#include <timemory/utility/type_list.hpp>

#include <cstddef>
#include <cstdlib>
#include <string>
#include <sys/resource.h>
#include <tuple>
#include <utility>

namespace rocprofsys
{
namespace cpu_freq
{
template <typename... Tp>
using type_list = tim::type_list<Tp...>;

namespace
{
using cpu_data_tuple_t = std::tuple<size_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                    int64_t, int64_t, component::cpu_freq>;
std::deque<cpu_data_tuple_t> data = {};

template <typename... Types>
void
init_perfetto_counter_tracks(type_list<Types...>)
{
    (perfetto_counter_track<Types>::init(), ...);
}

template <typename Category>
inline std::string
get_cpu_freq_track_name(uint64_t cpu_id)
{
    return std::string(trait::name<Category>::value) + " [" + std::to_string(cpu_id) +
           "]";
}

template <typename Func>
void
do_for_enabled_cpus(Func&& func)
{
    const auto& enabled_cpus = component::cpu_freq::get_enabled_cpus();
    for(const auto& cpu : enabled_cpus)
    {
        func(cpu);
    }
}

rocpd::data_processor&
get_data_processor()
{
    return rocpd::data_processor::get_instance();
}

void
metadata_initialize_cpu_freq_category()
{
    trace_cache::get_metadata_registry().add_string(
        trait::name<category::cpu_freq>::value);
}

void
metadata_initialize_cpu_freq_tracks()
{
    do_for_enabled_cpus([&](size_t cpu_id) {
        trace_cache::get_metadata_registry().add_track(
            { get_cpu_freq_track_name<category::cpu_freq>(cpu_id).c_str(), std::nullopt,
              "{}" });
    });
}

void
metadata_initialize_cpu_usage_tracks()
{
    trace_cache::get_metadata_registry().add_track(
        { trait::name<category::process_page>::value, std::nullopt, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trait::name<category::process_virt>::value, std::nullopt, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trait::name<category::process_peak>::value, std::nullopt, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trait::name<category::process_context_switch>::value, std::nullopt, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trait::name<category::process_page_fault>::value, std::nullopt, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trait::name<category::process_user_mode_time>::value, std::nullopt, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trait::name<category::process_kernel_mode_time>::value, std::nullopt, "{}" });
}

void
metadata_initialize_cpu_freq_pmc(size_t dev_id)
{
    // TODO: Find the proper values for a following definitions
    size_t      EVENT_CODE       = 0;
    size_t      INSTANCE_ID      = 0;
    const char* LONG_DESCRIPTION = "";
    const char* COMPONENT        = "";
    const char* BLOCK            = "";
    const char* EXPRESSION       = "";
    const char* MEMORY           = "MB";
    const char* TIME             = "sec";
    auto        ni               = node_info::get_instance();
    const auto* TARGET_ARCH      = "CPU";

    auto& agent_mngr = agent_manager::get_instance();
    auto  agent_handle =
        agent_mngr.get_agent_by_type_index(dev_id, agent_type::CPU).handle;

    do_for_enabled_cpus([&](size_t cpu_id) {
        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_handle, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              get_cpu_freq_track_name<category::cpu_freq>(cpu_id).c_str(), "Frequency",
              trait::name<category::cpu_freq>::description, LONG_DESCRIPTION, COMPONENT,
              component::cpu_freq::display_unit().c_str(), "ABS", BLOCK, EXPRESSION, 0,
              0 });
    });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_handle, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::process_page>::value, "Memory Usage",
          trait::name<category::process_page>::description, LONG_DESCRIPTION, COMPONENT,
          MEMORY, "ABS", BLOCK, EXPRESSION, 0, 0 });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_handle, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::process_virt>::value, "Virtual Memory Usage",
          trait::name<category::process_virt>::description, LONG_DESCRIPTION, COMPONENT,
          MEMORY, "ABS", BLOCK, EXPRESSION, 0, 0 });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_handle, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::process_peak>::value, "Peak Memory",
          trait::name<category::process_peak>::description, LONG_DESCRIPTION, COMPONENT,
          MEMORY, "ABS", BLOCK, EXPRESSION, 0, 0 });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_handle, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::process_context_switch>::value, "Context Switches",
          trait::name<category::process_context_switch>::description, LONG_DESCRIPTION,
          COMPONENT, "", "ABS", BLOCK, EXPRESSION, 0, 0 });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_handle, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::process_page_fault>::value, "Page Faults",
          trait::name<category::process_page_fault>::description, LONG_DESCRIPTION,
          COMPONENT, "", "ABS", BLOCK, EXPRESSION, 0, 0 });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_handle, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::process_user_mode_time>::value, "User Time",
          trait::name<category::process_user_mode_time>::description, LONG_DESCRIPTION,
          COMPONENT, TIME, "ABS", BLOCK, EXPRESSION, 0, 0 });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_handle, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::process_kernel_mode_time>::value, "Kernel Time",
          trait::name<category::process_kernel_mode_time>::description, LONG_DESCRIPTION,
          COMPONENT, TIME, "ABS", BLOCK, EXPRESSION, 0, 0 });
}

void
rocpd_process_cpu_usage_events(const uint32_t device_id, uint64_t timestamp,
                               const component::cpu_freq& freq, double mem_page,
                               double virt_mem_page, double peak_mem,
                               double context_switch, double page_fault, double user_time,
                               double kernel_time)
{
    auto&       data_processor   = get_data_processor();
    const auto* _name            = trait::name<category::cpu_freq>::value;
    auto        name_primary_key = data_processor.insert_string(_name);
    auto        event_id         = data_processor.insert_event(name_primary_key, 0, 0, 0);

    auto& agent_mngr = agent_manager::get_instance();
    auto base_id = agent_mngr.get_agent_by_type_index(device_id, agent_type::CPU).base_id;

    auto insert_event_and_sample = [&](const char* name, double value) {
        data_processor.insert_pmc_event(event_id, base_id, name, value);
        data_processor.insert_sample(name, timestamp, event_id);
    };

    do_for_enabled_cpus([&](size_t cpu_id) {
        insert_event_and_sample(
            get_cpu_freq_track_name<category::cpu_freq>(cpu_id).c_str(), freq.at(cpu_id));
    });

    insert_event_and_sample(trait::name<category::process_page>::value, mem_page);
    insert_event_and_sample(trait::name<category::process_virt>::value, virt_mem_page);
    insert_event_and_sample(trait::name<category::process_peak>::value, peak_mem);
    insert_event_and_sample(trait::name<category::process_context_switch>::value,
                            context_switch);
    insert_event_and_sample(trait::name<category::process_page_fault>::value, page_fault);
    insert_event_and_sample(trait::name<category::process_user_mode_time>::value,
                            user_time);
    insert_event_and_sample(trait::name<category::process_kernel_mode_time>::value,
                            kernel_time);
}

}  // namespace
}  // namespace cpu_freq
}  // namespace rocprofsys

namespace rocprofsys
{
namespace cpu_freq
{
void
setup()
{
    if(get_use_perfetto())
    {
        init_perfetto_counter_tracks(
            type_list<category::cpu_freq, category::process_page, category::process_virt,
                      category::process_peak, category::process_context_switch,
                      category::process_page_fault, category::process_user_mode_time,
                      category::process_kernel_mode_time>{});
    }

    metadata_initialize_cpu_freq_category();
    metadata_initialize_cpu_usage_tracks();
    metadata_initialize_cpu_freq_tracks();

    // `get_enabled_cpus()` returns the number of cores enabled for monitoring but
    // the actual device_id is 0, since there is a single device available. And
    // the agents seems to be assigned per device basis not per core.
    // TODO: `get_enabled_cpus()` should be fixed in the future to align with GPU
    // implementation.
    auto cpu_agents = agent_manager::get_instance().get_agents_by_type(agent_type::CPU);
    for(auto& agent : cpu_agents)
    {
        metadata_initialize_cpu_freq_pmc(agent->device_id);
    }
}

void
config()
{
    component::cpu_freq::configure();
}

void
sample()
{
    auto _ts = tim::get_clock_real_now<size_t, std::nano>();

    auto _rcache = tim::rusage_cache{ RUSAGE_SELF };
    auto _freqs  = component::cpu_freq{}.sample();

    // user and kernel mode times are in microseconds
    data.emplace_back(
        _ts, tim::get_page_rss(), tim::get_virt_mem(), _rcache.get_peak_rss(),
        _rcache.get_num_priority_context_switch() +
            _rcache.get_num_voluntary_context_switch(),
        _rcache.get_num_major_page_faults() + _rcache.get_num_minor_page_faults(),
        _rcache.get_user_mode_time() * 1000, _rcache.get_kernel_mode_time() * 1000,
        std::move(_freqs));
}

void
shutdown()
{}

namespace
{
template <typename... Types, size_t N = sizeof...(Types)>
void
config_perfetto_counter_tracks(type_list<Types...>, std::array<const char*, N> _labels,
                               std::array<const char*, N> _units)
{
    static_assert(sizeof...(Types) == N,
                  "Error! Number of types != number of labels/units");

    auto _config = [&](auto _t) {
        using type          = std::decay_t<decltype(_t)>;
        using track         = perfetto_counter_track<type>;
        constexpr auto _idx = tim::index_of<type, type_list<Types...>>::value;
        if(!track::exists(0))
        {
            auto addendum = [&](const char* _v) { return JOIN(" ", "CPU", _v, "(S)"); };
            track::emplace(0, addendum(_labels.at(_idx)), _units.at(_idx));
        }
    };

    (_config(Types{}), ...);
}

struct index
{
    size_t value = 0;
};

template <typename Tp, typename... Args>
void
write_perfetto_counter_track(Args... _args)
{
    using track = perfetto_counter_track<Tp>;
    TRACE_COUNTER(trait::name<Tp>::value, track::at(0, 0), _args...);
}

template <typename Tp, typename... Args>
void
write_perfetto_counter_track(index&& _idx, Args... _args)
{
    using track = perfetto_counter_track<Tp>;
    TRACE_COUNTER(trait::name<Tp>::value, track::at(_idx.value, 0), _args...);
}
}  // namespace

void
post_process()
{
    ROCPROFSYS_VERBOSE(1,
                       "Post-processing %zu cpu frequency and memory usage entries...\n",
                       data.size());

    auto& enabled_cpus = component::cpu_freq::get_enabled_cpus();

    auto _process_frequencies = [](size_t _idx, size_t _offset) {
        using freq_track = perfetto_counter_track<category::cpu_freq>;

        const auto& _thread_info = thread_info::get(0, InternalTID);
        ROCPROFSYS_CI_THROW(!_thread_info, "Missing thread info for thread 0");
        if(!_thread_info) return;

        if(!freq_track::exists(_idx))
        {
            auto addendum = [&](const char* _v) {
                return JOIN(" ", "CPU", _v, JOIN("", '[', _idx, ']'), "(S)");
            };
            freq_track::emplace(_idx, addendum("Frequency"), "MHz");
        }

        for(auto& itr : data)
        {
            uint64_t _ts   = std::get<0>(itr);
            double   _freq = static_cast<double>(std::get<8>(itr).at(_offset));
            if(!_thread_info->is_valid_time(_ts)) continue;
            write_perfetto_counter_track<category::cpu_freq>(index{ _idx }, _ts, _freq);
        }

        auto _end_ts = _thread_info->get_stop();
        write_perfetto_counter_track<category::cpu_freq>(index{ _idx }, _end_ts, 0);
    };

    auto _process_cpu_rusage = []() {
        if(get_use_perfetto())
        {
            config_perfetto_counter_tracks(
                type_list<category::process_page, category::process_virt,
                          category::process_peak, category::process_context_switch,
                          category::process_page_fault, category::process_user_mode_time,
                          category::process_kernel_mode_time>{},
                { "Memory Usage", "Virtual Memory Usage", "Peak Memory",
                  "Context Switches", "Page Faults", "User Time", "Kernel Time" },
                { "MB", "MB", "MB", "", "", "sec", "sec" });
        }

        const auto& _thread_info = thread_info::get(0, InternalTID);
        ROCPROFSYS_CI_THROW(!_thread_info, "Missing thread info for thread 0");
        if(!_thread_info) return;

        for(auto& itr : data)
        {
            uint64_t _ts = std::get<0>(itr);
            if(!_thread_info->is_valid_time(_ts)) continue;

            double   _page = std::get<1>(itr) / units::megabyte;
            double   _virt = std::get<2>(itr) / units::megabyte;
            double   _peak = std::get<3>(itr) / units::megabyte;
            uint64_t _cntx = std::get<4>(itr);
            uint64_t _flts = std::get<5>(itr);
            double   _user = std::get<6>(itr) / units::sec;
            double   _kern = std::get<7>(itr) / units::sec;
            if(get_use_perfetto())
            {
                write_perfetto_counter_track<category::process_page>(_ts, _page);
                write_perfetto_counter_track<category::process_virt>(_ts, _virt);
                write_perfetto_counter_track<category::process_peak>(_ts, _peak);
                write_perfetto_counter_track<category::process_context_switch>(_ts,
                                                                               _cntx);
                write_perfetto_counter_track<category::process_page_fault>(_ts, _flts);
                write_perfetto_counter_track<category::process_user_mode_time>(_ts,
                                                                               _user);
                write_perfetto_counter_track<category::process_kernel_mode_time>(_ts,
                                                                                 _kern);
            }
            if(get_use_rocpd())
            {
                const auto& freq_data = std::get<8>(itr);
                rocpd_process_cpu_usage_events(0, _ts, freq_data, _page, _virt, _peak,
                                               _cntx, _flts, _user, _kern);
            }
        }

        if(get_use_perfetto())
        {
            auto _end_ts = _thread_info->get_stop();
            write_perfetto_counter_track<category::process_page>(_end_ts, 0.0);
            write_perfetto_counter_track<category::process_virt>(_end_ts, 0.0);
            write_perfetto_counter_track<category::process_peak>(_end_ts, 0.0);
            write_perfetto_counter_track<category::process_context_switch>(_end_ts, 0);
            write_perfetto_counter_track<category::process_page_fault>(_end_ts, 0);
            write_perfetto_counter_track<category::process_user_mode_time>(_end_ts, 0.0);
            write_perfetto_counter_track<category::process_kernel_mode_time>(_end_ts,
                                                                             0.0);
        }
    };

    _process_cpu_rusage();

    if(get_use_perfetto())
    {
        for(auto itr = enabled_cpus.begin(); itr != enabled_cpus.end(); ++itr)
        {
            auto _idx    = *itr;
            auto _offset = std::distance(enabled_cpus.begin(), itr);
            _process_frequencies(_idx, _offset);
        }
    }
    enabled_cpus.clear();
}

}  // namespace cpu_freq
}  // namespace rocprofsys
