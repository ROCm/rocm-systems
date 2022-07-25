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

#pragma once

#include "library/defines.hpp"

#if defined(OMNITRACE_PERFETTO_CATEGORIES)
#    error "OMNITRACE_PERFETTO_CATEGORIES is already defined. Please include \"" __FILE__ "\" before including any timemory files"
#endif

#define OMNITRACE_PERFETTO_CATEGORIES                                                    \
    perfetto::Category("host").SetDescription("Host-side function tracing"),             \
        perfetto::Category("user").SetDescription("User-defined regions"),               \
        perfetto::Category("sampling").SetDescription("Host-side function sampling"),    \
        perfetto::Category("device_hip")                                                 \
            .SetDescription("Device-side functions submitted via HSA API"),              \
        perfetto::Category("device_hsa")                                                 \
            .SetDescription("Device-side functions submitted via HIP API"),              \
        perfetto::Category("rocm_hip").SetDescription("Host-side HIP functions"),        \
        perfetto::Category("rocm_hsa").SetDescription("Host-side HSA functions"),        \
        perfetto::Category("rocm_roctx").SetDescription("Host-side ROCTX labels"),       \
        perfetto::Category("device_busy")                                                \
            .SetDescription("Busy percentage of a GPU device"),                          \
        perfetto::Category("device_temp")                                                \
            .SetDescription("Temperature of GPU device in degC"),                        \
        perfetto::Category("device_power")                                               \
            .SetDescription("Power consumption of GPU device in watts"),                 \
        perfetto::Category("device_memory_usage")                                        \
            .SetDescription("Memory usage of GPU device in MB"),                         \
        perfetto::Category("thread_peak_memory")                                         \
            .SetDescription(                                                             \
                "Peak memory usage on thread in MB (derived from sampling)"),            \
        perfetto::Category("thread_context_switch")                                      \
            .SetDescription("Context switches on thread (derived from sampling)"),       \
        perfetto::Category("thread_page_fault")                                          \
            .SetDescription("Memory page faults on thread (derived from sampling)"),     \
        perfetto::Category("hardware_counter")                                           \
            .SetDescription("Hardware counter value on thread (derived from sampling)"), \
        perfetto::Category("cpu_freq")                                                   \
            .SetDescription("CPU frequency in MHz (collected in background thread)"),    \
        perfetto::Category("process_page_fault")                                         \
            .SetDescription(                                                             \
                "Memory page faults in process (collected in background thread)"),       \
        perfetto::Category("process_memory_hwm")                                         \
            .SetDescription("Memory High-Water Mark i.e. peak memory usage (collected "  \
                            "in background thread)"),                                    \
        perfetto::Category("process_virtual_memory")                                     \
            .SetDescription("Virtual memory usage in process in MB (collected in "       \
                            "background thread)"),                                       \
        perfetto::Category("process_context_switch")                                     \
            .SetDescription(                                                             \
                "Context switches in process (collected in background thread)"),         \
        perfetto::Category("process_page_fault")                                         \
            .SetDescription(                                                             \
                "Memory page faults in process (collected in background thread)"),       \
        perfetto::Category("process_user_cpu_time")                                      \
            .SetDescription("CPU time of functions executing in user-space in process "  \
                            "in seconds (collected in background thread)"),              \
        perfetto::Category("process_kernel_cpu_time")                                    \
            .SetDescription("CPU time of functions executing in kernel-space in "        \
                            "process in seconds (collected in background thread)"),      \
        perfetto::Category("pthread").SetDescription("Pthread functions"),               \
        perfetto::Category("kokkos").SetDescription("Kokkos regions"),                   \
        perfetto::Category("mpi").SetDescription("MPI regions"),                         \
        perfetto::Category("ompt").SetDescription("OpenMP Tools regions"),               \
        perfetto::Category("rccl").SetDescription(                                       \
            "ROCm Communication Collectives Library (RCCL) regions"),                    \
        perfetto::Category("critical-trace").SetDescription("Combined critical traces"), \
        perfetto::Category("host-critical-trace")                                        \
            .SetDescription("Host-side critical traces"),                                \
        perfetto::Category("device-critical-trace")                                      \
            .SetDescription("Device-side critical traces"),                              \
        perfetto::Category("timemory").SetDescription("Events from the timemory API")

#if defined(TIMEMORY_USE_PERFETTO)
#    define TIMEMORY_PERFETTO_CATEGORIES OMNITRACE_PERFETTO_CATEGORIES
#    include <timemory/components/perfetto/backends.hpp>
#else
#    include <perfetto.h>
PERFETTO_DEFINE_CATEGORIES(OMNITRACE_PERFETTO_CATEGORIES);
#endif

#include "library/debug.hpp"

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace omnitrace
{
#if defined(CUSTOM_DATA_SOURCE)
class CustomDataSource : public perfetto::DataSource<CustomDataSource>
{
public:
    void OnSetup(const SetupArgs&) override
    {
        // Use this callback to apply any custom configuration to your data source
        // based on the TraceConfig in SetupArgs.
        OMNITRACE_PRINT_F("[CustomDataSource] setup\n");
    }

    void OnStart(const StartArgs&) override
    {
        // This notification can be used to initialize the GPU driver, enable
        // counters, etc. StartArgs will contains the DataSourceDescriptor,
        // which can be extended.
        OMNITRACE_PRINT_F("[CustomDataSource] start\n");
    }

    void OnStop(const StopArgs&) override
    {
        // Undo any initialization done in OnStart.
        OMNITRACE_PRINT_F("[CustomDataSource] stop\n");
    }

    // Data sources can also have per-instance state.
    int my_custom_state = 0;
};

PERFETTO_DECLARE_DATA_SOURCE_STATIC_MEMBERS(CustomDataSource);
#endif

template <typename Tp>
struct perfetto_counter_track
{
    using track_map_t = std::map<uint32_t, std::vector<perfetto::CounterTrack>>;
    using name_map_t  = std::map<uint32_t, std::vector<std::unique_ptr<std::string>>>;
    using data_t      = std::pair<name_map_t, track_map_t>;

    static auto init() { (void) get_data(); }

    static auto exists(size_t _idx, int64_t _n = -1)
    {
        bool _v = get_data().second.count(_idx) != 0;
        if(_n < 0 || !_v) return _v;
        return static_cast<size_t>(_n) < get_data().second.at(_idx).size();
    }

    static size_t size(size_t _idx)
    {
        bool _v = get_data().second.count(_idx) != 0;
        if(!_v) return 0;
        return get_data().second.at(_idx).size();
    }

    static auto emplace(size_t _idx, const std::string& _v, const char* _units = nullptr,
                        const char* _category = nullptr, int64_t _mult = 1,
                        bool _incr = false)
    {
        auto& _name_data  = get_data().first[_idx];
        auto& _track_data = get_data().second[_idx];
        std::vector<std::tuple<std::string, const char*, bool>> _missing = {};
        if(config::get_is_continuous_integration())
        {
            for(const auto& itr : _name_data)
            {
                _missing.emplace_back(std::make_tuple(*itr, itr->c_str(), false));
            }
        }
        auto        _index = _track_data.size();
        auto&       _name  = _name_data.emplace_back(std::make_unique<std::string>(_v));
        const char* _unit_name = (_units && strlen(_units) > 0) ? _units : nullptr;
        _track_data.emplace_back(perfetto::CounterTrack{ _name->c_str() }
                                     .set_unit_name(_unit_name)
                                     .set_category(_category)
                                     .set_unit_multiplier(_mult)
                                     .set_is_incremental(_incr));
        if(config::get_is_continuous_integration())
        {
            for(auto& itr : _missing)
            {
                const char* citr = std::get<1>(itr);
                for(const auto& ditr : _name_data)
                {
                    if(citr == ditr->c_str() && strcmp(citr, ditr->c_str()) == 0)
                    {
                        std::get<2>(itr) = true;
                        break;
                    }
                }
                if(!std::get<2>(itr))
                {
                    std::set<void*> _prev = {};
                    std::set<void*> _curr = {};
                    for(const auto& eitr : _missing)
                        _prev.emplace(
                            static_cast<void*>(const_cast<char*>(std::get<1>(eitr))));
                    for(const auto& eitr : _name_data)
                        _curr.emplace(
                            static_cast<void*>(const_cast<char*>(eitr->c_str())));
                    std::stringstream _pss{};
                    for(auto&& eitr : _prev)
                        _pss << " " << std::hex << std::setw(12) << std::left << eitr;
                    std::stringstream _css{};
                    for(auto&& eitr : _curr)
                        _css << " " << std::hex << std::setw(12) << std::left << eitr;
                    OMNITRACE_THROW("perfetto_counter_track emplace method for '%s' (%p) "
                                    "invalidated C-string '%s' (%p).\n%8s: %s\n%8s: %s\n",
                                    _v.c_str(), _name->c_str(), std::get<0>(itr).c_str(),
                                    std::get<0>(itr).c_str(), "previous",
                                    _pss.str().c_str(), "current", _css.str().c_str());
                }
            }
        }
        return _index;
    }

    static auto& at(size_t _idx, size_t _n) { return get_data().second.at(_idx).at(_n); }

private:
    static data_t& get_data()
    {
        static auto _v = data_t{};
        return _v;
    }
};
}  // namespace omnitrace
