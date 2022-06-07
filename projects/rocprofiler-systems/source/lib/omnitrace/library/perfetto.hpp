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

#if defined(PERFETTO_CATEGORIES)
#    error "PERFETTO_CATEGORIES is already defined. Please include \"" __FILE__ "\" before including any timemory files"
#endif

#if !defined(TIMEMORY_USE_PERFETTO)
#    include <perfetto.h>
#    define PERFETTO_CATEGORIES                                                          \
        perfetto::Category("host").SetDescription("Host-side function tracing"),         \
            perfetto::Category("device").SetDescription("Device-side function tracing"), \
            perfetto::Category("user").SetDescription("User-defined regions"),           \
            perfetto::Category("rocm_smi").SetDescription("Device-level metrics"),       \
            perfetto::Category("sampling")                                               \
                .SetDescription("Metrics derived from sampling"),                        \
            perfetto::Category("critical-trace")                                         \
                .SetDescription("Combined critical traces"),                             \
            perfetto::Category("host-critical-trace")                                    \
                .SetDescription("Host-side critical traces"),                            \
            perfetto::Category("device-critical-trace")                                  \
                .SetDescription("Device-side critical traces")
#else
#    define PERFETTO_CATEGORIES                                                          \
        perfetto::Category("host").SetDescription("Host-side function tracing"),         \
            perfetto::Category("device").SetDescription("Device-side function tracing"), \
            perfetto::Category("user").SetDescription("User-defined regions"),           \
            perfetto::Category("rocm_smi").SetDescription("Device-level metrics"),       \
            perfetto::Category("sampling")                                               \
                .SetDescription("Metrics derived from sampling"),                        \
            perfetto::Category("critical-trace")                                         \
                .SetDescription("Combined critical traces"),                             \
            perfetto::Category("host-critical-trace")                                    \
                .SetDescription("Host-side critical traces"),                            \
            perfetto::Category("device-critical-trace")                                  \
                .SetDescription("Device-side critical traces"),                          \
            perfetto::Category("timemory")                                               \
                .SetDescription("Events from the timemory API")
#    define TIMEMORY_PERFETTO_CATEGORIES PERFETTO_CATEGORIES
#endif

#if !defined(TIMEMORY_USE_PERFETTO)
PERFETTO_DEFINE_CATEGORIES(PERFETTO_CATEGORIES);
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
        std::vector<std::tuple<std::string, const char*, bool>> _missing = {};
        if(config::get_is_continuous_integration())
        {
            for(const auto& itr : get_data().first[_idx])
            {
                _missing.emplace_back(std::make_tuple(*itr, itr->c_str(), false));
            }
        }
        auto& _name =
            get_data().first[_idx].emplace_back(std::make_unique<std::string>(_v));
        const char* _unit_name = (_units && strlen(_units) > 0) ? _units : nullptr;
        get_data().second[_idx].emplace_back(perfetto::CounterTrack{ _name->c_str() }
                                                 .set_unit_name(_unit_name)
                                                 .set_category(_category)
                                                 .set_unit_multiplier(_mult)
                                                 .set_is_incremental(_incr));
        if(config::get_is_continuous_integration())
        {
            for(auto& itr : _missing)
            {
                for(const auto& ditr : get_data().first.at(_idx))
                {
                    if(*ditr == _v) continue;
                    const char* citr = std::get<1>(itr);
                    if(citr == ditr->c_str() && strcmp(citr, ditr->c_str()) == 0)
                    {
                        std::get<2>(itr) = true;
                        break;
                    }
                }
                if(!std::get<2>(itr))
                {
                    OMNITRACE_THROW("perfetto_counter_track emplace method for '%s' "
                                    "invalidated C-string '%s'\n",
                                    _v.c_str(), std::get<0>(itr).c_str());
                }
            }
        }
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
