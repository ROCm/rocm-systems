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
            perfetto::Category("rocm_smi").SetDescription("Device-level metrics"),       \
            perfetto::Category("sampling")                                               \
                .SetDescription("Metrics derived from sampling"),                        \
            perfetto::Category("host-critical-trace")                                    \
                .SetDescription("Host-side critical traces"),                            \
            perfetto::Category("device-critical-trace")                                  \
                .SetDescription("Device-side critical traces")
#else
#    define PERFETTO_CATEGORIES                                                          \
        perfetto::Category("host").SetDescription("Host-side function tracing"),         \
            perfetto::Category("device").SetDescription("Device-side function tracing"), \
            perfetto::Category("rocm_smi").SetDescription("Device-level metrics"),       \
            perfetto::Category("sampling")                                               \
                .SetDescription("Metrics derived from sampling"),                        \
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
        PRINT_HERE("%s", "setup");
    }

    void OnStart(const StartArgs&) override
    {
        // This notification can be used to initialize the GPU driver, enable
        // counters, etc. StartArgs will contains the DataSourceDescriptor,
        // which can be extended.
        PRINT_HERE("%s", "start");
    }

    void OnStop(const StopArgs&) override
    {
        // Undo any initialization done in OnStart.
        PRINT_HERE("%s", "stop");
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
    using name_map_t  = std::map<uint32_t, std::vector<std::string>>;
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

    static auto emplace(size_t _idx, const std::string& _v, const char* _units)
    {
        get_data().first[_idx].emplace_back(_v);
        get_data().second[_idx].emplace_back(get_data().first[_idx].back().c_str(),
                                             _units);
    }

    static auto& at(size_t _idx, size_t _n) { return get_data().second.at(_idx).at(_n); }

private:
    static data_t& get_data()
    {
        static auto* _v = new data_t{};
        return *_v;
    }
};
}  // namespace omnitrace
