// Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#if defined(PERFETTO_CATEGORIES)
#    error "PERFETTO_CATEGORIES is already defined. Please include \"" __FILE__ "\" before including any timemory files"
#endif

#if !defined(TIMEMORY_USE_PERFETTO)
#    include <perfetto.h>
#    define PERFETTO_CATEGORIES                                                          \
        perfetto::Category("host").SetDescription("Host-side function tracing"),         \
            perfetto::Category("device").SetDescription("Device-side function tracing"), \
            perfetto::Category("host-critical-trace")                                    \
                .SetDescription("Host-side critical traces"),                            \
            perfetto::Category("device-critical-trace")                                  \
                .SetDescription("Device-side critical traces")
#else
#    define PERFETTO_CATEGORIES                                                          \
        perfetto::Category("host").SetDescription("Host-side function tracing"),         \
            perfetto::Category("device").SetDescription("Device-side function tracing"), \
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
