// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// The public headers must compile as C (C99 and later): this translation unit is built
// as C into the unit test executable.

#include <rocprofiler-sdk-collector/collector.h>
#include <rocprofiler-sdk-collector/records.h>

#include <stddef.h>

int
rocprofiler_collector_c_api_compiles(void);

int
rocprofiler_collector_c_api_compiles(void)
{
    rocprofiler_collector_config_t          config      = {0};
    rocprofiler_collector_t                 collector   = {0};
    const char*                             providers[] = {"rocm_hip", NULL};
    const char*                             argv[]      = {"true", NULL};
    const char*                             path        = NULL;
    int                                     pid         = 0;
    int                                     exit_status = 0;
    uint32_t                                major       = 0;
    rocprofiler_collector_read_config_t     read_config = {0};
    rocprofiler_collector_read_statistics_t statistics  = {0};

    config.size       = sizeof(config);
    config.output_dir = "/proc/rocprofiler-collector-c-api"; /* not creatable, even as root */
    config.providers  = providers;

    (void) rocprofiler_collector_get_version(&major, NULL, NULL);
    (void) rocprofiler_collector_create(&config, &collector);
    (void) rocprofiler_collector_start(collector);
    (void) rocprofiler_collector_launch(collector, argv, &pid);
    (void) rocprofiler_collector_wait(collector, pid, &exit_status);
    (void) rocprofiler_collector_stop(collector);
    (void) rocprofiler_collector_get_trace_path(collector, &path);
    read_config.size = sizeof(read_config);
    (void) rocprofiler_collector_read(collector, &read_config, &statistics);
    (void) rocprofiler_collector_destroy(collector);
    return rocprofiler_collector_get_status_name(ROCPROFILER_COLLECTOR_STATUS_SUCCESS) != NULL;
}
