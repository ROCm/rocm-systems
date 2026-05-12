// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

/**
 * @brief ROCProfiler enumeration of rocSHMEM API tracing operations
 */
typedef enum rocprofiler_rocshmem_api_id_t  // NOLINT(performance-enum-size)
{
    ROCPROFILER_ROCSHMEM_API_ID_NONE = -1,

    ROCPROFILER_ROCSHMEM_API_ID_barrier_all_on_stream = 0,
    ROCPROFILER_ROCSHMEM_API_ID_quiet_on_stream,
    ROCPROFILER_ROCSHMEM_API_ID_sync_all_on_stream,
    ROCPROFILER_ROCSHMEM_API_ID_alltoallmem_on_stream,
    ROCPROFILER_ROCSHMEM_API_ID_broadcastmem_on_stream,
    ROCPROFILER_ROCSHMEM_API_ID_getmem_on_stream,
    ROCPROFILER_ROCSHMEM_API_ID_putmem_on_stream,
    ROCPROFILER_ROCSHMEM_API_ID_putmem_signal_on_stream,
    ROCPROFILER_ROCSHMEM_API_ID_signal_wait_until_on_stream,
    ROCPROFILER_ROCSHMEM_API_ID_LAST,
} rocprofiler_rocshmem_api_id_t;
