// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * @brief ROCProfiler enumeration of OMPT (OpenMP tools) tracing operations
 * NOTE: These are callbacks into the ROCProfiler SDK from the vendor-provided OMPT implementation
 */
typedef enum rocprofiler_ompt_operation_t  // NOLINT(performance-enum-size)
{
    ROCPROFILER_OMPT_ID_NONE         = -1,
    ROCPROFILER_OMPT_ID_thread_begin = 0,
    ROCPROFILER_OMPT_ID_thread_end,
    ROCPROFILER_OMPT_ID_parallel_begin,
    ROCPROFILER_OMPT_ID_parallel_end,
    ROCPROFILER_OMPT_ID_task_create,
    ROCPROFILER_OMPT_ID_task_schedule,
    ROCPROFILER_OMPT_ID_implicit_task,
    ROCPROFILER_OMPT_ID_device_initialize,
    ROCPROFILER_OMPT_ID_device_finalize,
    ROCPROFILER_OMPT_ID_device_load,
    // ROCPROFILER_OMPT_ID_device_unload,
    ROCPROFILER_OMPT_ID_sync_region_wait,
    ROCPROFILER_OMPT_ID_mutex_released,
    ROCPROFILER_OMPT_ID_dependences,
    ROCPROFILER_OMPT_ID_task_dependence,
    ROCPROFILER_OMPT_ID_work,
    ROCPROFILER_OMPT_ID_masked,
    ROCPROFILER_OMPT_ID_sync_region,
    ROCPROFILER_OMPT_ID_lock_init,
    ROCPROFILER_OMPT_ID_lock_destroy,
    ROCPROFILER_OMPT_ID_mutex_acquire,
    ROCPROFILER_OMPT_ID_mutex_acquired,
    ROCPROFILER_OMPT_ID_nest_lock,
    ROCPROFILER_OMPT_ID_flush,
    ROCPROFILER_OMPT_ID_cancel,
    ROCPROFILER_OMPT_ID_reduction,
    ROCPROFILER_OMPT_ID_dispatch,
    ROCPROFILER_OMPT_ID_target_emi,
    ROCPROFILER_OMPT_ID_target_data_op_emi,
    ROCPROFILER_OMPT_ID_target_submit_emi,
    // ROCPROFILER_OMPT_ID_target_map_emi,
    ROCPROFILER_OMPT_ID_error,
    ROCPROFILER_OMPT_ID_callback_functions,  // fake to return struct of ompt callback function
                                             // pointers
    ROCPROFILER_OMPT_ID_LAST
} rocprofiler_ompt_operation_t;
