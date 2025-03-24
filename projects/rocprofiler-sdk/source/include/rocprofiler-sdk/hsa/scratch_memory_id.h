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

#include <hsa/hsa_amd_tool.h>

/**
 * @brief Allocation flags for @see rocprofiler_buffer_tracing_scratch_memory_record_t
 */
// NOLINTNEXTLINE(performance-enum-size)
typedef enum rocprofiler_scratch_alloc_flag_t
{
    ROCPROFILER_SCRATCH_ALLOC_FLAG_NONE = 0,
    ROCPROFILER_SCRATCH_ALLOC_FLAG_USE_ONCE =
        HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_USE_ONCE,  ///< This scratch allocation is only valid for 1
                                                    ///< dispatch.
    ROCPROFILER_SCRATCH_ALLOC_FLAG_ALT =
        HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_ALT,  ///< Used alternate scratch instead of main scratch
} rocprofiler_scratch_alloc_flag_t;
