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

#include <rocprofiler-sdk/ext_version.h>

/**
 * @brief ROCProfiler enumeration of HSA Image Extended API tracing operations
 */
typedef enum rocprofiler_hsa_finalize_ext_api_id_t  // NOLINT(performance-enum-size)
{
    ROCPROFILER_HSA_FINALIZE_EXT_API_ID_NONE                   = -1,
    ROCPROFILER_HSA_FINALIZE_EXT_API_ID_hsa_ext_program_create = 0,
    ROCPROFILER_HSA_FINALIZE_EXT_API_ID_hsa_ext_program_destroy,
    ROCPROFILER_HSA_FINALIZE_EXT_API_ID_hsa_ext_program_add_module,
    ROCPROFILER_HSA_FINALIZE_EXT_API_ID_hsa_ext_program_iterate_modules,
    ROCPROFILER_HSA_FINALIZE_EXT_API_ID_hsa_ext_program_get_info,
    ROCPROFILER_HSA_FINALIZE_EXT_API_ID_hsa_ext_program_finalize,
    ROCPROFILER_HSA_FINALIZE_EXT_API_ID_LAST,
} rocprofiler_hsa_finalize_ext_api_id_t;
