// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/openmp/omp-tools.h>
#include <rocprofiler-sdk/registration.h>

/**
 * @defgroup OMPT_REGISTRATION Tool registration for OpenMP Tools
 *
 * Functions for enabling OpenMP support in tools which provide their own ompt_start_tool symbol but
 * want to defer to rocprofiler-sdk for OMPT.
 *
 * @{
 */

ROCPROFILER_EXTERN_C_INIT

void
rocprofiler_ompt_finalize_tool() ROCPROFILER_API;

rocprofiler_status_t
rocprofiler_ompt_is_initialized(int* status) ROCPROFILER_API ROCPROFILER_NONNULL(1);

rocprofiler_status_t
rocprofiler_ompt_is_finalized(int* status) ROCPROFILER_API ROCPROFILER_NONNULL(1);

ompt_start_tool_result_t*
rocprofiler_ompt_start_tool(unsigned int omp_version, const char* runtime_version) ROCPROFILER_API;

ROCPROFILER_EXTERN_C_FINI

/** @} */
