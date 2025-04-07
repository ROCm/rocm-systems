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

#if defined(__cplusplus)
#    include <hsa/hsa_api_trace.h>  // safe to include from C++
#elif defined(__has_include)
#    if __has_include(<hsa/hsa_api_trace_version.h>)
#        include <hsa/hsa_api_trace_version.h>
#    endif
#endif

#ifndef HSA_API_TABLE_MAJOR_VERSION
#    ifdef ROCPROFILER_HSA_API_TABLE_MAJOR_VERSION
#        define HSA_API_TABLE_MAJOR_VERSION ROCPROFILER_HSA_API_TABLE_MAJOR_VERSION
#    endif
#endif

#ifndef HSA_CORE_API_TABLE_MAJOR_VERSION
#    ifdef ROCPROFILER_HSA_CORE_API_TABLE_MAJOR_VERSION
#        define HSA_CORE_API_TABLE_MAJOR_VERSION ROCPROFILER_HSA_CORE_API_TABLE_MAJOR_VERSION
#    endif
#endif

#ifndef HSA_AMD_EXT_API_TABLE_MAJOR_VERSION
#    ifdef ROCPROFILER_HSA_AMD_EXT_API_TABLE_MAJOR_VERSION
#        define HSA_AMD_EXT_API_TABLE_MAJOR_VERSION ROCPROFILER_HSA_AMD_EXT_API_TABLE_MAJOR_VERSION
#    endif
#endif

#ifndef HSA_FINALIZER_API_TABLE_MAJOR_VERSION
#    ifdef ROCPROFILER_HSA_FINALIZER_API_TABLE_MAJOR_VERSION
#        define HSA_FINALIZER_API_TABLE_MAJOR_VERSION                                              \
            ROCPROFILER_HSA_FINALIZER_API_TABLE_MAJOR_VERSION
#    endif
#endif

#ifndef HSA_IMAGE_API_TABLE_MAJOR_VERSION
#    ifdef ROCPROFILER_HSA_IMAGE_API_TABLE_MAJOR_VERSION
#        define HSA_IMAGE_API_TABLE_MAJOR_VERSION ROCPROFILER_HSA_IMAGE_API_TABLE_MAJOR_VERSION
#    endif
#endif

#ifndef HSA_AQLPROFILE_API_TABLE_MAJOR_VERSION
#    ifdef ROCPROFILER_HSA_AQLPROFILE_API_TABLE_MAJOR_VERSION
#        define HSA_AQLPROFILE_API_TABLE_MAJOR_VERSION                                             \
            ROCPROFILER_HSA_AQLPROFILE_API_TABLE_MAJOR_VERSION
#    endif
#endif

#ifndef HSA_TOOLS_API_TABLE_MAJOR_VERSION
#    ifdef ROCPROFILER_HSA_TOOLS_API_TABLE_MAJOR_VERSION
#        define HSA_TOOLS_API_TABLE_MAJOR_VERSION ROCPROFILER_HSA_TOOLS_API_TABLE_MAJOR_VERSION
#    endif
#endif

#ifndef HSA_API_TABLE_STEP_VERSION
#    ifdef ROCPROFILER_HSA_API_TABLE_STEP_VERSION
#        define HSA_API_TABLE_STEP_VERSION ROCPROFILER_HSA_API_TABLE_STEP_VERSION
#    endif
#endif

#ifndef HSA_CORE_API_TABLE_STEP_VERSION
#    ifdef ROCPROFILER_HSA_CORE_API_TABLE_STEP_VERSION
#        define HSA_CORE_API_TABLE_STEP_VERSION ROCPROFILER_HSA_CORE_API_TABLE_STEP_VERSION
#    endif
#endif

#ifndef HSA_AMD_EXT_API_TABLE_STEP_VERSION
#    ifdef ROCPROFILER_HSA_AMD_EXT_API_TABLE_STEP_VERSION
#        define HSA_AMD_EXT_API_TABLE_STEP_VERSION ROCPROFILER_HSA_AMD_EXT_API_TABLE_STEP_VERSION
#    endif
#endif

#ifndef HSA_FINALIZER_API_TABLE_STEP_VERSION
#    ifdef ROCPROFILER_HSA_FINALIZER_API_TABLE_STEP_VERSION
#        define HSA_FINALIZER_API_TABLE_STEP_VERSION                                               \
            ROCPROFILER_HSA_FINALIZER_API_TABLE_STEP_VERSION
#    endif
#endif

#ifndef HSA_IMAGE_API_TABLE_STEP_VERSION
#    ifdef ROCPROFILER_HSA_IMAGE_API_TABLE_STEP_VERSION
#        define HSA_IMAGE_API_TABLE_STEP_VERSION ROCPROFILER_HSA_IMAGE_API_TABLE_STEP_VERSION
#    endif
#endif

#ifndef HSA_AQLPROFILE_API_TABLE_STEP_VERSION
#    ifdef ROCPROFILER_HSA_AQLPROFILE_API_TABLE_STEP_VERSION
#        define HSA_AQLPROFILE_API_TABLE_STEP_VERSION                                              \
            ROCPROFILER_HSA_AQLPROFILE_API_TABLE_STEP_VERSION
#    endif
#endif

#ifndef HSA_TOOLS_API_TABLE_STEP_VERSION
#    ifdef ROCPROFILER_HSA_TOOLS_API_TABLE_STEP_VERSION
#        define HSA_TOOLS_API_TABLE_STEP_VERSION ROCPROFILER_HSA_TOOLS_API_TABLE_STEP_VERSION
#    endif
#endif
