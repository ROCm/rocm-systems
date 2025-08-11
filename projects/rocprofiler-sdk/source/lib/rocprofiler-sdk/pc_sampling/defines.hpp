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

#include <rocprofiler-sdk/hsa.h>

#if defined(HSA_PC_SAMPLING_API_TABLE_MAJOR_VERSION) &&                                            \
    HSA_PC_SAMPLING_API_TABLE_MAJOR_VERSION > 0x0
#    define ROCPROFILER_SDK_HSA_PC_SAMPLING 1
#else
#    define ROCPROFILER_SDK_HSA_PC_SAMPLING 0
#endif

// redundant check based on whether the pc sampling API header was found
#if defined __has_include
#    if __has_include(<hsa/hsa_ven_amd_pc_sampling.h>)
#        if ROCPROFILER_SDK_HSA_PC_SAMPLING == 0
#            error                                                                                 \
                "rocprofiler-sdk disabled the HSA PC sampling table even though the hsa_ven_amd_pc_sampling.h was found"
#        endif
#    else
#        if ROCPROFILER_SDK_HSA_PC_SAMPLING == 1
#            error                                                                                 \
                "rocprofiler-sdk enabled the HSA PC sampling table even though the hsa_ven_amd_pc_sampling.h was not found"
#        endif
#    endif
#endif
