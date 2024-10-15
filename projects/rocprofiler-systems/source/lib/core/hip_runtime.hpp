// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "core/defines.hpp"

#if defined(ROCPROFSYS_USE_HIP) && ROCPROFSYS_USE_HIP > 0

#    if defined(HIP_INCLUDE_HIP_HIP_RUNTIME_H) ||                                        \
        defined(HIP_INCLUDE_HIP_HIP_RUNTIME_API_H)
#        error                                                                           \
            "include core/hip_runtime.hpp before <hip/hip_runtime.h> or <hip/hip_runtime_api.h>"
#    endif

#    define HIP_PROF_HIP_API_STRING 1

// following must be included before <roctracer_hip.h> for ROCm 6.0+
#    if ROCPROFSYS_HIP_VERSION >= 60000
#        if defined(USE_PROF_API)
#            undef USE_PROF_API
#        endif
#        include <hip/hip_runtime.h>
#        include <hip/hip_runtime_api.h>
// must be included after hip_runtime_api.h
#        include <hip/hip_deprecated.h>
// must be included after hip_runtime_api.h
#        include <hip_ostream_ops.h>
// must be included after hip_runtime_api.h
#        include <hip/amd_detail/hip_prof_str.h>
#    else
#        include <hip/hip_runtime.h>
#        include <hip/hip_runtime_api.h>
#    endif

#    include <hip/hip_version.h>
#endif
