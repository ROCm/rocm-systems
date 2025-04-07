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

// without AMD_INTERNAL_BUILD defined, including the hsa/hsa.h looks for headers in inc/ folder
// so we always want it defined but we set ROCPROFILER_DEFINED_AMD_INTERNAL_BUILD to 1 to tell
// us that after this include, we should undefine it
#ifndef AMD_INTERNAL_BUILD
#    define AMD_INTERNAL_BUILD
#    ifndef ROCPROFILER_DEFINED_AMD_INTERNAL_BUILD
#        define ROCPROFILER_DEFINED_AMD_INTERNAL_BUILD 1
#    endif
#endif

#include <hsa/hsa.h>

#include <rocprofiler-sdk/hsa/api_trace_version.h>

#include <rocprofiler-sdk/hsa/api_args.h>
#include <rocprofiler-sdk/hsa/api_id.h>
#include <rocprofiler-sdk/hsa/scratch_memory_args.h>
#include <rocprofiler-sdk/hsa/scratch_memory_id.h>
#include <rocprofiler-sdk/hsa/table_id.h>

#if defined(ROCPROFILER_DEFINED_AMD_INTERNAL_BUILD) && ROCPROFILER_DEFINED_AMD_INTERNAL_BUILD > 0
#    undef AMD_INTERNAL_BUILD
#endif
