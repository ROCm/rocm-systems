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

#include <rocprofiler-sdk/defines.h>

#if !defined(ROCPROFILER_SDK_USE_SYSTEM_ROCSHMEM)
#    if defined __has_include
#        if __has_include(<rocshmem/api_trace.h>)
#            define ROCPROFILER_SDK_USE_SYSTEM_ROCSHMEM 1
#        else
#            define ROCPROFILER_SDK_USE_SYSTEM_ROCSHMEM 0
#        endif
#    else
#        define ROCPROFILER_SDK_USE_SYSTEM_ROCSHMEM 0
#    endif
#endif

#if ROCPROFILER_SDK_USE_SYSTEM_ROCSHMEM > 0
#    include <rocshmem/api_trace.h>
#else
#    include <rocprofiler-sdk/rocshmem/details/api_trace.h>
#endif

#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

// Empty struct has a size of 0 in C but size of 1 in C++.
// This struct is added to the union members which represent
// functions with no arguments or a void return value to ensure
// ABI compatibility between C and C++ tools.
typedef struct rocprofiler_rocshmem_api_no_args
{
    char empty;
} rocprofiler_rocshmem_api_no_args;

// All currently traced rocSHMEM host stream APIs return void; the union below
// exists so the tracing record layout remains uniform with other services.
typedef union rocprofiler_rocshmem_api_retval_t
{
    rocprofiler_rocshmem_api_no_args no_retval;
} rocprofiler_rocshmem_api_retval_t;

typedef union rocprofiler_rocshmem_api_args_t
{
    struct
    {
        hipStream_t stream;
    } barrier_all_on_stream;
    struct
    {
        hipStream_t stream;
    } quiet_on_stream;
    struct
    {
        hipStream_t stream;
    } sync_all_on_stream;
    struct
    {
        rocshmem_team_t team;
        void*           dest;
        const void*     source;
        size_t          size;
        hipStream_t     stream;
    } alltoallmem_on_stream;
    struct
    {
        rocshmem_team_t team;
        void*           dest;
        const void*     source;
        size_t          nelems;
        int             pe_root;
        hipStream_t     stream;
    } broadcastmem_on_stream;
    struct
    {
        void*       dest;
        const void* source;
        size_t      nelems;
        int         pe;
        hipStream_t stream;
    } getmem_on_stream;
    struct
    {
        void*       dest;
        const void* source;
        size_t      nelems;
        int         pe;
        hipStream_t stream;
    } putmem_on_stream;
    struct
    {
        void*       dest;
        const void* source;
        size_t      nelems;
        uint64_t*   sig_addr;
        uint64_t    signal;
        int         sig_op;
        int         pe;
        hipStream_t stream;
    } putmem_signal_on_stream;
    struct
    {
        uint64_t*   sig_addr;
        int         cmp;
        uint64_t    cmp_value;
        hipStream_t stream;
    } signal_wait_until_on_stream;
} rocprofiler_rocshmem_api_args_t;

ROCPROFILER_EXTERN_C_FINI
