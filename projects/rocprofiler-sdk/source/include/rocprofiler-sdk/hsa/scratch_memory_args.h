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

#include <rocprofiler-sdk/defines.h>

#include <stddef.h>
#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

// Empty struct has a size of 0 in C but size of 1 in C++.
// This struct is added to the union members which represent
// functions with no arguments to ensure ABI compatibility
typedef struct rocprofiler_scratch_memory_no_args
{
    char empty;
} rocprofiler_scratch_memory_no_args;

typedef union rocprofiler_scratch_memory_args_t
{
    struct
    {
        uint64_t dispatch_id;
    } alloc_start;
    struct
    {
        uint64_t dispatch_id;
        size_t   size;
        size_t   num_slots;
    } alloc_end;
    struct
    {
        rocprofiler_scratch_memory_no_args no_args;
    } free_start;
    struct
    {
        rocprofiler_scratch_memory_no_args no_args;
    } free_end;
    struct
    {
        rocprofiler_scratch_memory_no_args no_args;
    } async_reclaim_start;
    struct
    {
        rocprofiler_scratch_memory_no_args no_args;
    } async_reclaim_end;
} rocprofiler_scratch_memory_args_t;

ROCPROFILER_EXTERN_C_FINI
