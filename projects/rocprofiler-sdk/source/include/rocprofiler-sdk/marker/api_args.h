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

#include <stdint.h>

typedef uint64_t roctx_range_id_t;

typedef union rocprofiler_marker_api_retval_u
{
    uint32_t uint32_t_retval;
    uint64_t uint64_t_retval;
} rocprofiler_marker_api_retval_t;

typedef union rocprofiler_marker_api_args_u
{
    struct
    {
        const char* message;
    } roctxMarkA;
    struct
    {
        const char* message;
    } roctxRangePushA;
    struct
    {
    } roctxRangePop;
    struct
    {
        const char* message;
    } roctxRangeStartA;
    struct
    {
        roctx_range_id_t id;
    } roctxRangeStop;
} rocprofiler_marker_api_args_t;
