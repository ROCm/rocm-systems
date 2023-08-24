// Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
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

// NOLINTNEXTLINE(performance-enum-size)
typedef enum
{
    ROCPROFILER_ROCTX_API_ID_NONE       = -1,
    ROCPROFILER_ROCTX_API_ID_roctxMarkA = 0,
    ROCPROFILER_ROCTX_API_ID_roctxRangePushA,
    ROCPROFILER_ROCTX_API_ID_roctxRangePop,
    ROCPROFILER_ROCTX_API_ID_roctxRangeStartA,
    ROCPROFILER_ROCTX_API_ID_roctxRangeStop,
    ROCPROFILER_ROCTX_API_ID_LAST,
} rocprofiler_roctx_api_id_t;
