// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#define RCCL_API_TRACE_VERSION_MAJOR 0
#define RCCL_API_TRACE_VERSION_PATCH 0

#include <cstddef>
#include <cstdint>

extern "C" {
// fake rccl function
typedef int ncclResult_t;

enum ncclDataType_t
{
};

ncclResult_t
ncclGetVersion(int* version) __attribute__((visibility("default")));
}

namespace rccl
{
struct rcclApiFuncTable
{
    uint64_t                    size              = 0;
    decltype(::ncclGetVersion)* ncclGetVersion_fn = nullptr;
};

ncclResult_t
ncclGetVersion(int* version);

// populates rccl api table with function pointers
inline void
initialize_rccl_api_table(rcclApiFuncTable* dst)
{
    dst->size              = sizeof(rcclApiFuncTable);
    dst->ncclGetVersion_fn = &::rccl::ncclGetVersion;
}

// copies the api table from src to dst
inline void
copy_rccl_api_table(rcclApiFuncTable* dst, const rcclApiFuncTable* src)
{
    *dst = *src;
}
}  // namespace rccl
