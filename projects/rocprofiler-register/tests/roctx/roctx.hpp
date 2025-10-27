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

#define ROCTX_VERSION_MAJOR 4
#define ROCTX_VERSION_MINOR 6
#define ROCTX_VERSION_PATCH 1

#include <cstdint>

extern "C" {
void
roctxRangePush(const char*) __attribute__((visibility("default")));
void
roctxRangePop(const char*) __attribute__((visibility("default")));
}

namespace roctx
{
struct ROCTxApiTable
{
    uint64_t                    size              = 0;
    decltype(::roctxRangePush)* roctxRangePush_fn = nullptr;
    decltype(::roctxRangePop)*  roctxRangePop_fn  = nullptr;
};

void
roctx_range_push(const char*);

void
roctx_range_pop(const char*);

// populates roctx api table with function pointers
inline void
initialize_roctx_api_table(ROCTxApiTable* dst)
{
    dst->size              = sizeof(ROCTxApiTable);
    dst->roctxRangePush_fn = &::roctx::roctx_range_push;
    dst->roctxRangePop_fn  = &::roctx::roctx_range_pop;
}

// copies the api table from src to dst
inline void
copy_roctx_api_table(ROCTxApiTable* dst, const ROCTxApiTable* src)
{
    *dst = *src;
}
}  // namespace roctx
