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

#define ROCSHMEM_API_TRACE_VERSION_MAJOR 0
#define ROCSHMEM_API_TRACE_VERSION_PATCH 0

#include <cstddef>
#include <cstdint>

extern "C" {
// fake rocshmem function
typedef int rocshmem_status_t;

rocshmem_status_t
rocshmem_init_mock(void) __attribute__((visibility("default")));
}

namespace rocshmem
{
struct rocshmemApiFuncTable
{
    uint64_t                        size             = 0;
    decltype(::rocshmem_init_mock)* rocshmem_init_fn = nullptr;
};

rocshmem_status_t
rocshmem_init_mock(void);

// populates rocshmem api table with function pointers
inline void
initialize_rocshmem_api_table(rocshmemApiFuncTable* dst)
{
    dst->size             = sizeof(rocshmemApiFuncTable);
    dst->rocshmem_init_fn = &::rocshmem::rocshmem_init_mock;
}

// copies the api table from src to dst
inline void
copy_rocshmem_api_table(rocshmemApiFuncTable* dst, const rocshmemApiFuncTable* src)
{
    *dst = *src;
}
}  // namespace rocshmem
