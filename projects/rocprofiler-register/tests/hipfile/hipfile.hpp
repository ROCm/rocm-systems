// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#define HIPFILE_ROCP_REG_VERSION_MAJOR 0
#define HIPFILE_ROCP_REG_VERSION_MINOR 2
#define HIPFILE_ROCP_REG_VERSION_PATCH 0

#include <cstdint>

extern "C" {
enum hipFileError_t
{
};

hipFileError_t
hipFileGetVersion(unsigned* major, unsigned* minor, unsigned* patch)
    __attribute__((visibility("default")));
}

namespace hipFile
{
struct hipFileDispatchTable
{
    uint64_t                          size                       = 0;
    decltype(::hipFileGetVersion)*    pfn_hipfile_get_version    = nullptr;
};

hipFileError_t
hipFileGetVersion(unsigned* major, unsigned* minor, unsigned* patch);

// populates hipFILE api table with function pointers
inline void
initialize_hipfile_api_table(hipFileDispatchTable* dst)
{
    dst->size                    = sizeof(hipFileDispatchTable);
    dst->pfn_hipfile_get_version = &::hipFile::hipFileGetVersion;
}

// copies the api table from src to dst
inline void
copy_hipfile_api_table(hipFileDispatchTable* dst, const hipFileDispatchTable* src)
{
    *dst = *src;
}
}  // namespace hipFile
