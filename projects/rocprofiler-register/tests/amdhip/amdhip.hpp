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

#define HIP_VERSION_MAJOR 6
#define HIP_VERSION_MINOR 0
#define HIP_VERSION_PATCH 1

#include <cstdint>

// WINDOWS-DIVERGENCE: GCC's __attribute__((visibility("default"))) is not
// understood by MSVC; the equivalent for symbols exported from a DLL is
// __declspec(dllexport). The Linux side relies on default visibility being
// off via -fvisibility=hidden so this attribute makes the symbol resolvable
// via dlsym; on Windows GetProcAddress requires the symbol to be in the
// DLL's export table, which __declspec(dllexport) accomplishes.
#if defined(_WIN32)
#    define ROCP_REG_TEST_MOCK_EXPORT __declspec(dllexport)
#else
#    define ROCP_REG_TEST_MOCK_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
// fake hip function
ROCP_REG_TEST_MOCK_EXPORT void
hip_init(void);
}

namespace hip
{
struct HipApiTable
{
    uint64_t              size        = 0;
    decltype(::hip_init)* hip_init_fn = nullptr;
};

void
hip_init();

// populates hip api table with function pointers
inline void
initialize_hip_api_table(HipApiTable* dst)
{
    dst->size        = sizeof(HipApiTable);
    dst->hip_init_fn = &::hip::hip_init;
}

// copies the api table from src to dst
inline void
copy_hip_api_table(HipApiTable* dst, const HipApiTable* src)
{
    *dst = *src;
}
}  // namespace hip
