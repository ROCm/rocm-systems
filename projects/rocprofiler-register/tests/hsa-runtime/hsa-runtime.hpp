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

#define HSA_VERSION_MAJOR 2
#define HSA_VERSION_MINOR 1
#define HSA_VERSION_PATCH 0

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
// fake hsa function
ROCP_REG_TEST_MOCK_EXPORT void
hsa_init(void);
}

namespace hsa
{
struct HsaApiTable
{
    uint64_t              size        = 0;
    decltype(::hsa_init)* hsa_init_fn = nullptr;
};

void
hsa_init();

// populates hsa api table with function pointers
inline void
initialize_hsa_api_table(HsaApiTable* dst)
{
    dst->size        = sizeof(HsaApiTable);
    dst->hsa_init_fn = &::hsa::hsa_init;
}

// copies the api table from src to dst
inline void
copy_hsa_api_table(HsaApiTable* dst, const HsaApiTable* src)
{
    *dst = *src;
}
}  // namespace hsa
