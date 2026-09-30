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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "common/defines.hpp"

#include <rocprofiler-register/rocprofiler-register.h>

#include <cstdint>
#include <cstdio>

namespace
{
int detach_status = 0;
}

extern "C" {
int
rocprofiler_set_api_table(const char*, uint64_t, uint64_t, void**, uint64_t)
    ROCPROFILER_REGISTER_TEST_PUBLIC_API;

int
rocprofiler_attach(void) ROCPROFILER_REGISTER_TEST_PUBLIC_API;

int
rocprofiler_detach(void) ROCPROFILER_REGISTER_TEST_PUBLIC_API;

rocprofiler_register_error_code_t
rocprofiler_register_attach(const char*, const char*);

rocprofiler_register_error_code_t
rocprofiler_register_detach();

int
rocprofiler_set_api_table(const char*, uint64_t, uint64_t, void**, uint64_t)
{
    return 0;
}

int
rocprofiler_attach(void)
{
    return 0;
}

int
rocprofiler_detach(void)
{
    return detach_status;
}
}

int
main()
{
    auto  table        = int{};
    void* api_tables[] = { &table };
    auto  register_id  = rocprofiler_register_library_indentifier_t{};

    auto status = rocprofiler_register_library_api_table(
        "rocattach", nullptr, 10000, api_tables, 1, &register_id);
    if(status != ROCP_REG_SUCCESS)
    {
        std::fprintf(stderr, "Test FAILED: rocattach registration returned %d\n", status);
        return 1;
    }

    status = rocprofiler_register_attach(nullptr, "mock-tool.so");
    if(status != ROCP_REG_SUCCESS)
    {
        std::fprintf(stderr, "Test FAILED: attach returned %d\n", status);
        return 2;
    }

    detach_status = 0;
    status        = rocprofiler_register_detach();
    if(status != ROCP_REG_SUCCESS)
    {
        std::fprintf(stderr, "Test FAILED: successful detach returned %d\n", status);
        return 3;
    }

    detach_status = -1;
    status        = rocprofiler_register_detach();
    if(status != ROCP_REG_ROCPROFILER_ERROR)
    {
        std::fprintf(stderr, "Test FAILED: rejected detach returned %d\n", status);
        return 4;
    }

    std::puts("Test PASSED: detach status propagated through rocprofiler-register");
    return 0;
}
