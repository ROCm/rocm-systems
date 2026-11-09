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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "common/defines.hpp"

#include <rocprofiler-register/rocprofiler-register.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(ROCPROFILER_REGISTER_TEST_SDK_FIXTURE)

extern "C" {
int
rocprofiler_set_api_table(const char*, uint64_t, uint64_t, void**, uint64_t)
    ROCPROFILER_REGISTER_TEST_PUBLIC_API;

int
rocprofiler_attach() ROCPROFILER_REGISTER_TEST_PUBLIC_API;

int
rocprofiler_detach() ROCPROFILER_REGISTER_TEST_PUBLIC_API;

int
rocprofiler_set_api_table(const char*, uint64_t, uint64_t, void**, uint64_t)
{
    std::puts("runtime SONAME SDK fixture loaded");
    if(std::getenv("ROCPROFILER_REGISTER_TOOL_ATTACHED") != nullptr)
    {
        const auto* contents = std::getenv("ROCPROF_EXTRA_COUNTERS_CONTENTS");
        if(contents == nullptr || std::strlen(contents) <= 256 * 1024) return 1;
    }
    return 0;
}

int
rocprofiler_attach()
{
    const auto* contents = std::getenv("ROCPROF_EXTRA_COUNTERS_CONTENTS");
    return (contents != nullptr && std::strlen(contents) > 256 * 1024) ? 0 : 1;
}

int
rocprofiler_detach()
{
    return (std::getenv("ROCPROF_EXTRA_COUNTERS_CONTENTS") == nullptr) ? 0 : 1;
}
}

#elif defined(ROCPROFILER_REGISTER_TEST_ATTACH_FIXTURE)

using register_api_table_func_t = decltype(&rocprofiler_register_library_api_table);

extern "C" {
int
rocprofiler_attach_set_api_table(const char*,
                                 uint64_t,
                                 uint64_t,
                                 void**,
                                 uint64_t,
                                 register_api_table_func_t)
    ROCPROFILER_REGISTER_TEST_PUBLIC_API;

int
rocprofiler_attach_set_api_table(const char*,
                                 uint64_t,
                                 uint64_t,
                                 void**,
                                 uint64_t,
                                 register_api_table_func_t register_api_table)
{
    std::puts("runtime SONAME attach fixture loaded");

    // register like the real attach library so rocprofiler_register_attach can proceed
    static void* api_table = nullptr;
    auto         lib_id    = rocprofiler_register_library_indentifier_t{};
    return register_api_table("rocattach", nullptr, 1, &api_table, 1, &lib_id);
}
}

#else
#    error "A runtime SONAME fixture type must be selected"
#endif
