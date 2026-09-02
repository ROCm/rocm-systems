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

#include <rocprofiler-register/rocprofiler-register.h>

#include <cstdint>
#include <cstdio>

#if defined(ROCPROFILER_REGISTER_TEST_SDK_FIXTURE)

extern "C"
{
int
rocprofiler_set_api_table(const char*, uint64_t, uint64_t, void**, uint64_t)
    ROCPROFILER_REGISTER_EXPORT_DECORATOR;

int
rocprofiler_set_api_table(const char*, uint64_t, uint64_t, void**, uint64_t)
{
    std::puts("runtime SONAME SDK fixture loaded");
    return 0;
}
}

#elif defined(ROCPROFILER_REGISTER_TEST_ATTACH_FIXTURE)

using register_api_table_func_t = decltype(&rocprofiler_register_library_api_table);

extern "C"
{
int
rocprofiler_attach_set_api_table(const char*,
                                 uint64_t,
                                 uint64_t,
                                 void**,
                                 uint64_t,
                                 register_api_table_func_t)
    ROCPROFILER_REGISTER_EXPORT_DECORATOR;

int
rocprofiler_attach_set_api_table(const char*,
                                 uint64_t,
                                 uint64_t,
                                 void**,
                                 uint64_t,
                                 register_api_table_func_t)
{
    std::puts("runtime SONAME attach fixture loaded");
    return 0;
}
}

#else
#    error "A runtime SONAME fixture type must be selected"
#endif
