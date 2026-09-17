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

#include <cstdint>
#include <iostream>
#include <string_view>

#include <rocprofiler-register/rocprofiler-register.h>

using register_api_table_func_t = decltype(::rocprofiler_register_library_api_table)*;

namespace
{
bool attachment_initialized = false;
}

extern "C" int
rocprofiler_test_attachment_initialized() ROCPROFILER_REGISTER_TEST_PUBLIC_API;

extern "C" int
rocprofiler_test_attachment_initialized()
{
    return attachment_initialized ? 1 : 0;
}

extern "C" int
rocprofiler_attach_initialize(register_api_table_func_t register_functor)
    ROCPROFILER_REGISTER_TEST_PUBLIC_API;

extern "C" int
rocprofiler_attach_initialize(register_api_table_func_t register_functor)
{
    if(register_functor == nullptr)
    {
        std::cerr << "Test FAILED: invalid registration functor\n";
        return 1;
    }
    if(attachment_initialized) return 0;

    auto        library_id      = rocprofiler_register_library_indentifier_t{};
    static auto attach_table    = uint64_t{};
    void*       attach_tables[] = { &attach_table };
    if(register_functor("rocattach", nullptr, 1, attach_tables, 1, &library_id) !=
       ROCP_REG_SUCCESS)
    {
        std::cerr << "Test FAILED: mock attachment library could not register itself\n";
        return 1;
    }

    attachment_initialized = true;
    std::cout << "Test fixture: attachment capability initialized\n";
    return 0;
}

extern "C" int
rocprofiler_attach_set_api_table(const char* name,
                                 uint64_t,
                                 uint64_t,
                                 void**,
                                 uint64_t,
                                 register_api_table_func_t register_functor)
    ROCPROFILER_REGISTER_TEST_PUBLIC_API;

extern "C" int
rocprofiler_attach_set_api_table(const char* name,
                                 uint64_t,
                                 uint64_t,
                                 void**,
                                 uint64_t,
                                 register_api_table_func_t register_functor)
{
    if(std::string_view{ name } != "hsa" || !attachment_initialized ||
       register_functor == nullptr)
    {
        std::cerr << "Test FAILED: invalid API table passed to mock attachment library\n";
        return 1;
    }

    return 0;
}
