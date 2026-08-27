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

#include <rocprofiler-sdk/registration.h>

#include <dlfcn.h>

#include <cstdlib>
#include <iostream>

namespace
{
using is_initialized_func_t = rocprofiler_status_t (*)(int*);

int
get_initialization_status(void* handle)
{
    auto is_initialized =
        reinterpret_cast<is_initialized_func_t>(dlsym(handle, "rocprofiler_is_initialized"));
    if(is_initialized == nullptr) return -1;

    auto status = int{-1};
    if(is_initialized(&status) != ROCPROFILER_STATUS_SUCCESS) return -1;
    return status;
}
}  // namespace

int
main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "FAIL: expected the path to the second rocprofiler-sdk\n";
        return EXIT_FAILURE;
    }

    if(get_initialization_status(RTLD_DEFAULT) != 1)
    {
        std::cerr << "FAIL: the preloaded rocprofiler-sdk did not initialize\n";
        return EXIT_FAILURE;
    }

    auto* duplicate_handle = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    if(duplicate_handle == nullptr)
    {
        std::cerr << "FAIL: failed to load the second rocprofiler-sdk: " << dlerror() << '\n';
        return EXIT_FAILURE;
    }

    if(get_initialization_status(duplicate_handle) != 0)
    {
        std::cerr << "FAIL: the second rocprofiler-sdk initialized\n";
        return EXIT_FAILURE;
    }

    std::cout << "PASS: only the preloaded SDK initialized\n";
    return EXIT_SUCCESS;
}
