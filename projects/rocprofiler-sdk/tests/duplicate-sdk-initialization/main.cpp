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
fail(const char* message)
{
    std::cerr << "FAIL: " << message << '\n';
    return EXIT_FAILURE;
}

is_initialized_func_t
get_is_initialized(void* handle)
{
    return reinterpret_cast<is_initialized_func_t>(
        dlsym(handle, "rocprofiler_is_initialized"));
}
}  // namespace

int
main(int argc, char** argv)
{
    if(argc != 2) return fail("expected the absolute path to the second rocprofiler-sdk");

    auto first_is_initialized = get_is_initialized(RTLD_DEFAULT);
    if(first_is_initialized == nullptr)
        return fail("the preloaded rocprofiler-sdk is not visible through RTLD_DEFAULT");

    auto first_status = int{-1};
    if(first_is_initialized(&first_status) != ROCPROFILER_STATUS_SUCCESS)
        return fail("failed to query the preloaded rocprofiler-sdk initialization status");
    if(first_status != 1) return fail("the preloaded rocprofiler-sdk did not initialize");

    const char* ctor_value = std::getenv("ROCPROFILER_LIBRARY_CTOR");

    void* second_handle = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    if(second_handle == nullptr)
    {
        std::cerr << "FAIL: failed to load the second rocprofiler-sdk: " << dlerror() << '\n';
        return EXIT_FAILURE;
    }

    auto second_is_initialized = get_is_initialized(second_handle);
    if(second_is_initialized == nullptr)
        return fail("the second rocprofiler-sdk is missing rocprofiler_is_initialized");

    auto first_info  = Dl_info{};
    auto second_info = Dl_info{};
    if(dladdr(reinterpret_cast<void*>(first_is_initialized), &first_info) == 0)
        return fail("dladdr failed for the preloaded rocprofiler-sdk");
    if(dladdr(reinterpret_cast<void*>(second_is_initialized), &second_info) == 0)
        return fail("dladdr failed for the second rocprofiler-sdk");
    if(first_info.dli_fbase == second_info.dli_fbase)
        return fail("the dynamic loader reused the preloaded rocprofiler-sdk mapping");

    auto second_status = int{-1};
    if(second_is_initialized(&second_status) != ROCPROFILER_STATUS_SUCCESS)
        return fail("failed to query the second rocprofiler-sdk initialization status");

    std::cout << "preloaded SDK: path='" << first_info.dli_fname << "', status=" << first_status
              << '\n';
    std::cout << "second SDK: path='" << second_info.dli_fname << "', status=" << second_status
              << '\n';
    std::cout << "ROCPROFILER_LIBRARY_CTOR="
              << ((ctor_value != nullptr) ? ctor_value : "<unset>") << '\n';

    if(second_status != 0) return fail("the second rocprofiler-sdk initialized");

    if(dlclose(second_handle) != 0)
    {
        std::cerr << "FAIL: failed to close the second rocprofiler-sdk: " << dlerror() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "PASS: two SDKs mapped, only the preloaded SDK initialized\n";
    return EXIT_SUCCESS;
}
