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

// Loaded with dlopen by the test app. Its constructor, and then the registration of its fat
// binary, run while dlopen holds glibc's dl_load_lock.

#include <hip/hip_runtime.h>

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

__global__ void
plugin_kernel(int* data)
{
    data[threadIdx.x] = threadIdx.x;
}

// priority 101 runs before the compiler-generated constructor that registers this library's fat
// binary, so in code-object-load mode the worker is inside its code-object load by then
__attribute__((constructor(101))) static void
plugin_ctor()
{
    using hook_t = void (*)();
    auto* hook   = reinterpret_cast<hook_t>(dlsym(RTLD_DEFAULT, "dlopen_ctor_deadlock_hook"));
    if(!hook)
    {
        fprintf(stderr, "dlopen_ctor_deadlock_hook not found: %s\n", dlerror());
        std::abort();
    }
    hook();
}
