/*
Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <sys/socket.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <unistd.h>
#include <cstdio>
#include <string>
#include <chrono>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <cstdlib>
#include <fstream>
#include <iostream> //cerr
#include <cstring>
#include "non-caching-store.hpp"
#include "non_caching_store.h"
#include "non_caching_store_vec4.h"

typedef uint32_t uint32x4 __attribute__((ext_vector_type(4)));

template<typename T>
__global__ void nonCachingStore(T* out){
    T val[4];
    if constexpr (std::is_same<T, uint32x4>::value){
        val[4] = {22, 22, 22, 22};
        __non_caching_store_vec4<T>(val[0], out);
    }
    else{
        val[0] = 22;
        __non_caching_store<T>(val[0], out);
    }
    return;
}

template<typename T>
void cachingStore() {
    T* out1;
    size_t size = sizeof(out1);
    hipMalloc(&out1, size);

    hipLaunchKernelGGL(nonCachingStore<T>, dim3(1), dim3(1), 0, 0, out1);
    hipDeviceSynchronize();

    T* h_o1 = (T*)malloc(size);
    hipMemcpy(h_o1, out1, size, hipMemcpyDeviceToHost);

    hipFree(out1);
    hipFree(h_o1);
    return;
}

int main(int argc, char **argv)
{
    cachingStore<uint64_t>();
    cachingStore<uint32_t>();
    cachingStore<uint16_t>();
    cachingStore<uint8_t>();
    using V2 = unsigned __attribute__((ext_vector_type(4)));
    cachingStore<V2>();

    return 0;
}

