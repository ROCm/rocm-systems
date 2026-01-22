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
#include "memoryRAR.hpp"
#include "non_caching_load.h"
#include "non_caching_store.h"
#include "non_caching_store_vec4.h"

constexpr int N = 1024;
typedef uint32_t uint32x4 __attribute__((ext_vector_type(4)));

template<typename T>
__global__ void readKernel(T* data, T* out){
    T temp;
    temp = __non_caching_load<T>(out);

    T temp1;
    temp1 = __non_caching_load<T>(out);

    if constexpr (std::is_same<T, uint32x4>::value){
        if(temp[0] == temp1[0])
            printf("PASSED\n");
    }
    else{
        if(temp == temp1)
            printf("PASSED\n");
    }
}

template<typename T>
void caching_load_store() {
    T* d_data;
    T* out;
    size_t size = N * sizeof(d_data);

    hipMalloc(&d_data, size);
    hipMalloc(&out, size);

    int* h_data = new int[N];
    for(int i = 0; i < N; ++i){
        h_data[i] = 105;
    }

    HIP_CALL(hipMemcpy(d_data, h_data, size, hipMemcpyHostToDevice));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    std::cout << "Running Read After Read(RAR) Test" << std::endl;
    hipLaunchKernelGGL(readKernel, blocksPerGrid, threadsPerBlock, 0, 0, d_data, out);
    HIP_CALL(hipDeviceSynchronize());

    HIP_CALL(hipMemcpy(h_data, d_data, size, hipMemcpyDeviceToHost));

    HIP_CALL(hipFree(d_data));
    delete[] h_data;

    std::cout << "Testing done" << std::endl;

    return;
}

int main(int argc, char **argv){
    caching_load_store<uint64_t>();
    caching_load_store<uint32_t>();
    caching_load_store<uint16_t>();
    caching_load_store<uint8_t>();
    using V2 = unsigned __attribute__((ext_vector_type(4)));
    caching_load_store<V2>();

    return 0;
}

