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
#include "non-caching-load.hpp"
#include "non_caching_load.h"

typedef uint32_t uint32x4 __attribute__((ext_vector_type(4)));

template<typename T>
__global__ void nonCachingLoad(T* p, T* out){
  if constexpr (std::is_same<T, uint32x4>::value)
    p[0] = {22, 22, 22, 22};
  else
    p[0] = 22;
	out[0] = __non_caching_load<T>(p);
}

template<typename T>
__global__ void builtinTemporalLoad(T* p, T* out){
  if constexpr (std::is_same<T, uint32x4>::value)
    p[0] = {22, 22, 22, 22};
  else
    p[0] = 22;
	out[0] = __builtin_nontemporal_load(p);
}

template<typename T>
void caching_load() {
  T* data;
  T* out1;
  T* out2;
  size_t size = sizeof(data);

  hipMalloc(&data, size);
  hipMalloc(&out1, size);
  hipMalloc(&out2, size);

  hipLaunchKernelGGL(nonCachingLoad<T>, dim3(1), dim3(1), 0, 0, data, out1);
  hipLaunchKernelGGL(builtinTemporalLoad<T>, dim3(1), dim3(1), 0, 0, data, out2);

  hipDeviceSynchronize();

  T* host_data = (T*)malloc(size);
  T* h_o1 = (T*)malloc(size);
  T* h_o2 = (T*)malloc(size);

  hipMemcpy(host_data, data, size, hipMemcpyDeviceToHost);
  hipMemcpy(h_o1, out1, size, hipMemcpyDeviceToHost);
  hipMemcpy(h_o2, out2, size, hipMemcpyDeviceToHost);

  if constexpr (std::is_same<T, uint32x4>::value)
  {
    if ( ((*h_o1)[0] == (*h_o2)[0]) && ((*h_o1)[1] == (*h_o2)[1]) && ((*h_o1)[2] == (*h_o2)[2])
     && ((*h_o1)[3] == (*h_o2)[3]))
      std::cout << "PASS" << std::endl;
    else
      std::cout << "FAIL" << std::endl; 
  }
  else {
    if(*h_o1 == *h_o2)
    {
      std::cout << "PASS" << std::endl; 
    }
    else{
      std::cout << "FAIL" << std::endl;
    }
  }

  hipFree(data);
  return;
}

int main(int argc, char **argv)
{
  caching_load<uint64_t>();
  caching_load<uint32_t>();
  caching_load<uint16_t>();
  caching_load<uint8_t>();
  using V2 = unsigned __attribute__((ext_vector_type(4)));
  caching_load<V2>();

  return 0;
}

