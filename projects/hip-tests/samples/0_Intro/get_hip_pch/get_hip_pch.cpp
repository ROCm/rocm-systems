/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <cstdio>
#include "hip/hip_runtime.h"

int main(int argc, char* argv[]) {
#ifdef __HIP_ENABLE_PCH
#ifndef __HIP_HAS_GET_PCH
#error "Macro __HIP_ENABLE_PCH defined but __HIP_HAS_GET_PCH not defined"
#endif

#if __HIP_HAS_GET_PCH != 1
#error "Macro __HIP_ENABLE_PCH defined but __HIP_HAS_GET_PCH value not equal to 1"
#endif
  // Verify hip_pch.o
  const char* pch = nullptr;
  unsigned int size = 0;
  __hipGetPCH(&pch, &size);
  printf("pch size: %u\n", size);
  if (size == 0) {
    printf("__hipGetPCH failed!\n");
    return -1;
  }
#endif
  // CLR exposes hip_runtime.h and hip_fp16.h in a single precompiled header
  // through the __hipGetPCH function.
  // This PCH could be passed to Comgr by the user to compile hip kernels.
  // This was the case for MIOpen, which has been dropped since ROCm 6.3 for
  // the preferred hiprtc API.
  printf("__hipGetPCH succeeded!\n");
  printf("PASSED!\n");
  return 0;
}
