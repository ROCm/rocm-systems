/*
   Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <thread>

#define HIP_CHECK(call)                                                                            \
  {                                                                                                \
    auto res_ = (call);                                                                            \
    if (res_ != hipSuccess) {                                                                      \
      std::cout << "Failed in: " << #call << std::endl;                                            \
      return -1;                                                                                   \
    }                                                                                              \
  }

int main() {
  size_t freeMem = 0, totalMem = 0;
  HIP_CHECK(hipMemGetInfo(&freeMem, &totalMem));

  void* ptr;
  HIP_CHECK(hipMalloc(&ptr, 0.4 * totalMem));  // hold 40% of total gpu memory
  std::cout << "Sleeping..." << std::endl;
  std::this_thread::sleep_for(
      std::chrono::seconds(4));  //  sleep for few seconds till test complete
  std::cout << "Waking up..." << std::endl;
  HIP_CHECK(hipFree(ptr));
}