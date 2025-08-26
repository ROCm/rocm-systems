/*
 * Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * */

#include <hip/hip_runtime_api.h>
#include <iostream>
#include "hipMalloc.h"

int MallocFunc() {
  int* Ad;
  hipError_t err;
  err = hipMalloc(reinterpret_cast<void**>(&Ad), 1024);
  if (err == hipSuccess) {
    std::cout << "hipMalloc PASSED!" << std::endl;
    err = hipFree(Ad);
    if (err == hipSuccess) {
      std::cout << "hipFree PASSED!" << std::endl;
      return 1;
    } else {
      std::cout << "hipFree FAILED!" << std::endl;
      return 0;
    }
  } else {
    std::cout << "hipMalloc FAILED!" << std::endl;
    return 0;
  }
}
