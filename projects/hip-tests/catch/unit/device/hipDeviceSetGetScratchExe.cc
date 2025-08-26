/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip/hip_runtime.h"

#define HIP_CHECK(error)                                                                           \
  {                                                                                                \
    hipError_t localError = error;                                                                 \
    if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {          \
      printf("error: '%s'(%d) from %s at %s:%d\n", hipGetErrorString(localError), localError,      \
             #error, __FUNCTION__, __LINE__);                                                      \
      return -1;                                                                                   \
    }                                                                                              \
  }

#define REQUIRE(res)                                                                               \
  {                                                                                                \
    if (res) {                                                                                     \
      return 0;                                                                                    \
    } else {                                                                                       \
      return -1;                                                                                   \
    }                                                                                              \
  }

/*
 * This funtion perform the below operations,
 * 1) Get the Minimum Scratch limit and validate
 * 2) Get the Maximum Scratch limit and validate
 * 3) Get the Current Scratch limit and validate
 * 4) Set the Current Scratch limit to 50% of Maximum Scratch limit
 * 5) Get the Current Scratch limit and validate it
 * 6) Set the Current Scratch limit to original value
 */
int main() {
  size_t min = 0, max = 0, orgCurrent = 0;

  HIP_CHECK(hipDeviceGetLimit(&min, hipExtLimitScratchMin));
  REQUIRE(min == 0);

  HIP_CHECK(hipDeviceGetLimit(&max, hipExtLimitScratchMax));
  REQUIRE(max > 0);

  HIP_CHECK(hipDeviceGetLimit(&orgCurrent, hipExtLimitScratchCurrent));
  REQUIRE(orgCurrent >= 0);

  size_t setCurrent = 0.5 * max;
  HIP_CHECK(hipDeviceSetLimit(hipExtLimitScratchCurrent, setCurrent));

  size_t getCurrent = 0;
  HIP_CHECK(hipDeviceGetLimit(&getCurrent, hipExtLimitScratchCurrent));
  REQUIRE(getCurrent == setCurrent);

  HIP_CHECK(hipDeviceSetLimit(hipExtLimitScratchCurrent, orgCurrent));

  return 0;
}
