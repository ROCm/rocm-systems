/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

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
  do {                                                                                             \
    if (res) {                                                                                     \
      return 0;                                                                                    \
    } else {                                                                                       \
      return -1;                                                                                   \
    }                                                                                              \
  } while (0)

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

  HIP_CHECK(hipDeviceGetLimit(&min, hipExtLimitScratchMin))
  REQUIRE(min == 0);

  HIP_CHECK(hipDeviceGetLimit(&max, hipExtLimitScratchMax))
  REQUIRE(max > 0);

  HIP_CHECK(hipDeviceGetLimit(&orgCurrent, hipExtLimitScratchCurrent))

  size_t setCurrent = max / 2;
  HIP_CHECK(hipDeviceSetLimit(hipExtLimitScratchCurrent, setCurrent))

  size_t getCurrent = 0;
  HIP_CHECK(hipDeviceGetLimit(&getCurrent, hipExtLimitScratchCurrent))
  REQUIRE(getCurrent == setCurrent);

  HIP_CHECK(hipDeviceSetLimit(hipExtLimitScratchCurrent, orgCurrent))

  return 0;
}
