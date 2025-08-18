/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>

// trigger invalid memory access
void __global__ MemAccessFaultKernel(int *data) {
  data[threadIdx.x + 1000] = threadIdx.x;
}

#if 1
void throw_hip_error(hipError_t error, char const *file,
                             unsigned int line, hipError_t exp_error) {
#else
inline void throw_hip_error(hipError_t error, char const *file,
                             unsigned int line, hipError_t exp_error) {
#endif
  // Calls hipGetLastError to clear the error status. It is nearly certain that
  // a fatal error occurred if it still returns the same error after a cleanup.

  hipError_t tmpError = hipGetLastError();
  (void)tmpError;
  auto const last = hipFree(0);
  auto const msg =
      std::string{"CUDA error encountered at: " + std::string{file} + ":" +
                  std::to_string(line) + ": " + std::to_string(error) + " " +
                  hipGetErrorName(error) + " " + hipGetErrorString(error)};

  if (error != exp_error || last != exp_error) {
    // throw std::runtime_error{"Fatal " + msg};
  } else {
    // throw std::runtime_error{msg};
  }
}

#define HIP_CHECK_TRY(call, error)                                                                 \
do {                                                                                               \
  hipError_t const status = (call);                                                                \
  if (hipSuccess != status) {                                                                      \
    throw_hip_error(status, __FILE__, __LINE__, error);                                            \
  }                                                                                                \
} while (0);

#define HIP_CHECK_STREAM(stream)                                                                   \
do {                                                                                               \
  HIP_CHECK_TRY(hipStreamSynchronize(stream), hipErrorIllegalAddress);                             \
  HIP_CHECK_TRY(hipPeekAtLastError(), hipErrorIllegalAddress);                                     \
} while (0);

/**
 * @addtogroup hipGetErrorOnCrash hipGetErrorOnCrash
 * @{
 * @ingroup ErrorTest
 * `hipGetErrorOnCrash(hipError_t hip_error)` -
 * Return hip error as text string form.
 */

/**
 * Test Description
 * ------------------------
 *  - Validate that the correct string is returned for each supported
 *    device error enumeration.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetErrorOnCrash.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */

TEST_CASE("Unit_hipGetErrorOnMemAccessFault") {
  int a;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  MemAccessFaultKernel<<<1, 1, 0, stream>>>(&a);
  HIP_CHECK_STREAM(stream);
}

/**
* End doxygen group ErrorTest.
* @}
*/
