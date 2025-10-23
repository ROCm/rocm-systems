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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define NUM_THREADS 128 // Per block
#define NUM_BLOCKS  1

typedef int v1024i __attribute__((ext_vector_type(1024)));

__launch_bounds__(NUM_THREADS, NUM_BLOCKS)
__global__ void test1024(v1024i *in, v1024i *out) {
  out[threadIdx.x] = in[threadIdx.x];
}

void test_vgprs_value() {
  size_t n = NUM_BLOCKS * NUM_THREADS;
  size_t bufferSize = n * sizeof(v1024i);

  v1024i *dX, *dY;
  HIP_CHECK(hipMalloc(&dX, bufferSize));
  HIP_CHECK(hipMalloc(&dY, bufferSize));

  std::unique_ptr<v1024i[]> hX{new v1024i[n]};
  std::unique_ptr<v1024i[]> hY{new v1024i[n]};
  const int sizeofv1024i =  sizeof(v1024i)/sizeof(int);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < sizeofv1024i; j++) {
      hX[i][j] = (i + 1) * (j + 1);
      hY[i][j] = 0;
    }
  }
  HIP_CHECK(hipMemcpy(dX, hX.get(), bufferSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dY, hY.get(), bufferSize, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test1024, NUM_BLOCKS, NUM_THREADS, 0, 0, dX, dY);
  HIP_CHECK(hipMemcpy(hY.get(), dY, bufferSize, hipMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < sizeofv1024i; j++) {
      if (hY[i][j] != (i + 1) * (j + 1)) {
        std::cout << "Failed on "
            << i << ", " << j << ":" << hX[i][j] << " with: " << (i + 1) * (j + 1) << std::endl;
        REQUIRE(false);
      }
    }
  }
  HIP_CHECK(hipFree(dX));
  HIP_CHECK(hipFree(dY));
  REQUIRE(true);
}

TEST_CASE("Unit_Device__hip_check_VGPRs") {
  hipDeviceProp_t props;
  hipFuncAttributes attr;
  int maxAvailableVgprsPerThread = 0;
  constexpr int device = 0;
  HIP_CHECK(hipSetDevice(device));
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  HIP_CHECK(hipDeviceGetAttribute(&maxAvailableVgprsPerThread,
                                hipDeviceAttributeMaxAvailableVgprsPerThread, device));
  if (maxAvailableVgprsPerThread > 1024) {
    // The test should work on all current devices as of writing.
    HipTest::HIP_SKIP_TEST("maxAvailableVgprsPerThread > 1024 isn't supported in this test!");
  }
  HIP_CHECK(hipFuncGetAttributes(&attr, reinterpret_cast<void*>(test1024)));
  std::cout << "Info: running on device #" << device << " " << props.name << ": arch = "
        << props.gcnArchName << ", major = " << props.major << ", minor = " << props.minor
        << ", warpSize = " << props.warpSize << ", numRegs of test1024() = "
        << attr.numRegs << " DWORDs, MaxAvailableVgprsPerThread = " << maxAvailableVgprsPerThread
        << " DWORDs\n";
  const int usedVGPRs_ = attr.numRegs; // Used VGPRs in DWORDS.
  const int extraOffset = 20; // Empirical offset due to extra VGPRs consumed.
  // Verify VGPRs usage
  if (maxAvailableVgprsPerThread < usedVGPRs_ ||
      usedVGPRs_ < (maxAvailableVgprsPerThread - extraOffset)) {
    REQUIRE(false);
  }
  // Verify VGPRs data
  test_vgprs_value();
}
