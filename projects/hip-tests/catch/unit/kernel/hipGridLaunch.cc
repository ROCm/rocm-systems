/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Test the Grid_Launch syntax.

#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <utils.hh>


static unsigned threadsPerBlock = 256;
static unsigned blocksPerCU = 6;

// __device__ maps to __attribute__((hc))
__device__ int foo(int i) { return i + 1; }


template <typename T> __global__ void vectorADD2(T* A_d, T* B_d, T* C_d, size_t N) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < N; i += stride) {
    double foo = __hiloint2double(A_d[i], B_d[i]);
    C_d[i] = __double2loint(foo) + __double2hiint(foo);
  }
}

int test_gl2(size_t N) {
  size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  // Full vadd in one large chunk, to get things started:
  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(vectorADD2, dim3(blocks), dim3(threadsPerBlock), 0, 0, A_d, B_d, C_d, N);
  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  // verify
  HipTest::checkVectorADD(A_h, B_h, C_h, N);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  return 0;
}

#if __HIP__
int test_triple_chevron(size_t N) {
  size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  // Full vadd in one large chunk, to get things started:
  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));
  vectorADD2<<<dim3(blocks), dim3(threadsPerBlock)>>>(A_d, B_d, C_d, N);
  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  // verify
  HipTest::checkVectorADD(A_h, B_h, C_h, N);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  return 0;
}
#endif

/**
* @addtogroup hipLaunchKernelGGL hipLaunchKernelGGL
* @{
* @ingroup KernelTest
* `void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
   std::uint32_t sharedMemBytes, hipStream_t stream, Args... args)` -
* Method to invocate kernel functions
*/

/**
 * Test Description
 * ------------------------
 *    - Test case to verify the Grid_Launch syntax.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipGridLaunch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 */

HIP_TEST_CASE(Unit_hipGridLaunch) {
  size_t N = isQuickLevel() ? 100 * 1024 : 4 * 1024 * 1024;
  SECTION("Test test_gl2") { test_gl2(N); }

#if __HIP__
  SECTION("Test triple_chevron") { test_triple_chevron(N); }
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Test case to verify maxGrid limits from device properties.
 *    - Fetches maxGridSize from hipDeviceProp_t and verifies kernel launch
 *    - at maximum grid dimensions succeeds.
 *    - Also verifies that exceeding maxGridSize returns appropriate error codes.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipGridLaunch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 */

__global__ void emptyKernel() {}

TEST_CASE("Unit_hipGridLaunch_MaxGridDim_DeviceProperties") {
  hipDeviceProp_t deviceProp;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, device));

  unsigned int maxGridX = deviceProp.maxGridSize[0];
  unsigned int maxGridY = deviceProp.maxGridSize[1];
  unsigned int maxGridZ = deviceProp.maxGridSize[2];

  INFO("Device: " << deviceProp.name);
  INFO("maxGridSize[0]: " << maxGridX);
  INFO("maxGridSize[1]: " << maxGridY);
  INFO("maxGridSize[2]: " << maxGridZ);

  SECTION("Launch kernel with gridDim.x == maxGridDimX") {
    hipLaunchKernelGGL(emptyKernel, dim3(maxGridX, 1, 1), dim3(1, 1, 1), 0, 0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Launch kernel with gridDim.y == maxGridDimY") {
    hipLaunchKernelGGL(emptyKernel, dim3(1, maxGridY, 1), dim3(1, 1, 1), 0, 0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Launch kernel with gridDim.z == maxGridDimZ") {
    hipLaunchKernelGGL(emptyKernel, dim3(1, 1, maxGridZ), dim3(1, 1, 1), 0, 0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }
}

TEST_CASE("Unit_hipGridLaunch_MaxGridDim_GetDeviceAttribute") {
  const unsigned int maxGridX = GetDeviceAttribute(hipDeviceAttributeMaxGridDimX, 0);
  const unsigned int maxGridY = GetDeviceAttribute(hipDeviceAttributeMaxGridDimY, 0);
  const unsigned int maxGridZ = GetDeviceAttribute(hipDeviceAttributeMaxGridDimZ, 0);

  INFO("maxGridDimX: " << maxGridX);
  INFO("maxGridDimY: " << maxGridY);
  INFO("maxGridDimZ: " << maxGridZ);

  SECTION("Launch kernel with gridDim.x == maxGridDimX using attribute") {
    hipLaunchKernelGGL(emptyKernel, dim3(maxGridX, 1, 1), dim3(1, 1, 1), 0, 0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Launch kernel with gridDim.y == maxGridDimY using attribute") {
    hipLaunchKernelGGL(emptyKernel, dim3(1, maxGridY, 1), dim3(1, 1, 1), 0, 0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Launch kernel with gridDim.z == maxGridDimZ using attribute") {
    hipLaunchKernelGGL(emptyKernel, dim3(1, 1, maxGridZ), dim3(1, 1, 1), 0, 0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }
}

HIP_TEST_CASE(Unit_hipGridLaunch_ExceedMaxGridDim_Negative) {
  const unsigned int maxGridX = GetDeviceAttribute(hipDeviceAttributeMaxGridDimX, 0);
  const unsigned int maxGridY = GetDeviceAttribute(hipDeviceAttributeMaxGridDimY, 0);
  const unsigned int maxGridZ = GetDeviceAttribute(hipDeviceAttributeMaxGridDimZ, 0);

  SECTION("Launch kernel with gridDim.x > maxGridDimX returns error") {
    // Only test if we can exceed the max (avoid overflow)
    if (maxGridX < UINT_MAX) {
      hipLaunchKernelGGL(emptyKernel, dim3(maxGridX + 1, 1, 1), dim3(1, 1, 1), 0, 0);
      hipError_t err = hipGetLastError();
      REQUIRE(err == hipErrorInvalidConfiguration);
    }
  }

  SECTION("Launch kernel with gridDim.y > maxGridDimY returns error") {
    if (maxGridY < UINT_MAX) {
      hipLaunchKernelGGL(emptyKernel, dim3(1, maxGridY + 1, 1), dim3(1, 1, 1), 0, 0);
      hipError_t err = hipGetLastError();
      REQUIRE(err == hipErrorInvalidConfiguration);
    }
  }

  SECTION("Launch kernel with gridDim.z > maxGridDimZ returns error") {
    if (maxGridZ < UINT_MAX) {
      hipLaunchKernelGGL(emptyKernel, dim3(1, 1, maxGridZ + 1), dim3(1, 1, 1), 0, 0);
      hipError_t err = hipGetLastError();
      REQUIRE(err == hipErrorInvalidConfiguration);
    }
  }
}

/**
 * End doxygen group KernelTest.
 * @}
 */
