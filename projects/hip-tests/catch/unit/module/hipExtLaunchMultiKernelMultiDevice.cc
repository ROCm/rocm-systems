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
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

/**
 * @addtogroup hipExtLaunchMultiKernelMultiDevice
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList,
 *                                          int  numDevices, unsigned int  flags)` -
 * Launches kernels on multiple devices and guarantees all specified kernels are dispatched
 * on respective streams before enqueuing any other work on the specified streams from any
 * other threads
 */

/**
 * Test Description
 * ------------------------
 * - Test case to Launche Multiple kernels on single device or multiple devices.
 * Test source
 * ------------------------
 * - catch/unit/module/hipExtLaunchMultiKernelMultiDevice.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */

// Square each element in the array A and write to array C.
#define NUM_KERNEL_ARGS 3
__global__ void vector_square(float* C_d, float* A_d, size_t N) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < N; i += stride) {
    C_d[i] = A_d[i] * A_d[i];
  }
}

TEST_CASE("Unit_hipExtLaunchMultiKernelMultiDevice_Functional") {
  constexpr int MAX_GPUS = 8;
  float *A_d[MAX_GPUS], *C_d[MAX_GPUS];
  float *A_h, *C_h;
  size_t N = 1000000;
  size_t Nbytes = N * sizeof(float);

  int nGpu = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpu));
  if (nGpu < 1) {
    INFO("info: didn't find any GPU!\n");
    REQUIRE(false);
  }
  if (nGpu > MAX_GPUS) {
    nGpu = MAX_GPUS;
  }
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  HIP_CHECK(A_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  HIP_CHECK(C_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 1.618f + i;
  }

  const unsigned blocks = 512;
  const unsigned threadsPerBlock = 256;

  hipStream_t stream[MAX_GPUS];
  for (int i = 0; i < nGpu; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipStreamCreateWithFlags(&stream[i], hipStreamNonBlocking));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, i));
    HIP_CHECK(hipMalloc(&A_d[i], Nbytes));
    HIP_CHECK(hipMalloc(&C_d[i], Nbytes));


    INFO("info: copy Host2Device\n");
    HIP_CHECK(hipMemcpy(A_d[i], A_h, Nbytes, hipMemcpyHostToDevice));
  }

  hipLaunchParams* launchParamsList =
      reinterpret_cast<hipLaunchParams*>(malloc(sizeof(hipLaunchParams) * nGpu));

  void* args[MAX_GPUS * NUM_KERNEL_ARGS];

  for (int i = 0; i < nGpu; i++) {
    args[i * NUM_KERNEL_ARGS] = &C_d[i];
    args[i * NUM_KERNEL_ARGS + 1] = &A_d[i];
    args[i * NUM_KERNEL_ARGS + 2] = &N;
    launchParamsList[i].func = reinterpret_cast<void*>(vector_square);
    launchParamsList[i].gridDim = dim3(blocks);
    launchParamsList[i].blockDim = dim3(threadsPerBlock);
    launchParamsList[i].sharedMem = 0;
    launchParamsList[i].stream = stream[i];
    launchParamsList[i].args = args + i * NUM_KERNEL_ARGS;
  }

  INFO("info: launch vector_square kernel with")
  INFO("hipExtLaunchMultiKernelMultiDevice API\n");
  HIP_CHECK(hipExtLaunchMultiKernelMultiDevice(launchParamsList, nGpu, 0));

  for (int j = 0; j < nGpu; j++) {
    HIP_CHECK(hipStreamSynchronize(stream[j]));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, j));
    INFO("info: copy Device2Host\n");
    HIP_CHECK(hipSetDevice(j));
    HIP_CHECK(hipMemcpy(C_h, C_d[j], Nbytes, hipMemcpyDeviceToHost));

    INFO("info: check result\n");
    for (size_t i = 0; i < N; i++) {
      REQUIRE(fabs(C_h[i] - (A_h[i] * A_h[i])) < 0.00000000001);
    }
  }

  for (int i = 0; i < nGpu; i++) {
    HIP_CHECK(hipFree(A_d[i]));
    HIP_CHECK(hipFree(C_d[i]));
    HIP_CHECK(hipStreamDestroy(stream[i]));
  }

  free(launchParamsList);
  free(A_h);
  free(C_h);
}
