/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/*
 * Square each element in the array A and write to array C.
 */
template <typename T> __global__ void vector_square(T* C_d, const T* A_d, size_t N) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < N; i += stride) {
    C_d[i] = A_d[i] * A_d[i];
  }
}

TEST_CASE(Unit_test_compressed_codeobject) {
  float *A_d, *C_d;
  float *A_h, *C_h;
  size_t N = 1000000;
  size_t Nbytes = N * sizeof(float);
  static int device = 0;
  HIP_CHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
  printf("info: running on device %s\n", props.name);
#ifdef __HIP_PLATFORM_AMD__
  printf("info: architecture on AMD GPU device is: %s\n", props.gcnArchName);
#endif
  printf("info: allocate host mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
  A_h = (float*)malloc(Nbytes);
  HIP_CHECK(A_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  C_h = (float*)malloc(Nbytes);
  HIP_CHECK(C_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 1.618f + i;
  }
  printf("info: allocate device mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));

  printf("info: copy Host2Device\n");
  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

  const unsigned blocks = 512;
  const unsigned threadsPerBlock = 256;

  printf("info: launch 'vector_square' kernel\n");
  hipLaunchKernelGGL(vector_square, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N);

  printf("info: copy Device2Host\n");
  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  printf("info: check result\n");
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      HIP_CHECK(hipErrorUnknown);
    }
  }
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
  free(A_h);
  free(C_h);
  printf("PASSED!\n");
  REQUIRE(true);
}
