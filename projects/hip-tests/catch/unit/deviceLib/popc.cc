/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <cstdio>

constexpr int WIDTH = 16;
constexpr int HEIGHT = 16;
constexpr int NUM = WIDTH * HEIGHT;

constexpr int THREADS_PER_BLOCK_X = 8;
constexpr int THREADS_PER_BLOCK_Y = 8;

// CPU implementation of popcount
template <typename T> unsigned int popcountCPU(T value) {
  unsigned int ret = 0;
  while (value) {
    if (value & 0x1) ++ret;
    value >>= 1;
  }
  return ret;
}

__global__ void HIP_kernel(unsigned int* a, unsigned int* b, unsigned int* c,
                           unsigned long long int* d, int width, int height) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int i = y * width + x;
  if (i < (width * height)) {
    a[i] = __popc(b[i]);
    c[i] = __popcll(d[i]);
  }
}


HIP_TEST_CASE(Unit_popc) {
  using namespace std;
  unsigned int* hostA;
  unsigned int* hostB;
  unsigned int* hostC;
  unsigned long long int* hostD;

  unsigned int* deviceA;
  unsigned int* deviceB;
  unsigned int* deviceC;
  unsigned long long int* deviceD;

  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  INFO("System minor : " << devProp.minor);
  INFO("System major : " << devProp.major);
  INFO("agent prop name : " << devProp.name);

  INFO("hip Device prop succeeded");


  int i;
  int errors;

  hostA = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostB = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostC = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostD = (unsigned long long int*)malloc(NUM * sizeof(unsigned long long int));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = i;
    hostD[i] = 1099511627776 - i;
  }

  HIP_CHECK(hipMalloc((void**)&deviceA, NUM * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc((void**)&deviceB, NUM * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc((void**)&deviceC, NUM * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc((void**)&deviceD, NUM * sizeof(unsigned long long int)));

  HIP_CHECK(hipMemcpy(deviceB, hostB, NUM * sizeof(unsigned int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(deviceD, hostD, NUM * sizeof(unsigned long long int), hipMemcpyHostToDevice));


  hipLaunchKernelGGL(HIP_kernel, dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                     dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, deviceA, deviceB,
                     deviceC, deviceD, WIDTH, HEIGHT);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy(hostA, deviceA, NUM * sizeof(unsigned int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(hostC, deviceC, NUM * sizeof(unsigned int), hipMemcpyDeviceToHost));

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != popcountCPU(hostB[i])) {
      errors++;
    }
  }

  CHECK(errors == 0);

  for (i = 0; i < NUM; i++) {
    if (hostC[i] != popcountCPU(hostD[i])) {
      errors++;
    }
  }


  HIP_CHECK(hipFree(deviceA));
  HIP_CHECK(hipFree(deviceB));
  HIP_CHECK(hipFree(deviceC));
  HIP_CHECK(hipFree(deviceD));

  free(hostA);
  free(hostB);
  free(hostC);
  free(hostD);

  REQUIRE(errors == 0);
}
