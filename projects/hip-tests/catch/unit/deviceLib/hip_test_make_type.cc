/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <hip_test_common.hh>

#define WIDTH 8
#define HEIGHT 8

#define NUM (WIDTH * HEIGHT)

#define THREADS_PER_BLOCK_X 8
#define THREADS_PER_BLOCK_Y 8
#define THREADS_PER_BLOCK_Z 1


__global__ void vectoradd_char1(char1* a, const char1* bm, const char1* cm, int width, int height) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int i = y * width + x;
  if (i < (width * height)) {
    a[i] = make_char1(bm[i].x) + make_char1(cm[i].x);
  }
}

__global__ void vectoradd_char2(char2* a, const char2* bm, const char2* cm, int width, int height) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int i = y * width + x;
  if (i < (width * height)) {
    a[i] = make_char2(bm[i].x, bm[i].y) + make_char2(cm[i].x, cm[i].y);
  }
}

__global__ void vectoradd_char3(char3* a, const char3* bm, const char3* cm, int width, int height) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int i = y * width + x;
  if (i < (width * height)) {
    a[i] = make_char3(bm[i].x, bm[i].y, bm[i].z) + make_char3(cm[i].x, cm[i].y, cm[i].z);
  }
}
__global__ void vectoradd_char4(char4* a, const char4* bm, const char4* cm, int width, int height) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int i = y * width + x;
  if (i < (width * height)) {
    a[i] = make_char4(bm[i].x, bm[i].y, bm[i].z, bm[i].w) +
           make_char4(cm[i].x, cm[i].y, cm[i].z, cm[i].w);
  }
}

template <typename T> bool dataTypesRunChar1() {
  T* hostA;
  T* hostB;
  T* hostC;

  T* deviceA;
  T* deviceB;
  T* deviceC;

  int i;
  int errors;

  hostA = reinterpret_cast<T*>(malloc(NUM * sizeof(T)));
  hostB = reinterpret_cast<T*>(malloc(NUM * sizeof(T)));
  hostC = reinterpret_cast<T*>(malloc(NUM * sizeof(T)));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = (T)i;
    hostC[i] = (T)i;
  }

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&deviceA), NUM * sizeof(T)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&deviceB), NUM * sizeof(T)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&deviceC), NUM * sizeof(T)));

  HIP_CHECK(hipMemcpy(deviceB, hostB, NUM * sizeof(T), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(deviceC, hostC, NUM * sizeof(T), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(HIP_KERNEL_NAME(vectoradd_char1),
                     dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                     dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, deviceA, deviceB,
                     deviceC, WIDTH, HEIGHT);

  HIP_CHECK(hipMemcpy(hostA, deviceA, NUM * sizeof(T), hipMemcpyDeviceToHost));

  bool ret = false;
  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != (hostB[i] + hostC[i])) {
      errors++;
    }
  }
  if (errors != 0) {
    ret = false;
  } else {
    ret = true;
  }

  HIP_CHECK(hipFree(deviceA));
  HIP_CHECK(hipFree(deviceB));
  HIP_CHECK(hipFree(deviceC));

  free(hostA);
  free(hostB);
  free(hostC);

  return ret;
}

template <typename T> bool dataTypesRunChar2() {
  T* hostA;
  T* hostB;
  T* hostC;

  T* deviceA;
  T* deviceB;
  T* deviceC;

  int i;
  int errors;

  hostA = reinterpret_cast<T*>(malloc(NUM * sizeof(T)));
  hostB = reinterpret_cast<T*>(malloc(NUM * sizeof(T)));
  hostC = reinterpret_cast<T*>(malloc(NUM * sizeof(T)));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = (T)i;
    hostC[i] = (T)i;
  }

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&deviceA), NUM * sizeof(T)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&deviceB), NUM * sizeof(T)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&deviceC), NUM * sizeof(T)));

  HIP_CHECK(hipMemcpy(deviceB, hostB, NUM * sizeof(T), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(deviceC, hostC, NUM * sizeof(T), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(HIP_KERNEL_NAME(vectoradd_char2),
                     dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                     dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, deviceA, deviceB,
                     deviceC, WIDTH, HEIGHT);

  HIP_CHECK(hipMemcpy(hostA, deviceA, NUM * sizeof(T), hipMemcpyDeviceToHost));

  bool ret = false;
  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != (hostB[i] + hostC[i])) {
      errors++;
    }
  }
  if (errors != 0) {
    ret = false;
  } else {
    ret = true;
  }

  HIP_CHECK(hipFree(deviceA));
  HIP_CHECK(hipFree(deviceB));
  HIP_CHECK(hipFree(deviceC));

  free(hostA);
  free(hostB);
  free(hostC);

  return ret;
}

template <typename T> bool dataTypesRunChar3() {
  T* hostA;
  T* hostB;
  T* hostC;

  T* deviceA;
  T* deviceB;
  T* deviceC;

  int i;
  int errors;

  hostA = reinterpret_cast<T*>(malloc(NUM * sizeof(T)));
  hostB = reinterpret_cast<T*>(malloc(NUM * sizeof(T)));
  hostC = reinterpret_cast<T*>(malloc(NUM * sizeof(T)));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = (T)i;
    hostC[i] = (T)i;
  }

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&deviceA), NUM * sizeof(T)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&deviceB), NUM * sizeof(T)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&deviceC), NUM * sizeof(T)));

  HIP_CHECK(hipMemcpy(deviceB, hostB, NUM * sizeof(T), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(deviceC, hostC, NUM * sizeof(T), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(HIP_KERNEL_NAME(vectoradd_char3),
                     dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                     dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, deviceA, deviceB,
                     deviceC, WIDTH, HEIGHT);

  HIP_CHECK(hipMemcpy(hostA, deviceA, NUM * sizeof(T), hipMemcpyDeviceToHost));

  bool ret = false;
  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != (hostB[i] + hostC[i])) {
      errors++;
    }
  }
  if (errors != 0) {
    ret = false;
  } else {
    ret = true;
  }
  HIP_CHECK(hipFree(deviceA));
  HIP_CHECK(hipFree(deviceB));
  HIP_CHECK(hipFree(deviceC));

  free(hostA);
  free(hostB);
  free(hostC);

  return ret;
}

template <typename T> bool dataTypesRunChar4() {
  char4* hostA;
  char4* hostB;
  char4* hostC;

  char4* deviceA;
  char4* deviceB;
  char4* deviceC;

  int i;
  int errors;

  hostA = reinterpret_cast<T*>(malloc(NUM * sizeof(T)));
  hostB = reinterpret_cast<T*>(malloc(NUM * sizeof(T)));
  hostC = reinterpret_cast<T*>(malloc(NUM * sizeof(T)));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = (T)i;
    hostC[i] = (T)i;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&deviceA), NUM * sizeof(T)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&deviceB), NUM * sizeof(T)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&deviceC), NUM * sizeof(T)));

  HIP_CHECK(hipMemcpy(deviceB, hostB, NUM * sizeof(T), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(deviceC, hostC, NUM * sizeof(T), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(HIP_KERNEL_NAME(vectoradd_char4),
                     dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                     dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, deviceA, deviceB,
                     deviceC, WIDTH, HEIGHT);

  HIP_CHECK(hipMemcpy(hostA, deviceA, NUM * sizeof(T), hipMemcpyDeviceToHost));

  bool ret = false;
  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != (hostB[i] + hostC[i])) {
      errors++;
    }
  }
  if (errors != 0) {
    ret = false;
  } else {
    ret = true;
  }
  HIP_CHECK(hipFree(deviceA));
  HIP_CHECK(hipFree(deviceB));
  HIP_CHECK(hipFree(deviceC));

  free(hostA);
  free(hostB);
  free(hostC);

  return ret;
}

HIP_TEST_CASE(Unit_Test_makechar_functionality) {
  bool errors;

  errors = dataTypesRunChar1<char1>() && dataTypesRunChar2<char2>() && dataTypesRunChar3<char3>() &&
           dataTypesRunChar4<char4>();

  REQUIRE(errors == true);
}
