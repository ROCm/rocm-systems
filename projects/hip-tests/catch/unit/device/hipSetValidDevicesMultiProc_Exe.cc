/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Child process used by Unit_hipSetValidDevices_MultiProcess.

// This runs as a freshly exec'd process, so that the HIP runtime is initialized
// cleanly in this process. The parent test process spawns it, runs its own
// hipSetValidDevices path concurrently, and asserts that this process exits
// with status 0.

// Returns 0 on success, 1 on identifying the failing step otherwise.

#include <hip/hip_runtime.h>
#include <cstdio>

#define N 1024

static __global__ void doubleKernel(int* arr, size_t arrSize) {
  size_t offset = blockDim.x * blockIdx.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < arrSize; i += stride) {
    arr[i] += arr[i];
  }
}

static int performOperations() {
  int hostMem[N];
  for (int i = 0; i < N; i++) {
    hostMem[i] = 5;
  }

  int rc = 0;
  int* devMem = nullptr;
  if (hipMalloc(&devMem, N * sizeof(int)) != hipSuccess) return 1;

  if (hipMemcpy(devMem, hostMem, N * sizeof(int), hipMemcpyHostToDevice) != hipSuccess) {
    rc = 1;
    goto cleanup;
  }

  doubleKernel<<<1, N>>>(devMem, N);

  if (hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
    rc = 1;
    goto cleanup;
  }

  for (int i = 0; i < N; i++) {
    if (hostMem[i] != 10) {
      rc = 1;
      goto cleanup;
    }
  }

  cleanup:
    if (devMem != nullptr && hipFree(devMem) != hipSuccess) {
     rc = 1;
    }
    return rc;
}

int main() {
  int device = -1;
  if (hipGetDevice(&device) != hipSuccess || device != 0) {
    printf("child: expected current device 0, got %d\n", device);
    return 1;
  }

  int deviceArr[2] = {0, 1};
  if (hipSetValidDevices(deviceArr, 2) != hipSuccess) {
    printf("child: hipSetValidDevices failed\n");
    return 1;
  }

  if (hipGetDevice(&device) != hipSuccess || device != 0) {
    printf("child: expected current device 0 after set, got %d\n", device);
    return 1;
  }

  int rc = performOperations();
  if (rc != 0) {
    printf("child: performOperations failed with code %d\n", rc);
    return rc;
  }

  return 0;
}
