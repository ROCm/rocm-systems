/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <string>
#include <cstring>

int main() {
  hipDevice_t device;
  int devCount = 0;
  hipError_t localError;
  localError = hipGetDeviceCount(&devCount);
  if (localError != hipSuccess) {
    printf("hipGetDeviceCount is failed.");
    return 1;
  }
  std::string uuid = "";
  for (int i = 0; i < devCount; i++) {
    localError = hipSetDevice(i);
    if (localError != hipSuccess) {
      printf("hipSetDevice is failed.");
      return 1;
    }
    localError = hipDeviceGet(&device, i);
    if (localError != hipSuccess) {
      printf("hipDeviceGet is failed.");
      return 1;
    }
    hipUUID d_uuid{0};
    localError = hipDeviceGetUuid(&d_uuid, device);
    if (localError != hipSuccess) {
      printf("hipDeviceGetUuid is failed.");
      return 1;
    }
    if (i == devCount - 1) {
      uuid = uuid + "GPU-" + d_uuid.bytes;
    } else {
      uuid = uuid + "GPU-" + d_uuid.bytes + ",";
    }
  }
  std::cout << uuid;
  return 0;
}
