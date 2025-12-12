/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <cstring>
#include <string>
int main(int argc, char** argv) {
  if (argc < 0) {
    return -1;
  }
  int testPassed = 0;
  hipDevice_t device;
  std::string uuid = argv[1];
  hipError_t localError;
  localError = hipSetDevice(0);
  if (localError == hipSuccess) {
    printf("HIP Api returned hipSuccess");
  }
  localError = hipDeviceGet(&device, 0);
  if (localError == hipSuccess) {
    printf("HIP Api returned hipSuccess");
  }
  hipUUID d_uuid{0};
  localError = hipDeviceGetUuid(&d_uuid, device);
  if (localError == hipSuccess) {
    printf("HIP Api returned hipSuccess");
  }
  if (memcmp(d_uuid.bytes, argv[1], 16) == 0) {
    testPassed = 1;
  } else {
    testPassed = 0;
  }
  return testPassed;
}
