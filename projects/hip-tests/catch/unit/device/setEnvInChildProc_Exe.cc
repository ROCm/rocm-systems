/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <string>
#ifdef _WIN64
#define setenv(x, y, z) _putenv_s(x, y)
#define unsetenv(x) _putenv(x)
#endif

int main(int argc, char** argv) {
  if (argc < 0) {
    return -1;
  }
  std::string uuid = argv[1];
  unsetenv("HIP_VISIBLE_DEVICES");
  setenv("HIP_VISIBLE_DEVICES", uuid.c_str(), 1);
  int devCount = 0;
  hipError_t localError;
  localError = hipGetDeviceCount(&devCount);
  if (localError == hipSuccess) {
    printf("HIP Api returned hipSuccess");
  }
  return devCount;
}
