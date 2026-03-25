/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <cstring>
#include <string>
#include <vector>
int main(int argc, char** argv) {
  if (argc < 0) {
    return -1;
  }
  int testPassed = 0;
  std::string s = argv[1];
  std::string delimiter = ",";

  size_t pos = 0;
  std::vector<std::string> token;
  while ((pos = s.find(delimiter)) != std::string::npos) {
    token.push_back(s.substr(4, 16));
    s.erase(0, pos + delimiter.length());
  }
  token.push_back(s.substr(4, 16));
  int devCount = 0;
  hipError_t localError;
  localError = hipGetDeviceCount(&devCount);
  if (localError == hipSuccess) {
    printf("HIP Api returned hipSuccess");
  }
  hipDevice_t device;
  for (int i = 0; i < devCount; i++) {
    localError = hipSetDevice(i);
    if (localError == hipSuccess) {
      printf("HIP Api returned hipSuccess");
    }
    localError = hipDeviceGet(&device, i);
    if (localError == hipSuccess) {
      printf("HIP Api returned hipSuccess");
    }
    hipUUID d_uuid{0};
    localError = hipDeviceGetUuid(&d_uuid, device);
    if (localError == hipSuccess) {
      printf("HIP Api returned hipSuccess");
    }
    std::string uuid = token[i];
    if (memcmp(d_uuid.bytes, uuid.c_str(), 16) == 0) {
      testPassed += 1;
    }
  }
  if (testPassed == devCount) {
    return 1;
  } else {
    return 0;
  }
  return testPassed;
}
