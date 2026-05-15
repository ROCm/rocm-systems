/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include "hip/hip_runtime_api.h"
#include <hip_test_context.hh>
#include <stdlib.h>

bool UNSETENV(std::string var) {
  int result = -1;
#ifdef __unix__
  result = unsetenv(var.c_str());
#else
  result = _putenv((var + '=').c_str());
#endif
  return (result == 0) ? true : false;
}

bool SETENV(std::string var, std::string value) {
  int result = -1;
#ifdef __unix__
  result = setenv(var.c_str(), value.c_str(), 1);
#else
  result = _putenv((var + '=' + value).c_str());
#endif
  return (result == 0) ? true : false;
}

void inline hideDevices(const char* devices) {
#if HT_NVIDIA
  SETENV("CUDA_VISIBLE_DEVICES", devices);
#else
  SETENV("HIP_VISIBLE_DEVICES", devices);
  SETENV("ROCR_VISIBLE_DEVICES", devices);
#endif
}

void inline unhideAllDevices() {
#if HT_NVIDIA
  UNSETENV("CUDA_VISIBLE_DEVICES");
#else
  UNSETENV("HIP_VISIBLE_DEVICES");
  UNSETENV("ROCR_VISIBLE_DEVICES");
#endif
}

/**
 * @brief Runs hipDeviceGetP2PAttribute with srcDevice = 0 and dstDevice = 1
 *        Expects 1 command line arg, which is the Device Visible String
 *
 * @return the error code returned by hipDeviceGetP2PAttribute
 */
int main(int argc, char** argv) {
  int value;
  const int srcDevice = 0;
  const int dstDevice = 1;
  const hipDeviceP2PAttr validAttr = hipDevP2PAttrAccessSupported;

  if (argc == 2) {
    hideDevices(argv[1]);
  }

  hipError_t error = hipDeviceGetP2PAttribute(&value, validAttr, srcDevice, dstDevice);

  if (argc == 2) {
    unhideAllDevices();
  }

  return error;
}
