/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * hipGetDeviceCount tests
 * Scenario: Validates the value of numDevices when devices are hidden.
 */

#include <hip_test_common.hh>
#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>

#define MAX_SIZE 30
#define VISIBLE_DEVICE 0

/**
 * Validate behavior of hipGetDeviceCount for masked devices.
 */
TEST_CASE(Unit_hipGetDeviceCount_MaskedDevices) {
  int numDevices = 0;
  char visibleDeviceString[MAX_SIZE] = {};
  snprintf(visibleDeviceString, MAX_SIZE, "%d", VISIBLE_DEVICE);

#ifdef __HIP_PLATFORM_NVIDIA__
  unsetenv("CUDA_VISIBLE_DEVICES");
  setenv("CUDA_VISIBLE_DEVICES", visibleDeviceString, 1);
#else
  unsetenv("ROCR_VISIBLE_DEVICES");
  unsetenv("HIP_VISIBLE_DEVICES");
  setenv("ROCR_VISIBLE_DEVICES", visibleDeviceString, 1);
  setenv("HIP_VISIBLE_DEVICES", visibleDeviceString, 1);
#endif

  HIP_CHECK(hipGetDeviceCount(&numDevices));
  REQUIRE(numDevices == 1);
}
#endif
