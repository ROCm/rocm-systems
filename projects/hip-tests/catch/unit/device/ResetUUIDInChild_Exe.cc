/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_process.hh>

#ifdef _WIN64
#define setenv(x, y, z) _putenv_s(x, y)
#endif

int main() {
  int testPassed = 0;
  setenv("HIP_VISIBLE_DEVICES", "", 1);
  hip::SpawnProc proc("setuuidGetDevCount", true);
  if (proc.run() == 0) {
    testPassed = 1;
  } else {
    testPassed = 0;
  }
  unsetenv("HIP_VISIBLE_DEVICES");
  return testPassed;
}
