/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_process.hh>
#include <cstring>
int main(int argc, char** argv) {
  if (argc < 0) {
    return -1;
  }
  int testPassed = 0;
  std::string uuid = argv[1];
  hip::SpawnProc proc("chkUUIDInGrandChild_Exe", true);
  std::string t_uuid = uuid.substr(4, 19);
  if (proc.run(t_uuid) == 1) {
    testPassed = 1;
  } else {
    testPassed = 0;
  }
  return testPassed;
}
