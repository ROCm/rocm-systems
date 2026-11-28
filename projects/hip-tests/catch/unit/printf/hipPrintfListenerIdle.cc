/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

#include <sstream>
#include <string>

/**
 * Test Description
 * ------------------------
 *    - The first kernel that uses printf (or a device-side assert) starts the hostcall listener
 *      thread, which then lives for the rest of the process. While no kernel calls the host, the
 *      process must stay idle. When the runtime has no interrupt signals (WSL,
 *      HSA_ENABLE_INTERRUPT=0) the listener used to spin a CPU core forever
 *      (ROCm/rocm-systems#15002). The companion executable runs with HSA_ENABLE_INTERRUPT=0 in
 *      its environment from the start, so the test covers that path on any Linux machine.
 * ------------------------
 *    - unit/printf/hipPrintfListenerIdle.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_Printf_ListenerIdleCpu) {
  CHECK_PCIE_ATOMIC_SUPPORT

  // In the child's environment from its start, so it is in place even if the runtime were
  // initialized during static registration.
  hip::SpawnProc proc("hipPrintfListenerIdle_exe", true);
  proc.setEnv("HSA_ENABLE_INTERRUPT", "0");
  const int exit_code = proc.run();
  REQUIRE(exit_code == 0);

  std::istringstream output(proc.getOutput());
  double percent = -1;
  int serviced = 0;
  for (std::string line; std::getline(output, line);) {
    if (line == "hostcall serviced") {
      ++serviced;
    } else if (line.rfind("idle_cpu_percent ", 0) == 0) {
      percent = std::stod(line.substr(17));
    }
  }
  REQUIRE(serviced == 1);
  INFO("idle process used " << percent << "% of a core with HSA_ENABLE_INTERRUPT=0");
  REQUIRE(percent >= 0);
  REQUIRE(percent <= 30);
}
