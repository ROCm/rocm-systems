/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <string>

/**
 * @addtogroup hipInit hipInit
 * @{
 * @ingroup DriverTest
 */

/**
 * Test Description
 * ------------------------
 *  - In a fresh process, initialize HIP and query every device, then count the hardware queues the
 *    process created. The child intercepts libc's ioctl() and counts the KFD create-queue
 *    requests; init must run in a process of its own to be observed.
 *    -# After init and device queries
 *      - Expected output: no queue
 *  - A queue created at init on every GPU costs each HIP process a doorbell, a runlist slot and
 *    VRAM on all GPUs, used or not (ROCM-34268: hipErrorInvalidValue under 64-way fork on MI350X
 *    CPX; ROCM-34585: out of VRAM on MI325X; ROCM-34588: oversubscribed hardware scheduler with
 *    16 ranks on MI250X).
 *  - A kernel on device 0 afterwards must create a queue; if none is seen, the interception no
 *    longer sees the runtime's ioctl() calls and the test fails.
 * Test source
 * ------------------------
 *  - unit/device/hipInitQueues.cc
 * Test requirements
 * ------------------------
 *  - Linux, AMD
 */
HIP_TEST_CASE(Unit_hipInit_Positive_NoQueuesAtInit) {
  hip::SpawnProc proc("hipInitQueues_Exe", true, true);
  const int status = proc.run();
  const std::string output = proc.getOutput();
  INFO("hipInitQueues_Exe: " << output);
  REQUIRE(status == 0);

  // Only the child's report; stderr, also captured for the INFO above, may hold other "a=b" text.
  const std::set<std::string> keys = {"init_queues", "used_queues"};
  std::map<std::string, int> result;
  std::istringstream tokens(output);
  std::string token;
  while (tokens >> token) {
    auto eq = token.find('=');
    if (eq == std::string::npos || keys.count(token.substr(0, eq)) == 0) continue;
    char* end = nullptr;
    const std::string value = token.substr(eq + 1);
    const long number = std::strtol(value.c_str(), &end, 10);
    if (!value.empty() && *end == '\0') result[token.substr(0, eq)] = static_cast<int>(number);
  }
  REQUIRE(result.size() == keys.size());
  // The kernel creates a queue; if none was seen, the interception is broken and 0 at init would
  // prove nothing.
  REQUIRE(result["used_queues"] > 0);
  REQUIRE(result["init_queues"] == 0);
}

/**
 * End doxygen group DriverTest.
 * @}
 */
