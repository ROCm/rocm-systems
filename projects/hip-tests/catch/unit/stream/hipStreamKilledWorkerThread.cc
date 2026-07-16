/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

/**
 * Test Description
 * ------------------------
 *  - A worker thread killed during teardown should not hang the process.
 * Test source
 * ------------------------
 *  - catch\unit\stream\hipStreamKilledWorkerThread.cc
 * Test requirements
 * ------------------------
 *  - Windows
 */
HIP_TEST_CASE(Unit_hipStreamKilledWorkerThread) {
  hip::SpawnProc proc("hipStreamKilledWorkerThread_exe", true);
  REQUIRE(proc.run() == 0);
}
