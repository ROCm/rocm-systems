/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

/**
 * @addtogroup hipExtStreamCreateWithCUMask hipExtStreamCreateWithCUMask
 * @{
 * @ingroup StreamTest
 */

/**
 * Test Description
 * ------------------------
 *  - Runs unit/stream/hipStreamCUMaskTeardown_exe.cc, which creates and destroys streams with a
 *    CU mask while a tracer callback is registered. The runtime's async signal handler may then
 *    release the last reference to a stream; its hardware queue must not be destroyed on that
 *    thread, which would wait for itself. The helper fails on a hang (60 s alarm) or when stream
 *    creation fails because queues were never destroyed.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamCUMaskTeardown.cc
 * Test requirements
 * ------------------------
 *  - Linux
 *  - HIP_VERSION >= 7.14
 */
HIP_TEST_CASE(Unit_hipExtStreamCreateWithCUMask_TeardownWithTracer) {
  hip::SpawnProc proc("hipStreamCUMaskTeardown_exe", true);
  const int result = proc.run();
  INFO("Output from process : " << proc.getOutput());
  REQUIRE(result == 0);
}

/**
 * End doxygen group StreamTest.
 * @}
 */
