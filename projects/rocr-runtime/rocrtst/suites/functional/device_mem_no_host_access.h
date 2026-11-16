/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ROCRTST_SUITES_FUNCTIONAL_DEVICE_MEM_NO_HOST_ACCESS_H_
#define ROCRTST_SUITES_FUNCTIONAL_DEVICE_MEM_NO_HOST_ACCESS_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

// Validates the HSA_DEVICE_MEM_NO_HOST_ACCESS runtime switch. ROCr reads the
// variable during hsa_init(), which the rocrtst harness has already performed
// by the time a test runs, so this test cannot toggle the mode itself. It
// instead behaves according to the variable as set in the environment of the
// rocrtst process:
//   * set (truthy): CPU access to a GPU coarse-grain device pool allocation
//     must fault (SIGSEGV), the CPU->pool access query must report
//     NEVER_ALLOWED, the pool must not be signal capable, and GPU-side
//     fill/copy must still work;
//   * unset: CPU access must succeed, unless the pool is not CPU accessible
//     (small BAR), in which case the test is skipped.
// Run the test twice (with and without the variable) to cover both modes.
class DeviceMemNoHostAccess : public TestBase {
 public:
  DeviceMemNoHostAccess();
  virtual ~DeviceMemNoHostAccess();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

  // @Brief: Check CPU access to device memory faults / works as dictated by
  // HSA_DEVICE_MEM_NO_HOST_ACCESS.
  void CpuAccessTest(void);
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_DEVICE_MEM_NO_HOST_ACCESS_H_
