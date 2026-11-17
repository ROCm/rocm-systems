/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ROCRTST_SUITES_FUNCTIONAL_DEVICE_MEM_NO_HOST_ACCESS_H_
#define ROCRTST_SUITES_FUNCTIONAL_DEVICE_MEM_NO_HOST_ACCESS_H_

#include <string>

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

// Validates the HSA_DEVICE_MEM_NO_HOST_ACCESS runtime switch. ROCr reads the
// variable during hsa_init(), so the test sets or unsets it itself in SetUp()
// (before TestBase::SetUp() initializes the runtime) and restores the previous
// environment state in the destructor, so the mode does not depend on the
// caller's environment. Two tests are registered, selected by the constructor
// argument:
//   * Device_Mem_No_Host_Access_Enabled_Test (true): variable set to 1. For
//     every GPU device-local global pool that allows runtime allocation
//     (coarse, fine and extended-scope fine grained), CPU access to the pool
//     must be reported NEVER_ALLOWED, the pool must not be signal capable,
//     the allocation must not carry the HOST_ACCESS pointer-info flag, CPU
//     reads and writes must fault (SIGSEGV), and GPU-side fill/copy must work.
//   * Device_Mem_No_Host_Access_Default_Test (false): variable unset. CPU
//     reads and writes must succeed, except for pools whose allocations are
//     not host mapped (HOST_ACCESS pointer-info flag clear, e.g. small-BAR
//     dGPUs), which are skipped. The pool ACCESS enum is not used for that
//     decision because it reports DISALLOWED_BY_DEFAULT on small BAR.
class DeviceMemNoHostAccess : public TestBase {
 public:
  explicit DeviceMemNoHostAccess(bool no_host_access);
  virtual ~DeviceMemNoHostAccess();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

  // @Brief: Check CPU access to device memory faults / works as dictated by
  // HSA_DEVICE_MEM_NO_HOST_ACCESS.
  void CpuAccessTest(void);

 private:
  bool no_host_access_;     // Mode under test.
  bool env_changed_;        // True once SetUp modified the environment.
  bool saved_env_was_set_;  // Original state of HSA_DEVICE_MEM_NO_HOST_ACCESS.
  std::string saved_env_value_;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_DEVICE_MEM_NO_HOST_ACCESS_H_
