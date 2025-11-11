/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: NCSA
 */
#ifndef ROCRTST_SUITES_FUNCTIONAL_GPU_COREDUMP_H_
#define ROCRTST_SUITES_FUNCTIONAL_GPU_COREDUMP_H_

#include <string>
#include <sys/resource.h>
#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

class GpuCoreDumpTest : public TestBase {
 public:
  GpuCoreDumpTest();
  virtual ~GpuCoreDumpTest();

  // Override to avoid HSA init in parent
  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayTestInfo(void);
  virtual void DisplayResults(void) const;

  // Test cases
  void TestDefaultPattern(void);
  void TestCustomPattern(void);
  void TestDisableFlag(void);
  void TestPatternSubstitution(void);
  void TestInvalidPath(void);

 private:
  // Run faulting kernel in child process (returns child PID)
  pid_t RunFaultingKernelInChild();

  // Verify core dump file exists and is valid
  bool VerifyCoreDumpFile(const std::string& filename);

  // Check if file is a valid GPU core dump (ELF format)
  bool IsValidGPUCoreDump(const std::string& filename);

  // Clean up core dump files
  void CleanupCoreDumps(const std::string& pattern);

  std::string test_dir_;
  struct rlimit original_rlimit_;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_GPU_COREDUMP_H_
