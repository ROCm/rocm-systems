/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * 
 * SPDX-License-Identifier: NCSA
 */
#ifndef ROCRTST_SUITES_FUNCTIONAL_GPU_COREDUMP_H_
#define ROCRTST_SUITES_FUNCTIONAL_GPU_COREDUMP_H_

#include <string>
#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

class GpuCoreDumpTest : public TestBase {
 public:
  GpuCoreDumpTest();

  // @Brief: Destructor for test case of GpuCoreDumpTest
  virtual ~GpuCoreDumpTest();

  // @Brief: Setup the environment for measurement
  virtual void SetUp();

  // @Brief: Core measurement execution
  virtual void Run();

  // @Brief: Clean up and retrieve the resource
  virtual void Close();

  // @Brief: Display results
  virtual void DisplayResults() const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);

  // Test cases
  void TestDefaultPattern(void);
  void TestCustomPattern(void);
  void TestDisableFlag(void);
  void TestPatternSubstitution(void);
  void TestInvalidPath(void);

 private:
  // Helper to dispatch the faulting kernel
  void DispatchFaultingKernel();
  
  // Helper to verify core dump file exists and is valid
  bool VerifyCoreDumpFile(const std::string& filename);
  
  // Helper to check if file is a valid GPU core dump (ELF format)
  bool IsValidGPUCoreDump(const std::string& filename);
  
  // Helper to clean up core dump files
  void CleanupCoreDumps(const std::string& pattern);
  
  std::string test_dir_;
  void* a_buffer_ = nullptr;
  void* b_buffer_ = nullptr;
  void* c_buffer_ = nullptr;
  void* d_buffer_ = nullptr;
  void* e_buffer_ = nullptr;
  struct rlimit original_rlimit_;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_GPU_COREDUMP_H_
