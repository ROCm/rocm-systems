/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ROCRTST_SUITES_FUNCTIONAL_MEMORY_FILL_BYTES_H_
#define ROCRTST_SUITES_FUNCTIONAL_MEMORY_FILL_BYTES_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

class MemoryFillBytesTest : public TestBase {
 public:
  MemoryFillBytesTest();

  // @Brief: Destructor for test case
  virtual ~MemoryFillBytesTest();

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

  // Main test method that runs all sub-tests
  void MemoryFillBytesAllTests();

 private:
  // Test aligned byte fill with various patterns
  void TestAlignedByteFill(hsa_agent_t agent, hsa_amd_memory_pool_t pool);

  // Test unaligned byte fill (non-dword-aligned address)
  void TestUnalignedByteFill(hsa_agent_t agent, hsa_amd_memory_pool_t pool);

  // Test zero fill operation
  void TestZeroFill(hsa_agent_t agent, hsa_amd_memory_pool_t pool);

  // Test byte fill matches dword fill for uniform patterns
  void TestCompareWithDwordFill(hsa_agent_t agent, hsa_amd_memory_pool_t pool);

  // Test edge cases (null pointer, zero size)
  void TestEdgeCases(hsa_agent_t agent, hsa_amd_memory_pool_t pool);

  // Test large fills (1MB)
  void TestLargeFill(hsa_agent_t agent, hsa_amd_memory_pool_t pool);

  // Test fine-grained GPU memory
  void TestFineGrainedGpuMemory(hsa_agent_t agent, hsa_amd_memory_pool_t pool);

  // Test fine-grained system memory (CPU memory accessible by GPU)
  void TestFineGrainedSystemMemory(hsa_agent_t cpu_agent, hsa_agent_t gpu_agent,
                                   hsa_amd_memory_pool_t pool);

  // Test system memory (host-only path via memset)
  void TestSystemMemoryHostPath(hsa_agent_t cpu_agent, hsa_amd_memory_pool_t pool);
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_MEMORY_FILL_BYTES_H_
