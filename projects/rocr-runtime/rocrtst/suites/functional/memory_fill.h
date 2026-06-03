/*
* Copyright © Advanced Micro Devices, Inc., or its affiliates.
*
* SPDX-License-Identifier: MIT
*/

#ifndef ROCRTST_SUITES_FUNCTIONAL_MEMORY_FILL_H_
#define ROCRTST_SUITES_FUNCTIONAL_MEMORY_FILL_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

// Memory Fill Test type
enum MemeoryFill {
  ByteBasic,  // For BasicByte test
  ByteUnAligned,  // For Unaligned tests
  NoTest
};

class MemoryFillTest : public TestBase {
 public:
  MemoryFillTest();
  MemoryFillTest(MemeoryFill type);

  // @Brief: Destructor for the MemoryFillTest class
  virtual ~MemoryFillTest();

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

  // @Brief: Tests error conditions for memory fill APIs
  void TestMemoryFillErrors();
  void MemoryFillByteTest();
 private:
   // Test 2: Fill GPU memory (if accessible)
  uint8_t* gpuBuf = NULL;
  uint8_t* g_gpuBuf = NULL;
  uint8_t* staging = NULL;
  hsa_signal_t signal;

  
  // Test 1: Fill system memory
  uint8_t* sysBuf = NULL;

  bool resources_free = false;
   MemeoryFill testtype_;
    // @Brief: Tests hsa_amd_memory_fill_byte (byte-level fills)
  void TestMemoryFillByte(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent);

  // @Brief: Tests byte fill with unaligned addresses and odd sizes
  void TestMemoryFillByteUnaligned(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent);
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_MEMORY_FILL_H_
