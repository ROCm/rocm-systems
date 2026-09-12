/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_SDMA_MULTICAST_H_
#define ROCRTST_SUITES_FUNCTIONAL_SDMA_MULTICAST_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "suites/test_common/test_base.h"
#include <vector>

// =============================================================================
// SDMA Multicast Test Suite
// =============================================================================
// Tests for SDMA COPY LINEAR MULTICAST packet functionality (GFX13+).
// These tests validate:
// - Decision tree path selection (multicast vs broadcast vs fan-out)
// - Packet chunking for transfers > 1023 bytes
// - Data integrity across multiple destinations
// - Edge cases and boundary conditions
// =============================================================================

class SdmaMulticastTest : public TestBase {
 public:
  SdmaMulticastTest();
  virtual ~SdmaMulticastTest();

  virtual void SetUp() override;
  virtual void Run() override;
  virtual void Close() override;
  virtual void DisplayResults() const override;
  virtual void DisplayTestInfo() override;

  // ---------------------------------------------------------------------------
  // Decision Tree Tests
  // ---------------------------------------------------------------------------
  // Validate that the correct SDMA path is selected based on num_dsts and size.

  // TC-DT-001: 2 destinations should use HW BROADCAST path
  void TestDecisionTree_2Dst_UseBroadcast();

  // TC-DT-002: 3 destinations should use HW MULTICAST path (if supported)
  void TestDecisionTree_3Dst_UseMulticast();

  // TC-DT-003: 10 destinations should use HW MULTICAST path
  void TestDecisionTree_10Dst_UseMulticast();

  // TC-DT-004: 1 destination should use regular LINEAR COPY
  void TestDecisionTree_1Dst_UseLinearCopy();

  // TC-DT-005: 100 destinations should use HW MULTICAST
  void TestDecisionTree_100Dst_UseMulticast();

  // TC-DT-006: Fallback to SW fan-out when HW not supported
  void TestDecisionTree_Fallback_SwFanout();

  // ---------------------------------------------------------------------------
  // Chunking Tests
  // ---------------------------------------------------------------------------
  // Validate that transfers > 1023 bytes are correctly chunked.

  // TC-CHK-001: Size exactly 1023 bytes (max single packet)
  void TestChunking_ExactMax1023();

  // TC-CHK-002: Size 1024 bytes (requires 2 packets: 1023 + 1)
  void TestChunking_1024Bytes();

  // TC-CHK-003: Size 2046 bytes (2 full packets)
  void TestChunking_2046Bytes();

  // TC-CHK-004: Size 4KB (multiple chunks)
  void TestChunking_4KB();

  // TC-CHK-005: Size 1MB with 10 destinations
  void TestChunking_1MB_10Dst();

  // TC-CHK-006: Odd size that doesn't divide evenly
  void TestChunking_OddSize();

  // TC-CHK-007: Very small transfer (1 byte)
  void TestChunking_1Byte();

  // TC-CHK-008: Maximum destinations (1024) with small size
  void TestChunking_MaxDst_SmallSize();

  // ---------------------------------------------------------------------------
  // Data Integrity Tests
  // ---------------------------------------------------------------------------

  // TC-INT-001: Verify all destinations receive correct data
  void TestIntegrity_AllDstReceiveData();

  // TC-INT-002: Verify data pattern preservation
  void TestIntegrity_DataPattern();

  // TC-INT-003: Verify partial overwrite doesn't corrupt adjacent memory
  void TestIntegrity_NoOverwrite();

 private:
  // Helper to allocate GPU-accessible memory
  hsa_status_t AllocateMemory(void** ptr, size_t size, hsa_agent_t agent);

  // Helper to free memory
  void FreeMemory(void* ptr);

  // Helper to verify data at destinations
  bool VerifyData(const std::vector<void*>& dsts, const void* expected, size_t size);

  // Check if multicast is supported on current hardware
  bool IsMulticastSupported();

  // Get GFX major version
  uint32_t GetGfxMajorVersion();

  hsa_agent_t gpu_agent_;
  hsa_agent_t cpu_agent_;
  hsa_amd_memory_pool_t gpu_pool_;
  hsa_amd_memory_pool_t cpu_pool_;
  bool multicast_supported_;
  uint32_t gfx_major_version_;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_SDMA_MULTICAST_H_
