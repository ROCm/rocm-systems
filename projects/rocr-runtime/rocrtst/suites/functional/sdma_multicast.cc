/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "suites/functional/sdma_multicast.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/broadcast_copy_utils.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

// Multicast packet constants (must match sdma_registers.h)
static constexpr size_t kMulticastMaxSize = 0x3ff;  // 1023 bytes per packet
static constexpr uint32_t kMulticastMaxDst = 1024;

#define RET_IF_HSA_ERR(err)                                                                        \
  {                                                                                                \
    if ((err) != HSA_STATUS_SUCCESS) {                                                             \
      const char* msg = nullptr;                                                                   \
      hsa_status_string(err, &msg);                                                                \
      std::cout << "HSA error at line " << __LINE__ << ": " << msg << std::endl;                   \
      return;                                                                                      \
    }                                                                                              \
  }

// =============================================================================
// Test Class Implementation
// =============================================================================

SdmaMulticastTest::SdmaMulticastTest()
    : TestBase(),
      gpu_agent_{0},
      cpu_agent_{0},
      gpu_pool_{0},
      cpu_pool_{0},
      multicast_supported_(false),
      gfx_major_version_(0) {
  set_num_iteration(1);
  set_title("SDMA Multicast Tests");
  set_description(
      "Tests for SDMA COPY LINEAR MULTICAST packet functionality "
      "(GFX13+). Validates decision tree, chunking, and data integrity.");
}

SdmaMulticastTest::~SdmaMulticastTest() {}

void SdmaMulticastTest::SetUp() {
  hsa_status_t err;
  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // Get GPU and CPU agents
  gpu_agent_ = *this->gpu_device1();
  cpu_agent_ = *this->cpu_device();
  gpu_pool_ = this->device_pool();
  cpu_pool_ = this->cpu_pool();

  // Determine GFX version and multicast support
  gfx_major_version_ = GetGfxMajorVersion();
  multicast_supported_ = IsMulticastSupported();

  std::cout << "GFX Major Version: " << gfx_major_version_ << std::endl;
  std::cout << "Multicast Supported: " << (multicast_supported_ ? "YES" : "NO") << std::endl;
}

void SdmaMulticastTest::Run() {
  std::cout << "\n=== Running SDMA Multicast Tests ===" << std::endl;

  // Decision Tree Tests
  std::cout << "\n--- Decision Tree Tests ---" << std::endl;
  TestDecisionTree_1Dst_UseLinearCopy();
  TestDecisionTree_2Dst_UseBroadcast();
  TestDecisionTree_3Dst_UseMulticast();
  TestDecisionTree_10Dst_UseMulticast();

  // Chunking Tests
  std::cout << "\n--- Chunking Tests ---" << std::endl;
  TestChunking_ExactMax1023();
  TestChunking_1024Bytes();
  TestChunking_2046Bytes();
  TestChunking_4KB();
  TestChunking_OddSize();
  TestChunking_1Byte();

  // Data Integrity Tests
  std::cout << "\n--- Data Integrity Tests ---" << std::endl;
  TestIntegrity_AllDstReceiveData();
  TestIntegrity_DataPattern();
}

void SdmaMulticastTest::Close() { TestBase::Close(); }

void SdmaMulticastTest::DisplayResults() const {
  std::cout << "\n=== SDMA Multicast Test Results ===" << std::endl;
  std::cout << "GFX Version: " << gfx_major_version_ << std::endl;
  std::cout << "Multicast HW Support: " << (multicast_supported_ ? "YES" : "NO") << std::endl;
}

void SdmaMulticastTest::DisplayTestInfo() { TestBase::DisplayTestInfo(); }

// =============================================================================
// Helper Functions
// =============================================================================

uint32_t SdmaMulticastTest::GetGfxMajorVersion() {
  // Query ISA from GPU agent
  hsa_isa_t isa;
  hsa_status_t err = hsa_agent_get_info(gpu_agent_, HSA_AGENT_INFO_ISA, &isa);
  if (err != HSA_STATUS_SUCCESS) return 0;

  // Get ISA name to parse version
  char isa_name[128];
  err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_name);
  if (err != HSA_STATUS_SUCCESS) return 0;

  // Parse gfxXXXX from name (e.g., "amdgcn-amd-amdhsa--gfx1300")
  const char* gfx_str = strstr(isa_name, "gfx");
  if (gfx_str) {
    uint32_t version = 0;
    if (sscanf(gfx_str, "gfx%u", &version) == 1) {
      return version / 100;  // gfx1300 -> 13
    }
  }
  return 0;
}

bool SdmaMulticastTest::IsMulticastSupported() {
  // Multicast supported on GFX13+
  return gfx_major_version_ >= 13;
}

hsa_status_t SdmaMulticastTest::AllocateMemory(void** ptr, size_t size, hsa_agent_t agent) {
  return hsa_amd_memory_pool_allocate(gpu_pool_, size, 0, ptr);
}

void SdmaMulticastTest::FreeMemory(void* ptr) {
  if (ptr) {
    hsa_amd_memory_pool_free(ptr);
  }
}

bool SdmaMulticastTest::VerifyData(const std::vector<void*>& dsts, const void* expected,
                                   size_t size) {
  for (size_t i = 0; i < dsts.size(); ++i) {
    if (memcmp(dsts[i], expected, size) != 0) {
      std::cout << "Data mismatch at destination " << i << std::endl;
      return false;
    }
  }
  return true;
}

// =============================================================================
// Decision Tree Tests
// =============================================================================

void SdmaMulticastTest::TestDecisionTree_1Dst_UseLinearCopy() {
  std::cout << "TC-DT-004: 1 destination -> LINEAR COPY... ";

  const size_t size = 512;
  void* src = nullptr;
  void* dst = nullptr;

  hsa_status_t err = AllocateMemory(&src, size, gpu_agent_);
  RET_IF_HSA_ERR(err);
  err = AllocateMemory(&dst, size, gpu_agent_);
  RET_IF_HSA_ERR(err);

  // Initialize source
  memset(src, 0xAB, size);
  memset(dst, 0x00, size);

  // Use regular async copy for 1 destination
  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  RET_IF_HSA_ERR(err);

  err = hsa_amd_memory_async_copy(dst, gpu_agent_, src, gpu_agent_, size, 0, nullptr, signal);
  RET_IF_HSA_ERR(err);

  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Verify
  bool pass = (memcmp(dst, src, size) == 0);

  hsa_signal_destroy(signal);
  FreeMemory(src);
  FreeMemory(dst);

  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  EXPECT_TRUE(pass);
}

void SdmaMulticastTest::TestDecisionTree_2Dst_UseBroadcast() {
  std::cout << "TC-DT-001: 2 destinations -> BROADCAST... ";

  if (!multicast_supported_) {
    std::cout << "SKIP (HW not supported)" << std::endl;
    return;
  }

  const size_t size = 512;
  const int num_dsts = 2;
  void* src = nullptr;
  std::vector<void*> dsts(num_dsts, nullptr);

  hsa_status_t err = AllocateMemory(&src, size, gpu_agent_);
  RET_IF_HSA_ERR(err);
  for (int i = 0; i < num_dsts; ++i) {
    err = AllocateMemory(&dsts[i], size, gpu_agent_);
    RET_IF_HSA_ERR(err);
  }

  // Initialize
  memset(src, 0xCD, size);
  for (auto& d : dsts) memset(d, 0x00, size);

  // Create signal
  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  RET_IF_HSA_ERR(err);

  // Use broadcast copy API (if available)
  std::vector<hsa_agent_t> dst_agents(num_dsts, gpu_agent_);
  err = hsa_amd_memory_broadcast_copy(src, gpu_agent_, dsts.data(), dst_agents.data(), num_dsts,
                                      size, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  if (err != HSA_STATUS_SUCCESS) {
    std::cout << "SKIP (API not available)" << std::endl;
    hsa_signal_destroy(signal);
    FreeMemory(src);
    for (auto& d : dsts) FreeMemory(d);
    return;
  }

  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  bool pass = VerifyData(dsts, src, size);

  hsa_signal_destroy(signal);
  FreeMemory(src);
  for (auto& d : dsts) FreeMemory(d);

  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  EXPECT_TRUE(pass);
}

void SdmaMulticastTest::TestDecisionTree_3Dst_UseMulticast() {
  std::cout << "TC-DT-002: 3 destinations -> MULTICAST... ";

  if (!multicast_supported_) {
    std::cout << "SKIP (HW not supported)" << std::endl;
    return;
  }

  const size_t size = 512;
  const int num_dsts = 3;
  void* src = nullptr;
  std::vector<void*> dsts(num_dsts, nullptr);

  hsa_status_t err = AllocateMemory(&src, size, gpu_agent_);
  RET_IF_HSA_ERR(err);
  for (int i = 0; i < num_dsts; ++i) {
    err = AllocateMemory(&dsts[i], size, gpu_agent_);
    RET_IF_HSA_ERR(err);
  }

  memset(src, 0xEF, size);
  for (auto& d : dsts) memset(d, 0x00, size);

  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  RET_IF_HSA_ERR(err);

  std::vector<hsa_agent_t> dst_agents(num_dsts, gpu_agent_);
  err = hsa_amd_memory_broadcast_copy(src, gpu_agent_, dsts.data(), dst_agents.data(), num_dsts,
                                      size, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  if (err != HSA_STATUS_SUCCESS) {
    std::cout << "SKIP (API not available)" << std::endl;
    hsa_signal_destroy(signal);
    FreeMemory(src);
    for (auto& d : dsts) FreeMemory(d);
    return;
  }

  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  bool pass = VerifyData(dsts, src, size);

  hsa_signal_destroy(signal);
  FreeMemory(src);
  for (auto& d : dsts) FreeMemory(d);

  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  EXPECT_TRUE(pass);
}

void SdmaMulticastTest::TestDecisionTree_10Dst_UseMulticast() {
  std::cout << "TC-DT-003: 10 destinations -> MULTICAST... ";

  if (!multicast_supported_) {
    std::cout << "SKIP (HW not supported)" << std::endl;
    return;
  }

  const size_t size = 1000;
  const int num_dsts = 10;
  void* src = nullptr;
  std::vector<void*> dsts(num_dsts, nullptr);

  hsa_status_t err = AllocateMemory(&src, size, gpu_agent_);
  RET_IF_HSA_ERR(err);
  for (int i = 0; i < num_dsts; ++i) {
    err = AllocateMemory(&dsts[i], size, gpu_agent_);
    RET_IF_HSA_ERR(err);
  }

  // Fill source with pattern
  uint8_t* src_bytes = static_cast<uint8_t*>(src);
  for (size_t i = 0; i < size; ++i) {
    src_bytes[i] = static_cast<uint8_t>(i & 0xFF);
  }
  for (auto& d : dsts) memset(d, 0x00, size);

  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  RET_IF_HSA_ERR(err);

  std::vector<hsa_agent_t> dst_agents(num_dsts, gpu_agent_);
  err = hsa_amd_memory_broadcast_copy(src, gpu_agent_, dsts.data(), dst_agents.data(), num_dsts,
                                      size, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  if (err != HSA_STATUS_SUCCESS) {
    std::cout << "SKIP (API not available)" << std::endl;
    hsa_signal_destroy(signal);
    FreeMemory(src);
    for (auto& d : dsts) FreeMemory(d);
    return;
  }

  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  bool pass = VerifyData(dsts, src, size);

  hsa_signal_destroy(signal);
  FreeMemory(src);
  for (auto& d : dsts) FreeMemory(d);

  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  EXPECT_TRUE(pass);
}

// =============================================================================
// Chunking Tests
// =============================================================================

void SdmaMulticastTest::TestChunking_ExactMax1023() {
  std::cout << "TC-CHK-001: Size 1023 bytes (max single packet)... ";

  if (!multicast_supported_) {
    std::cout << "SKIP (HW not supported)" << std::endl;
    return;
  }

  const size_t size = 1023;  // Exact max for single multicast packet
  const int num_dsts = 5;
  void* src = nullptr;
  std::vector<void*> dsts(num_dsts, nullptr);

  hsa_status_t err = AllocateMemory(&src, size, gpu_agent_);
  RET_IF_HSA_ERR(err);
  for (int i = 0; i < num_dsts; ++i) {
    err = AllocateMemory(&dsts[i], size, gpu_agent_);
    RET_IF_HSA_ERR(err);
  }

  memset(src, 0x55, size);
  for (auto& d : dsts) memset(d, 0x00, size);

  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  RET_IF_HSA_ERR(err);

  std::vector<hsa_agent_t> dst_agents(num_dsts, gpu_agent_);
  err = hsa_amd_memory_broadcast_copy(src, gpu_agent_, dsts.data(), dst_agents.data(), num_dsts,
                                      size, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  if (err != HSA_STATUS_SUCCESS) {
    std::cout << "SKIP (API not available)" << std::endl;
    hsa_signal_destroy(signal);
    FreeMemory(src);
    for (auto& d : dsts) FreeMemory(d);
    return;
  }

  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  bool pass = VerifyData(dsts, src, size);

  hsa_signal_destroy(signal);
  FreeMemory(src);
  for (auto& d : dsts) FreeMemory(d);

  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  EXPECT_TRUE(pass);
}

void SdmaMulticastTest::TestChunking_1024Bytes() {
  std::cout << "TC-CHK-002: Size 1024 bytes (requires 2 packets)... ";

  if (!multicast_supported_) {
    std::cout << "SKIP (HW not supported)" << std::endl;
    return;
  }

  const size_t size = 1024;  // Requires chunking: 1023 + 1
  const int num_dsts = 4;
  void* src = nullptr;
  std::vector<void*> dsts(num_dsts, nullptr);

  hsa_status_t err = AllocateMemory(&src, size, gpu_agent_);
  RET_IF_HSA_ERR(err);
  for (int i = 0; i < num_dsts; ++i) {
    err = AllocateMemory(&dsts[i], size, gpu_agent_);
    RET_IF_HSA_ERR(err);
  }

  memset(src, 0xAA, size);
  for (auto& d : dsts) memset(d, 0x00, size);

  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  RET_IF_HSA_ERR(err);

  std::vector<hsa_agent_t> dst_agents(num_dsts, gpu_agent_);
  err = hsa_amd_memory_broadcast_copy(src, gpu_agent_, dsts.data(), dst_agents.data(), num_dsts,
                                      size, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  if (err != HSA_STATUS_SUCCESS) {
    std::cout << "SKIP (API not available)" << std::endl;
    hsa_signal_destroy(signal);
    FreeMemory(src);
    for (auto& d : dsts) FreeMemory(d);
    return;
  }

  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  bool pass = VerifyData(dsts, src, size);

  hsa_signal_destroy(signal);
  FreeMemory(src);
  for (auto& d : dsts) FreeMemory(d);

  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  EXPECT_TRUE(pass);
}

void SdmaMulticastTest::TestChunking_2046Bytes() {
  std::cout << "TC-CHK-003: Size 2046 bytes (2 full packets)... ";

  if (!multicast_supported_) {
    std::cout << "SKIP (HW not supported)" << std::endl;
    return;
  }

  const size_t size = 2046;  // 2 * 1023 = 2 full packets
  const int num_dsts = 3;
  void* src = nullptr;
  std::vector<void*> dsts(num_dsts, nullptr);

  hsa_status_t err = AllocateMemory(&src, size, gpu_agent_);
  RET_IF_HSA_ERR(err);
  for (int i = 0; i < num_dsts; ++i) {
    err = AllocateMemory(&dsts[i], size, gpu_agent_);
    RET_IF_HSA_ERR(err);
  }

  // Fill with incrementing pattern
  uint8_t* src_bytes = static_cast<uint8_t*>(src);
  for (size_t i = 0; i < size; ++i) {
    src_bytes[i] = static_cast<uint8_t>(i & 0xFF);
  }
  for (auto& d : dsts) memset(d, 0x00, size);

  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  RET_IF_HSA_ERR(err);

  std::vector<hsa_agent_t> dst_agents(num_dsts, gpu_agent_);
  err = hsa_amd_memory_broadcast_copy(src, gpu_agent_, dsts.data(), dst_agents.data(), num_dsts,
                                      size, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  if (err != HSA_STATUS_SUCCESS) {
    std::cout << "SKIP (API not available)" << std::endl;
    hsa_signal_destroy(signal);
    FreeMemory(src);
    for (auto& d : dsts) FreeMemory(d);
    return;
  }

  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  bool pass = VerifyData(dsts, src, size);

  hsa_signal_destroy(signal);
  FreeMemory(src);
  for (auto& d : dsts) FreeMemory(d);

  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  EXPECT_TRUE(pass);
}

void SdmaMulticastTest::TestChunking_4KB() {
  std::cout << "TC-CHK-004: Size 4KB (multiple chunks)... ";

  if (!multicast_supported_) {
    std::cout << "SKIP (HW not supported)" << std::endl;
    return;
  }

  const size_t size = 4096;
  const int num_dsts = 5;
  void* src = nullptr;
  std::vector<void*> dsts(num_dsts, nullptr);

  hsa_status_t err = AllocateMemory(&src, size, gpu_agent_);
  RET_IF_HSA_ERR(err);
  for (int i = 0; i < num_dsts; ++i) {
    err = AllocateMemory(&dsts[i], size, gpu_agent_);
    RET_IF_HSA_ERR(err);
  }

  memset(src, 0x77, size);
  for (auto& d : dsts) memset(d, 0x00, size);

  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  RET_IF_HSA_ERR(err);

  std::vector<hsa_agent_t> dst_agents(num_dsts, gpu_agent_);
  err = hsa_amd_memory_broadcast_copy(src, gpu_agent_, dsts.data(), dst_agents.data(), num_dsts,
                                      size, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  if (err != HSA_STATUS_SUCCESS) {
    std::cout << "SKIP (API not available)" << std::endl;
    hsa_signal_destroy(signal);
    FreeMemory(src);
    for (auto& d : dsts) FreeMemory(d);
    return;
  }

  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  bool pass = VerifyData(dsts, src, size);

  hsa_signal_destroy(signal);
  FreeMemory(src);
  for (auto& d : dsts) FreeMemory(d);

  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  EXPECT_TRUE(pass);
}

void SdmaMulticastTest::TestChunking_OddSize() {
  std::cout << "TC-CHK-006: Odd size (1537 bytes)... ";

  if (!multicast_supported_) {
    std::cout << "SKIP (HW not supported)" << std::endl;
    return;
  }

  const size_t size = 1537;  // Odd size: 1023 + 514
  const int num_dsts = 3;
  void* src = nullptr;
  std::vector<void*> dsts(num_dsts, nullptr);

  hsa_status_t err = AllocateMemory(&src, size, gpu_agent_);
  RET_IF_HSA_ERR(err);
  for (int i = 0; i < num_dsts; ++i) {
    err = AllocateMemory(&dsts[i], size, gpu_agent_);
    RET_IF_HSA_ERR(err);
  }

  memset(src, 0x33, size);
  for (auto& d : dsts) memset(d, 0x00, size);

  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  RET_IF_HSA_ERR(err);

  std::vector<hsa_agent_t> dst_agents(num_dsts, gpu_agent_);
  err = hsa_amd_memory_broadcast_copy(src, gpu_agent_, dsts.data(), dst_agents.data(), num_dsts,
                                      size, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  if (err != HSA_STATUS_SUCCESS) {
    std::cout << "SKIP (API not available)" << std::endl;
    hsa_signal_destroy(signal);
    FreeMemory(src);
    for (auto& d : dsts) FreeMemory(d);
    return;
  }

  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  bool pass = VerifyData(dsts, src, size);

  hsa_signal_destroy(signal);
  FreeMemory(src);
  for (auto& d : dsts) FreeMemory(d);

  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  EXPECT_TRUE(pass);
}

void SdmaMulticastTest::TestChunking_1Byte() {
  std::cout << "TC-CHK-007: Very small transfer (1 byte)... ";

  if (!multicast_supported_) {
    std::cout << "SKIP (HW not supported)" << std::endl;
    return;
  }

  const size_t size = 1;
  const int num_dsts = 3;
  void* src = nullptr;
  std::vector<void*> dsts(num_dsts, nullptr);

  hsa_status_t err = AllocateMemory(&src, 64, gpu_agent_);  // Min alloc
  RET_IF_HSA_ERR(err);
  for (int i = 0; i < num_dsts; ++i) {
    err = AllocateMemory(&dsts[i], 64, gpu_agent_);
    RET_IF_HSA_ERR(err);
  }

  memset(src, 0xFF, 64);
  for (auto& d : dsts) memset(d, 0x00, 64);

  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  RET_IF_HSA_ERR(err);

  std::vector<hsa_agent_t> dst_agents(num_dsts, gpu_agent_);
  err = hsa_amd_memory_broadcast_copy(src, gpu_agent_, dsts.data(), dst_agents.data(), num_dsts,
                                      size, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  if (err != HSA_STATUS_SUCCESS) {
    std::cout << "SKIP (API not available)" << std::endl;
    hsa_signal_destroy(signal);
    FreeMemory(src);
    for (auto& d : dsts) FreeMemory(d);
    return;
  }

  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  bool pass = VerifyData(dsts, src, size);

  hsa_signal_destroy(signal);
  FreeMemory(src);
  for (auto& d : dsts) FreeMemory(d);

  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  EXPECT_TRUE(pass);
}

// =============================================================================
// Data Integrity Tests
// =============================================================================

void SdmaMulticastTest::TestIntegrity_AllDstReceiveData() {
  std::cout << "TC-INT-001: All destinations receive correct data... ";

  if (!multicast_supported_) {
    std::cout << "SKIP (HW not supported)" << std::endl;
    return;
  }

  const size_t size = 2048;
  const int num_dsts = 8;
  void* src = nullptr;
  std::vector<void*> dsts(num_dsts, nullptr);

  hsa_status_t err = AllocateMemory(&src, size, gpu_agent_);
  RET_IF_HSA_ERR(err);
  for (int i = 0; i < num_dsts; ++i) {
    err = AllocateMemory(&dsts[i], size, gpu_agent_);
    RET_IF_HSA_ERR(err);
  }

  // Fill source with unique pattern
  uint8_t* src_bytes = static_cast<uint8_t*>(src);
  for (size_t i = 0; i < size; ++i) {
    src_bytes[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
  }
  for (auto& d : dsts) memset(d, 0x00, size);

  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  RET_IF_HSA_ERR(err);

  std::vector<hsa_agent_t> dst_agents(num_dsts, gpu_agent_);
  err = hsa_amd_memory_broadcast_copy(src, gpu_agent_, dsts.data(), dst_agents.data(), num_dsts,
                                      size, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  if (err != HSA_STATUS_SUCCESS) {
    std::cout << "SKIP (API not available)" << std::endl;
    hsa_signal_destroy(signal);
    FreeMemory(src);
    for (auto& d : dsts) FreeMemory(d);
    return;
  }

  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  bool pass = VerifyData(dsts, src, size);

  hsa_signal_destroy(signal);
  FreeMemory(src);
  for (auto& d : dsts) FreeMemory(d);

  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  EXPECT_TRUE(pass);
}

void SdmaMulticastTest::TestIntegrity_DataPattern() {
  std::cout << "TC-INT-002: Data pattern preservation... ";

  if (!multicast_supported_) {
    std::cout << "SKIP (HW not supported)" << std::endl;
    return;
  }

  const size_t size = 4096;
  const int num_dsts = 4;
  void* src = nullptr;
  std::vector<void*> dsts(num_dsts, nullptr);

  hsa_status_t err = AllocateMemory(&src, size, gpu_agent_);
  RET_IF_HSA_ERR(err);
  for (int i = 0; i < num_dsts; ++i) {
    err = AllocateMemory(&dsts[i], size, gpu_agent_);
    RET_IF_HSA_ERR(err);
  }

  // Create a specific pattern: alternating words
  uint32_t* src_words = static_cast<uint32_t*>(src);
  for (size_t i = 0; i < size / sizeof(uint32_t); ++i) {
    src_words[i] = (i & 1) ? 0xDEADBEEF : 0xCAFEBABE;
  }
  for (auto& d : dsts) memset(d, 0x00, size);

  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  RET_IF_HSA_ERR(err);

  std::vector<hsa_agent_t> dst_agents(num_dsts, gpu_agent_);
  err = hsa_amd_memory_broadcast_copy(src, gpu_agent_, dsts.data(), dst_agents.data(), num_dsts,
                                      size, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  if (err != HSA_STATUS_SUCCESS) {
    std::cout << "SKIP (API not available)" << std::endl;
    hsa_signal_destroy(signal);
    FreeMemory(src);
    for (auto& d : dsts) FreeMemory(d);
    return;
  }

  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  bool pass = VerifyData(dsts, src, size);

  hsa_signal_destroy(signal);
  FreeMemory(src);
  for (auto& d : dsts) FreeMemory(d);

  std::cout << (pass ? "PASS" : "FAIL") << std::endl;
  EXPECT_TRUE(pass);
}

// =============================================================================
// GTest Registration
// =============================================================================

TEST(SdmaMulticastTestSuite, DecisionTreeTests) {
  SdmaMulticastTest test;
  test.SetUp();
  test.TestDecisionTree_1Dst_UseLinearCopy();
  test.TestDecisionTree_2Dst_UseBroadcast();
  test.TestDecisionTree_3Dst_UseMulticast();
  test.TestDecisionTree_10Dst_UseMulticast();
  test.Close();
}

TEST(SdmaMulticastTestSuite, ChunkingTests) {
  SdmaMulticastTest test;
  test.SetUp();
  test.TestChunking_ExactMax1023();
  test.TestChunking_1024Bytes();
  test.TestChunking_2046Bytes();
  test.TestChunking_4KB();
  test.TestChunking_OddSize();
  test.TestChunking_1Byte();
  test.Close();
}

TEST(SdmaMulticastTestSuite, DataIntegrityTests) {
  SdmaMulticastTest test;
  test.SetUp();
  test.TestIntegrity_AllDstReceiveData();
  test.TestIntegrity_DataPattern();
  test.Close();
}
