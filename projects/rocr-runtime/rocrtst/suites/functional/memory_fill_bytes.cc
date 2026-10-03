/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#include "suites/functional/memory_fill_bytes.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#define RET_IF_HSA_ERR(err)                                                                        \
  do {                                                                                             \
    hsa_status_t _err = (err);                                                                     \
    if (_err != HSA_STATUS_SUCCESS) {                                                              \
      const char* msg = 0;                                                                         \
      hsa_status_string(_err, &msg);                                                               \
      FAIL() << "HSA API failure at line " << __LINE__ << ", file: " << __FILE__                   \
             << ". Call returned " << _err << " (" << (msg ? msg : "unknown") << ")";              \
      return;                                                                                      \
    }                                                                                              \
  } while (0)

MemoryFillBytesTest::MemoryFillBytesTest(void) : TestBase() {
  set_num_iteration(1);
  set_title("RocR Memory Fill Bytes Tests");
  set_description(
      "This test verifies the hsa_amd_memory_fill_bytes API "
      "which fills memory with a byte value (similar to cudaMemset).");
}

MemoryFillBytesTest::~MemoryFillBytesTest(void) {}

void MemoryFillBytesTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();
  if (test_skipped_) return;

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
}

void MemoryFillBytesTest::Run(void) {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void MemoryFillBytesTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void MemoryFillBytesTest::DisplayResults(void) const {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
}

void MemoryFillBytesTest::Close() { TestBase::Close(); }

void MemoryFillBytesTest::TestAlignedByteFill(hsa_agent_t agent, hsa_amd_memory_pool_t pool) {
  const size_t size = 1024;
  void* ptr = nullptr;

  hsa_status_t err = hsa_amd_memory_pool_allocate(pool, size, 0, &ptr);
  RET_IF_HSA_ERR(err);

  // Fill with 0xAB pattern
  const uint8_t fill_value = 0xAB;
  err = hsa_amd_memory_fill_bytes(ptr, fill_value, size);
  RET_IF_HSA_ERR(err);

  // Verify by checking content
  std::vector<uint8_t> host_buf(size);

  // Copy back to host for verification using HSA copy API
  err = hsa_memory_copy(host_buf.data(), ptr, size);
  RET_IF_HSA_ERR(err);

  for (size_t i = 0; i < size; i++) {
    ASSERT_EQ(host_buf[i], fill_value)
        << "TestAlignedByteFill: Mismatch at index " << i << " expected 0x" << std::hex
        << (int)fill_value << " got 0x" << (int)host_buf[i] << std::dec;
  }

  err = hsa_amd_memory_pool_free(ptr);
  RET_IF_HSA_ERR(err);

  std::cout << "  TestAlignedByteFill: PASSED" << std::endl;
}

void MemoryFillBytesTest::TestUnalignedByteFill(hsa_agent_t agent, hsa_amd_memory_pool_t pool) {
  const size_t alloc_size = 1024;
  const size_t fill_offset = 3;  // Start at unaligned offset
  const size_t fill_size = 100;  // Fill non-multiple-of-4 bytes

  void* base_ptr = nullptr;
  hsa_status_t err = hsa_amd_memory_pool_allocate(pool, alloc_size, 0, &base_ptr);
  RET_IF_HSA_ERR(err);

  // Zero the buffer first
  err = hsa_amd_memory_fill_bytes(base_ptr, 0x00, alloc_size);
  RET_IF_HSA_ERR(err);

  // Fill starting at unaligned offset
  const uint8_t fill_value = 0xCD;
  void* fill_ptr = static_cast<char*>(base_ptr) + fill_offset;
  err = hsa_amd_memory_fill_bytes(fill_ptr, fill_value, fill_size);
  RET_IF_HSA_ERR(err);

  // Verify - copy back to host using HSA copy API
  std::vector<uint8_t> host_buf(alloc_size);

  err = hsa_memory_copy(host_buf.data(), base_ptr, alloc_size);
  RET_IF_HSA_ERR(err);

  // Check prefix zeros
  for (size_t i = 0; i < fill_offset; i++) {
    ASSERT_EQ(host_buf[i], 0x00) << "TestUnalignedByteFill: Prefix at " << i << " should be 0";
  }

  // Check filled region
  for (size_t i = fill_offset; i < fill_offset + fill_size; i++) {
    ASSERT_EQ(host_buf[i], fill_value)
        << "TestUnalignedByteFill: Fill at " << i << " expected 0x" << std::hex << (int)fill_value
        << " got 0x" << (int)host_buf[i] << std::dec;
  }

  // Check suffix zeros
  for (size_t i = fill_offset + fill_size; i < alloc_size; i++) {
    ASSERT_EQ(host_buf[i], 0x00) << "TestUnalignedByteFill: Suffix at " << i << " should be 0";
  }

  err = hsa_amd_memory_pool_free(base_ptr);
  RET_IF_HSA_ERR(err);

  std::cout << "  TestUnalignedByteFill: PASSED" << std::endl;
}

void MemoryFillBytesTest::TestZeroFill(hsa_agent_t agent, hsa_amd_memory_pool_t pool) {
  const size_t size = 2048;
  void* ptr = nullptr;

  hsa_status_t err = hsa_amd_memory_pool_allocate(pool, size, 0, &ptr);
  RET_IF_HSA_ERR(err);

  // First fill with non-zero pattern
  err = hsa_amd_memory_fill_bytes(ptr, 0xFF, size);
  RET_IF_HSA_ERR(err);

  // Then zero it
  err = hsa_amd_memory_fill_bytes(ptr, 0x00, size);
  RET_IF_HSA_ERR(err);

  // Verify - copy back to host using HSA copy API
  std::vector<uint8_t> host_buf(size);

  err = hsa_memory_copy(host_buf.data(), ptr, size);
  RET_IF_HSA_ERR(err);

  for (size_t i = 0; i < size; i++) {
    ASSERT_EQ(host_buf[i], 0x00) << "TestZeroFill: Mismatch at index " << i;
  }

  err = hsa_amd_memory_pool_free(ptr);
  RET_IF_HSA_ERR(err);

  std::cout << "  TestZeroFill: PASSED" << std::endl;
}

void MemoryFillBytesTest::TestCompareWithDwordFill(hsa_agent_t agent, hsa_amd_memory_pool_t pool) {
  const size_t size = 256;  // Must be multiple of 4
  void* ptr_dword = nullptr;
  void* ptr_byte = nullptr;

  hsa_status_t err = hsa_amd_memory_pool_allocate(pool, size, 0, &ptr_dword);
  RET_IF_HSA_ERR(err);
  err = hsa_amd_memory_pool_allocate(pool, size, 0, &ptr_byte);
  RET_IF_HSA_ERR(err);

  // Fill with dword API: value 0x42424242
  const uint32_t dword_value = 0x42424242;
  err = hsa_amd_memory_fill(ptr_dword, dword_value, size / sizeof(uint32_t));
  RET_IF_HSA_ERR(err);

  // Fill with byte API: value 0x42 (should produce same result)
  const uint8_t byte_value = 0x42;
  err = hsa_amd_memory_fill_bytes(ptr_byte, byte_value, size);
  RET_IF_HSA_ERR(err);

  // Compare results - copy back to host using HSA copy API
  std::vector<uint8_t> host_dword(size);
  std::vector<uint8_t> host_byte(size);

  err = hsa_memory_copy(host_dword.data(), ptr_dword, size);
  RET_IF_HSA_ERR(err);
  err = hsa_memory_copy(host_byte.data(), ptr_byte, size);
  RET_IF_HSA_ERR(err);

  ASSERT_EQ(memcmp(host_dword.data(), host_byte.data(), size), 0)
      << "TestCompareWithDwordFill: Byte fill and dword fill results differ";

  err = hsa_amd_memory_pool_free(ptr_dword);
  RET_IF_HSA_ERR(err);
  err = hsa_amd_memory_pool_free(ptr_byte);
  RET_IF_HSA_ERR(err);

  std::cout << "  TestCompareWithDwordFill: PASSED" << std::endl;
}

void MemoryFillBytesTest::TestEdgeCases(hsa_agent_t agent, hsa_amd_memory_pool_t pool) {
  void* ptr = nullptr;
  hsa_status_t err = hsa_amd_memory_pool_allocate(pool, 64, 0, &ptr);
  RET_IF_HSA_ERR(err);

  // Test zero-size fill (should succeed immediately)
  err = hsa_amd_memory_fill_bytes(ptr, 0xAA, 0);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS) << "TestEdgeCases: Zero-size fill should succeed";

  // Test null pointer (should fail with INVALID_ARGUMENT)
  err = hsa_amd_memory_fill_bytes(nullptr, 0xAA, 64);
  ASSERT_EQ(err, HSA_STATUS_ERROR_INVALID_ARGUMENT)
      << "TestEdgeCases: Null pointer should return INVALID_ARGUMENT";

  err = hsa_amd_memory_pool_free(ptr);
  RET_IF_HSA_ERR(err);

  std::cout << "  TestEdgeCases: PASSED" << std::endl;
}

void MemoryFillBytesTest::TestLargeFill(hsa_agent_t agent, hsa_amd_memory_pool_t pool) {
  const size_t size = 1024 * 1024;  // 1MB
  void* ptr = nullptr;

  hsa_status_t err = hsa_amd_memory_pool_allocate(pool, size, 0, &ptr);
  RET_IF_HSA_ERR(err);

  // Fill with pattern
  const uint8_t fill_value = 0x55;
  err = hsa_amd_memory_fill_bytes(ptr, fill_value, size);
  RET_IF_HSA_ERR(err);

  // Verify by sampling a few locations - copy back using HSA copy API
  std::vector<uint8_t> host_buf(size);

  err = hsa_memory_copy(host_buf.data(), ptr, size);
  RET_IF_HSA_ERR(err);

  // Check start, middle, and end
  size_t check_points[] = {0, size / 4, size / 2, 3 * size / 4, size - 1};
  for (size_t i : check_points) {
    ASSERT_EQ(host_buf[i], fill_value) << "TestLargeFill: Mismatch at index " << i;
  }

  // Full verification
  for (size_t i = 0; i < size; i++) {
    if (host_buf[i] != fill_value) {
      FAIL() << "TestLargeFill: Mismatch at index " << i << " expected 0x" << std::hex
             << (int)fill_value << " got 0x" << (int)host_buf[i] << std::dec;
      break;
    }
  }

  err = hsa_amd_memory_pool_free(ptr);
  RET_IF_HSA_ERR(err);

  std::cout << "  TestLargeFill: PASSED" << std::endl;
}

// Helper to find a coarse-grained GPU pool
static hsa_status_t FindGpuCoarsePool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_segment_t segment;
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (segment != HSA_AMD_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_amd_memory_pool_global_flag_t flags;
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
    *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

// Helper to find a fine-grained GPU pool
static hsa_status_t FindGpuFinePool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_segment_t segment;
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (segment != HSA_AMD_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_amd_memory_pool_global_flag_t flags;
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
    *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

// Helper to find a fine-grained system memory pool from CPU agent
static hsa_status_t FindCpuFinePool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_segment_t segment;
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (segment != HSA_AMD_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_amd_memory_pool_global_flag_t flags;
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
    *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

// Helper to find a CPU agent
static hsa_status_t FindCpuAgent(hsa_agent_t agent, void* data) {
  hsa_device_type_t device_type;
  hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (device_type == HSA_DEVICE_TYPE_CPU) {
    *static_cast<hsa_agent_t*>(data) = agent;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

void MemoryFillBytesTest::TestFineGrainedGpuMemory(hsa_agent_t agent, hsa_amd_memory_pool_t pool) {
  const size_t alloc_size = 1024;
  const size_t fill_offset = 3;  // Unaligned offset
  const size_t fill_size = 100;  // Non-multiple-of-4 bytes

  void* base_ptr = nullptr;
  hsa_status_t err = hsa_amd_memory_pool_allocate(pool, alloc_size, 0, &base_ptr);
  RET_IF_HSA_ERR(err);

  // Zero the buffer first
  err = hsa_amd_memory_fill_bytes(base_ptr, 0x00, alloc_size);
  RET_IF_HSA_ERR(err);

  // Fill starting at unaligned offset
  const uint8_t fill_value = 0xEF;
  void* fill_ptr = static_cast<char*>(base_ptr) + fill_offset;
  err = hsa_amd_memory_fill_bytes(fill_ptr, fill_value, fill_size);
  RET_IF_HSA_ERR(err);

  // Fine-grained memory is CPU accessible, so we can verify directly
  uint8_t* byte_ptr = static_cast<uint8_t*>(base_ptr);

  // Check prefix zeros
  for (size_t i = 0; i < fill_offset; i++) {
    ASSERT_EQ(byte_ptr[i], 0x00) << "TestFineGrainedGpuMemory: Prefix at " << i << " should be 0";
  }

  // Check filled region
  for (size_t i = fill_offset; i < fill_offset + fill_size; i++) {
    ASSERT_EQ(byte_ptr[i], fill_value)
        << "TestFineGrainedGpuMemory: Fill at " << i << " expected 0x" << std::hex
        << (int)fill_value << " got 0x" << (int)byte_ptr[i] << std::dec;
  }

  // Check suffix zeros
  for (size_t i = fill_offset + fill_size; i < alloc_size; i++) {
    ASSERT_EQ(byte_ptr[i], 0x00) << "TestFineGrainedGpuMemory: Suffix at " << i << " should be 0";
  }

  err = hsa_amd_memory_pool_free(base_ptr);
  RET_IF_HSA_ERR(err);

  std::cout << "  TestFineGrainedGpuMemory: PASSED" << std::endl;
}

void MemoryFillBytesTest::TestFineGrainedSystemMemory(hsa_agent_t cpu_agent, hsa_agent_t gpu_agent,
                                                      hsa_amd_memory_pool_t pool) {
  const size_t alloc_size = 2048;
  const size_t fill_offset = 5;  // Unaligned offset
  const size_t fill_size = 500;  // Non-multiple-of-4 bytes

  void* base_ptr = nullptr;
  hsa_status_t err = hsa_amd_memory_pool_allocate(pool, alloc_size, 0, &base_ptr);
  RET_IF_HSA_ERR(err);

  // Allow GPU access to system memory
  err = hsa_amd_agents_allow_access(1, &gpu_agent, nullptr, base_ptr);
  RET_IF_HSA_ERR(err);

  // Zero the buffer first
  err = hsa_amd_memory_fill_bytes(base_ptr, 0x00, alloc_size);
  RET_IF_HSA_ERR(err);

  // Fill starting at unaligned offset with a pattern
  const uint8_t fill_value = 0xBC;
  void* fill_ptr = static_cast<char*>(base_ptr) + fill_offset;
  err = hsa_amd_memory_fill_bytes(fill_ptr, fill_value, fill_size);
  RET_IF_HSA_ERR(err);

  // System memory is CPU accessible, so we can verify directly
  uint8_t* byte_ptr = static_cast<uint8_t*>(base_ptr);

  // Check prefix zeros
  for (size_t i = 0; i < fill_offset; i++) {
    ASSERT_EQ(byte_ptr[i], 0x00) << "TestFineGrainedSystemMemory: Prefix at " << i
                                 << " should be 0";
  }

  // Check filled region
  for (size_t i = fill_offset; i < fill_offset + fill_size; i++) {
    ASSERT_EQ(byte_ptr[i], fill_value)
        << "TestFineGrainedSystemMemory: Fill at " << i << " expected 0x" << std::hex
        << (int)fill_value << " got 0x" << (int)byte_ptr[i] << std::dec;
  }

  // Check suffix zeros
  for (size_t i = fill_offset + fill_size; i < alloc_size; i++) {
    ASSERT_EQ(byte_ptr[i], 0x00) << "TestFineGrainedSystemMemory: Suffix at " << i
                                 << " should be 0";
  }

  err = hsa_amd_memory_pool_free(base_ptr);
  RET_IF_HSA_ERR(err);

  std::cout << "  TestFineGrainedSystemMemory: PASSED" << std::endl;
}

void MemoryFillBytesTest::TestSystemMemoryHostPath(hsa_agent_t cpu_agent,
                                                   hsa_amd_memory_pool_t pool) {
  // This test verifies the host memset path in Runtime::FillMemoryBytes
  // when memory is not GPU-mapped.
  const size_t alloc_size = 1024;
  const size_t fill_offset = 7;  // Unaligned offset
  const size_t fill_size = 200;  // Non-multiple-of-4 bytes

  void* base_ptr = nullptr;
  hsa_status_t err = hsa_amd_memory_pool_allocate(pool, alloc_size, 0, &base_ptr);
  RET_IF_HSA_ERR(err);

  // Zero the buffer using hsa_amd_memory_fill_bytes (should use memset path)
  err = hsa_amd_memory_fill_bytes(base_ptr, 0x00, alloc_size);
  RET_IF_HSA_ERR(err);

  // Fill with a pattern
  const uint8_t fill_value = 0x99;
  void* fill_ptr = static_cast<char*>(base_ptr) + fill_offset;
  err = hsa_amd_memory_fill_bytes(fill_ptr, fill_value, fill_size);
  RET_IF_HSA_ERR(err);

  // Verify directly (host memory)
  uint8_t* byte_ptr = static_cast<uint8_t*>(base_ptr);

  // Check prefix zeros
  for (size_t i = 0; i < fill_offset; i++) {
    ASSERT_EQ(byte_ptr[i], 0x00) << "TestSystemMemoryHostPath: Prefix at " << i << " should be 0";
  }

  // Check filled region
  for (size_t i = fill_offset; i < fill_offset + fill_size; i++) {
    ASSERT_EQ(byte_ptr[i], fill_value)
        << "TestSystemMemoryHostPath: Fill at " << i << " expected 0x" << std::hex
        << (int)fill_value << " got 0x" << (int)byte_ptr[i] << std::dec;
  }

  // Check suffix zeros
  for (size_t i = fill_offset + fill_size; i < alloc_size; i++) {
    ASSERT_EQ(byte_ptr[i], 0x00) << "TestSystemMemoryHostPath: Suffix at " << i << " should be 0";
  }

  err = hsa_amd_memory_pool_free(base_ptr);
  RET_IF_HSA_ERR(err);

  std::cout << "  TestSystemMemoryHostPath: PASSED" << std::endl;
}

void MemoryFillBytesTest::MemoryFillBytesAllTests() {
  // Get GPU agent
  hsa_agent_t gpu_agent = *gpu_device1();
  if (gpu_agent.handle == 0) {
    std::cout << "No GPU agent found, skipping test" << std::endl;
    return;
  }

  // Find CPU agent
  hsa_agent_t cpu_agent = {0};
  hsa_status_t err = hsa_iterate_agents(FindCpuAgent, &cpu_agent);
  if (cpu_agent.handle == 0) {
    std::cout << "No CPU agent found, skipping system memory tests" << std::endl;
  }

  // Find coarse-grained GPU memory pool
  hsa_amd_memory_pool_t gpu_coarse_pool = {0};
  err = hsa_amd_agent_iterate_memory_pools(gpu_agent, FindGpuCoarsePool, &gpu_coarse_pool);

  // Find fine-grained GPU memory pool
  hsa_amd_memory_pool_t gpu_fine_pool = {0};
  err = hsa_amd_agent_iterate_memory_pools(gpu_agent, FindGpuFinePool, &gpu_fine_pool);

  // Find fine-grained system memory pool from CPU agent
  hsa_amd_memory_pool_t system_fine_pool = {0};
  if (cpu_agent.handle != 0) {
    err = hsa_amd_agent_iterate_memory_pools(cpu_agent, FindCpuFinePool, &system_fine_pool);
  }

  std::cout << "Running Memory Fill Bytes Tests..." << std::endl;

  // Test 1: Coarse-grained GPU memory (original tests)
  if (gpu_coarse_pool.handle != 0) {
    std::cout << "\n[Coarse-grained GPU memory tests]" << std::endl;
    TestAlignedByteFill(gpu_agent, gpu_coarse_pool);
    TestUnalignedByteFill(gpu_agent, gpu_coarse_pool);
    TestZeroFill(gpu_agent, gpu_coarse_pool);
    TestCompareWithDwordFill(gpu_agent, gpu_coarse_pool);
    TestEdgeCases(gpu_agent, gpu_coarse_pool);
    TestLargeFill(gpu_agent, gpu_coarse_pool);
  } else {
    std::cout << "No coarse-grained GPU memory pool found, skipping coarse-grained tests"
              << std::endl;
  }

  // Test 2: Fine-grained GPU memory
  if (gpu_fine_pool.handle != 0) {
    std::cout << "\n[Fine-grained GPU memory tests]" << std::endl;
    TestFineGrainedGpuMemory(gpu_agent, gpu_fine_pool);
    // Run aligned and unaligned tests on fine-grained GPU memory too
    TestAlignedByteFill(gpu_agent, gpu_fine_pool);
    TestUnalignedByteFill(gpu_agent, gpu_fine_pool);
  } else {
    std::cout << "No fine-grained GPU memory pool found, skipping fine-grained GPU tests"
              << std::endl;
  }

  // Test 3: Fine-grained system memory (CPU memory with GPU access)
  if (system_fine_pool.handle != 0 && cpu_agent.handle != 0) {
    std::cout << "\n[Fine-grained system memory tests]" << std::endl;
    TestFineGrainedSystemMemory(cpu_agent, gpu_agent, system_fine_pool);
  } else {
    std::cout << "No fine-grained system memory pool found, skipping system memory tests"
              << std::endl;
  }

  // Test 4: System memory host-only path (memset fallback)
  if (system_fine_pool.handle != 0 && cpu_agent.handle != 0) {
    std::cout << "\n[System memory host path tests]" << std::endl;
    TestSystemMemoryHostPath(cpu_agent, system_fine_pool);
  }

  std::cout << "\nAll Memory Fill Bytes Tests PASSED" << std::endl;
}
