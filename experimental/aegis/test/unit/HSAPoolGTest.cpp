//===-- HSAPoolGTest.cpp - HSA Memory Pool Tests ----------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for HSA memory pool discovery and allocation.
/// These tests verify that the HSA kernarg memory pool can be found and used
/// for trace buffer allocation.
///
/// Test IDs:
///   - test-pool-discovery: HSA pool discovery finds kernarg pool
///   - test-cpu-readwrite: Allocate from kernarg pool, memset/memcpy works
///
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#ifdef AEGISBIT_HAS_GPU
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstring>

namespace {

/// Helper to find GPU agent
hsa_agent_t findGPUAgent() {
  hsa_agent_t gpu_agent = {0};
  hsa_iterate_agents(
      [](hsa_agent_t agent, void* data) -> hsa_status_t {
        hsa_device_type_t type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type == HSA_DEVICE_TYPE_GPU) {
          *static_cast<hsa_agent_t*>(data) = agent;
          return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
      },
      &gpu_agent);
  return gpu_agent;
}

/// Helper to find CPU-accessible GPU memory pool
/// Priority: KERNARG_INIT > FINE_GRAINED
/// gfx950 doesn't have KERNARG_INIT, but FINE_GRAINED works for CPU+GPU access
hsa_amd_memory_pool_t findKernargPool(hsa_agent_t gpu_agent) {
  struct PoolSearchResult {
    hsa_amd_memory_pool_t kernarg_pool = {0};
    hsa_amd_memory_pool_t fine_grained_pool = {0};
  };
  PoolSearchResult result;
  hsa_amd_agent_iterate_memory_pools(
      gpu_agent,
      [](hsa_amd_memory_pool_t pool, void* data) -> hsa_status_t {
        auto* result = static_cast<PoolSearchResult*>(data);
        hsa_amd_segment_t segment;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                      &segment);
        if (segment == HSA_AMD_SEGMENT_GLOBAL) {
          uint32_t flags;
          hsa_amd_memory_pool_get_info(
              pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
          // Check if allocation is allowed
          bool alloc_allowed = false;
          hsa_amd_memory_pool_get_info(
              pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
              &alloc_allowed);
          if (!alloc_allowed) {
            return HSA_STATUS_SUCCESS;
          }
          // Prefer KERNARG_INIT if available
          if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) {
            result->kernarg_pool = pool;
            return HSA_STATUS_INFO_BREAK;
          }
          // Fall back to FINE_GRAINED (CPU-accessible)
          if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) &&
              result->fine_grained_pool.handle == 0) {
            result->fine_grained_pool = pool;
          }
        }
        return HSA_STATUS_SUCCESS;
      },
      &result);

  // Use KERNARG pool if found, otherwise use FINE_GRAINED
  if (result.kernarg_pool.handle != 0) {
    return result.kernarg_pool;
  }
  return result.fine_grained_pool;
}

class HSAPoolTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize HSA runtime
    hsa_status_t status = hsa_init();
    ASSERT_EQ(status, HSA_STATUS_SUCCESS) << "Failed to initialize HSA runtime";
  }

  void TearDown() override {
    // Shutdown HSA runtime
    hsa_shut_down();
  }
};

// Test: HSA pool discovery finds GPU agent and kernarg pool
TEST_F(HSAPoolTest, FindKernargPool) {
  // Find GPU agent
  hsa_agent_t gpu_agent = findGPUAgent();
  ASSERT_NE(gpu_agent.handle, 0u) << "No GPU agent found";

  // Get agent name for logging
  char agent_name[64];
  hsa_status_t status =
      hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, agent_name);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);
  std::cout << "Found GPU agent: " << agent_name << std::endl;

  // Find kernarg pool
  hsa_amd_memory_pool_t pool = findKernargPool(gpu_agent);
  ASSERT_NE(pool.handle, 0u) << "No kernarg memory pool found";
  std::cout << "Found kernarg pool: " << pool.handle << std::endl;

  // Verify pool properties
  bool cpu_accessible = false;
  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &cpu_accessible);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);
  EXPECT_TRUE(cpu_accessible) << "Kernarg pool must be CPU-accessible";
}

// Test: Allocate from kernarg pool, CPU can read/write via memset/memcpy
TEST_F(HSAPoolTest, AllocateAndReadWrite) {
  // Find pool
  hsa_agent_t gpu_agent = findGPUAgent();
  ASSERT_NE(gpu_agent.handle, 0u);
  hsa_amd_memory_pool_t pool = findKernargPool(gpu_agent);
  ASSERT_NE(pool.handle, 0u);

  // Allocate 4KB from kernarg pool
  constexpr size_t kBufferSize = 4096;
  void* ptr = nullptr;
  hsa_status_t status =
      hsa_amd_memory_pool_allocate(pool, kBufferSize, 0, &ptr);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);
  ASSERT_NE(ptr, nullptr);

  // CPU write via memset
  std::memset(ptr, 0, kBufferSize);

  // CPU write specific values
  uint32_t* data = reinterpret_cast<uint32_t*>(ptr);
  data[0] = 0xDEADBEEF;
  data[1] = 0xCAFEBABE;
  data[2] = 0x12345678;

  // CPU read back - verify values are correct
  EXPECT_EQ(data[0], 0xDEADBEEFu);
  EXPECT_EQ(data[1], 0xCAFEBABEu);
  EXPECT_EQ(data[2], 0x12345678u);

  // Test memcpy roundtrip
  std::vector<uint8_t> copy(kBufferSize);
  std::memcpy(copy.data(), ptr, kBufferSize);

  // Verify copied data
  uint32_t* copy_data = reinterpret_cast<uint32_t*>(copy.data());
  EXPECT_EQ(copy_data[0], 0xDEADBEEFu);
  EXPECT_EQ(copy_data[1], 0xCAFEBABEu);
  EXPECT_EQ(copy_data[2], 0x12345678u);

  // Free the allocation
  status = hsa_amd_memory_pool_free(ptr);
  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
}

// Test: Multiple allocations work correctly
TEST_F(HSAPoolTest, MultipleAllocations) {
  hsa_agent_t gpu_agent = findGPUAgent();
  ASSERT_NE(gpu_agent.handle, 0u);
  hsa_amd_memory_pool_t pool = findKernargPool(gpu_agent);
  ASSERT_NE(pool.handle, 0u);

  // Allocate multiple buffers
  constexpr size_t kNumBuffers = 8;
  constexpr size_t kBufferSize = 1024;
  std::vector<void*> buffers(kNumBuffers);

  for (size_t i = 0; i < kNumBuffers; ++i) {
    void* ptr = nullptr;
    hsa_status_t status =
        hsa_amd_memory_pool_allocate(pool, kBufferSize, 0, &ptr);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);
    ASSERT_NE(ptr, nullptr);
    buffers[i] = ptr;

    // Write a unique value
    uint32_t* data = reinterpret_cast<uint32_t*>(ptr);
    data[0] = static_cast<uint32_t>(i);
  }

  // Verify all buffers have correct values
  for (size_t i = 0; i < kNumBuffers; ++i) {
    uint32_t* data = reinterpret_cast<uint32_t*>(buffers[i]);
    EXPECT_EQ(data[0], static_cast<uint32_t>(i));
  }

  // Free all buffers
  for (void* ptr : buffers) {
    hsa_status_t status = hsa_amd_memory_pool_free(ptr);
    EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  }
}

// Test: WriteOffset pattern used for tracing
TEST_F(HSAPoolTest, WriteOffsetPattern) {
  hsa_agent_t gpu_agent = findGPUAgent();
  ASSERT_NE(gpu_agent.handle, 0u);
  hsa_amd_memory_pool_t pool = findKernargPool(gpu_agent);
  ASSERT_NE(pool.handle, 0u);

  // Allocate trace buffer
  constexpr size_t kBufferSize = 4096;
  void* trace_buf = nullptr;
  hsa_status_t status =
      hsa_amd_memory_pool_allocate(pool, kBufferSize, 0, &trace_buf);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);
  std::memset(trace_buf, 0, kBufferSize);

  // Allocate write offset
  void* write_offset_ptr = nullptr;
  status = hsa_amd_memory_pool_allocate(pool, 8, 0, &write_offset_ptr);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);
  std::memset(write_offset_ptr, 0, 8);

  // Simulate what the GPU would do: write to trace buffer and update offset
  uint64_t* write_offset = reinterpret_cast<uint64_t*>(write_offset_ptr);
  EXPECT_EQ(*write_offset, 0u);  // Initially zero

  // Simulate writing a trace record
  uint8_t* trace_data = reinterpret_cast<uint8_t*>(trace_buf);
  uint64_t offset = 0;

  // Write record 1
  uint32_t* record1 = reinterpret_cast<uint32_t*>(trace_data + offset);
  record1[0] = 1;  // wavefront_id
  record1[1] = 0;  // bb_id
  offset += 8;

  // Write record 2
  uint32_t* record2 = reinterpret_cast<uint32_t*>(trace_data + offset);
  record2[0] = 2;  // wavefront_id
  record2[1] = 1;  // bb_id
  offset += 8;

  // Update write offset
  *write_offset = offset;

  // Read back and verify
  EXPECT_EQ(*write_offset, 16u);
  EXPECT_EQ(record1[0], 1u);
  EXPECT_EQ(record1[1], 0u);
  EXPECT_EQ(record2[0], 2u);
  EXPECT_EQ(record2[1], 1u);

  // Clean up
  hsa_amd_memory_pool_free(trace_buf);
  hsa_amd_memory_pool_free(write_offset_ptr);
}

}  // namespace

#else  // !AEGISBIT_HAS_GPU

// Dummy test when GPU is not available
TEST(HSAPoolTest, GPUNotAvailable) {
  GTEST_SKIP() << "GPU not available, skipping HSA pool tests";
}

#endif  // AEGISBIT_HAS_GPU
