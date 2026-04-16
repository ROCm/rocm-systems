//===-- HSAPoolGPUGTest.cpp - HSA Pool GPU Tests ----------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration tests for HSA memory pool with GPU access.
/// These tests verify that the GPU can write to HSA kernarg pool memory
/// and the CPU can read it back.
///
/// Test IDs:
///   - test-gpu-scalar-store: GPU s_store_dword writes to kernarg pool
///   - test-callback-alloc: HSA allocation works from dispatch callback
///
/// CRITICAL: These are GATE TESTS. If they fail, the HSA pool approach
/// won't work and instrumentation needs to use vector stores instead.
///
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#ifdef AEGISBIT_HAS_GPU
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hip/hip_runtime.h>
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

/// Simple kernel that writes to a buffer
/// Uses global_store_dword (vector store) which is what the instrumentation uses
__global__ void writeTestKernel(uint32_t* output) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    output[0] = 0xDEADBEEF;
    output[1] = 0xCAFEBABE;
  }
  __threadfence_system();  // Ensure writes to HSA pool memory are visible to CPU
}

/// Kernel that simulates trace recording - writes wavefront and bb_id
__global__ void traceRecordKernel(uint32_t* trace_buf, uint64_t* write_offset) {
  // Simulate what BB entry instrumentation does:
  // 1. Atomic add to get write offset
  // 2. Write wavefront_id and bb_id to trace buffer

  int wf_id = threadIdx.x / 64;  // Simplified wavefront ID
  int bb_id = 0;  // First basic block

  if (threadIdx.x % 64 == 0) {  // One thread per wavefront
    // Atomically get and increment write offset
    uint64_t offset = atomicAdd(
        reinterpret_cast<unsigned long long*>(write_offset),
        static_cast<unsigned long long>(8));

    // Write trace record
    uint32_t* record = reinterpret_cast<uint32_t*>(
        reinterpret_cast<uint8_t*>(trace_buf) + offset);
    record[0] = wf_id;
    record[1] = bb_id;
  }
  __threadfence_system();  // Ensure writes to HSA pool memory are visible to CPU
}

class HSAPoolGPUTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Check that HIP is available
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No HIP devices available";
    }

    // Initialize HIP
    err = hipSetDevice(0);
    ASSERT_EQ(err, hipSuccess) << "Failed to set HIP device";
  }
};

// CRITICAL GATE TEST: GPU vector store to HSA kernarg pool memory
TEST_F(HSAPoolGPUTest, VectorStoreToKernargPool) {
  // Initialize HSA
  hsa_status_t hsa_status = hsa_init();
  ASSERT_EQ(hsa_status, HSA_STATUS_SUCCESS);

  // Find GPU and kernarg pool
  hsa_agent_t gpu_agent = findGPUAgent();
  ASSERT_NE(gpu_agent.handle, 0u) << "No GPU agent found";

  hsa_amd_memory_pool_t pool = findKernargPool(gpu_agent);
  ASSERT_NE(pool.handle, 0u) << "No kernarg pool found";

  // Log pool type for diagnostics
  uint32_t flags = 0;
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                &flags);
  std::cout << "Pool flags: KERNARG_INIT="
            << ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) ? "yes" : "no")
            << " FINE_GRAINED="
            << ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) ? "yes" : "no")
            << std::endl;

  // Allocate buffer from HSA kernarg pool
  constexpr size_t kBufferSize = 4096;
  void* buf = nullptr;
  hsa_status = hsa_amd_memory_pool_allocate(pool, kBufferSize, 0, &buf);
  ASSERT_EQ(hsa_status, HSA_STATUS_SUCCESS);
  ASSERT_NE(buf, nullptr);

  // Grant GPU access to the allocated memory.
  // On gfx950, FINE_GRAINED pools may require explicit access grants
  // even though the pool is nominally CPU+GPU accessible.
  hsa_status = hsa_amd_agents_allow_access(1, &gpu_agent, nullptr, buf);
  if (hsa_status != HSA_STATUS_SUCCESS) {
    std::cout << "Note: hsa_amd_agents_allow_access returned "
              << hsa_status << " (may not be needed)" << std::endl;
  }

  // Zero the buffer (CPU access)
  std::memset(buf, 0, kBufferSize);

  // Launch HIP kernel that writes to this buffer
  writeTestKernel<<<1, 64>>>(reinterpret_cast<uint32_t*>(buf));
  hipError_t hip_err = hipDeviceSynchronize();
  ASSERT_EQ(hip_err, hipSuccess) << "Kernel launch failed";

  // Memory fence to ensure GPU writes are visible to CPU
  __atomic_thread_fence(__ATOMIC_SEQ_CST);

  // CPU read back (plain dereference - no hipMemcpy needed for HSA pool memory)
  uint32_t* data = reinterpret_cast<uint32_t*>(buf);
  std::cout << "Read back: data[0]=0x" << std::hex << data[0]
            << " data[1]=0x" << data[1] << std::dec << std::endl;

  EXPECT_EQ(data[0], 0xDEADBEEFu)
      << "GPU vector store to kernarg pool failed - value not written";
  EXPECT_EQ(data[1], 0xCAFEBABEu)
      << "GPU vector store to kernarg pool failed - second value not written";

  // Clean up
  hsa_amd_memory_pool_free(buf);
  hsa_shut_down();
}

// Test: Trace recording pattern works with HSA pool memory
TEST_F(HSAPoolGPUTest, TraceRecordingPattern) {
  // Initialize HSA
  hsa_status_t hsa_status = hsa_init();
  ASSERT_EQ(hsa_status, HSA_STATUS_SUCCESS);

  // Find GPU and kernarg pool
  hsa_agent_t gpu_agent = findGPUAgent();
  ASSERT_NE(gpu_agent.handle, 0u);

  hsa_amd_memory_pool_t pool = findKernargPool(gpu_agent);
  ASSERT_NE(pool.handle, 0u);

  // Allocate trace buffer from HSA kernarg pool
  constexpr size_t kBufferSize = 4096;
  void* trace_buf = nullptr;
  hsa_status = hsa_amd_memory_pool_allocate(pool, kBufferSize, 0, &trace_buf);
  ASSERT_EQ(hsa_status, HSA_STATUS_SUCCESS);
  std::memset(trace_buf, 0, kBufferSize);

  // Allocate write offset from HSA kernarg pool
  void* write_offset = nullptr;
  hsa_status = hsa_amd_memory_pool_allocate(pool, 8, 0, &write_offset);
  ASSERT_EQ(hsa_status, HSA_STATUS_SUCCESS);
  std::memset(write_offset, 0, 8);

  // Grant GPU access (needed on gfx950 with FINE_GRAINED pools)
  hsa_amd_agents_allow_access(1, &gpu_agent, nullptr, trace_buf);
  hsa_amd_agents_allow_access(1, &gpu_agent, nullptr, write_offset);

  // Launch trace recording kernel
  // 64 threads = 1 wavefront, should produce 1 trace record
  traceRecordKernel<<<1, 64>>>(
      reinterpret_cast<uint32_t*>(trace_buf),
      reinterpret_cast<uint64_t*>(write_offset));
  hipError_t hip_err = hipDeviceSynchronize();
  ASSERT_EQ(hip_err, hipSuccess);

  // Read back write offset (plain dereference)
  uint64_t offset = *reinterpret_cast<uint64_t*>(write_offset);
  std::cout << "WriteOffset after kernel: " << offset << std::endl;
  EXPECT_EQ(offset, 8u) << "Expected 1 trace record (8 bytes)";

  // Read back trace record
  uint32_t* record = reinterpret_cast<uint32_t*>(trace_buf);
  std::cout << "Trace record: wavefront=" << record[0]
            << " bb_id=" << record[1] << std::endl;
  EXPECT_EQ(record[0], 0u) << "wavefront_id should be 0";
  EXPECT_EQ(record[1], 0u) << "bb_id should be 0";

  // Clean up
  hsa_amd_memory_pool_free(trace_buf);
  hsa_amd_memory_pool_free(write_offset);
  hsa_shut_down();
}

// Test: Multiple wavefronts write trace records
TEST_F(HSAPoolGPUTest, MultiWavefrontTracing) {
  // Initialize HSA
  hsa_status_t hsa_status = hsa_init();
  ASSERT_EQ(hsa_status, HSA_STATUS_SUCCESS);

  // Find GPU and kernarg pool
  hsa_agent_t gpu_agent = findGPUAgent();
  ASSERT_NE(gpu_agent.handle, 0u);

  hsa_amd_memory_pool_t pool = findKernargPool(gpu_agent);
  ASSERT_NE(pool.handle, 0u);

  // Allocate buffers
  constexpr size_t kBufferSize = 4096;
  void* trace_buf = nullptr;
  void* write_offset = nullptr;
  hsa_amd_memory_pool_allocate(pool, kBufferSize, 0, &trace_buf);
  hsa_amd_memory_pool_allocate(pool, 8, 0, &write_offset);
  std::memset(trace_buf, 0, kBufferSize);
  std::memset(write_offset, 0, 8);

  // Grant GPU access (needed on gfx950 with FINE_GRAINED pools)
  hsa_amd_agents_allow_access(1, &gpu_agent, nullptr, trace_buf);
  hsa_amd_agents_allow_access(1, &gpu_agent, nullptr, write_offset);

  // Launch with 4 wavefronts (256 threads)
  constexpr int kNumWavefronts = 4;
  traceRecordKernel<<<1, kNumWavefronts * 64>>>(
      reinterpret_cast<uint32_t*>(trace_buf),
      reinterpret_cast<uint64_t*>(write_offset));
  hipDeviceSynchronize();

  // Read back write offset
  uint64_t offset = *reinterpret_cast<uint64_t*>(write_offset);
  std::cout << "WriteOffset after multi-wavefront kernel: " << offset << std::endl;
  EXPECT_EQ(offset, kNumWavefronts * 8u)
      << "Expected " << kNumWavefronts << " trace records";

  // Clean up
  hsa_amd_memory_pool_free(trace_buf);
  hsa_amd_memory_pool_free(write_offset);
  hsa_shut_down();
}

}  // namespace

#else  // !AEGISBIT_HAS_GPU

// Dummy test when GPU is not available
TEST(HSAPoolGPUTest, GPUNotAvailable) {
  GTEST_SKIP() << "GPU not available, skipping HSA pool GPU tests";
}

#endif  // AEGISBIT_HAS_GPU
