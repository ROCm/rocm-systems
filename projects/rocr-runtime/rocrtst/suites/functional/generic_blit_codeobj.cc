/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#include "suites/functional/generic_blit_codeobj.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#define RET_IF_HSA_ERR(err)                                                                        \
  {                                                                                                \
    if ((err) != HSA_STATUS_SUCCESS) {                                                             \
      const char* msg = 0;                                                                         \
      hsa_status_string(err, &msg);                                                                \
      std::cout << "HSA API call failure at line " << __LINE__ << ", file: " << __FILE__           \
                << ". Call returned " << err << std::endl;                                         \
      std::cout << msg << std::endl;                                                               \
      return;                                                                                      \
    }                                                                                              \
  }

static const char kSubTestSeparator[] = "  **************************";

static void PrintSubtestHeader(const char* header) {
  std::cout << "  *** Generic Blit Code Object Subtest: " << header << " ***" << std::endl;
}

// Callback to find a GPU agent
static hsa_status_t FindGpuAgent(hsa_agent_t agent, void* data) {
  hsa_device_type_t device_type;
  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (device_type == HSA_DEVICE_TYPE_GPU) {
    hsa_agent_t* ret = reinterpret_cast<hsa_agent_t*>(data);
    *ret = agent;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

// Callback to find a CPU agent
static hsa_status_t FindCpuAgent(hsa_agent_t agent, void* data) {
  hsa_device_type_t device_type;
  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (device_type == HSA_DEVICE_TYPE_CPU) {
    hsa_agent_t* ret = reinterpret_cast<hsa_agent_t*>(data);
    *ret = agent;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

// Callback to find a global memory pool
static hsa_status_t FindGlobalPool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_segment_t segment;
  hsa_status_t status =
      hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

  bool accessible = false;
  status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                                        &accessible);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (accessible) {
    hsa_amd_memory_pool_t* ret = reinterpret_cast<hsa_amd_memory_pool_t*>(data);
    *ret = pool;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

GenericBlitCodeObjTest::GenericBlitCodeObjTest() : TestBase(), has_gpu_(false) {
  set_num_iteration(1);
  set_title("Generic Blit Code Object Test");
  set_description(
      "This test validates that blit operations (memory copy and fill) "
      "work correctly with generic code objects (COV 6+). Generic code objects "
      "provide architecture-wide compatibility, reducing the number of binaries "
      "while maintaining functionality across all GPUs in an architecture generation.");
}

GenericBlitCodeObjTest::~GenericBlitCodeObjTest() {}

void GenericBlitCodeObjTest::SetUp() {
  hsa_status_t err;

  TestBase::SetUp();

  // Find CPU agent
  cpu_agent_.handle = 0;
  err = hsa_iterate_agents(FindCpuAgent, &cpu_agent_);
  ASSERT_NE(cpu_agent_.handle, 0u) << "No CPU agent found";

  // Find GPU agent
  gpu_agent_.handle = 0;
  err = hsa_iterate_agents(FindGpuAgent, &gpu_agent_);
  if (gpu_agent_.handle == 0) {
    std::cout << "No GPU agent found, skipping GPU-specific tests" << std::endl;
    has_gpu_ = false;
    return;
  }
  has_gpu_ = true;

  // Print GPU info for debugging
  char name[64];
  err = hsa_agent_get_info(gpu_agent_, HSA_AGENT_INFO_NAME, name);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  std::cout << "Testing with GPU: " << name << std::endl;

  // Find CPU memory pool
  cpu_pool_.handle = 0;
  err = hsa_amd_agent_iterate_memory_pools(cpu_agent_, FindGlobalPool, &cpu_pool_);
  ASSERT_NE(cpu_pool_.handle, 0u) << "No CPU memory pool found";

  // Find GPU memory pool
  gpu_pool_.handle = 0;
  err = hsa_amd_agent_iterate_memory_pools(gpu_agent_, FindGlobalPool, &gpu_pool_);
  ASSERT_NE(gpu_pool_.handle, 0u) << "No GPU memory pool found";
}

void GenericBlitCodeObjTest::Run() {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::Run();
}

void GenericBlitCodeObjTest::DisplayTestInfo() { TestBase::DisplayTestInfo(); }

void GenericBlitCodeObjTest::DisplayResults() const {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
}

void GenericBlitCodeObjTest::Close() { TestBase::Close(); }

void GenericBlitCodeObjTest::AsyncMemoryCopyTest() {
  PrintSubtestHeader("Async Memory Copy");

  if (!has_gpu_) {
    std::cout << "  Skipping - no GPU available" << std::endl;
    return;
  }

  hsa_status_t err;
  const size_t buffer_size = 1024 * 1024;  // 1 MB

  // Allocate source buffer on CPU
  void* src_ptr = nullptr;
  err = hsa_amd_memory_pool_allocate(cpu_pool_, buffer_size, 0, &src_ptr);
  RET_IF_HSA_ERR(err);

  // Allocate destination buffer on GPU
  void* dst_ptr = nullptr;
  err = hsa_amd_memory_pool_allocate(gpu_pool_, buffer_size, 0, &dst_ptr);
  if (err != HSA_STATUS_SUCCESS) {
    hsa_amd_memory_pool_free(src_ptr);
    RET_IF_HSA_ERR(err);
  }

  // Allocate result buffer on CPU for readback
  void* result_ptr = nullptr;
  err = hsa_amd_memory_pool_allocate(cpu_pool_, buffer_size, 0, &result_ptr);
  if (err != HSA_STATUS_SUCCESS) {
    hsa_amd_memory_pool_free(src_ptr);
    hsa_amd_memory_pool_free(dst_ptr);
    RET_IF_HSA_ERR(err);
  }

  // Allow GPU to access CPU memory and vice versa
  hsa_agent_t agents[2] = {cpu_agent_, gpu_agent_};
  err = hsa_amd_agents_allow_access(2, agents, nullptr, src_ptr);
  RET_IF_HSA_ERR(err);
  err = hsa_amd_agents_allow_access(2, agents, nullptr, dst_ptr);
  RET_IF_HSA_ERR(err);
  err = hsa_amd_agents_allow_access(2, agents, nullptr, result_ptr);
  RET_IF_HSA_ERR(err);

  // Initialize source with pattern
  uint32_t* src_data = reinterpret_cast<uint32_t*>(src_ptr);
  for (size_t i = 0; i < buffer_size / sizeof(uint32_t); i++) {
    src_data[i] = static_cast<uint32_t>(i);
  }
  memset(result_ptr, 0, buffer_size);

  // Create signal for completion
  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  RET_IF_HSA_ERR(err);

  // Copy from CPU to GPU (exercises blit copy shader)
  err = hsa_amd_memory_async_copy(dst_ptr, gpu_agent_, src_ptr, cpu_agent_, buffer_size, 0, nullptr,
                                  signal);
  RET_IF_HSA_ERR(err);

  // Wait for copy to complete
  hsa_signal_value_t result = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                                        UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  ASSERT_EQ(result, 0) << "Async copy CPU->GPU failed";

  // Reset signal
  hsa_signal_store_screlease(signal, 1);

  // Copy from GPU back to CPU
  err = hsa_amd_memory_async_copy(result_ptr, cpu_agent_, dst_ptr, gpu_agent_, buffer_size, 0,
                                  nullptr, signal);
  RET_IF_HSA_ERR(err);

  // Wait for copy to complete
  result = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                     HSA_WAIT_STATE_BLOCKED);
  ASSERT_EQ(result, 0) << "Async copy GPU->CPU failed";

  // Verify data integrity
  uint32_t* result_data = reinterpret_cast<uint32_t*>(result_ptr);
  bool data_valid = true;
  for (size_t i = 0; i < buffer_size / sizeof(uint32_t); i++) {
    if (result_data[i] != static_cast<uint32_t>(i)) {
      std::cout << "  Data mismatch at index " << i << ": expected " << i << ", got "
                << result_data[i] << std::endl;
      data_valid = false;
      break;
    }
  }
  ASSERT_TRUE(data_valid) << "Data verification failed after async copy";

  std::cout << "  PASSED: Async memory copy test (1MB roundtrip verified)" << std::endl;

  // Cleanup
  hsa_signal_destroy(signal);
  hsa_amd_memory_pool_free(src_ptr);
  hsa_amd_memory_pool_free(dst_ptr);
  hsa_amd_memory_pool_free(result_ptr);

  std::cout << kSubTestSeparator << std::endl;
}

void GenericBlitCodeObjTest::MemoryFillTest() {
  PrintSubtestHeader("Memory Fill");

  if (!has_gpu_) {
    std::cout << "  Skipping - no GPU available" << std::endl;
    return;
  }

  hsa_status_t err;
  const size_t buffer_size = 1024 * 1024;  // 1 MB
  const uint32_t fill_value = 0xDEADBEEF;

  // Allocate GPU buffer
  void* gpu_ptr = nullptr;
  err = hsa_amd_memory_pool_allocate(gpu_pool_, buffer_size, 0, &gpu_ptr);
  RET_IF_HSA_ERR(err);

  // Allocate CPU buffer for readback
  void* cpu_ptr = nullptr;
  err = hsa_amd_memory_pool_allocate(cpu_pool_, buffer_size, 0, &cpu_ptr);
  if (err != HSA_STATUS_SUCCESS) {
    hsa_amd_memory_pool_free(gpu_ptr);
    RET_IF_HSA_ERR(err);
  }

  // Allow access
  hsa_agent_t agents[2] = {cpu_agent_, gpu_agent_};
  err = hsa_amd_agents_allow_access(2, agents, nullptr, gpu_ptr);
  RET_IF_HSA_ERR(err);
  err = hsa_amd_agents_allow_access(2, agents, nullptr, cpu_ptr);
  RET_IF_HSA_ERR(err);

  // Initialize GPU buffer to zero
  memset(cpu_ptr, 0, buffer_size);

  // Create signal for completion
  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  RET_IF_HSA_ERR(err);

  // First copy zeros to GPU
  err = hsa_amd_memory_async_copy(gpu_ptr, gpu_agent_, cpu_ptr, cpu_agent_, buffer_size, 0, nullptr,
                                  signal);
  RET_IF_HSA_ERR(err);
  hsa_signal_value_t wait_result = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                                             UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  ASSERT_EQ(wait_result, 0) << "Async copy CPU->GPU (zero init) failed";
  hsa_signal_store_screlease(signal, 1);

  // Fill GPU memory (exercises blit fill shader)
  err = hsa_amd_memory_fill(gpu_ptr, fill_value, buffer_size / sizeof(uint32_t));
  RET_IF_HSA_ERR(err);

  // Copy back to CPU
  err = hsa_amd_memory_async_copy(cpu_ptr, cpu_agent_, gpu_ptr, gpu_agent_, buffer_size, 0, nullptr,
                                  signal);
  RET_IF_HSA_ERR(err);
  wait_result = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                          HSA_WAIT_STATE_BLOCKED);
  ASSERT_EQ(wait_result, 0) << "Async copy GPU->CPU (readback) failed";

  // Verify fill
  uint32_t* result_data = reinterpret_cast<uint32_t*>(cpu_ptr);
  bool data_valid = true;
  for (size_t i = 0; i < buffer_size / sizeof(uint32_t); i++) {
    if (result_data[i] != fill_value) {
      std::cout << "  Fill mismatch at index " << i << ": expected 0x" << std::hex << fill_value
                << ", got 0x" << result_data[i] << std::dec << std::endl;
      data_valid = false;
      break;
    }
  }
  ASSERT_TRUE(data_valid) << "Data verification failed after memory fill";

  std::cout << "  PASSED: Memory fill test (1MB fill with 0xDEADBEEF verified)" << std::endl;

  // Cleanup
  hsa_signal_destroy(signal);
  hsa_amd_memory_pool_free(gpu_ptr);
  hsa_amd_memory_pool_free(cpu_ptr);

  std::cout << kSubTestSeparator << std::endl;
}

void GenericBlitCodeObjTest::VariousCopySizesTest() {
  PrintSubtestHeader("Various Copy Sizes");

  if (!has_gpu_) {
    std::cout << "  Skipping - no GPU available" << std::endl;
    return;
  }

  hsa_status_t err;

  // Test various copy sizes to exercise different code paths:
  // - Small unaligned copies (byte-by-byte path)
  // - Aligned copies (vectorized path)
  // - Large copies (unrolled path)
  // - Misaligned copies (misaligned path)
  const size_t test_sizes[] = {
      1,        // 1 byte - smallest possible
      3,        // 3 bytes - odd size
      15,       // 15 bytes - misaligned
      16,       // 16 bytes - aligned
      63,       // 63 bytes - just under cache line
      64,       // 64 bytes - cache line
      255,      // 255 bytes - misaligned
      256,      // 256 bytes - aligned
      1023,     // 1KB - 1
      1024,     // 1KB
      4096,     // 4KB - page size
      65536,    // 64KB
      1048576,  // 1MB
  };

  const size_t max_size = 1048576;

  // Allocate buffers
  void* src_ptr = nullptr;
  void* gpu_ptr = nullptr;
  void* result_ptr = nullptr;

  err = hsa_amd_memory_pool_allocate(cpu_pool_, max_size, 0, &src_ptr);
  RET_IF_HSA_ERR(err);

  err = hsa_amd_memory_pool_allocate(gpu_pool_, max_size, 0, &gpu_ptr);
  if (err != HSA_STATUS_SUCCESS) {
    hsa_amd_memory_pool_free(src_ptr);
    RET_IF_HSA_ERR(err);
  }

  err = hsa_amd_memory_pool_allocate(cpu_pool_, max_size, 0, &result_ptr);
  if (err != HSA_STATUS_SUCCESS) {
    hsa_amd_memory_pool_free(src_ptr);
    hsa_amd_memory_pool_free(gpu_ptr);
    RET_IF_HSA_ERR(err);
  }

  // Allow access
  hsa_agent_t agents[2] = {cpu_agent_, gpu_agent_};
  err = hsa_amd_agents_allow_access(2, agents, nullptr, src_ptr);
  RET_IF_HSA_ERR(err);
  err = hsa_amd_agents_allow_access(2, agents, nullptr, gpu_ptr);
  RET_IF_HSA_ERR(err);
  err = hsa_amd_agents_allow_access(2, agents, nullptr, result_ptr);
  RET_IF_HSA_ERR(err);

  // Create signal
  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  RET_IF_HSA_ERR(err);

  // Test each size
  int passed = 0;
  int failed = 0;

  for (size_t size : test_sizes) {
    // Initialize source
    uint8_t* src_data = reinterpret_cast<uint8_t*>(src_ptr);
    for (size_t i = 0; i < size; i++) {
      src_data[i] = static_cast<uint8_t>(i & 0xFF);
    }
    memset(result_ptr, 0, size);

    // Copy CPU -> GPU
    hsa_signal_store_screlease(signal, 1);
    err = hsa_amd_memory_async_copy(gpu_ptr, gpu_agent_, src_ptr, cpu_agent_, size, 0, nullptr,
                                    signal);
    if (err != HSA_STATUS_SUCCESS) {
      std::cout << "  FAILED: Size " << size << " - copy to GPU failed" << std::endl;
      failed++;
      continue;
    }
    hsa_signal_value_t wait_val = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                                            UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    if (wait_val != 0) {
      std::cout << "  FAILED: Size " << size << " - copy to GPU wait failed" << std::endl;
      failed++;
      continue;
    }

    // Copy GPU -> CPU
    hsa_signal_store_screlease(signal, 1);
    err = hsa_amd_memory_async_copy(result_ptr, cpu_agent_, gpu_ptr, gpu_agent_, size, 0, nullptr,
                                    signal);
    if (err != HSA_STATUS_SUCCESS) {
      std::cout << "  FAILED: Size " << size << " - copy from GPU failed" << std::endl;
      failed++;
      continue;
    }
    wait_val = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                         HSA_WAIT_STATE_BLOCKED);
    if (wait_val != 0) {
      std::cout << "  FAILED: Size " << size << " - copy from GPU wait failed" << std::endl;
      failed++;
      continue;
    }

    // Verify
    uint8_t* result_data = reinterpret_cast<uint8_t*>(result_ptr);
    bool data_valid = true;
    for (size_t i = 0; i < size; i++) {
      if (result_data[i] != static_cast<uint8_t>(i & 0xFF)) {
        std::cout << "  FAILED: Size " << size << " - data mismatch at byte " << i
                  << ": expected 0x" << std::hex << (i & 0xFF) << ", got 0x"
                  << static_cast<int>(result_data[i]) << std::dec << std::endl;
        data_valid = false;
        break;
      }
    }

    if (data_valid) {
      passed++;
    } else {
      failed++;
    }
  }

  std::cout << "  Results: " << passed << " passed, " << failed << " failed out of "
            << (sizeof(test_sizes) / sizeof(test_sizes[0])) << " sizes" << std::endl;

  ASSERT_EQ(failed, 0) << "Some copy size tests failed";

  std::cout << "  PASSED: Various copy sizes test" << std::endl;

  // Cleanup
  hsa_signal_destroy(signal);
  hsa_amd_memory_pool_free(src_ptr);
  hsa_amd_memory_pool_free(gpu_ptr);
  hsa_amd_memory_pool_free(result_ptr);

  std::cout << kSubTestSeparator << std::endl;
}
