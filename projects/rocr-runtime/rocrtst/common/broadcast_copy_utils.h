/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_COMMON_BROADCAST_COPY_UTILS_H_
#define ROCRTST_COMMON_BROADCAST_COPY_UTILS_H_

// =============================================================================
// Broadcast Copy Test Utilities
// =============================================================================
// This file provides utilities for broadcast/swap/indirect copy tests:
//   - IsaVersion: General helper for extracting GFX major/minor/stepping
//   - HsaTestContext: Standalone HSA context manager (doesn't require BaseRocR)
//   - BroadcastTestUtils: Pattern generation and verification helpers
//
// Device discovery uses existing rocrtst helpers from common/common.h:
//   - rocrtst::FindGPUDevice, rocrtst::FindCPUDevice, rocrtst::IterateGPUAgents
// =============================================================================

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include "common/common.h"  // For rocrtst::FindGPUDevice, FindCPUDevice, IterateGPUAgents

// =============================================================================
// GTest compatibility macros for older versions that lack GTEST_SKIP()
// =============================================================================
#ifndef GTEST_SKIP
// Older gtest versions don't have GTEST_SKIP, so we provide a fallback
// that returns early with SUCCEED(). This isn't perfect (test shows as passed
// rather than skipped) but it's the best we can do without modifying gtest.
#define GTEST_SKIP()                                                                               \
  do {                                                                                             \
    std::cout << "[  SKIPPED ] ";                                                                  \
    return;                                                                                        \
  } while (0);                                                                                     \
  std::cout

// Alternative implementation that can be used with << operator
#define GTEST_SKIP_REASON(reason)                                                                  \
  do {                                                                                             \
    std::cout << "[  SKIPPED ] " << reason << std::endl;                                           \
    return;                                                                                        \
  } while (0)
#endif  // GTEST_SKIP
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// SDMA Engine ID enum is now defined in hsa_ext_amd.h

// =============================================================================
// ISA Version Helper
// =============================================================================
// General helper to extract ISA major/minor/stepping from an HSA agent.
// Can be reused by other tests that need GFX version information.
// =============================================================================

struct IsaVersion {
  uint32_t major;    // e.g., 9, 10, 11, 12, 13
  uint32_t minor;    // e.g., 0, 4, 5
  uint32_t stepping; // e.g., 0, 2

  IsaVersion() : major(0), minor(0), stepping(0) {}

  // Parse ISA version from agent
  static IsaVersion FromAgent(hsa_agent_t agent) {
    IsaVersion ver;

    hsa_isa_t isa;
    hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa);
    if (status != HSA_STATUS_SUCCESS) return ver;

    char isa_name[128];
    status = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_name);
    if (status != HSA_STATUS_SUCCESS) return ver;

    // Parse gfxXXXX from name (e.g., "amdgcn-amd-amdhsa--gfx942", "gfx1250")
    const char* gfx_str = strstr(isa_name, "gfx");
    if (gfx_str && strlen(gfx_str) > 3) {
      uint32_t version = 0;
      if (sscanf(gfx_str, "gfx%u", &version) == 1) {
        // gfx942  -> major=9, minor=4, stepping=2
        // gfx1200 -> major=12, minor=0, stepping=0
        // gfx1250 -> major=12, minor=5, stepping=0
        ver.major = version / 100;
        ver.minor = (version / 10) % 10;
        ver.stepping = version % 10;
      }
    }
    return ver;
  }

  bool IsValid() const { return major > 0; }
  bool IsGfx9() const { return major == 9; }
  bool IsGfx10() const { return major == 10; }
  bool IsGfx11() const { return major == 11; }
  bool IsGfx12() const { return major == 12; }
  bool IsGfx13() const { return major == 13; }
  bool IsGfx12OrLater() const { return major >= 12; }
  bool IsGfx1250() const { return major == 12 && minor == 5; }
};

// =============================================================================
// Query broadcast copy capability for an agent
// =============================================================================
// Returns the maximum number of destinations supported for broadcast copy.
// For GFX12+, returns 1024. For older GPUs, returns 0.
// =============================================================================

/**
 * @brief Query broadcast copy capability for an HSA agent.
 *
 * @param[in]  agent            The HSA agent to query.
 * @param[out] max_destinations Maximum supported destinations (0 if not supported).
 *
 * @return HSA_STATUS_SUCCESS on success, or error code on failure.
 */
static inline hsa_status_t hsa_amd_memory_broadcast_capability(hsa_agent_t agent,
                                                               uint32_t* max_destinations) {
  if (max_destinations == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Check if agent is a GPU
  hsa_device_type_t device_type;
  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  // Non-GPU agents have no broadcast capability (not an error, just 0 capability)
  if (device_type != HSA_DEVICE_TYPE_GPU) {
    *max_destinations = 0;
    return HSA_STATUS_SUCCESS;
  }

  // Use the general ISA version helper
  IsaVersion ver = IsaVersion::FromAgent(agent);

  // GFX9+ supports broadcast copy via shader fallback path (up to 1024 destinations)
  // GFX12+ has native SDMA multicast packet support
  // Both paths support the same max destinations
  if (ver.major >= 9) {
    *max_destinations = 1024;
    return HSA_STATUS_SUCCESS;
  }

  *max_destinations = 0;
  return HSA_STATUS_SUCCESS;
}

// =============================================================================
// Convenience wrapper: hsa_amd_memory_broadcast_copy
// =============================================================================
// This wrapper provides a simplified interface for broadcast copy operations
// by internally using the hsa_amd_memory_async_batch_copy API with
// HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST type.
// =============================================================================

/**
 * @brief Perform a broadcast memory copy (1 source -> N destinations).
 *
 * This is a convenience wrapper around hsa_amd_memory_async_batch_copy.
 *
 * @param[in] src              Pointer to source memory.
 * @param[in] src_agent        Agent owning the source memory.
 * @param[in] dst              Array of destination pointers.
 * @param[in] dst_agents       Array of agents owning each destination.
 * @param[in] num_dst          Number of destinations (1-1024).
 * @param[in] size             Number of bytes to copy.
 * @param[in] num_dep_signals  Number of dependency signals.
 * @param[in] dep_signals      Array of dependency signals to wait on.
 * @param[in] out_signal       Completion signal (decremented by 1 on completion).
 * @param[in] sdma_engine      SDMA engine selection hint (unused in wrapper).
 * @param[in] force_copy_on_sdma  Force SDMA engine (unused in wrapper).
 *
 * @return HSA_STATUS_SUCCESS on success, or appropriate error code.
 */
static inline hsa_status_t hsa_amd_memory_broadcast_copy(
    const void* src, hsa_agent_t src_agent, void* const* dst, const hsa_agent_t* dst_agents,
    uint32_t num_dst, size_t size, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t out_signal, hsa_amd_sdma_engine_id_t sdma_engine, bool force_copy_on_sdma) {
  // Validate inputs
  if (src == nullptr || dst == nullptr || num_dst == 0 || num_dst > 1024) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // size == 0 is a no-op (matches runtime behavior)
  if (size == 0) {
    return HSA_STATUS_SUCCESS;
  }

  // Build copy operation descriptor
  hsa_amd_memory_copy_op_t op;
  memset(&op, 0, sizeof(op));

  op.version = HSA_AMD_MEMORY_COPY_OP_VERSION;
  op.type = HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST;
  op.num_entries = num_dst;
  op.traffic_class = 0;  // Default QoS
  op.completion_signal = out_signal;
  op.src = const_cast<void*>(src);
  op.src_agent = src_agent;
  op.dst_list = const_cast<void**>(dst);
  op.dst_agent_list = const_cast<hsa_agent_t*>(dst_agents);
  op.size = size;
  op.unused_size = 0;

  // Submit via batch copy API
  return hsa_amd_memory_async_batch_copy(&op, 1, num_dep_signals, dep_signals);
}

// =============================================================================
// HsaTestContext: Simple HSA context manager for tests
// =============================================================================

class HsaTestContext {
 public:
  hsa_agent_t gpu_agent;
  hsa_agent_t cpu_agent;
  hsa_amd_memory_pool_t gpu_pool;
  hsa_amd_memory_pool_t cpu_pool;
  std::vector<hsa_agent_t> all_gpu_agents;

  HsaTestContext() : gpu_agent{0}, cpu_agent{0}, gpu_pool{0}, cpu_pool{0} {
    hsa_status_t status = hsa_init();
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
      std::cerr << "Warning: hsa_init failed" << std::endl;
      return;
    }

    // Find all GPU agents
    hsa_iterate_agents(
        [](hsa_agent_t agent, void* data) -> hsa_status_t {
          hsa_device_type_t type;
          hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
          if (err != HSA_STATUS_SUCCESS) return HSA_STATUS_SUCCESS;  // Skip on error
          if (type == HSA_DEVICE_TYPE_GPU) {
            static_cast<std::vector<hsa_agent_t>*>(data)->push_back(agent);
          }
          return HSA_STATUS_SUCCESS;
        },
        &all_gpu_agents);

    // Set first GPU as primary
    if (!all_gpu_agents.empty()) {
      gpu_agent = all_gpu_agents[0];
    }

    // Find CPU agent
    hsa_iterate_agents(
        [](hsa_agent_t agent, void* data) -> hsa_status_t {
          hsa_device_type_t type;
          hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
          if (err != HSA_STATUS_SUCCESS) return HSA_STATUS_SUCCESS;  // Skip on error
          if (type == HSA_DEVICE_TYPE_CPU) {
            *static_cast<hsa_agent_t*>(data) = agent;
            return HSA_STATUS_INFO_BREAK;
          }
          return HSA_STATUS_SUCCESS;
        },
        &cpu_agent);

    // Find GPU memory pool (device-local, coarse-grained)
    if (gpu_agent.handle != 0) {
      hsa_amd_agent_iterate_memory_pools(
          gpu_agent,
          [](hsa_amd_memory_pool_t pool, void* data) -> hsa_status_t {
            hsa_amd_segment_t segment;
            hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
            if (segment == HSA_AMD_SEGMENT_GLOBAL) {
              uint32_t flags;
              hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
              if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) != 0) {
                *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
                return HSA_STATUS_INFO_BREAK;
              }
            }
            return HSA_STATUS_SUCCESS;
          },
          &gpu_pool);
    }

    // Find CPU memory pool (system, coarse-grained)
    if (cpu_agent.handle != 0) {
      hsa_amd_agent_iterate_memory_pools(
          cpu_agent,
          [](hsa_amd_memory_pool_t pool, void* data) -> hsa_status_t {
            hsa_amd_segment_t segment;
            hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
            if (segment == HSA_AMD_SEGMENT_GLOBAL) {
              uint32_t flags;
              hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
              if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) != 0) {
                *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
                return HSA_STATUS_INFO_BREAK;
              }
            }
            return HSA_STATUS_SUCCESS;
          },
          &cpu_pool);
    }
  }

  ~HsaTestContext() {
    // Note: Don't call hsa_shut_down here as other tests may still be running
  }

  bool HasGPUAgent() const { return gpu_agent.handle != 0 && gpu_pool.handle != 0; }

  bool HasCPUAgent() const { return cpu_agent.handle != 0 && cpu_pool.handle != 0; }

  // Allocates CPU-pool memory accessible by GPU (for validation convenience).
  // Named "GPUBuffer" for caller semantics: buffer is used by GPU operations.
  // Returns nullptr on allocation failure.
  void* AllocateGPUBuffer(size_t size) {
    void* ptr = nullptr;
    // Use CPU pool for CPU-accessible GPU buffers (coarse-grained GPU memory
    // may not be directly CPU accessible on some hardware).
    // The CPU pool memory is still GPU-accessible via allow_access.
    hsa_status_t status = hsa_amd_memory_pool_allocate(cpu_pool, size, 0, &ptr);
    if (status != HSA_STATUS_SUCCESS) {
      return nullptr;
    }
    // Allow access from GPU
    if (gpu_agent.handle != 0) {
      hsa_amd_agents_allow_access(1, &gpu_agent, nullptr, ptr);
    }
    return ptr;
  }

  void* AllocateCPUBuffer(size_t size) {
    void* ptr = nullptr;
    hsa_status_t status = hsa_amd_memory_pool_allocate(cpu_pool, size, 0, &ptr);
    if (status != HSA_STATUS_SUCCESS) {
      return nullptr;
    }
    // Allow access from GPU
    if (gpu_agent.handle != 0) {
      hsa_amd_agents_allow_access(1, &gpu_agent, nullptr, ptr);
    }
    return ptr;
  }

  void Free(void* ptr) {
    if (ptr != nullptr) {
      hsa_amd_memory_pool_free(ptr);
    }
  }

  // Get ISA version using the general helper
  IsaVersion GetIsaVersion() const {
    if (gpu_agent.handle == 0) return IsaVersion();
    return IsaVersion::FromAgent(gpu_agent);
  }

  uint32_t GetGfxMajorVersion() const {
    return GetIsaVersion().major;
  }

  bool IsMulticastSupported() const { return GetIsaVersion().IsGfx13(); }

  bool IsBroadcastSupported() const { return GetIsaVersion().IsGfx12OrLater(); }
};

// =============================================================================
// BroadcastTestUtils: Helper functions for broadcast copy tests
// =============================================================================

class BroadcastTestUtils {
 public:
  // Pattern types for test data
  enum PatternType {
    ZERO,
    ONES,
    SEQUENTIAL,
    RANDOM,
    ALTERNATING,
    CHECKERBOARD,
    INCREMENTAL,  // Alias for SEQUENTIAL
    WALKING_BIT   // Walking bit pattern
  };

  // Alias for compatibility
  typedef PatternType Pattern;

  // Fill buffer with specified pattern
  static void FillPattern(void* buffer, size_t size, PatternType pattern, uint32_t seed = 0) {
    uint8_t* ptr = static_cast<uint8_t*>(buffer);

    switch (pattern) {
      case ZERO:
        memset(buffer, 0x00, size);
        break;

      case ONES:
        memset(buffer, 0xFF, size);
        break;

      case INCREMENTAL:
        // INCREMENTAL is an alias for SEQUENTIAL - fall through
        [[fallthrough]];
      case SEQUENTIAL:
        for (size_t i = 0; i < size; ++i) {
          ptr[i] = static_cast<uint8_t>((i + seed) & 0xFF);
        }
        break;

      case RANDOM: {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint32_t> dist(0, 255);
        for (size_t i = 0; i < size; ++i) {
          ptr[i] = static_cast<uint8_t>(dist(rng));
        }
        break;
      }

      case ALTERNATING:
        for (size_t i = 0; i < size; ++i) {
          ptr[i] = (i & 1) ? 0xAA : 0x55;
        }
        break;

      case CHECKERBOARD: {
        uint32_t* ptr32 = static_cast<uint32_t*>(buffer);
        size_t count = size / sizeof(uint32_t);
        for (size_t i = 0; i < count; ++i) {
          ptr32[i] = (i & 1) ? 0xDEADBEEF : 0xCAFEBABE;
        }
        // Handle remaining bytes
        size_t remaining = size % sizeof(uint32_t);
        if (remaining > 0) {
          memset(ptr + (count * sizeof(uint32_t)), 0xCC, remaining);
        }
        break;
      }

      case WALKING_BIT:
        // Walking bit pattern - a single bit walks through each byte
        for (size_t i = 0; i < size; ++i) {
          ptr[i] = static_cast<uint8_t>(1 << ((i + seed) % 8));
        }
        break;
    }
  }

  // Verify buffer matches expected pattern
  static bool VerifyPattern(const void* buffer, size_t size, PatternType pattern,
                            uint32_t seed = 0) {
    if (buffer == nullptr || size == 0) return false;
    std::vector<uint8_t> expected(size);
    FillPattern(expected.data(), size, pattern, seed);
    return memcmp(buffer, expected.data(), size) == 0;
  }

  // Compare two buffers
  static bool CompareBuffers(const void* buf1, const void* buf2, size_t size) {
    if (buf1 == nullptr || buf2 == nullptr) return false;
    if (size == 0) return true;
    return memcmp(buf1, buf2, size) == 0;
  }

  // Find first mismatch between buffers
  static bool FindMismatch(const void* buf1, const void* buf2, size_t size,
                           size_t* mismatch_offset = nullptr) {
    if (buf1 == nullptr || buf2 == nullptr || size == 0) return false;
    const uint8_t* p1 = static_cast<const uint8_t*>(buf1);
    const uint8_t* p2 = static_cast<const uint8_t*>(buf2);

    for (size_t i = 0; i < size; ++i) {
      if (p1[i] != p2[i]) {
        if (mismatch_offset) *mismatch_offset = i;
        return true;
      }
    }
    return false;
  }

  // Create HSA signal
  static hsa_signal_t CreateSignal(hsa_signal_value_t initial_value) {
    hsa_signal_t signal;
    hsa_status_t status = hsa_signal_create(initial_value, 0, nullptr, &signal);
    if (status != HSA_STATUS_SUCCESS) {
      signal.handle = 0;
    }
    return signal;
  }

  // Destroy HSA signal
  static void DestroySignal(hsa_signal_t signal) {
    if (signal.handle != 0) {
      hsa_signal_destroy(signal);
    }
  }

  // Wait for signal to reach expected value (default: 0)
  static void WaitSignal(hsa_signal_t signal,
                         hsa_signal_condition_t condition = HSA_SIGNAL_CONDITION_LT,
                         hsa_signal_value_t compare_value = 1, uint64_t timeout_ns = UINT64_MAX) {
    hsa_signal_wait_scacquire(signal, condition, compare_value, timeout_ns, HSA_WAIT_STATE_BLOCKED);
  }

  // Reset signal to new value
  static void ResetSignal(hsa_signal_t signal, hsa_signal_value_t value) {
    hsa_signal_store_screlease(signal, value);
  }

  // Print hex dump of buffer (for debugging)
  static void HexDump(const void* buffer, size_t size, size_t max_bytes = 64) {
    const uint8_t* ptr = static_cast<const uint8_t*>(buffer);
    size_t to_print = (size < max_bytes) ? size : max_bytes;

    for (size_t i = 0; i < to_print; ++i) {
      if (i % 16 == 0) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << i << ": ";
      }
      std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)ptr[i] << " ";
      if ((i + 1) % 16 == 0) {
        std::cout << std::endl;
      }
    }
    if (to_print % 16 != 0) {
      std::cout << std::endl;
    }
    if (to_print < size) {
      std::cout << "... (" << std::dec << (size - to_print) << " more bytes)" << std::endl;
    }
    std::cout << std::dec;  // Reset to decimal
  }

  // Calculate bandwidth in GB/s
  static double CalculateBandwidthGBps(size_t bytes, uint64_t time_us) {
    if (time_us == 0) return 0.0;
    double bytes_per_second = (double)bytes / ((double)time_us / 1e6);
    return bytes_per_second / (1024.0 * 1024.0 * 1024.0);
  }

  // Calculate effective bandwidth for broadcast (total data written)
  static double CalculateEffectiveBandwidthGBps(size_t bytes, uint32_t num_dsts, uint64_t time_us) {
    // Cast to uint64_t to prevent overflow when bytes is large and num_dsts is up to 1024
    return CalculateBandwidthGBps(static_cast<uint64_t>(bytes) * num_dsts, time_us);
  }

  // Find the default memory region for an agent (for hsa_memory_allocate)
  static hsa_region_t FindDefaultRegion(hsa_agent_t agent) {
    hsa_region_t result = {0};
    hsa_agent_iterate_regions(
        agent,
        [](hsa_region_t region, void* data) -> hsa_status_t {
          hsa_region_segment_t segment;
          hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
          if (segment == HSA_REGION_SEGMENT_GLOBAL) {
            hsa_region_global_flag_t flags;
            hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
            if ((flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0) {
              *static_cast<hsa_region_t*>(data) = region;
              return HSA_STATUS_INFO_BREAK;
            }
          }
          return HSA_STATUS_SUCCESS;
        },
        &result);
    return result;
  }

  // Find all GPU agents in the system
  // Uses rocrtst::IterateGPUAgents from common/common.h
  static std::vector<hsa_agent_t> FindAllGPUAgents() {
    std::vector<hsa_agent_t> gpus;
    hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
    return gpus;
  }

  // Find first GPU agent in the system
  // Uses rocrtst::FindGPUDevice from common/common.h
  static hsa_agent_t FindGPUAgent() {
    hsa_agent_t gpu_agent = {0};
    hsa_iterate_agents(rocrtst::FindGPUDevice, &gpu_agent);
    return gpu_agent;
  }

  // Find first CPU agent in the system
  // Uses rocrtst::FindCPUDevice from common/common.h
  static hsa_agent_t FindCPUAgent() {
    hsa_agent_t cpu_agent = {0};
    hsa_iterate_agents(rocrtst::FindCPUDevice, &cpu_agent);
    return cpu_agent;
  }

  // Calculate bandwidth (alias for CalculateBandwidthGBps)
  static double CalculateBandwidth(size_t bytes, uint64_t time_us) {
    return CalculateBandwidthGBps(bytes, time_us);
  }

  // Get agent name string
  static std::string GetAgentName(hsa_agent_t agent) {
    char name[64] = {0};
    hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
    return std::string(name);
  }
};

#endif  // ROCRTST_COMMON_BROADCAST_COPY_UTILS_H_
