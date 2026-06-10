/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_COMMON_INDIRECT_COPY_UTILS_H_
#define ROCRTST_COMMON_INDIRECT_COPY_UTILS_H_

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

// Include shared utilities from broadcast_copy_utils.h
#include "broadcast_copy_utils.h"

// =============================================================================
// Convenience wrapper: hsa_amd_memory_indirect_copy
// =============================================================================
// This wrapper provides a simplified interface for indirect copy operations
// by internally using the hsa_amd_memory_async_batch_copy API with
// HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_* types.
// =============================================================================

/**
 * @brief Perform an indirect memory copy where source address is resolved via indirection.
 *
 * @param[in] src_ptr          Pointer to pointer containing actual source address (*src_ptr).
 * @param[in] src_agent        Agent owning the source memory.
 * @param[in] dst              Destination pointer (direct).
 * @param[in] dst_agent        Agent owning the destination memory.
 * @param[in] size             Number of bytes to copy.
 * @param[in] num_dep_signals  Number of dependency signals.
 * @param[in] dep_signals      Array of dependency signals to wait on.
 * @param[in] out_signal       Completion signal (decremented by 1 on completion).
 *
 * @return HSA_STATUS_SUCCESS on success, or appropriate error code.
 */
static inline hsa_status_t hsa_amd_memory_indirect_copy_src(void** src_ptr, hsa_agent_t src_agent,
                                                            void* dst, hsa_agent_t dst_agent,
                                                            size_t size, uint32_t num_dep_signals,
                                                            const hsa_signal_t* dep_signals,
                                                            hsa_signal_t out_signal) {
  if (src_ptr == nullptr || dst == nullptr) {
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
  op.type = HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRC;
  op.num_entries = 0;  // Required for indirect operations
  op.traffic_class = 0;
  op.completion_signal = out_signal;
  op.src = reinterpret_cast<void*>(src_ptr);  // Pointer-to-pointer
  op.src_agent = src_agent;
  op.dst = dst;
  op.dst_agent = dst_agent;
  op.size = size;
  op.unused_size = 0;

  return hsa_amd_memory_async_batch_copy(&op, 1, num_dep_signals, dep_signals);
}

/**
 * @brief Perform an indirect memory copy where destination address is resolved via indirection.
 *
 * @param[in] src              Source pointer (direct).
 * @param[in] src_agent        Agent owning the source memory.
 * @param[in] dst_ptr          Pointer to pointer containing actual destination address (*dst_ptr).
 * @param[in] dst_agent        Agent owning the destination memory.
 * @param[in] size             Number of bytes to copy.
 * @param[in] num_dep_signals  Number of dependency signals.
 * @param[in] dep_signals      Array of dependency signals to wait on.
 * @param[in] out_signal       Completion signal (decremented by 1 on completion).
 *
 * @return HSA_STATUS_SUCCESS on success, or appropriate error code.
 */
static inline hsa_status_t hsa_amd_memory_indirect_copy_dst(void* src, hsa_agent_t src_agent,
                                                            void** dst_ptr, hsa_agent_t dst_agent,
                                                            size_t size, uint32_t num_dep_signals,
                                                            const hsa_signal_t* dep_signals,
                                                            hsa_signal_t out_signal) {
  if (src == nullptr || dst_ptr == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // size == 0 is a no-op (matches runtime behavior)
  if (size == 0) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_amd_memory_copy_op_t op;
  memset(&op, 0, sizeof(op));

  op.version = HSA_AMD_MEMORY_COPY_OP_VERSION;
  op.type = HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_DST;
  op.num_entries = 0;
  op.traffic_class = 0;
  op.completion_signal = out_signal;
  op.src = src;
  op.src_agent = src_agent;
  op.dst = reinterpret_cast<void*>(dst_ptr);  // Pointer-to-pointer
  op.dst_agent = dst_agent;
  op.size = size;
  op.unused_size = 0;

  return hsa_amd_memory_async_batch_copy(&op, 1, num_dep_signals, dep_signals);
}

/**
 * @brief Perform an indirect memory copy where both addresses are resolved via indirection.
 *
 * @param[in] src_ptr          Pointer to pointer containing actual source address (*src_ptr).
 * @param[in] src_agent        Agent owning the source memory.
 * @param[in] dst_ptr          Pointer to pointer containing actual destination address (*dst_ptr).
 * @param[in] dst_agent        Agent owning the destination memory.
 * @param[in] size             Number of bytes to copy.
 * @param[in] num_dep_signals  Number of dependency signals.
 * @param[in] dep_signals      Array of dependency signals to wait on.
 * @param[in] out_signal       Completion signal (decremented by 1 on completion).
 *
 * @return HSA_STATUS_SUCCESS on success, or appropriate error code.
 */
static inline hsa_status_t hsa_amd_memory_indirect_copy_srcdst(
    void** src_ptr, hsa_agent_t src_agent, void** dst_ptr, hsa_agent_t dst_agent, size_t size,
    uint32_t num_dep_signals, const hsa_signal_t* dep_signals, hsa_signal_t out_signal) {
  if (src_ptr == nullptr || dst_ptr == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // size == 0 is a no-op (matches runtime behavior)
  if (size == 0) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_amd_memory_copy_op_t op;
  memset(&op, 0, sizeof(op));

  op.version = HSA_AMD_MEMORY_COPY_OP_VERSION;
  op.type = HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRCDST;
  op.num_entries = 0;
  op.traffic_class = 0;
  op.completion_signal = out_signal;
  op.src = reinterpret_cast<void*>(src_ptr);  // Pointer-to-pointer
  op.src_agent = src_agent;
  op.dst = reinterpret_cast<void*>(dst_ptr);  // Pointer-to-pointer
  op.dst_agent = dst_agent;
  op.size = size;
  op.unused_size = 0;

  return hsa_amd_memory_async_batch_copy(&op, 1, num_dep_signals, dep_signals);
}

// =============================================================================
// IndirectCopyTestUtils: Helper functions for indirect copy tests
// =============================================================================

class IndirectCopyTestUtils {
 public:
  // Allocate a GPU-accessible buffer containing a pointer value
  // Returns pointer to the allocated buffer; the buffer contains ptr_value
  // Returns nullptr on allocation failure; caller must check.
  static void** AllocatePointerBuffer(HsaTestContext& ctx, void* ptr_value) {
    void** ptr_buf = reinterpret_cast<void**>(ctx.AllocateGPUBuffer(sizeof(void*)));
    if (ptr_buf != nullptr) {
      *ptr_buf = ptr_value;
    }
    return ptr_buf;
  }

  // Verify copy result and print diagnostics
  static bool VerifyAndReport(const char* test_name, const void* src, const void* dst, size_t size,
                              bool verbose = true) {
    if (src == nullptr || dst == nullptr) {
      if (verbose) {
        std::cout << "  [" << test_name << "] [FAIL] NULL pointer passed" << std::endl;
      }
      return false;
    }
    bool match = (memcmp(src, dst, size) == 0);
    if (verbose) {
      if (match) {
        std::cout << "  [" << test_name << "] [PASS] Data integrity verified (" << size << " bytes)"
                  << std::endl;
      } else {
        std::cout << "  [" << test_name << "] [FAIL] Data mismatch detected" << std::endl;
        // Find first mismatch
        const uint8_t* s = static_cast<const uint8_t*>(src);
        const uint8_t* d = static_cast<const uint8_t*>(dst);
        for (size_t i = 0; i < size && i < 10; ++i) {
          if (s[i] != d[i]) {
            std::cout << "    First mismatch at offset " << i << ": expected=0x" << std::hex
                      << (int)s[i] << ", actual=0x" << (int)d[i] << std::dec << std::endl;
            break;
          }
        }
      }
    }
    return match;
  }
};

#endif  // ROCRTST_COMMON_INDIRECT_COPY_UTILS_H_
