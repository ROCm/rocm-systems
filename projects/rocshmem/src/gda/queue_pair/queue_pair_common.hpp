/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_GDA_QUEUE_PAIR_COMMON_HPP_
#define LIBRARY_SRC_GDA_QUEUE_PAIR_COMMON_HPP_

/**
 * @file queue_pair_common.hpp
 *
 * @section DESCRIPTION
 * An IB QueuePair (SQ and CQ) that the device can use to perform network
 * operations. Most important rocSHMEM operations are performed by this
 * class.
 */

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "util.hpp"

namespace rocshmem {

struct BufferInfo {
  uintptr_t addr;
  size_t    length;
  uint32_t  lkey;
};

/**
 * @brief One symmetric user-buffer registration, pre-specialized to a single
 * QueuePair's fixed (dest_pe, nic_idx).
 *
 * Because every QP talks to exactly one peer over exactly one NIC, all the
 * per-PE / per-NIC indexing can be resolved on the host at registration time,
 * leaving the device with a self-contained 32-byte record: matching an address
 * and producing the remote address + keys needs no pointer chasing. The host
 * fills one entry per (pe, nic) per registration (see GDABackend's flat entry
 * table), and each QP is handed the contiguous slice for its own (dest_pe,
 * nic_idx), scanned by registration slot.
 */
struct SymmBufferInfo {
  /**
   * @brief This PE's registered alias base (identical across all slices).
   *
   * Used to recognize whether a symmetric address falls in this registration
   * and to compute the intra-registration offset.
   */
  uintptr_t local_base;

  /**
   * @brief The peer's alias base for this slice's dest_pe.
   *
   * Remote address for a transfer is remote_base + (sym_addr - local_base).
   */
  uintptr_t remote_base;

  /**
   * @brief Registered length in bytes.
   */
  size_t length;

  /**
   * @brief Local key for this slice's nic_idx (locally-sourced buffers).
   */
  uint32_t lkey;

  /**
   * @brief Remote key for this slice's (dest_pe, nic_idx).
   */
  uint32_t rkey;
};

/**
 * @brief Scope at which WQEs are issued and completed. This is used to
 * determine how to synchronize threads and when to poll the CQ for
 * completions.
 * thread: Each thread issues WQEs independently
 * wave: Thread 0 in each wave issues WQE
 * wg: Thread 0 of WAVE 0 issues WQE
 */

enum class ThreadScope: int {
  thread,
  wave,
  wg
};

class ActiveWFInfo {
 public:
  uint64_t    activemask{0};                  // Mask of active threads in the wavefront
  uint64_t    pe_group_mask{0};               // Mask of active threads with the same PE
  int         pe{-1};                         // PE for the threads in pe_group_mask
  int         num_pe_group_lanes{0};          // Number of active lanes in pe_group_mask
  int         pe_group_logical_lane_id{0};    // Logical lane id of this thread in pe_group_mask
  int         pe_group_first_phys_lane_id{0}; // Physical lane id of first thread in pe_group_mask
  int         pe_group_last_phys_lane_id{0};  // Physical lane id of last thread in pe_group_mask
  ThreadScope scope{ThreadScope::thread};     // Threading scope
  bool        is_pe_group_first{false};       // True if this is the first thread in pe_group_mask
  bool        is_pe_group_last{false};        // True if this is the last thread in pe_group_mask

  __device__ explicit ActiveWFInfo(int pe, ThreadScope scope = ThreadScope::thread)
      : pe(pe), scope(scope) {
    // Get active lane mask
    activemask = get_active_lane_mask();

    // Get mask of active lanes with the same PE
    switch (scope) {
      case ThreadScope::thread: {
        pe_group_mask       = __match_any_sync(activemask, pe);
        num_pe_group_lanes  = get_active_lane_count(pe_group_mask);
        pe_group_logical_lane_id = get_active_lane_num(pe_group_mask);
        pe_group_first_phys_lane_id = get_first_active_lane_id(pe_group_mask);
        pe_group_last_phys_lane_id  = get_last_active_lane_id(pe_group_mask);
        break;
      }
      // Only thread 0 issues the WQE, so the group is just that thread
      case ThreadScope::wave:
      case ThreadScope::wg: {
        pe_group_mask       = 1;
        num_pe_group_lanes  = 1;
        pe_group_logical_lane_id = get_active_lane_num(activemask);
        pe_group_first_phys_lane_id = 0;
        pe_group_last_phys_lane_id  = 0;
      }
    }
    is_pe_group_first = (pe_group_logical_lane_id == 0);
    is_pe_group_last  = (pe_group_logical_lane_id == num_pe_group_lanes - 1);
  }

  // used in CAS based atomic operations at thread scope
  __device__ void update(int _pe, ThreadScope _scope = ThreadScope::thread) {
    // Get active lane mask
    activemask          = get_active_lane_mask();
    pe_group_mask       = __match_any_sync(activemask, pe);
    num_pe_group_lanes  = get_active_lane_count(pe_group_mask);
    pe_group_logical_lane_id = get_active_lane_num(pe_group_mask);
    pe_group_first_phys_lane_id = get_first_active_lane_id(pe_group_mask);
    pe_group_last_phys_lane_id  = get_last_active_lane_id(pe_group_mask);
    is_pe_group_first   = (pe_group_logical_lane_id == 0);
    is_pe_group_last    = (pe_group_logical_lane_id == num_pe_group_lanes - 1);
    scope               = _scope;
    pe                  = _pe;
  }

  __device__ void printInfo() {
    printf("PE: %d, Scope: %d, activemask: %llx, "
           "pe_group_mask: %llx, num_pe_group_lanes: %d, "
           "thread_id: %u, pe_group_logical_lane_id: %d, "
           "is_pe_group_first: %d, pe_group_first_phys_lane_id: %d, "
           "is_pe_group_last: %d, pe_group_last_phys_lane_id: %d\n",
           pe, static_cast<int>(scope), static_cast<unsigned long long>(activemask),
           static_cast<unsigned long long>(pe_group_mask), num_pe_group_lanes,
           threadIdx.x, pe_group_logical_lane_id,
           static_cast<int>(is_pe_group_first), pe_group_first_phys_lane_id,
           static_cast<int>(is_pe_group_last), pe_group_last_phys_lane_id);
  }
};

/*
 * @struct QueuePairTraits<Provider>
 * @brief Defines Provider-specific types and constants.
 *
 * Each Provider subclass of QueuePairBase<Provider> should also define a specialization
 * for QueuePairTraits<Provider> that defines the documented members:
 *   - QueuePairTraits<Provider>::OpCode
 *   - QueuePairTraits<Provider>::Endianness
 *   - QueuePairTraits<Provider>::InlineMax
 *   - QueuePairTraits<Provider>::InlineThreshold
 *
 * Sample specialization code for a QueuePairProvider subclass
 * of QueuePairBase<QueuePairProvider>:
 * @code
 * class QueuePairProvider;
 * template <> struct QueuePairTraits<QueuePairProvider> {
 *   enum class OpCode : uint8_t {
 *     RDMA_WRITE = ...,
 *     RDMA_READ  = ...,
 *     ATOMIC_CS  = ...,
 *     ATOMIC_FA  = ...,
 *   };
 *
 *   static constexpr endian::Order Endianness = endian::Order::...;
 *
 *   static constexpr size_t InlineMax       = ...;
 *   static constexpr size_t InlineThreshold = ...;
 * };
 * @endcode
 *
 * @tparam Provider Name of Provider class.
 */

/*
 * @enum QueuePairTraits<Provider>::OpCode
 * @brief Enumeration of the Provider-specific opcodes.
 *
 * @var QueuePairTraits<Provider>::OpCode::RDMA_WRITE
 * Provider-specific RDMA Write opcode.
 *
 * @var QueuePairTraits<Provider>::OpCode::RDMA_READ
 * Provider-specific RDMA Read opcode.
 *
 * @var QueuePairTraits<Provider>::OpCode::ATOMIC_CS
 * Provider-specific Compare-and-Swap opcode.
 *
 * @var QueuePairTraits<Provider>::OpCode::ATOMIC_FA
 * Provider-specific Fetch-Add opcode.
 */

/*
 * @var endian::Order QueuePairTraits<Provider>::Endianness
 * @brief Endianness order of data stored by the hardware for Provider.
 *
 * @qualifier static
 * @qualifier constexpr
 */

/*
 * @var size_t QueuePairTraits<Provider>::InlineMax
 * @brief Maximum number of bytes that can be sent inline in a WQE.
 *
 * @qualifer static
 * @qualifier constexpr
 */

/*
 * @var size_t QueuePairTraits<Provider>::InlineThreshold
 * @brief Preferred maximum number of bytes that should be sent inline in a WQE.
 *
 * @qualifer static
 * @qualifier constexpr
 */

template <typename Provider>
struct QueuePairTraits;

/*
 * @brief Atomic Memory Operation fetching behavior.
 */
enum class AMOFetchType {
  Blocking,
  NonBlocking,
  NonFetching,
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_QUEUE_PAIR_COMMON_HPP_
