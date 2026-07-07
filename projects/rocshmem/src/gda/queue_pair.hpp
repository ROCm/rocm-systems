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

#ifndef LIBRARY_SRC_GDA_QUEUE_PAIR_HPP_
#define LIBRARY_SRC_GDA_QUEUE_PAIR_HPP_

/**
 * @file queue_pair.hpp
 *
 * @section DESCRIPTION
 * An IB QueuePair (SQ and CQ) that the device can use to perform network
 * operations. Most important rocSHMEM operations are performed by this
 * class.
 */

#include "rocshmem_config.h"
#include "endian.h"
#include "constants.hpp"
#include "util.hpp"

#include "ibv_wrapper.hpp"

#include "gda/ionic/provider_gda_ionic.hpp"
#include "gda/mlx5/provider_gda_mlx5.hpp"
#include "gda/bnxt/provider_gda_bnxt.hpp"

#include "containers/free_list.hpp"
#include "memory/hip_allocator.hpp"

#include <map>

namespace rocshmem {

class GDABackend;

struct user_buf_info_t {
  uintptr_t addr;
  size_t    length;
  uint32_t  lkey;
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

class QueuePair {
 public:
  friend GDABackend;

  /**
   * @brief Constructor.
   */
  explicit QueuePair(struct ibv_pd* pd, int gda_provider);

  /**
   * @brief Destructor.
   */
  virtual ~QueuePair();

  /**
   * @brief Create and enqueue a non-blocking put work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] source Source address for data transmission.
   * @param[in] length Size in bytes of data transmission.
   * @param[in] wf_info Wavefront information.
   */
  __device__ void put_nbi(void *dest, const void *source, size_t length,
      ActiveWFInfo &wf_info);

  __device__ void put_nbi_single(void *dest, const void *source, size_t length,
      bool ring_db);

  __device__ void put_nbi_single(void *raddr, uint32_t rkey,
      const void *laddr, uint32_t lkey,
      size_t length, bool ring_db = true);

  /**
   * @brief Create and enqueue a non-blocking put with explicit rkey/lkey.
   *
   * Used when each buffer registration has its own keys, distinct from
   * the QP's default heap keys.
   *
   * @param[in] length Size in bytes of data transmission.
   * @param[in] raddr Remote destination address.
   * @param[in] rkey Remote key for the destination buffer.
   * @param[in] laddr Local source address.
   * @param[in] lkey Local key for the source buffer.
   * @param[in] wf_info Wavefront information.
   * @param[in] ring_db Ring doorbell after posting (default true).
   */
  __device__ void put_nbi(void *raddr, uint32_t rkey,
      const void *laddr, uint32_t lkey,
      size_t length, ActiveWFInfo &wf_info, bool ring_db = true);

  /**
   * @brief Create and enqueue a non-blocking get work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] source Source address for data transmission.
   * @param[in] length Size in bytes of data transmission.
   * @param[in] wf_info Wavefront information.
   */
  __device__ void get_nbi(void *dest, const void *source, size_t length,
      ActiveWFInfo &wf_info);

  __device__ void get_nbi_single(void *dest, const void *source, size_t length,
      bool ring_db);

  /**
   * @brief Empty all completions from the completion queue.
   * @param[in] wf_info Wavefront information.
   */
  __device__ void quiet(ActiveWFInfo &wf_info);

  __device__ void quiet_single();

  /**
   * @brief Empty all completions from the completion queue.
   * @param[in] wf_info Wavefront information.
   */
  __device__ void quiet_scope(ActiveWFInfo &wf_info);

  /**
   * @brief Create and enqueue an atomic fetch work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] wf_info Wavefront information.
   *
   * @return An atomic value
   */
  __device__ int64_t atomic_fetch(void *dest, int64_t value, int64_t cond,
      ActiveWFInfo &wf_info);

  /**
   * @brief Create and enqueue an atomic fetch work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] wf_info Wavefront information.
   */
  __device__ void atomic_nofetch(void *dest, int64_t value, int64_t cond,
      ActiveWFInfo &wf_info);

  __device__ void atomic_nofetch_single(void *dest, int64_t value);

  __device__ void atomic_add_single(void *raddr, uint32_t rkey,
      int64_t value, bool fence = false);

  /**
   * @brief Non-fetching atomic add with explicit remote key.
   *
   * Used when the signal buffer has its own rkey distinct from
   * the QP's default heap key.
   *
   * @param[in] raddr Remote destination address.
   * @param[in] rkey  Remote key for the destination buffer.
   * @param[in] value Atomic add value.
   * @param[in] wf_info Wavefront information.
   */
  __device__ void atomic_add(void *raddr, uint32_t rkey,
      int64_t value, ActiveWFInfo &wf_info, bool fence = false);

  /**
   * @brief Create and enqueue an atomic cas work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] wf_info Wavefront information.
   *
   * @return An atomic value
   */
  __device__ int64_t atomic_cas(void *dest, int64_t atomic_data,
      int64_t atomic_cmp, ActiveWFInfo &wf_info);

  /**
   * @brief Create and enqueue an atomic cas work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] wf_info Wavefront information.
   */
  __device__ int64_t atomic_cas_nofetch(void *dest, int64_t atomic_data,
      int64_t atomic_cmp, ActiveWFInfo &wf_info);

  uintptr_t base_heap = 0;
  size_t base_heap_size = 0;

 private:
  /**
   * @brief Helper method to build work requests for the send queue.
   *
   * @param[in] raddr Remote address.
   * @param[in] rkey Remote key.
   * @param[in] opcode Operation to be performed.
   * @param[in] atomic_data An atomic data value to be used.
   * @param[in] atomic_cmp An atomic comparison value.
   * @param[in] wf_info Wavefront information.
   * @param[in] fetching True if the operation returns a value.
   * @param[in] fence True to set fence flag on the WQE.
   */
  __device__ __attribute__((noinline)) uint64_t
  post_wqe_amo(uintptr_t raddr, uint32_t rkey, uint8_t opcode,
      int64_t atomic_data, int64_t atomic_cmp,
      ActiveWFInfo &wf_info, bool fetching = false, bool fence = false);

#if defined(GDA_IONIC)
  __device__ uint64_t ionic_post_wqe_amo(uintptr_t raddr, uint32_t rkey,
      uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp,
      ActiveWFInfo &wf_info, bool fetching = false, bool fence = false);
#endif
#if defined(GDA_BNXT)
  __device__ uint64_t bnxt_post_wqe_amo(uintptr_t raddr, uint32_t rkey,
      uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp,
      ActiveWFInfo &wf_info, bool fetching = false, bool fence = false);
#endif
#if defined(GDA_MLX5)
  __device__ uint64_t mlx5_post_wqe_amo(uintptr_t raddr, uint32_t rkey,
      uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp,
      ActiveWFInfo &wf_info, bool fetching = false, bool fence = false);
#endif

  __device__ __attribute__((noinline)) uint64_t post_wqe_amo_single(uintptr_t raddr,
      uint32_t rkey, uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp,
      bool fetching = false, bool fence = false);

  /**
   * @brief Build and post an RMA work queue entry with explicit keys.
   *
   * @param[in] size Size in bytes of data transmission.
   * @param[in] raddr Remote address.
   * @param[in] rkey Remote key.
   * @param[in] laddr Local address.
   * @param[in] lkey Local key.
   * @param[in] opcode Operation to be performed.
   * @param[in] wf_info Wavefront information.
   * @param[in] ring_db Ring doorbell after posting.
   */
  __device__ __attribute__((noinline)) void
  post_wqe_rma(int32_t length, uintptr_t raddr, uint32_t rkey,
      uintptr_t laddr, uint32_t lkey,
      uint8_t opcode, ActiveWFInfo &wf_info, bool ring_db);

  __device__ __attribute__((noinline)) void
  post_wqe_rma_single(int32_t length, uintptr_t laddr, uint32_t lkey,
      uintptr_t raddr, uint32_t rkey, uint8_t opcode, bool ring_db);

#if defined(GDA_MLX5)
  __device__ uint64_t mlx5_post_wqe_amo_single(uintptr_t raddr, uint32_t rkey,
      uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp,
      bool fetch = false, bool fence = false);
  __device__ void mlx5_post_wqe_rma_single(int32_t length, uintptr_t laddr,
      uint32_t lkey, uintptr_t raddr, uint32_t rkey,
      uint8_t opcode, bool ring_db);
  __device__ void mlx5_post_wqe_rma(int32_t length, uintptr_t raddr,
      uint32_t rkey, uintptr_t laddr, uint32_t lkey,
      uint8_t opcode, ActiveWFInfo &wf_info, bool ring_db);
  __device__ void mlx5_quiet();
  __device__ void mlx5_quiet_single();
#endif
#if defined(GDA_BNXT)

  __device__ void bnxt_write_rma_wqe(int32_t length, uintptr_t raddr,
      uint32_t rkey, uintptr_t laddr, uint32_t lkey, uint8_t opcode);
  __device__ uint32_t bnxt_write_amo_wqe(uintptr_t raddr, uint32_t rkey,
      uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp,
      bool fetching, bool fence);

  __device__ uint64_t bnxt_post_wqe_amo_single(uintptr_t raddr, uint32_t rkey,
      uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp,
      bool fetching = false, bool fence = false);
  __device__ void bnxt_post_wqe_rma_single(int32_t length, uintptr_t laddr,
      uint32_t lkey, uintptr_t raddr, uint32_t rkey,
      uint8_t opcode, bool ring_db);
  __device__ void bnxt_post_wqe_rma(int32_t length, uintptr_t raddr,
      uint32_t rkey, uintptr_t laddr, uint32_t lkey,
      uint8_t opcode, ActiveWFInfo &wf_info, bool ring_db);
  __device__ void bnxt_quiet();
  __device__ void bnxt_quiet_single();
#endif
#if defined(GDA_IONIC)
  __device__ uint64_t ionic_post_wqe_amo_single(uintptr_t raddr, uint32_t rkey,
      uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp,
      bool fetch = false, bool fence = false);
  __device__ void ionic_post_wqe_rma_single(int32_t length,
      uintptr_t laddr, uint32_t lkey, uintptr_t raddr,
      uint32_t rkey, uint8_t opcode, bool ring_db);
  __device__ void ionic_post_wqe_rma(int32_t length, uintptr_t raddr,
      uint32_t rkey, uintptr_t laddr, uint32_t lkey,
      uint8_t opcode, ActiveWFInfo &wf_info, bool ring_db);
  __device__ void ionic_quiet(ActiveWFInfo &wf_info);
  __device__ void ionic_quiet_single();
#endif

  /**
   * @brief Helper method to ring the doorbell
   *
   * @param[in] db_val Doorbell value is written by method.
   */
#if defined(GDA_MLX5)
  __device__ void mlx5_ring_doorbell(uint64_t sq_post, const gda_mlx5_wqe& wqe);
#endif
#if defined(GDA_BNXT)
  __device__ void bnxt_ring_doorbell(uint32_t slot_idx);
#endif
#if defined(GDA_IONIC)
  __device__ void ionic_ring_doorbell(uint32_t pos);
  __device__ void ionic_ring_doorbell_single(uint32_t pos);
#endif

  // TODO: make private once gin_qp_factory uses a proper init API
 public:
  /* GDAProvider::BNXT START */
  uint64_t *bnxt_dbr;
  struct bnxt_device_cq bnxt_cq;
  struct bnxt_device_sq bnxt_sq;

  __device__ void bnxt_poll_cq_until(uint32_t requested_available_slots);
  [[maybe_unused]] __device__ __attribute__((noinline)) void bnxt_print_cqe_error(uint8_t status);

  /* GDAProvider::BNXT END */

  /* GDAProvider::MLX5 START */

  gda_mlx5_device_cq mlx5_cq;
  gda_mlx5_device_sq mlx5_sq;

  __device__ void mlx5_poll_cq_until(uint16_t requested_available_slots);
  [[maybe_unused]] __device__ __attribute__((noinline)) void mlx5_print_cqe_error(const mlx5_cqe64* cqe, uint8_t opcode);

  /* GDAProvider::MLX5 END */

  /* GDAProvider::IONIC START */

  uint64_t *cq_dbreg{nullptr};
  uint64_t cq_dbval{0};
  uint64_t cq_mask{0};
  struct ionic_v1_cqe *ionic_cq_buf{nullptr};
  uint32_t cq_lock{SPIN_LOCK_UNLOCKED};
  uint32_t cq_pos{0};
  uint32_t cq_dbpos{0};

  uint64_t *sq_dbreg{nullptr};
  uint64_t sq_dbval{0};
  uint64_t sq_mask{0};
  struct ionic_v1_wqe *ionic_sq_buf{nullptr};
  uint32_t sq_lock{SPIN_LOCK_UNLOCKED};
  uint32_t sq_dbprod{0};
  uint32_t sq_prod{0};
  uint32_t sq_msn{0};

  __device__ uint64_t get_same_qp_lane_mask();

  /**
   * @brief Reserve space in the sq to post this many wqes.
   * @param wf_info Wavefront information.
   * @param num_wqes number of sq wqes to reserve for this wave.
   * @return position of my_tid=0's wqe.
   */
  __device__ uint32_t reserve_sq(ActiveWFInfo &wf_info, uint32_t num_wqes);
  __device__ uint32_t reserve_sq_single(uint32_t num_wqes);

  /**
   * @brief Ring the sq doorbell maintaining order between waves.
   * @param wf_info Wavefront information.
   * @param my_sq_prod position of my_tid=0's wqe.
   * @param num_wqes number of sq wqes posted in this wave.
   * @param wqe this thread's wqe.
   * @return doorbell producer index.
   */
  __device__ uint32_t commit_sq(ActiveWFInfo &wf_info, uint32_t my_sq_prod,
      uint32_t my_sq_pos, uint32_t num_wqes);
  __device__ uint32_t commit_sq_single(uint32_t my_sq_prod, uint32_t my_sq_pos,
      uint32_t num_wqes);

  /**
   * @brief Helper method to poll the next completion queue entry.
   */
  __device__ __attribute__((noinline))
  void poll_wave_cqes(uint64_t active_lane_mask);

  /**
   * @brief Helper method to drain completion queue entries.
   * @param wf_info Wavefront information.
   * @param cons wait for sq_msn to catch up to this position.
   */
  __device__ __attribute__((noinline))
  void ionic_quiet_internal_ccqe(ActiveWFInfo &wf_info, uint32_t cons);
  __device__ __attribute__((noinline))
  void ionic_quiet_internal_ccqe_single(uint32_t cons);

  /**
   * @brief Helper method to drain completion queue entries.
   * @param wf_info Wavefront information.
   * @param cons wait for sq_msn to catch up to this position.
   */
  __device__ __attribute__((noinline))
  void ionic_quiet_internal(ActiveWFInfo &wf_info, uint32_t cons);

  /* GDAProvider::IONIC END */

  uint32_t inline_threshold{0};

  char dev_name[24];
  uint32_t qp_num{0};
  uint32_t rkey{0};
  uint32_t lkey{0};

  uint64_t* nonfetching_atomic{nullptr};
  uint32_t nonfetching_atomic_lkey{0};
  struct ibv_mr *mr_nonfetching_atomic;

  uint64_t* fetching_atomic{nullptr};
  uint32_t fetching_atomic_lkey{0};
  uint32_t fetching_atomic_idx{0};
  struct ibv_mr *mr_fetching_atomic;

  static constexpr uint32_t FETCHING_ATOMIC_CNT{1024};
  static_assert(FETCHING_ATOMIC_CNT % WF_SIZE == 0);
  using FreeListT = FreeList<uint64_t*>;
  FreeListT* fetching_atomic_freelist{nullptr};

  HIPAllocator allocator{};

  uint8_t gda_op_rdma_write;
  uint8_t gda_op_rdma_read;
  uint8_t gda_op_atomic_fa;
  uint8_t gda_op_atomic_cs;

  struct ibv_pd* pd_;
  std::map<uintptr_t, struct ibv_mr*> user_buffer_mrs;

  struct user_buf_info_t *user_buf_info = nullptr;
  size_t num_user_buffers = 0;
  int gda_provider_{0};  // host-side only; device dispatch uses constmem.gda_provider

  int buffer_register(uintptr_t addr, size_t length);
  int buffer_unregister(uintptr_t addr);
  void buffer_unregister_all();

  __device__ uint32_t get_lkey(uintptr_t addr);
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_QUEUE_PAIR_HPP_
