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

namespace rocshmem {

class GDABackend;

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

  enum Collectivity { THREAD, WAVE };

  /**
   * @brief Create and enqueue a non-blocking put work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] source Source address for data transmission.
   * @param[in] nelems Size in bytes of data transmission.
   * @param[in] pe Destination processing element of data transmission.
   */
  __device__ void put_nbi(void *dest, const void *source, size_t nelems, int pe, Collectivity cy = THREAD);

  __device__ void put_nbi_single(void *dest, const void *source, size_t nelems, bool ring_db);

  /**
   * @brief Create and enqueue a non-blocking get work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] source Source address for data transmission.
   * @param[in] nelems Size in bytes of data transmission.
   * @param[in] pe Destination processing element of data transmission.
   */
  __device__ void get_nbi(void *dest, const void *source, size_t nelems, int pe, Collectivity cy = THREAD);

  __device__ void get_nbi_single(void *dest, const void *source, size_t nelems, bool ring_db);

  /**
   * @brief Empty all completions from the completion queue.
   */
  __device__ void quiet(Collectivity cy = THREAD);
  __device__ void quiet_single();

  /**
   * @brief Create and enqueue an atomic fetch work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] pe Destination processing element of data transmission.
   *
   * @return An atomic value
   */
  __device__ int64_t atomic_fetch(void *dest, int64_t value, int64_t cond, int pe);

  /**
   * @brief Create and enqueue an atomic fetch work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] pe Destination processing element of data transmission.
   */
  __device__ void atomic_nofetch(void *dest, int64_t value, int64_t cond, int pe);

  __device__ void atomic_nofetch_single(void *dest, int64_t value);

  /**
   * @brief Create and enqueue an atomic cas work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] pe Destination processing element of data transmission.
   *
   * @return An atomic value
   */
  __device__ int64_t atomic_cas(void *dest, int64_t atomic_data, int64_t atomic_cmp, int pe);

  /**
   * @brief Create and enqueue an atomic cas work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] pe Destination processing element of data transmission.
   */
  __device__ int64_t atomic_cas_nofetch(void *dest, int64_t atomic_data, int64_t atomic_cmp, int pe);

  char *const *base_heap{nullptr};

 private:
  /**
   * @brief Helper method to build work requests for the send queue.
   *
   * @param[in] pe Destination processing element of data transmission.
   * @param[in] size Size in bytes of data transmission.
   * @param[in] raddr Remote address.
   * @param[in] opcode Operation to be performed.
   * @param[in] atomic_data An atomic data value to be used.
   * @param[in] atomic_cmp An atomic comparison operation to be performed.
   * @param[in] fetching True if the operation returns a value.
   */
  __device__ __attribute__((noinline)) uint64_t post_wqe_amo(int pe, int32_t size, uintptr_t raddr, uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp, bool fetch);

  __device__ __attribute__((noinline)) uint64_t post_wqe_amo_single(uintptr_t raddr,
                                                                    uint8_t opcode,
                                                                    int64_t atomic_data,
                                                                    int64_t atomic_cmp,
                                                                    bool fetching);

  /**
   * @brief Helper method to build work requests for the send queue.
   *
   * @param[in] pe Destination processing element of data transmission.
   * @param[in] size Size in bytes of data transmission.
   * @param[in] laddr Local address.
   * @param[in] raddr Remote address.
   * @param[in] opcode Operation to be performed.
   */
  __device__ __attribute__((noinline)) void post_wqe_rma(int pe, int32_t size, uintptr_t laddr, uintptr_t raddr, uint8_t opcode, Collectivity cy);
  __device__ __attribute__((noinline)) void post_wqe_rma_turn(int pe, int32_t size, uintptr_t laddr, uintptr_t raddr, uint8_t opcode, Collectivity cy);

  __device__ __attribute__((noinline)) void post_wqe_rma_single(int32_t size, uintptr_t laddr, uintptr_t raddr, uint8_t opcode, bool ring_db);
  __device__ __attribute__((noinline)) void post_wqe_rma_mt(int pe, int32_t size, uintptr_t laddr, uintptr_t raddr, uint8_t opcode);

#if defined(GDA_MLX5)
  __device__ __forceinline__ void
  mlx5_wait_for_free_sq_slots(uint64_t wave_sq_counter,
      uint8_t num_active_lanes);

  __device__ __forceinline__ void
  mlx5_wait_for_db_touched_eq(uint64_t target_sq_counter);

  __device__ __forceinline__ void
  mlx5_build_rma_wqe(uint64_t my_sq_counter, uint64_t my_sq_index,
      uintptr_t laddr, uintptr_t raddr, int32_t size, uint8_t opcode);

  __device__ __forceinline__ void
  mlx5_build_amo_wqe(uint64_t my_sq_counter, uint64_t my_sq_index,
      uintptr_t raddr, uint8_t opcode, int64_t atomic_data,
      int64_t atomic_cmp, bool fetching, uint64_t *wave_fetch_atomic);

  __device__ __forceinline__ uint64_t*
  mlx5_allocate_wave_fetching_atomic_buffer(uint64_t wave_sq_counter,
      bool is_leader, uint64_t leader_phys_lane_id);

  __device__ __forceinline__ void
  mlx5_ring_doorbell(uint64_t wave_sq_counter, uint8_t num_wqes);

  __device__ uint64_t
  mlx5_post_wqe_amo(int32_t size, uintptr_t raddr, uint8_t opcode,
      int64_t atomic_data, int64_t atomic_cmp, bool fetch);

  __device__ void
  mlx5_post_wqe_rma(int32_t size, uintptr_t laddr,
      uintptr_t raddr, uint8_t opcode);

  __device__ void
  mlx5_quiet();

#endif
#if defined(GDA_BNXT)

  __device__ void bnxt_write_rma_wqe(uintptr_t raddr, uintptr_t laddr, int32_t length, uint8_t opcode);
  __device__ uint32_t bnxt_write_amo_wqe(uintptr_t raddr, uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp, bool fetching);

  __device__ uint64_t bnxt_post_wqe_amo_single(uintptr_t raddr, uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp, bool fetching);
  __device__ uint64_t bnxt_post_wqe_amo(uintptr_t raddr, uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp, bool fetching);

  __device__ void bnxt_post_wqe_rma(int pe, int32_t size, uintptr_t laddr, uintptr_t raddr, uint8_t opcode);

  __device__ void bnxt_post_wqe_rma_single(int32_t size, uintptr_t laddr, uintptr_t raddr, uint8_t opcode, bool ring_db);
  __device__ void bnxt_quiet();
  __device__ void bnxt_quiet_single();
#endif
#if defined(GDA_IONIC)
  __device__ uint64_t ionic_post_wqe_amo(int pe, int32_t size, uintptr_t raddr, uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp, bool fetch);
  __device__ void ionic_post_wqe_rma(int pe, int32_t size, uintptr_t laddr, uintptr_t raddr, uint8_t opcode, Collectivity cy);
  __device__ void ionic_quiet();
#endif

  /**
   * @brief Helper method to ring the doorbell
   *
   * @param[in] db_val Doorbell value is written by method.
   */
#if defined(GDA_MLX5)
  __device__ void mlx5_ring_doorbell(uint64_t db_val, uint64_t my_sq_counter);
#endif
#if defined(GDA_BNXT)
  __device__ void bnxt_ring_doorbell(uint32_t slot_idx);
#endif
#if defined(GDA_IONIC)
  __device__ void ionic_ring_doorbell(uint32_t pos);
#endif

  int gda_provider_{0};

  /* GDAProvider::BNXT START */
  uint64_t *bnxt_dbr;
  struct bnxt_device_cq bnxt_cq;
  struct bnxt_device_sq bnxt_sq;

  __device__ void bnxt_poll_cq_until(uint32_t requested_available_slots);
  __device__ void bnxt_check_cqe_error(struct bnxt_re_req_cqe *cqe);

  /* GDAProvider::BNXT END */

  /* GDAProvider::MLX5 START */

  db_reg_t db{};

  uint64_t cq_consumer{0};
  uint64_t quiet_posted{0};
  uint64_t quiet_active{0};
  uint64_t quiet_completed{0};

  /*
   * struct mlx5dv_cq {
   *   void                    *buf;
   *   __be32                  *dbrec;
   *   uint32_t                cqe_cnt;
   *   uint32_t                cqe_size;
   *   void                    *cq_uar;
   *   uint32_t                cqn;
   *   uint64_t                comp_mask;
   * };
  */
  mlx5_cqe64 *cq_buf{nullptr};
  volatile uint32_t *cq_dbrec{nullptr};
  uint32_t cq_cnt{0};
  uint32_t cq_log_cnt{0};

  /*
   * struct mlx5dv_qp {
   *   __be32 *dbrec;
   *   struct {
   *     void *buf;
   *     uint32_t wqe_cnt;
   *     uint32_t stride;
   *   } sq;
   *   struct {
   *     void *buf;
   *     uint32_t wqe_cnt;
   *     uint32_t stride;
   *   } rq;
   *   struct {
   *     void *reg;
   *     uint32_t size;
   *   } bf;
   *   uint64_t comp_mask;
   *   off_t uar_mmap_offset;
   *   uint32_t tirn;
   *   uint32_t tisn;
   *   uint32_t rqn;
   *   uint32_t sqn;
   *   uint64_t tir_icm_addr;
   * };
   */
  volatile uint32_t *dbrec{nullptr};
  uint64_t *sq_buf{nullptr};
  uint16_t sq_wqe_cnt{0};
  uint64_t sq_posted{0};
  uint64_t sq_db_touched{0};
  uint64_t sq_sunk{0};

  static constexpr size_t OUTSTANDING_TABLE_SIZE = 65536;
  uint64_t outstanding_wqes[OUTSTANDING_TABLE_SIZE]{0};

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
   * @param my_tid my logical thread id.
   * @param num_wqes number of sq wqes to reserve for this wave.
   * @return position of my_tid=0's wqe.
   */
  __device__ uint32_t reserve_sq(uint64_t active_lane_mask, uint32_t num_wqes);

  /**
   * @brief Ring the sq doorbell maintaining order between waves.
   * @param last this is the last wqe posted in this wave.
   * @param my_sq_prod position of my_tid=0's wqe.
   * @param num_wqes number of sq wqes posted in this wave.
   * @param wqe this thread's wqe.
   * @return doorbell producer index.
   */
  __device__ uint32_t commit_sq(uint64_t activemask, uint32_t my_sq_prod, uint32_t my_sq_pos, uint32_t num_wqes);

  /**
   * @brief Helper method to poll the next completion queue entry.
   */
  __device__ __attribute__((noinline)) void poll_wave_cqes(uint64_t active_lane_mask);

  /**
   * @brief Helper method to drain completion queue entries.
   * @param cons wait for sq_msn to catch up to this position.
   */
  __device__ __attribute__((noinline)) void ionic_quiet_internal_ccqe(uint64_t active_lane_mask, uint32_t cons);

  /**
   * @brief Helper method to drain completion queue entries.
   * @param cons wait for sq_msn to catch up to this position.
   */
  __device__ __attribute__((noinline)) void ionic_quiet_internal(uint64_t active_lane_mask, uint32_t cons);

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
  using FreeListT = FreeList<uint64_t*, HIPAllocator>;
  FreeListT* fetching_atomic_freelist{nullptr};

  HIPAllocator allocator{};

  uint8_t gda_op_rdma_write;
  uint8_t gda_op_rdma_read;
  uint8_t gda_op_atomic_fa;
  uint8_t gda_op_atomic_cs;
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_QUEUE_PAIR_HPP_
