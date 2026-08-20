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

#ifndef LIBRARY_SRC_GDA_MLX5_QUEUE_PAIR_HPP_
#define LIBRARY_SRC_GDA_MLX5_QUEUE_PAIR_HPP_

#include <utility>

#include "log.hpp"
#include "gda/endian.hpp"
#include "gda/mlx5/provider_gda_mlx5.hpp"
#include "gda/queue_pair/queue_pair_device.hpp"

#define GDA_MLX5_LOCK_USE_S_SLEEP  1
#define GDA_MLX5_LOCK_USE_S_WAKEUP (0 && GDA_MLX5_LOCK_USE_S_SLEEP)

namespace rocshmem {

class QueuePairMLX5;

template <> struct QueuePairTraits<QueuePairMLX5> {
  enum class OpCode : uint8_t {
    RDMA_WRITE = MLX5_OPCODE_RDMA_WRITE,
    RDMA_READ  = MLX5_OPCODE_RDMA_READ,
    ATOMIC_CS  = MLX5_OPCODE_ATOMIC_CS,
    ATOMIC_FA  = MLX5_OPCODE_ATOMIC_FA,
  };

  /**
   * @brief mlx5 uses big-endian ordering
   */
  static constexpr endian::Order Endianness = endian::Order::Big;

  /**
   * @brief mlx5 inlining maximum is 28 bytes
   *
   * mlx5 inlining can use up to 2 inline data WQE segments:
   * the first segment uses 4 (out of 16) bytes for encoding the data transfer size,
   * leaving 12 bytes from the first segment and the full 16 bytes of the second segment
   * for the actual inlined data.
   */
  static constexpr size_t InlineMax = sizeof(gda_mlx5_wqe_inline_data::data);

  /*
   * @brief mlx5 preferred inlining threshold is the same as the maximum
   */
  static constexpr size_t InlineThreshold = InlineMax;
};

class QueuePairMLX5 : public QueuePairDevice<QueuePairMLX5> {
private:
  gda_mlx5_device_sq sq;
  gda_mlx5_device_cq cq;

#if GDA_MLX5_LOCK_USE_S_SLEEP
  // sleep for up to 64 * LOCK_S_SLEEP_DELAY clock cycles
  static constexpr int LOCK_S_SLEEP_DELAY = 2;
#endif

public:
  __host__ explicit QueuePairMLX5(uint32_t qpn, uintptr_t heap_laddr, uint32_t heap_lkey,
                                  uintptr_t heap_raddr, uint32_t heap_rkey, size_t heap_size,
                                  uint64_t *fetching_atomic, uint32_t fetching_atomic_lkey,
                                  uint64_t *nonfetching_atomic, uint32_t nonfetching_atomic_lkey,
                                  FreeList<uint64_t*> *fetching_atomic_freelist,
                                  const BufferInfo *local_buffers, size_t num_user_buffers,
                                  const SymmBufferInfo *symm_buffers, const int *symm_count,
                                  gda_mlx5_device_sq&& sq, gda_mlx5_device_cq&& cq)
    : QueuePairDevice{qpn, heap_laddr, heap_lkey, heap_raddr, heap_rkey, heap_size,
                      fetching_atomic, fetching_atomic_lkey,
                      nonfetching_atomic, nonfetching_atomic_lkey,
                      fetching_atomic_freelist,
                      local_buffers, num_user_buffers,
                      symm_buffers, symm_count},
      sq{std::move(sq)}, cq{std::move(cq)} { }

  __host__ explicit QueuePairMLX5(uint32_t qpn,
                                  uint64_t *nonfetching_atomic, uint32_t nonfetching_atomic_lkey,
                                  gda_mlx5_device_sq&& sq, gda_mlx5_device_cq&& cq)
    : QueuePairDevice{qpn, nonfetching_atomic, nonfetching_atomic_lkey},
      sq{std::move(sq)}, cq{std::move(cq)} { }

  __host__ QueuePairMLX5(const QueuePairMLX5& other)            = delete;
  __host__ QueuePairMLX5& operator=(const QueuePairMLX5& other) = delete;
  __host__ QueuePairMLX5(QueuePairMLX5&& other) noexcept        = default;
  __host__ QueuePairMLX5& operator=(QueuePairMLX5&& other)      = default;
  __host__ ~QueuePairMLX5()                                     = default;

public:
  template <OpCode Op, typename... Options>
  __device__ __noinline__
  void post_wqe_rma(uintptr_t laddr, uint32_t lkey,
                    uintptr_t raddr, uint32_t rkey, size_t size,
                    const ActiveWFInfo& wf_info, PostOpt<Options...> = {});

  template <OpCode Op, typename... Options>
  __device__ __noinline__
  void post_wqe_rma_single(uintptr_t laddr, uint32_t lkey,
                           uintptr_t raddr, uint32_t rkey, size_t size,
                           PostOpt<Options...> = {});

  template <OpCode Op, AMOFetchType Fetch, typename... Options>
  __device__ __noinline__
  amo_ret_t<Fetch> post_wqe_amo(uintptr_t raddr, uint32_t rkey,
                                uint64_t swap_add, uint64_t compare,
                                const ActiveWFInfo& wf_info, PostOpt<Options...> = {});

  template <OpCode Op, AMOFetchType Fetch, typename... Options>
  __device__ __noinline__
  amo_ret_t<Fetch> post_wqe_amo_single(uintptr_t raddr, uint32_t rkey,
                                       uint64_t swap_add, uint64_t compare,
                                       PostOpt<Options...> = {});

  __device__ __noinline__ void quiet_single();

private:
#if GDA_MLX5_LOCK_USE_S_WAKEUP
  static __device__ __forceinline__ void amdgcn_s_wakeup() {
    /* why doesn't __builtin_amdgcn_s_wakeup() exist?
     * signals other wavefronts in the same workgroup to exit early from s_sleep */
    asm volatile("s_wakeup");
  }
#endif

  __device__ void ring_doorbell(uint64_t sq_post, const gda_mlx5_wqe& wqe);
  __device__ void poll_cq_until(uint16_t requested_available_slots);

  static __device__ void acquire_lock(uint32_t* lock);
  static __device__ void release_lock(uint32_t* lock);

  __device__ __forceinline__ uint16_t get_wqe_idx(uint8_t lane_id) {
    return static_cast<uint16_t>(sq.post + lane_id);
  }

  __device__ __forceinline__ uint16_t get_sq_idx(uint16_t wqe_idx) {
    // sq.depth is a power of 2, so just mask off everything above that
    return wqe_idx & sq.depth_mask;
  }

  template <typename PostOptions>
  __device__ void lock_pollcq(int wqe_count);

  template <typename PostOptions>
  __device__ void post_ringdb_unlock(int wqe_count, const gda_mlx5_wqe& wqe);

#if defined(BUILD_DEBUG_DEVICE)
  __device__ __noinline__ void print_cqe_error(const mlx5_cqe64* cqe,
                                               uint8_t opcode, uint8_t owner);
#endif
};

template <typename PostOptions>
__device__ void QueuePairMLX5::lock_pollcq(int wqe_count) {
  if constexpr (PostOptions::ThreadSafe || PostOptions::CheckSQ) {
    if constexpr (PostOptions::ThreadSafe) {
      acquire_lock(&sq.lock);
    }
    if constexpr (PostOptions::CheckSQ) {
      poll_cq_until(wqe_count);
    }
  } else {
    // need to at least acquire so that post counter is visible
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
}

template <typename PostOptions>
__device__ void QueuePairMLX5::post_ringdb_unlock(int wqe_count, const gda_mlx5_wqe& wqe) {
  sq.post += wqe_count;
  if constexpr (PostOptions::RingDB || PostOptions::ThreadSafe) {
    if constexpr (PostOptions::RingDB) {
      ring_doorbell(sq.post, wqe);
    }
    if constexpr (PostOptions::ThreadSafe) {
      release_lock(&sq.lock);
    }
  } else {
    // need to at least release so that post counter is available
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  }
}

// can be called with all active lanes using any number of different QPs, don't assume anything
template <QueuePairMLX5::OpCode Op, typename... Options>
__device__ __noinline__ void QueuePairMLX5::post_wqe_rma(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size,
    const ActiveWFInfo& wf_info, PostOpt<Options...>) {
  using PostOptions = PostOpt<Options...>;
  uint32_t byte_count = static_cast<uint32_t>(size);
  if (wf_info.is_pe_group_last) {
    // acquire SQ lock and poll until we have enough WQEBB for all lanes using this QP
    lock_pollcq<PostOptions>(wf_info.num_pe_group_lanes);
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = get_wqe_idx(wf_info.pe_group_logical_lane_id);
  uint16_t sq_idx  = get_sq_idx(wqe_idx);

  // can we inline the data into the WQE?
  bool send_inline = can_inline<Op>(size);

  // should we update CQ for this WQE?
  uint8_t fm_ce_se = PostOptions::signal_completion(wf_info) ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, fm_ce_se,
                   raddr, rkey, laddr, lkey, byte_count, send_inline};

  // copy to SQ
  sq.buf[sq_idx] = wqe;

  if (wf_info.is_pe_group_last) {
    /* increment post counter, ring doorbell, and release SQ lock
     * we are the last thread in the wavefront, so we have the last WQE posted */
    post_ringdb_unlock<PostOptions>(wf_info.num_pe_group_lanes, wqe);
  }
}

// precondition: called with all active lanes using different QPs
template <QueuePairMLX5::OpCode Op, typename... Options>
__device__ __noinline__ void QueuePairMLX5::post_wqe_rma_single(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size, PostOpt<Options...>) {
  using PostOptions = PostOpt<Options...>;
  uint32_t byte_count = static_cast<uint32_t>(size);
  // acquire SQ lock and poll until we have enough space for at least one WQEBB
  lock_pollcq<PostOptions>(1);

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = get_wqe_idx(0);
  uint16_t sq_idx  = get_sq_idx(wqe_idx);

  // can we inline the data into the WQE?
  bool send_inline = can_inline<Op>(size);

  // should we update CQ for this WQE?
  uint8_t fm_ce_se = PostOptions::signal_completion_single() ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, fm_ce_se,
                   raddr, rkey, laddr, lkey, byte_count, send_inline};

  // copy to SQ
  sq.buf[sq_idx] = wqe;

  // increment post counter, ring doorbell for this WQE, and release SQ lock
  post_ringdb_unlock<PostOptions>(1, wqe);
}

// can be called with all active lanes using any number of different QPs, don't assume anything
template <QueuePairMLX5::OpCode Op, AMOFetchType Fetch, typename... Options>
__device__ __noinline__ QueuePairMLX5::amo_ret_t<Fetch> QueuePairMLX5::post_wqe_amo(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare,
    const ActiveWFInfo& wf_info, PostOpt<Options...>) {
  static_assert(Fetch != AMOFetchType::NonBlocking, "non-blocking AMOs not yet implemented");
  using PostOptions = PostOpt<Options...>;
  if (wf_info.is_pe_group_last) {
    // acquire SQ lock and poll until we have enough WQEBB for all lanes using this QP
    lock_pollcq<PostOptions>(wf_info.num_pe_group_lanes);
  }

  uint64_t* atomic_laddr = get_atomic_addr<Fetch>();
  uint32_t atomic_lkey   = get_atomic_lkey<Fetch>();
  if constexpr (Fetch == AMOFetchType::Blocking) {
    uint32_t atomic_idx = (fetching_atomic_idx + wf_info.pe_group_logical_lane_id) % FETCHING_ATOMIC_CNT;
    atomic_laddr += atomic_idx;
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = get_wqe_idx(wf_info.pe_group_logical_lane_id);
  uint16_t sq_idx  = get_sq_idx(wqe_idx);

  // should we update CQ for this WQE?
  uint8_t fm_ce_se = PostOptions::signal_completion(wf_info) ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, fm_ce_se,
                   raddr, rkey, swap_add, compare, reinterpret_cast<uintptr_t>(atomic_laddr), atomic_lkey};

  // copy to SQ
  sq.buf[sq_idx] = wqe;

  if (wf_info.is_pe_group_last) {
    // increment fetching-atomic counter
    if constexpr (Fetch == AMOFetchType::Blocking) {
      fetching_atomic_idx += wf_info.num_pe_group_lanes;
    }
    /* increment post counter, ring doorbell, and release SQ lock
     * we are the last thread in the wavefront, so we have the last WQE posted */
    post_ringdb_unlock<PostOptions>(wf_info.num_pe_group_lanes, wqe);
    // wait until fetch completes
    if constexpr (Fetch == AMOFetchType::Blocking) {
      quiet_single();
    }
  }

  if constexpr (Fetch == AMOFetchType::Blocking) {
    return *atomic_laddr;
  }
}

// precondition: called with all active lanes using different QPs
template <QueuePairMLX5::OpCode Op, AMOFetchType Fetch, typename... Options>
__device__ __noinline__ QueuePairMLX5::amo_ret_t<Fetch> QueuePairMLX5::post_wqe_amo_single(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare, PostOpt<Options...>) {
  static_assert(Fetch != AMOFetchType::NonBlocking, "non-blocking AMOs not yet implemented");
  using PostOptions = PostOpt<Options...>;
  // acquire SQ lock and poll until we have enough space for at least one WQEBB
  lock_pollcq<PostOptions>(1);

  uint64_t* atomic_laddr = get_atomic_addr<Fetch>();
  uint32_t atomic_lkey   = get_atomic_lkey<Fetch>();
  if constexpr (Fetch == AMOFetchType::Blocking) {
    uint32_t atomic_idx = fetching_atomic_idx % FETCHING_ATOMIC_CNT;
    atomic_laddr += atomic_idx;
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = get_wqe_idx(0);
  uint16_t sq_idx  = get_sq_idx(wqe_idx);

  // should we update CQ for this WQE?
  uint8_t fm_ce_se = PostOptions::signal_completion_single() ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, fm_ce_se,
                   raddr, rkey, swap_add, compare, reinterpret_cast<uintptr_t>(atomic_laddr), atomic_lkey};

  // copy to SQ
  sq.buf[sq_idx] = wqe;

  // increment fetching-atomic counter
  if constexpr (Fetch == AMOFetchType::Blocking) {
    fetching_atomic_idx += 1;
  }
  // increment post counter, ring doorbell for this WQE, and release SQ lock
  post_ringdb_unlock<PostOptions>(1, wqe);
  // wait until fetch completes
  if constexpr (Fetch == AMOFetchType::Blocking) {
    quiet_single();
    return *atomic_laddr;
  }
}

// precondition: called with all active lanes using different QPs
__device__ inline __noinline__ void QueuePairMLX5::quiet_single() {
  poll_cq_until(sq.depth);
}

// precondition: called with all active lanes using different QPs
__device__ inline void QueuePairMLX5::poll_cq_until(uint16_t requested_available_slots) {
  uint16_t sq_depth = sq.depth;

  uint64_t sq_post = __hip_atomic_load(&sq.post, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  // don't need to check CQ if we haven't ever filled SQ and there's enough space left
  if (sq_post + requested_available_slots <= sq_depth) {
    return;
  }

  while (true) {
    struct mlx5_cqe64* cqe = cq.buf;

    /* Update the SQ head
     * This param provides us the sq_wqebb_counter; all our WQEs are exactly one WQEBB (64B) */
    // 32-bit load: big-endian 16-bit field, then two 8-bit fields
    uint32_t wqecnt_sig_op_own = __hip_atomic_load(reinterpret_cast<uint32_t*>(&cqe->wqe_counter),
                                                   __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    // GPU is little-endian, so wqe_counter is loaded into the low half of wqecnt_sig_op_own
    __be16 be_wqe_counter = static_cast<__be16>(wqecnt_sig_op_own);
    /* GPU is little-endian, so op_own is loaded into the top byte of wqecnt_sig_op_own;
     * opcode is the top 4 bits of op_own */
    uint8_t opcode = static_cast<uint8_t>(wqecnt_sig_op_own >> 28);
    uint16_t sq_head = endian::from_be(be_wqe_counter);

    // sq_tail is the least significant bits of the post counter
    uint16_t sq_tail = static_cast<uint16_t>(sq_post);

    // CQEs are initially invalid, retry until we see a valid CQE
    if (opcode == MLX5_CQE_INVALID) {
      LOGD_TRACE("CQ: invalid completion (%x)", opcode);
      continue;
    }

#if defined(BUILD_DEBUG_DEVICE)
    if (opcode != MLX5_CQE_REQ) {
      /* GPU is little-endian, so op_own is loaded into the top byte of wqecnt_sig_op_own;
       * opcode is the top 4 bits of op_own */
      uint8_t owner = static_cast<uint8_t>(wqecnt_sig_op_own >> 24) & MLX5_CQE_OWNER_MASK;
      print_cqe_error(cqe, opcode, owner);
    }
#endif  // BUILD_DEBUG_DEVICE

    /* sq_tail is an index to the next free WQE i.e. counts number of posted WQEs
     * sq_head is an index to the *last* completed WQE - need to add one to get *count* of completed WQEs */
    uint16_t posted    = sq_tail;
    uint16_t completed = sq_head + 1;

    /* posted >= completed, except when posted has wrapped around 0xFFFF and completed hasn't
     * but posted - completed is correct even when it wraps around
     * in some marginal cases it's maybe possible to see consumed_slots > sq_depth,
     * but in that case available_slots will be very large, > requested_available_slots,
     * and the loop will continue for another iteration */
    uint16_t consumed_slots  = posted   - completed;
    uint16_t available_slots = sq_depth - consumed_slots;

    /* continue until both:
     *   - no additional WQEs have been posted
     *   - the number of requested SQ slots are available */
    uint64_t prior_sq_post = sq_post;
    sq_post = __hip_atomic_load(&sq.post, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    if (sq_post == prior_sq_post && available_slots >= requested_available_slots) {
      return;
    }
  }
}

__device__ __forceinline__ void QueuePairMLX5::ring_doorbell(
    uint64_t sq_post, const gda_mlx5_wqe& wqe) {
  // sq_wqebb_counter is the least significant bits of the post counter
  uint16_t sq_wqebb_counter = static_cast<uint16_t>(sq_post);
  // gda_mlx5_db_register constructor extracts first 8 bytes of WQE
  gda_mlx5_db_register db_val{wqe};
  __be32 be_sq_wqebb_counter = endian::to_be<uint32_t>(sq_wqebb_counter);

  // get BlueFlame buffer from SQ
  gda_mlx5_bf_buffer* bf = sq.bf_buffer();

  // store sq_wqebb_counter to doorbell record
  __hip_atomic_store(sq.dbrec, be_sq_wqebb_counter, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  // ring doorbell by storing first 8B of WQE to the doorbell register
  __hip_atomic_store(&bf->db_reg, db_val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);

  LOGD_TRACE("SQ: posted WQEs with dbrec(%p)=%x (%hu), dbreg(%p)=%lx (%x, %x)",
             sq.dbrec, be_sq_wqebb_counter, sq_wqebb_counter, &bf->db_reg, db_val.val,
             db_val.wqe_header.opmod_idx_opcode, db_val.wqe_header.qpn_ds);
}

__device__ __forceinline__ void QueuePairMLX5::acquire_lock(uint32_t* lock) {
  /* acquire lock when new value 1 (locked) is exchanged with prior value 0 (unlocked)
   *
   * the __ATOMIC_ACQUIRE load synchronizes with the __ATOMIC_RELEASE store in release_lock(),
   * but not with the (implicit) __ATOMIC_RELAXED store part of the exchange
   * this is fine, since we only need to ensure happens-before between the threads
   * that released and acquired the lock, not between the different threads contending on the lock
   * when they (eventually) acquire the lock, *then* they will synchronize */
  while (__hip_atomic_exchange(lock, 1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) {
#if GDA_MLX5_LOCK_USE_S_SLEEP
    // sleep so we don't hammer the memory
    __builtin_amdgcn_s_sleep(LOCK_S_SLEEP_DELAY);
#endif
  }
}

__device__ __forceinline__ void QueuePairMLX5::release_lock(uint32_t* lock) {
  // release lock by storing 0 (unlocked)
  __hip_atomic_store(lock, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#if GDA_MLX5_LOCK_USE_S_WAKEUP
  // wake up any other sleeping waves (in the same workgroup)
  amdgcn_s_wakeup();
#endif
}

#if defined(BUILD_DEBUG_DEVICE)
__device__ inline __noinline__ void QueuePairMLX5::print_cqe_error(
    const mlx5_cqe64* cqe, uint8_t opcode, uint8_t owner) {
  const mlx5_err_cqe* err_cqe = reinterpret_cast<const mlx5_err_cqe*>(cqe);
  uint8_t syndrome = 0x0;

  switch (opcode) {
  case MLX5_CQE_RESP_WR_IMM:
  case MLX5_CQE_RESP_SEND:
  case MLX5_CQE_RESP_SEND_IMM:
  case MLX5_CQE_RESP_SEND_INV:
    // (valid) responder completion?!
    LOGD_ERROR("CQ: unexpected responder completion (%x)", opcode);
    break;
  case MLX5_CQE_RESIZE_CQ:
  case MLX5_CQE_NO_PACKET:
    LOGD_ERROR("CQ: unexpected completion type (%x)", opcode);
    break;
  case MLX5_CQE_SIG_ERR:
    LOGD_ERROR("CQ: unexpected signature error (%x)", opcode);
    break;
  case MLX5_CQE_REQ_ERR:
    syndrome = __hip_atomic_load(&err_cqe->syndrome, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    switch (syndrome) {
    case MLX5_CQE_SYNDROME_LOCAL_LENGTH_ERR:
      LOGD_ERROR("CQ requester error LOCAL_LENGTH_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_LOCAL_QP_OP_ERR:
      LOGD_ERROR("CQ requester error LOCAL_QP_OP_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_LOCAL_PROT_ERR:
      LOGD_ERROR("CQ requester error LOCAL_PROT_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_WR_FLUSH_ERR:
      LOGD_ERROR("CQ requester error WR_FLUSH_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_MW_BIND_ERR:
      LOGD_ERROR("CQ requester error MW_BIND_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_BAD_RESP_ERR:
      LOGD_ERROR("CQ requester error BAD_RESP_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_LOCAL_ACCESS_ERR:
      LOGD_ERROR("CQ requester error LOCAL_ACCESS_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_REMOTE_INVAL_REQ_ERR:
      LOGD_ERROR("CQ requester error REMOTE_INVAL_REQ_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_REMOTE_ACCESS_ERR:
      LOGD_ERROR("CQ requester error REMOTE_ACCESS_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_REMOTE_OP_ERR:
      LOGD_ERROR("CQ requester error REMOTE_OP_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_TRANSPORT_RETRY_EXC_ERR:
      LOGD_ERROR("CQ requester error TRANSPORT_RETRY_EXC_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_RNR_RETRY_EXC_ERR:
      LOGD_ERROR("CQ requester error RNR_RETRY_EXC_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_REMOTE_ABORTED_ERR:
      LOGD_ERROR("CQ requester error REMOTE_ABORTED_ERR (%x)", syndrome);
      break;
    default:
      LOGD_ERROR("CQ requester error unknown syndrome type (%x)", syndrome);
      break;
    }
    break;
  case MLX5_CQE_RESP_ERR:
    LOGD_ERROR("CQ: unexpected responder error (%x)", opcode);
    break;
  case MLX5_CQE_INVALID: {
    LOGD_ERROR("CQ: invalid completion (%x), check owner bit = %u?", opcode, owner);
    break;
  }
  default:
    LOGD_ERROR("CQ: unknown completion type (%x)", opcode);
    break;
  }
  abort();
}
#endif  // BUILD_DEBUG_DEVICE

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_MLX5_QUEUE_PAIR_HPP_
