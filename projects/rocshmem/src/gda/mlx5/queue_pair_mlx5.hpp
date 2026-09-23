// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
  __host__ explicit QueuePairMLX5(uint32_t qpn, QueuePairInitInfo&& init_info,
                                  gda_mlx5_device_sq&& sq, gda_mlx5_device_cq&& cq)
    : QueuePairDevice{qpn, std::move(init_info)},
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

  using sq_idx_t = gda_mlx5_device_sq::index_t;

  __device__ void ring_doorbell(sq_idx_t next_commit_idx, const gda_mlx5_wqe& wqe);
  __device__ void poll_cq_until(sq_idx_t requested_idx);

  __device__ __forceinline__ uint16_t get_sq_idx(uint16_t wqe_idx) {
    // sq.depth is a power of 2, so just mask off everything above that
    return wqe_idx & sq.depth_mask;
  }

  template <typename PostOptions>
  __device__ sq_idx_t reserve_sq(int wqe_count);

  template <typename PostOptions>
  __device__ void check_sq(sq_idx_t reserve_idx_base, int wqe_count);

  template <typename PostOptions>
  __device__ void commit_sq(sq_idx_t reserve_idx_base, int wqe_count, const gda_mlx5_wqe& wqe);

#if defined(BUILD_DEBUG_DEVICE)
  __device__ __noinline__ void print_cqe_error(const mlx5_cqe64* cqe,
                                               uint8_t opcode, uint8_t owner);
#endif
};

// precondition: called with all active lanes using different QPs
template <typename PostOptions>
__device__ __forceinline__ QueuePairMLX5::sq_idx_t QueuePairMLX5::reserve_sq(int wqe_count) {
  if constexpr (PostOptions::ThreadSafe) {
    // reserve wqe_count slots in SQ
    return __scoped_atomic_fetch_add(&sq.reserve_idx, wqe_count,
                                     __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  } else {
    /* invariant: this is the only wave that is concurrently accessing the SQ
     * therefore, we don't need to reserve and wait for our turn: use commit index directly */
    return sq.commit_idx;
  }
}

// precondition: called with all active lanes using different QPs
template <typename PostOptions>
__device__ __forceinline__ void QueuePairMLX5::check_sq(
    QueuePairMLX5::sq_idx_t reserve_idx_base, int wqe_count) {
  if constexpr (PostOptions::CheckSQ) {
    /* wait until all indices are available in SQ
     * [reserve_idx_base ... reserve_idx_base + wqe_count - 1] % sq_depth */
    sq_idx_t next_reserve_idx = reserve_idx_base + wqe_count;
    // can skip checking CQ for first sq_depth WQEs
    sq_idx_t sq_depth = static_cast<sq_idx_t>(sq.depth);
    // NOTE: rollover
    if (next_reserve_idx > sq_depth) {
      poll_cq_until(next_reserve_idx - sq_depth);
    }
  }
}

// precondition: called with all active lanes using different QPs
template <typename PostOptions>
__device__ __forceinline__ void QueuePairMLX5::commit_sq(
    QueuePairMLX5::sq_idx_t reserve_idx_base, int wqe_count, const gda_mlx5_wqe& wqe) {
  sq_idx_t next_commit_idx = reserve_idx_base + wqe_count;

  /* Q: Do we need to wait if we don't ring doorbell?
   * A: Even if we don't ring doorbell, others are: and sq.commit_idx must be incremented
   *    in the correct order, or the logic invariants will be violated. */
  if constexpr (PostOptions::ThreadSafe) {
    // wait until our turn to ring doorbell
    while (reserve_idx_base != __scoped_atomic_load_n(&sq.commit_idx,
                                                      __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)) {
#if GDA_MLX5_LOCK_USE_S_SLEEP
      // sleep so we don't hammer the memory
      __builtin_amdgcn_s_sleep(LOCK_S_SLEEP_DELAY);
#endif
      continue;
    }
  }

  if constexpr (PostOptions::RingDB) {
    ring_doorbell(next_commit_idx, wqe);
  }

  if constexpr (PostOptions::ThreadSafe) {
    // increment commit index and release any other waiting waves
    __scoped_atomic_store_n(&sq.commit_idx, next_commit_idx,
                            __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
#if GDA_MLX5_LOCK_USE_S_WAKEUP
    // wake up any other sleeping waves (in the same workgroup)
    amdgcn_s_wakeup();
#endif
  } else {
    // no one else is here, just set the SQ indices
    sq.reserve_idx = next_commit_idx;
    sq.commit_idx  = next_commit_idx;
  }
}

// can be called with all active lanes using any number of different QPs, don't assume anything
template <QueuePairMLX5::OpCode Op, typename... Options>
__device__ __noinline__ void QueuePairMLX5::post_wqe_rma(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size,
    const ActiveWFInfo& wf_info, PostOpt<Options...>) {
  using PostOptions = PostOpt<Options...>;

  uint32_t byte_count = static_cast<uint32_t>(size);
  int wqe_count = wf_info.num_pe_group_lanes;
  sq_idx_t reserve_idx_base = 0;
  if (wf_info.is_pe_group_last) {
    // reserve SQ indices for this wave
    reserve_idx_base = reserve_sq<PostOptions>(wqe_count);
  }
  // fetch reservation base index from PE group leader
  reserve_idx_base = __shfl(reserve_idx_base, wf_info.pe_group_last_phys_lane_id);
  // compute this thread's reserved index
  sq_idx_t reserve_idx = reserve_idx_base + wf_info.pe_group_logical_lane_id;

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = static_cast<uint16_t>(reserve_idx);
  uint16_t sq_idx  = get_sq_idx(wqe_idx);

  // can we inline the data into the WQE?
  bool send_inline = can_inline<Op>(size);
  // should we update CQ for this WQE?
  uint8_t fm_ce_se = PostOptions::signal_completion(wf_info) ? MLX5_WQE_CTRL_CQ_UPDATE : 0;
  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, fm_ce_se,
                   raddr, rkey, laddr, lkey, byte_count, send_inline};

  if (wf_info.is_pe_group_last) {
    // wait until all indices are available in SQ
    check_sq<PostOptions>(reserve_idx_base, wqe_count);
  }

  // copy to SQ
  sq.buf[sq_idx] = wqe;

  if (wf_info.is_pe_group_last) {
    // commit SQ: we are the last thread in the wavefront, so we have the last WQE posted
    commit_sq<PostOptions>(reserve_idx_base, wqe_count, wqe);
  }
}

// precondition: called with all active lanes using different QPs
template <QueuePairMLX5::OpCode Op, typename... Options>
__device__ __noinline__ void QueuePairMLX5::post_wqe_rma_single(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size, PostOpt<Options...>) {
  using PostOptions = PostOpt<Options...>;

  uint32_t byte_count = static_cast<uint32_t>(size);
  // reserve SQ index
  sq_idx_t reserve_idx = reserve_sq<PostOptions>(1);

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = static_cast<uint16_t>(reserve_idx);
  uint16_t sq_idx  = get_sq_idx(wqe_idx);

  // can we inline the data into the WQE?
  bool send_inline = can_inline<Op>(size);
  // should we update CQ for this WQE?
  uint8_t fm_ce_se = PostOptions::signal_completion_single() ? MLX5_WQE_CTRL_CQ_UPDATE : 0;
  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, fm_ce_se,
                   raddr, rkey, laddr, lkey, byte_count, send_inline};

  // wait until index is available in SQ
  check_sq<PostOptions>(reserve_idx, 1);
  // copy to SQ
  sq.buf[sq_idx] = wqe;
  // commit SQ
  commit_sq<PostOptions>(reserve_idx, 1, wqe);
}

// can be called with all active lanes using any number of different QPs, don't assume anything
template <QueuePairMLX5::OpCode Op, AMOFetchType Fetch, typename... Options>
__device__ __noinline__ QueuePairMLX5::amo_ret_t<Fetch> QueuePairMLX5::post_wqe_amo(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare,
    const ActiveWFInfo& wf_info, PostOpt<Options...>) {
  static_assert(Fetch != AMOFetchType::NonBlocking, "non-blocking AMOs not yet implemented");
  using PostOptions = PostOpt<Options...>;

  int wqe_count = wf_info.num_pe_group_lanes;
  sq_idx_t reserve_idx_base = 0;
  if (wf_info.is_pe_group_last) {
    // reserve SQ indices for this wave
    reserve_idx_base = reserve_sq<PostOptions>(wqe_count);
  }
  // fetch reservation base index from PE group leader
  reserve_idx_base = __shfl(reserve_idx_base, wf_info.pe_group_last_phys_lane_id);
  // compute this thread's reserved index
  sq_idx_t reserve_idx = reserve_idx_base + wf_info.pe_group_logical_lane_id;

  uint64_t* atomic_laddr = get_atomic_addr<Fetch>();
  uint32_t atomic_lkey   = get_atomic_lkey<Fetch>();
  if constexpr (Fetch == AMOFetchType::Blocking) {
    // TODO: atomic_fetch_add(&fetching_atomic_idx, num_wqes)
    /* assumes the FETCHING_ATOMIC_CNT >= sq_depth; this is not generally true
     * TODO: fix this for the case where there are > 1024 concurrent AMOs */
    uint32_t atomic_idx = static_cast<uint32_t>(reserve_idx);
    atomic_laddr += (atomic_idx % FETCHING_ATOMIC_CNT);
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = static_cast<uint16_t>(reserve_idx);
  uint16_t sq_idx  = get_sq_idx(wqe_idx);

  // should we update CQ for this WQE?
  uint8_t fm_ce_se = PostOptions::signal_completion(wf_info) ? MLX5_WQE_CTRL_CQ_UPDATE : 0;
  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, fm_ce_se,
                   raddr, rkey, swap_add, compare, reinterpret_cast<uintptr_t>(atomic_laddr), atomic_lkey};

  if (wf_info.is_pe_group_last) {
    // wait until all indices are available in SQ
    check_sq<PostOptions>(reserve_idx_base, wqe_count);
  }

  // copy to SQ
  sq.buf[sq_idx] = wqe;

  if (wf_info.is_pe_group_last) {
    // commit SQ: we are the last thread in the wavefront, so we have the last WQE posted
    commit_sq<PostOptions>(reserve_idx_base, wqe_count, wqe);
    // wait until leader's fetch completes; completion order ensures others are complete as well
    if constexpr (Fetch == AMOFetchType::Blocking) {
      poll_cq_until(reserve_idx_base + wqe_count);
    }
  }

  if constexpr (Fetch == AMOFetchType::Blocking) {
//    return *atomic_laddr;
    // system-scope (cache-bypassing) load of AMO return data
    return __scoped_atomic_load_n(atomic_laddr, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

// precondition: called with all active lanes using different QPs
template <QueuePairMLX5::OpCode Op, AMOFetchType Fetch, typename... Options>
__device__ __noinline__ QueuePairMLX5::amo_ret_t<Fetch> QueuePairMLX5::post_wqe_amo_single(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare, PostOpt<Options...>) {
  static_assert(Fetch != AMOFetchType::NonBlocking, "non-blocking AMOs not yet implemented");
  using PostOptions = PostOpt<Options...>;
  // reserve SQ index
  sq_idx_t reserve_idx = reserve_sq<PostOptions>(1);

  uint64_t* atomic_laddr = get_atomic_addr<Fetch>();
  uint32_t atomic_lkey   = get_atomic_lkey<Fetch>();
  if constexpr (Fetch == AMOFetchType::Blocking) {
    // TODO: atomic_fetch_add(&fetching_atomic_idx, num_wqes)
    /* assumes the FETCHING_ATOMIC_CNT >= sq_depth; this is not generally true
     * TODO: fix this for the case where there are > 1024 concurrent AMOs */
    uint32_t atomic_idx = static_cast<uint32_t>(reserve_idx);
    atomic_laddr += (atomic_idx % FETCHING_ATOMIC_CNT);
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = static_cast<uint16_t>(reserve_idx);
  uint16_t sq_idx  = get_sq_idx(wqe_idx);

  // should we update CQ for this WQE?
  uint8_t fm_ce_se = PostOptions::signal_completion_single() ? MLX5_WQE_CTRL_CQ_UPDATE : 0;
  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, fm_ce_se,
                   raddr, rkey, swap_add, compare, reinterpret_cast<uintptr_t>(atomic_laddr), atomic_lkey};

  // wait until index is available in SQ
  check_sq<PostOptions>(reserve_idx, 1);
  // copy to SQ
  sq.buf[sq_idx] = wqe;
  // commit SQ
  commit_sq<PostOptions>(reserve_idx, 1, wqe);

  // wait until fetch completes
  if constexpr (Fetch == AMOFetchType::Blocking) {
    poll_cq_until(reserve_idx + 1);
//    return *atomic_laddr;
    // system-scope (cache-bypassing) load of AMO return data
    return __scoped_atomic_load_n(atomic_laddr, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

// precondition: called with all active lanes using different QPs
__device__ inline __noinline__ void QueuePairMLX5::quiet_single() {
  // check how many WQEs have been posted so far
  sq_idx_t commit_idx = __scoped_atomic_load_n(&sq.commit_idx,
                                               __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  // poll until we have completed all these WQEs
  poll_cq_until(commit_idx);
  // system-scope acquire fence, to ensure we see updated get or AMO data
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

/* precondition: called with all active lanes using different QPs
 * wait until requested_idx number of WQEs have been completed
 * TODO: apply PostOpts to poll_cq_until */
__device__ __forceinline__ void QueuePairMLX5::poll_cq_until(
    QueuePairMLX5::sq_idx_t requested_idx) {
  sq_idx_t complete_idx = __scoped_atomic_load_n(&sq.complete_idx,
                                                 __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);

  /* complete_idx is an index to the next free WQE i.e. counts number of completed WQEs
   * requested_idx is the complete_idx you need to observe before returning */
  // NOTE: rollover
  while (complete_idx < requested_idx) {
    struct mlx5_cqe64* cqe = cq.buf;

    // lowered to s_waitcnt vmcnt(0) : order ld/rmw complete_idx -> ld wqe_counter
//    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    /* read wqe_counter and sig_op_own from CQE
     * 32-bit load: big-endian 16-bit field, then two 8-bit fields
     * wqe_counter is the sq_wqebb_counter; all our WQEs are exactly one WQEBB (64B) */
#if 0
    uint32_t wqecnt_sig_op_own = __scoped_atomic_load_n(reinterpret_cast<uint32_t*>(&cqe->wqe_counter),
                                                        __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
#endif
    uint32_t wqecnt_sig_op_own = *reinterpret_cast<volatile uint32_t*>(&cqe->wqe_counter);

    /* GPU is little-endian, so op_own is loaded into the top byte of wqecnt_sig_op_own;
     * opcode is the top 4 bits of op_own */
    uint8_t opcode = static_cast<uint8_t>(wqecnt_sig_op_own >> 28);

    // CQEs are initially invalid, retry until we see a valid CQE
    if (opcode == MLX5_CQE_INVALID) {
      LOGD_TRACE("CQ: invalid completion (%x)", opcode);
      continue;
    }

#if defined(BUILD_DEBUG_DEVICE)
    if (opcode != MLX5_CQE_REQ) {
      /* GPU is little-endian, so op_own is loaded into the top byte of wqecnt_sig_op_own;
       * owner is the low 4 bits of op_own */
      uint8_t owner = static_cast<uint8_t>(wqecnt_sig_op_own >> 24) & MLX5_CQE_OWNER_MASK;
      print_cqe_error(cqe, opcode, owner);
    }
#endif  // BUILD_DEBUG_DEVICE

    // GPU is little-endian, so wqe_counter is loaded into the low half of wqecnt_sig_op_own
    __be16 be_wqe_counter = static_cast<__be16>(wqecnt_sig_op_own);
    uint16_t wqe_counter = endian::from_be(be_wqe_counter);
    /* wqe_counter is an index to the *last* completed WQE;
     * need to add one to get *count* of completed WQEs */
    uint16_t complete_idx16 = wqe_counter + 1;

    /* complete_idx16_diff <= sq_depth
     * except when many more WQEs get committed & executed while inside poll_cq_until
     * but in that case complete_idx is necessarily updated since we need to call poll_cq_until
     * to ensure that the SQ slots are free and we catch it */
    uint16_t complete_idx16_diff = complete_idx16 - static_cast<uint16_t>(complete_idx);
    sq_idx_t next_complete_idx   = complete_idx   + static_cast<sq_idx_t>(complete_idx16_diff);

    // lowered to s_waitcnt vmcnt(0) : order ld wqe_counter -> rmw complete_idx
//    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    /* accumulate the newly-complete indices into sq.complete_idx,
     * if some other thread didn't get there first */
    // NOTE: cannot use fetch_max, otherwise can never wrap around (when sq_idx_t is uint32_t)
    if (__scoped_atomic_compare_exchange_n(&sq.complete_idx, &complete_idx, next_complete_idx,
                                           /* weak */ true, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                           __MEMORY_SCOPE_DEVICE)) {
      complete_idx = next_complete_idx;
    }
  }
  LOGD_TRACE("CQ: completed %zu > requested %zu",
             static_cast<size_t>(complete_idx), static_cast<size_t>(requested_idx));
}

__device__ __forceinline__ void QueuePairMLX5::ring_doorbell(
    QueuePairMLX5::sq_idx_t next_commit_idx, const gda_mlx5_wqe& wqe) {
  // sq_wqebb_counter is the least significant bits of the next commit index
  uint16_t sq_wqebb_counter = static_cast<uint16_t>(next_commit_idx);
  __be32 be_sq_wqebb_counter = endian::to_be<uint32_t>(sq_wqebb_counter);

  // get BlueFlame buffer from SQ
  gda_mlx5_bf_buffer* bf = sq.bf_buffer();

  // store sq_wqebb_counter to doorbell record
  __scoped_atomic_store_n(sq.dbrec, be_sq_wqebb_counter, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  /* ring doorbell by storing first 8B of WQE to the doorbell register
   * gda_mlx5_db_register constructor extracts first 8 bytes of WQE */
//  __scoped_atomic_store_n(&bf->db_reg.val, db_val.val, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  *reinterpret_cast<volatile uint64_t*>(&bf->db_reg.val) = gda_mlx5_db_register{wqe}.val;

  LOGD_TRACE("SQ: posted WQEs with dbrec(%p)=%x (%hu), dbreg(%p)=%lx (%x, %x)",
             sq.dbrec, be_sq_wqebb_counter, sq_wqebb_counter, &bf->db_reg, db_val.val,
             db_val.wqe_header.opmod_idx_opcode, db_val.wqe_header.qpn_ds);
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
    syndrome = __scoped_atomic_load_n(&err_cqe->syndrome, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
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
