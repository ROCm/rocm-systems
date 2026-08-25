/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

/**
 * Per-peer GDA queue-pair introspection for external WQE builders.
 *
 * rocSHMEM's GDA backend already creates, connects and registers an RC QP per
 * peer. This exposes the device-visible resources of that QP -- send/completion
 * ring addresses, doorbell, live producer counter, and the memory keys -- so a
 * consumer that compiles its own device code (e.g. a Triton kernel) can post
 * work requests into a QP rocSHMEM set up, instead of duplicating connection
 * establishment.
 *
 * The NIC providers do not have the same shape, so the payload is a
 * vendor-neutral core plus a tagged union:
 *
 *   ionic  doorbell register + a base doorbell value, and a UDMA index
 *          selecting which of the NIC's two engines the QP is bound to.
 *   mlx5   doorbell record + BlueFlame register; the WQE is written to BF.
 *   bnxt   a single doorbell record pointer.
 *
 * Callers switch on `vendor`. Reading a union arm other than the one `vendor`
 * names is undefined.
 *
 * Only ionic is implemented. mlx5 and bnxt are TODO: their union arms are
 * declared so adding them later does not change this struct's ABI, but
 * rocshmem_query_qp_info returns false for those providers today.
 *
 * Deliberately a flat POD plus a free function: callers (including language
 * bindings) need no internal rocSHMEM headers.
 *
 * IMPORTANT -- choosing ctx_id. ctx_id 0 is the default context, the queue
 * rocSHMEM drives for its own operations. Its producer index, send-queue lock
 * and cached doorbell position are private to rocSHMEM, and an external builder
 * updates none of them: it bumps the shared producer index and rings the
 * doorbell on its own. Pass ctx_id > 0 to borrow a user-context QP that
 * rocSHMEM does not drive itself, so the two never share that state.
 *
 * Measured on gfx950/ionic, sharing ctx 0 does not fail the way one might
 * guess. Posting external descriptors and then calling rocshmem_barrier_all
 * is fine -- the barrier completes. But a full workload that mixes external
 * posts with the caller's own completion polling on that same queue does not
 * run to completion, while the identical workload on a user context does.
 * The exact stall was not characterised; the queue is simply not safely
 * shared.
 *
 * This is enforced, not merely advised: rocshmem_query_qp_info() returns false
 * for ctx_id <= 0. Describing the default-context QP is the one thing this API
 * must not do, so a caller cannot opt into that state by mistake.
 */
#ifndef ROCSHMEM_QP_INTROSPECT_HPP
#define ROCSHMEM_QP_INTROSPECT_HPP

#include <cstdint>

namespace rocshmem {

/**
 * @brief Which NIC provider owns the QP described by a QpInfo.
 *
 * Mirrors the internal GDAProvider enum. Duplicated here so the public header
 * stays free of internal includes; kept in sync by a static_assert in
 * qp_introspect.cpp.
 */
enum class QpInfoVendor : uint32_t {
  UNKNOWN = 0,
  IONIC   = 1,
  BNXT    = 2,
  MLX5    = 3,
};

/**
 * @brief Device-visible resources of one peer's RC QP.
 *
 * All addresses are device virtual addresses, valid in the address space of the
 * GPU that owns the QP. Depths are in ring slots, not bytes.
 */
struct QpInfo {
  /* ---- vendor-neutral ------------------------------------------------- */
  uint64_t sq_buf;     //!< Send queue ring buffer.
  uint64_t sq_prod;    //!< Address of the live SQ producer counter. An external
                       //!<   builder must continue this sequence, not restart
                       //!<   it, or it will collide with rocSHMEM's own posts.
  uint64_t cq_buf;     //!< Completion queue ring buffer.
  uint64_t base_heap;  //!< Local symmetric heap base for this QP.
  uint32_t sq_depth;   //!< SQ capacity in WQE slots.
  uint32_t cq_depth;   //!< CQ capacity in CQE slots.
  uint32_t lkey;       //!< Local heap memory key.
  uint32_t rkey;       //!< Peer heap memory key for this QP's connection.
  uint32_t qpn;        //!< QP number.
  QpInfoVendor vendor; //!< Selects the valid union arm below.

  /* ---- provider-specific ---------------------------------------------- */
  union {
    struct {
      uint64_t dbrec;  //!< SQ doorbell record.
      uint64_t bf;     //!< BlueFlame register; the WQE itself is written here.
    } mlx5;

    struct {
      uint64_t db;        //!< SQ doorbell register.
      uint64_t dbval;     //!< Base doorbell value; the producer index is OR'ed
                          //!<   in when ringing.
      uint64_t sq_mask;   //!< SQ index wrap mask (depth - 1).
      uint64_t cq_mask;   //!< CQ index wrap mask (depth - 1).
      uint8_t  udma_idx;  //!< Which of the NIC's two UDMA engines this QP uses.
    } ionic;

    struct {
      uint64_t dbr;    //!< Doorbell record.
    } bnxt;
  };
};

/**
 * @brief Which GDA provider is active, whether or not it is supported here.
 *
 * Returns QpInfoVendor::UNKNOWN when rocSHMEM is not initialised or the active
 * backend is not GDA. Otherwise returns the detected provider, including mlx5
 * and bnxt, which this API does not yet describe.
 *
 * Pair with rocshmem_qp_introspect_available() to tell the two failure modes
 * apart: UNKNOWN means there is no GDA backend to introspect, while a known
 * provider with available() == false means the NIC was detected but is not
 * implemented yet. A caller that only sees a null result cannot distinguish
 * them, and the remedy for each is different.
 *
 * @return the detected provider, or QpInfoVendor::UNKNOWN. Host-side, safe to
 *         call before init.
 */
QpInfoVendor rocshmem_qp_introspect_provider();

/**
 * @brief Whether QP introspection is usable in this process.
 *
 * True only when rocSHMEM is initialised, the active backend is the GDA
 * backend, and its RDMA provider is one this API can describe -- today that
 * means ionic. Everything here is specific to GDA's RC queue pairs, so on any
 * other backend (IPC, reverse offload) there is nothing to introspect.
 *
 * This deliberately checks the provider and not merely the backend: reporting
 * "available" on an mlx5 or bnxt GDA backend would promise a capability that
 * rocshmem_query_qp_info then refuses.
 *
 * Callers should use this to branch up front rather than inferring capability
 * from a null result: rocshmem_query_qp_info() returns false both when the
 * environment cannot support the call and when a particular peer/ctx_id names
 * no QP, and those deserve different handling.
 *
 * @return true if rocshmem_query_qp_info() can succeed for some peer/ctx_id.
 *         Host-side, collective-independent, safe to call before init (returns
 *         false).
 */
bool rocshmem_qp_introspect_available();

/**
 * @brief Describe the RC QP connecting this PE to @p peer within @p ctx_id.
 *
 * @param peer   Destination PE.
 * @param ctx_id Context whose QP to describe. Must be > 0; ctx_id <= 0 is
 *               rejected -- see the header comment on why the default-context
 *               QP is unsafe for external WQE posting.
 * @param out    Filled on success; untouched on failure.
 *
 * @return false if the active backend is not the GDA backend, if ctx_id <= 0,
 *         if the provider is unrecognised, or if peer/ctx_id do not name a QP.
 *         Host-side and collective-independent.
 */
bool rocshmem_query_qp_info(int peer, int ctx_id, QpInfo* out);

}  // namespace rocshmem

#endif  // ROCSHMEM_QP_INTROSPECT_HPP
