/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_IB_OPS_FAULT_H_
#define RCCL_NET_IB_OPS_FAULT_H_

#ifdef ENABLE_FAULT_INJECTION

/*
 * Test-only IB ops-overload fault injection for the net-ib CAST transport.
 *
 * After wrap_ibv_open_device() returns an ibv_context, ncclIbOpsFaultInstall()
 * saves the original post_send/post_recv/poll_cq pointers from the context's
 * embedded ops table and replaces them with fault-aware shims. The shims look
 * up per-(context, qp_num) fault config in a registry; when armed they inject an
 * errno return (post_send/post_recv) or an error completion status (poll_cq),
 * otherwise they forward to the saved real op. This exercises the REAL libibverbs
 * call boundary, unlike the pre-call intercept in p2p.cc (faultQpError/Delay).
 *
 * Implemented in src/transport/net_ib_cast/net_ib_ops_fault.cc.
 */

#include "ibvwrap.h"  /* selects ibvcore.h or infiniband/verbs.h per build mode */

#ifdef __cplusplus
extern "C" {
#endif

/* qp_num sentinel: arm/clear applies to every QP on the context. */
#define NCCL_IB_OPS_FAULT_QP_ANY (~0u)

/* wr_id for synthesized idle completions. Must stay outside the receiver's
 * recv-slot and flush wr_id ranges so the receiver rejects it cleanly; a
 * static_assert in p2p.cc ties this to those ranges. */
#define NCCL_IB_OPS_FAULT_SYNTH_WR_ID 0xF00000ull

/* Install fault shims onto ctx's ops table. Returns ncclInvalidArgument if ctx
 * is NULL. Otherwise idempotent and a no-op (returns ncclSuccess without
 * modifying the ops table) if ctx is already installed or any of the three
 * target ops is NULL (logged). */
ncclResult_t ncclIbOpsFaultInstall(struct ibv_context* ctx);

/* Restore original ops and erase the registry entry for ctx. No-op if not
 * installed. Call before wrap_ibv_close_device closes the context. */
ncclResult_t ncclIbOpsFaultRemove(struct ibv_context* ctx);

/* Arm a post_send errno fault on (ctx, qpNum). errnoVal=0 disarms. */
ncclResult_t ncclIbOpsFaultArmPostSend(struct ibv_context* ctx, uint32_t qpNum, int errnoVal);

/* Arm a post_recv errno fault on (ctx, qpNum). errnoVal=0 disarms. */
ncclResult_t ncclIbOpsFaultArmPostRecv(struct ibv_context* ctx, uint32_t qpNum, int errnoVal);

/* Arm a poll_cq error fault on (ctx, qpNum).
 *  wcStatus      : ibv_wc_status to apply (0 = IBV_WC_SUCCESS disarms).
 *  injectCount   : number of completions to corrupt; <0 = unlimited.
 *  injectWhenIdle: if true, synthesize one error WC when the real poll_cq
 *                  returns 0 completions; if false, only rewrite the status of
 *                  real completions whose qp_num matches. */
ncclResult_t ncclIbOpsFaultArmPollCq(struct ibv_context* ctx, uint32_t qpNum,
                                     int wcStatus, int injectCount, bool injectWhenIdle);

/* Clear all ops-fault config for ctx (does not uninstall the shims). */
ncclResult_t ncclIbOpsFaultClear(struct ibv_context* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ENABLE_FAULT_INJECTION */

#endif /* RCCL_NET_IB_OPS_FAULT_H_ */
