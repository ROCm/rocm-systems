/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_IB_FAULT_INJECT_H_
#define RCCL_NET_IB_FAULT_INJECT_H_

#ifdef ENABLE_FAULT_INJECTION

#include <stdint.h>

#ifdef __cplusplus
#include "nccl.h"  /* ncclResult_t */
extern "C" {
#else
#include <stdbool.h>
#include "nccl.h"
#endif

/*
 * Test-only per-QP fault injection API for the net-ib CAST transport.
 *
 * Implemented in src/transport/net_ib_cast/p2p.cc (CAST multi-QP path).
 */
#include "net_ib_cast_inspect.h"
#ifdef __cplusplus
static_assert(NCCL_IB_MAX_QPS == 128, "fault injection arrays sized for 128 QPs; update if NCCL_IB_MAX_QPS changes");
#endif

/* ── CAST path (src/transport/net_ib_cast/p2p.cc) ────────────────────── */

/* Set an artificial delay (microseconds) on a specific QP index.
 * qpIdx must be in [0, NCCL_IB_MAX_QPS).
 * Set delayUs=0 to clear the delay. */
ncclResult_t ncclIbCastFaultSetQpDelay(void* sendComm, int qpIdx, uint32_t delayUs);

/* Arm error injection on a specific QP index.
 * When armed, the hook calls IbCastStatsFatalError and then returns
 * ncclSystemError instead of calling wrap_ibv_post_send.
 * Set inject=false to disarm. */
ncclResult_t ncclIbCastFaultSetQpError(void* sendComm, int qpIdx, bool inject);

/* Clear all fault state (delays and errors) on the connection. */
ncclResult_t ncclIbCastFaultClear(void* sendComm);

/* Return the current fatalErrorCount from the connection's stats. */
ncclResult_t ncclIbCastFaultGetFatalCount(void* sendComm, int* out);

/* Drive the QP at the given flat index into IBV_QPS_ERR via ibv_modify_qp.
 * All outstanding and future WRs on this QP will produce WC with
 * IBV_WC_WR_FLUSH_ERR — the status code accepted by the resiliency
 * error whitelist.  Unlike ncclIbCastFaultSetQpError, this injects a
 * real CQE error rather than bypassing ibv_post_send.
 * qpIdx must be in [0, nqps). */
ncclResult_t ncclIbCastFaultDriveQpToError(void* sendComm, int qpIdx);

/* Same as above but for receiver-side QPs. In real port failures both
 * sides see errors simultaneously. */
ncclResult_t ncclIbCastFaultDriveRecvQpToError(void* recvComm, int qpIdx);

/* Thin wrapper exposing IbCastResiliencyCheckErrorNotFatal for unit tests.
 * Returns *isFatal = true if the given WC status code would be treated as
 * unrecoverable (not eligible for failover). */
ncclResult_t ncclIbCastFaultCheckErrorFatal(void* sendComm, int wcStatus, bool* isFatal);

/* ── CAST ops-overload path (src/transport/net_ib_cast/net_ib_ops_fault.cc) ──
 *
 * These arm faults on the shimmed ibv_context ops table installed at
 * wrap_ibv_open_device time, so the fault is applied at the REAL libibverbs
 * call boundary (post_send/post_recv/poll_cq) rather than the pre-call
 * intercept in p2p.cc. qpIdx must be in [0, nqps). */

/* Make the real ibv_post_send on the QP at qpIdx return errnoVal (e.g. EAGAIN,
 * ENOMEM). errnoVal=0 disarms. */
ncclResult_t ncclIbCastFaultOpsSetPostSendError(void* sendComm, int qpIdx, int errnoVal);

/* Make the real ibv_post_recv on the QP at qpIdx return errnoVal. 0 disarms. */
ncclResult_t ncclIbCastFaultOpsSetPostRecvError(void* recvComm, int qpIdx, int errnoVal);

/* Inject an error completion for the QP at qpIdx.
 *  wcStatus      : ibv_wc_status to apply (0 disarms).
 *  injectCount   : completions to corrupt; <0 = unlimited.
 *  injectWhenIdle: if true, synthesize an error WC when poll_cq is idle;
 *                  if false, only rewrite the status of real matching completions. */
ncclResult_t ncclIbCastFaultOpsSetPollCqError(void* comm, int qpIdx, int wcStatus,
                                              int injectCount, bool injectWhenIdle);

/* Clear all ops-overload fault config on the connection (shims stay installed). */
ncclResult_t ncclIbCastFaultOpsClear(void* comm);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ENABLE_FAULT_INJECTION */

#endif /* RCCL_NET_IB_FAULT_INJECT_H_ */
